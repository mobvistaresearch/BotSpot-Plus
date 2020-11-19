from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from  torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.preprocessing import *


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Mean_agg(torch.nn.Module):
    '''
    Mean aggregator, following the same logic as in GraphSage
    input x: node features with feature dimensions-${input_dim}  to be averaged and aggregated.
    return: aggregated features of ${hidden_dims} dimensions
    '''

    def __init__(self, input_dim, hidden_dim):
        super(Mean_agg, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h = torch.mean(x, dim=0, keepdim=True)
        h = self.linear(h)
        return h


class Pooling_agg(torch.nn.Module):
    '''
    Pooling aggregator
    input x: node features with feature dimensions-${input_dim}  to be averaged and aggregated.
    return: aggregated features of ${hidden_dims} dimensions
    '''

    def __init__(self, input_dim, hidden_dim):
        super(Pooling_agg, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h, _ = torch.max(x, dim=0, keepdim=True)
        h = self.linear(h)
        return h


class Lstm_agg(torch.nn.Module):
    """
    Aggregation layer using LSTM module
    input_dim:int. dimension for input node
    hidden_dim:int. dimension for hidden output
    Return:
    aggregated features of ${hidden_dims} dimensions
    """

    def __init__(self, input_dim, hidden_dim, num_layer=1):
        super(Lstm_agg, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layer)

    def forward(self, x):
        x = x.unsqueeze(0)
        h = self.lstm(x)
        return h[1][0][0]


class GAT_agg(torch.nn.Module):
    '''
    Graph Attention Aggregation
    linear_dev: linear transformation for device nodes
    linear_channel: linear transformation for channel nodes
    attn: attention layer for weighted combination
    '''

    def __init__(self, input_channel_dim, input_dev_dim, hidden_dim, num_layer=1):
        super(GAT_agg, self).__init__()
        self.linear_dev = nn.Linear(input_dev_dim, hidden_dim)
        self.linear_channel = nn.Linear(input_channel_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_self):
        x = self.linear_dev(x)
        h_self = self.linear_channel(h_self)
        attn_coef = torch.matmul(self.attn(x), h_self.t().contiguous())
        attn_coef = torch.softmax(attn_coef.t().contiguous(), dim=-1)
        out = torch.matmul(attn_coef, x)
        return out


class Msg_out(torch.nn.Module):
    '''
    nn module for node output after aggregating its neighboring nodes.
    input
    out: aggregated node features after aggregation layer
    x: self node in the lower layer in the graph, with linear transformation and then concat with aggregated node features.
    '''

    def __init__(self, self_dim, hidden_dim):
        super(Msg_out, self).__init__()
        self.linear = nn.Linear(self_dim, hidden_dim)
        self.linear_cluster = nn.Linear(self_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, out, x):
        h = self.linear(x)
        h = self.relu(torch.cat((out, h), dim=1))
        return h


class Graph_Conv(torch.nn.Module):
    '''
    A tailored graphsage model that aggregate bot node features and normal node features separately and obtain the final node embedding
    using convex combination of the two kind of node features.
    '''

    def __init__(self, channel_feats, device_feats, dims, edge_index_train, edge_index_test, agg='mean',
                 channel_split_val=0, device_split_val=0,
                 split_bots_normal=True, use_hierachy=False, device_info_list=None,component = None):
        '''
        :param dims: List[Int], holding input dimensions and hidden dimensions for source node and neighboring node
        :param num_hop: num_hop=1 if only aggregating neighborin nodes for once, if num_hop>1, aggregating nodes revursively.
        :param num_neighbor: maximum number of neighbors sampled when performing aggregation
        :param edge_index: Tuple:: <source node, dst node, label>
        :param agg: agg method, using Mean Aggregation as default
        :param split_num: indicate the rows for which channel node is less than split_num and device node otherwise.
        :param reverse: reverse--> device node to channel node; not reverse: channel node --> device node
        :param data: feature matrix for device node and channel node
        :param is_left: True if channel node else device node
        :param data_more_feats: feature matrix with more sparse high-cardinality categorical features
        :param channel_id_emb: pytorch embedding matrix for channel id
        :param carrier_emb: pytorch embedding matrix for service carrier
        :param language_emb: pytorch embedding matrix for device language
        :param device_brand_emb: pytorch embedding matrix for device brand
        :param device_name_emb: pytorch embedding matrix for device name
        :param plat_os_emb:pytorch embedding matrix for platform os-version combinations
        '''

        super(Graph_Conv, self).__init__()
        self.channel_feats = torch.from_numpy(channel_feats).float()
        self.device_feats = torch.from_numpy(device_feats).float()

        self.agg = agg
        self.edge_index = np.vstack((edge_index_train, edge_index_test))
        self.device2gr_channels = self.preprocess_device_to_channel_group(device_info_list,
                                                                          self.edge_index)  # this include devices for test dataset
        self.edge_index_train_normal = edge_index_train[edge_index_train[:, 2] == 0]
        self.edge_index_train_bots = edge_index_train[edge_index_train[:, 2] == 1]

        # do some preprocessing work here
        try:
            with open('components/components_1021.pickle','rb') as f:
                self.components = pickle.load(f)
        except:
            print('component pickle file not in disk,need to process')
            # self.components = self.calc_components_for_super_device(
            #     self.device2gr_channels,date = '1021')  # component for device node, to reduce training time for super device aggregation

        self.super_device_cache = {}  # used for cache channel_conv for super_device conv
        self.channel_cache = {}  # used for cache channel node features

        # need to fill various id emb size values!!!!!!
        emb_size = 16
        channel_id_max = int(channel_feats[:, channel_split_val].max() + 1)
        temp = np.max(device_feats[:, device_split_val:], axis=0) + 1
        temp = [int(i) for i in temp]
        lang, plat, os, country, carrier, device_brand, plat_os = temp  # bypass install city, be careful
        self.channel_id_emb = nn.Embedding(channel_id_max, emb_size)
        self.carrier_emb = nn.Embedding(carrier, emb_size)
        self.language_emb = nn.Embedding(lang, emb_size)
        self.device_brand_emb = nn.Embedding(device_brand, emb_size)
        self.plat_os_emb = nn.Embedding(plat_os, emb_size)
        self.plat_emb = nn.Embedding(plat, emb_size)
        self.os_emb = nn.Embedding(os, emb_size)
        self.country_emb = nn.Embedding(country, emb_size)
        self.split_bots_normal = split_bots_normal
        self.use_hierchy = use_hierachy
        self.channel_split_val = channel_split_val
        self.device_split_val = device_split_val
        input1_dim, input2_dim, hidden1_dim, hidden2_dim = dims[0], dims[1], dims[2], dims[3]
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        # input1 & hidden1: ad_channel, input2 & hidden2:device
        self.mean_agg = Mean_agg(input2_dim, hidden2_dim)
        self.msg_out = Msg_out(input1_dim, hidden1_dim)
        self.mean_agg_for_device_side = Mean_agg(input1_dim, hidden1_dim)
        self.msg_out_device = Msg_out(input2_dim, hidden2_dim)
        # in case if not using graph convolution in device side, only MLP for device nodes
        self.device_linear = nn.Linear(input2_dim, int(0.6 * input2_dim))
        self.device_relu = nn.ReLU()
        self.device_linear2 = nn.Linear(int(0.6 * input2_dim), hidden2_dim)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.attn = nn.Linear(hidden2_dim + input1_dim,
                              84)  # 20 channel-campaign node features + 32 embedding feature for channel-id
        self.relu = nn.ReLU()
        self.attn_layer2 = nn.Linear(84, 1)

    def calc_components_for_super_device(self, x, date='04xx'):
        # x: device2gr_channels
        N = len(x)
        marked = [False] * N
        components = [-1] * N
        c = 0
        for i in range(N):
            if i % 100 == 0:
                print(f'get components progess:{i} out of total:{N}')
            if marked[i]:
                continue
            marked[i] = True
            c += 1
            components[i] = c
            v = x[i]
            anchor = set(v)
            for j in range(i + 1, N):
                if marked[j]:
                    continue
                v = set(x[j])
                if anchor == v:
                    components[j] = c
                    marked[j] = True
        # with open(f'components/components_{date}.pickle', 'wb') as f:
        #     pickle.dump(components, f)
        return components

    def to_emb(self, arr, *models):
        '''
        :param arr: matrix for holding high-cardinality features, without one-hot encoding
        :param left: channel node if left is True else device node
        :param models: a list of embedding matrices to embed each high-cardinality feature to dense embeddings
        :return: 2-d tensor with dense embeddings for all the high-cardinality features.
        '''

        out_arr = []
        arr = arr.long().to(Device)
        # device node sparse features

        # N = arr.shape[0]
        num_models = len(models)
        for i in range(len(models)):
            # if num_models > 2 and i == 4:  # bypass install city, hardcoded
            #     continueed
            #             print (i,models[i])
            out_arr.append(models[i](arr[:, i]))
        return torch.cat(out_arr, dim=1)

    def concat_device_feats(self, dev_feats):  # NEED TO MODIFY IT
        '''
        :param dev_feats: normalized device features
        :param more_dev_feats: feature matrix with high-cardinality features
        :return: feature matrix with dense embeddings
        '''

        cat_dev_feats = dev_feats[:, self.device_split_val:].to(Device)
        emb_tensor = self.to_emb(cat_dev_feats, self.language_emb,
                                 self.plat_emb, self.os_emb, self.country_emb, self.carrier_emb,
                                 self.device_brand_emb, self.plat_os_emb)

        dev_emb_feats = torch.cat((dev_feats[:, :self.device_split_val].to(Device), emb_tensor), dim=1)
        return dev_emb_feats

    def concat_channel_feats(self, chan_feats):
        '''
        similar to concat_device_feats, to add dense embeddings
        '''
        emb_tensor = self.to_emb(chan_feats[:, self.channel_split_val:], self.channel_id_emb)
        return torch.cat((chan_feats[:, :self.channel_split_val].to(Device), emb_tensor), dim=1)

    def forward(self, vertices, neibr_vertices):  # x:->List of vertices
        '''
        :param vertice: ad channel vertices
        :param nei_vertice:  List of corresponding device vertices
        :return: extracted node features for every node in x
        '''
        # first,we extract representations for channel nodes
        channel_vertex_feats = self.concat_channel_feats(self.channel_feats[vertices].to(Device))
        if not self.split_bots_normal:  # use vannila message passing mechanism
            channel_conv_device_feats = []
            for idx, v in enumerate(vertices):
                channel_conv_device_feat = self.device_to_channel_conv(v, self.edge_index, self.channel_feats,
                                                                       self.device_feats)
                channel_conv_device_feats.append(channel_conv_device_feat)
            gcn_feats = torch.cat(channel_conv_device_feats, dim=0)
            channel_representation = self.msg_out(gcn_feats, channel_vertex_feats)
        else:  # use tailored message passing scheme
            channel_conv_normal_device_feats = []
            channel_conv_bot_device_feats = []
            for idx, v in enumerate(vertices):
                channel_conv_bot_device_feat = self.device_to_channel_conv(v, self.edge_index_train_bots,
                                                                           self.channel_feats,
                                                                           self.device_feats,
                                                                           neighbor=neibr_vertices[idx])
                channel_conv_normal_device_feat = self.device_to_channel_conv(v, self.edge_index_train_normal,
                                                                              self.channel_feats,
                                                                              self.device_feats,
                                                                              neighbor=neibr_vertices[idx])
                channel_conv_bot_device_feats.append(channel_conv_bot_device_feat)
                channel_conv_normal_device_feats.append(channel_conv_normal_device_feat)
            bots_feats = torch.cat(channel_conv_bot_device_feats, dim=0)
            normal_feats = torch.cat(channel_conv_normal_device_feats, dim=0)
            # start attention mechanism
            concat_feats = torch.cat((channel_vertex_feats, bots_feats), dim=1)
            bot_score = self.attn_layer2(self.relu(self.attn(concat_feats)))
            normal_concat_feats = torch.cat((channel_vertex_feats, normal_feats), dim=1)
            normal_score = self.attn_layer2(self.relu(self.attn(normal_concat_feats)))
            score = F.softmax(torch.cat((normal_score, bot_score), dim=1), dim=1)
            score = score.unsqueeze(1)
            bots_feats = bots_feats.unsqueeze(-1)
            normal_feats = normal_feats.unsqueeze(-1)
            stack_feats = torch.cat((normal_feats, bots_feats), dim=-1)
            hybrid_feats = torch.mean(stack_feats * score, dim=-1)  # this is the device feats after linear combination
            channel_representation = self.msg_out(hybrid_feats, channel_vertex_feats)
        # below is to extract device feats
        if not self.use_hierchy:
            device_vertex_feats = self.concat_device_feats(self.device_feats[neibr_vertices])
            device_representation = self.device_linear2(self.device_relu(self.device_linear(device_vertex_feats)))
        else:
            device_vertex_feats = self.concat_device_feats(self.device_feats[neibr_vertices])
            device_conv_feats_from_channel = []
            #             channel_emb_feats = self.concat_channel_feats(self.channel_feats)
            for v in neibr_vertices:
                v = v.item()
                channels = self.device2gr_channels[v]
                channels = torch.tensor(channels).long()
                device_conv_feats_from_channel.append(
                    self.channel_to_super_device_conv(v, self.edge_index, self.channel_feats,
                                                      self.device_feats, self.device2gr_channels))

            device_conv_feats = torch.cat(device_conv_feats_from_channel, dim=0)
            #             print(device_vertex_feats.shape, device_conv_feats.shape)
            device_representation = self.msg_out_device(device_conv_feats, device_vertex_feats)
            # device_representation = torch.cat((device_vertex_feats,device_conv_feats),dim=1)
        return channel_representation, device_representation

    def device_to_channel_conv(self, vertice, edge_index, channel_feats_matrix, device_feats_matrix, neighbor=None):
        # print (num_hop)
        '''
        :param vertice: the vertice in the graph as the root node for feature extraction
        '''
        try:
            vertice = vertice.item()
        except:
            pass
        if np.sum(edge_index[:, 0] == vertice) > 30 and vertice in self.channel_cache:
            return self.channel_cache[vertice]
        # sample neighbors from edge_index for vertex {vertice}
        if self.split_bots_normal:
            neighbors = self.sample_neighbor(edge_index, vertice=vertice, num_samples=40, neighbor=neighbor)
        else:
            neighbors = self.sample_neighbor(edge_index, vertice=vertice, num_samples=40, neighbor=None)
        # print (neighbors)
        if isinstance(neighbors, int) or len(neighbors) == 0:
            return torch.zeros(1, self.hidden2_dim).to(Device)

        neighbor_device_feats = self.concat_device_feats(device_feats_matrix[neighbors])  # !!!!!!
        #         print (neighbor_device_feats.shape)
        #         print (self.mean_agg)
        val = self.mean_agg(neighbor_device_feats)
        self.channel_cache[vertice] = val
        return val

    def preprocess_device_to_channel_group(self, device_info_list, edge_index):
        # device_info_list: [0,1,1,2,90,18,...]
        # index corresponds to device, value corresonds to some metric, such as package id ,or some other entity
        device_info_arr = np.asarray(device_info_list)
        uniq_values = np.unique(device_info_arr)
        device2channels = {}
        print('total package number:', len(uniq_values))
        idx = 0
        for v in uniq_values:
            idx += 1
            if idx % 50 == 0:
                print('current idx of packahe names:', idx)
            device_indices = np.where(device_info_arr == v)[0]
            mask = np.isin(edge_index[:, 1], device_indices)
            grouped_channels = edge_index[mask][:, 0]
            grouped_channels = tuple(set(grouped_channels))
            for d in device_indices:
                device2channels[d] = grouped_channels
        return device2channels  # finally, return the dictionary where key is device_id, value is grouped channel_ids or channel_campaign_ids

    def channel_to_super_device_conv(self, vertice, edge_index, channel_feats_matrix, device_feats_matrix,
                                     device2channels):
        # print (num_hop)
        '''
        :param vertice: the vertice in the graph as the root node for feature extraction
        edge_index: edge_index should still be in format of (channel,device ,label)
        '''
        try:
            vertice = vertice.item()  # vertice should be DEVICE vertice
        except:
            pass
        c = self.components[vertice]
        if c in self.super_device_cache:
            return self.super_device_cache[c]

        # sample neighbors from edge_index for vertex {vertice}

        neighbors = device2channels[vertice]
        neighbors = torch.tensor(list(neighbors)).long()
        if len(neighbors) > 30:
            neighbors = neighbors[torch.randperm(len(neighbors))][:30]

        neighbor_channel_feats = self.concat_channel_feats(channel_feats_matrix[neighbors])  # !!!!!!
        val = self.mean_agg_for_device_side(neighbor_channel_feats)
        self.super_device_cache[c] = val
        return val

    def sample_neighbor(self, edges, num_samples=40, vertice=0, neighbor=None):
        try:
            vertice = vertice.item()
        except:
            pass
        try:
            neighbor = neighbor.item()
        except:
            pass

        nei = edges[edges[:, 0] == vertice][:, 1]
        if len(nei) == 0:
            return -1
        if len(nei) == 1 and neighbor in nei: return -1

        vertices = np.random.choice(nei, size=num_samples)
        if neighbor is not None:
            try:
                vertices = np.delete(vertices, np.where(vertices == neighbor))
            except:
                pass
            # print ('channel neighbor num:',len(nei))
            return vertices
        else:
            return vertices


class Self_Attention(nn.Module):
    def __init__(self, emb_size, num_head):
        super(Self_Attention, self).__init__()
        self.Q = nn.ModuleList([])
        self.K = nn.ModuleList([])
        self.V = nn.ModuleList([])
        output_size = emb_size // num_head
        self.output_size = output_size
        self.num_head = num_head
        self.final_linear = nn.Linear(emb_size, emb_size)

        for i in range(num_head):
            self.Q.append(nn.Linear(emb_size, output_size))
            self.K.append(nn.Linear(emb_size, output_size))
            self.V.append(nn.Linear(emb_size, output_size))

    def calc_attention(self, X, Q, K, V):
        query = Q(X)
        key = K(X)
        value = V(X)
        key_ = key.transpose(2, 1).contiguous()
        attn_weights = torch.softmax(torch.bmm(query, key_) / math.sqrt(self.output_size), dim=-1)
        output = torch.bmm(attn_weights, value)
        return output

    def forward(self, X):
        outputs = []
        for i in range(self.num_head):
            q, k, v = self.Q[i], self.K[i], self.V[i]
            out = self.calc_attention(X, q, k, v)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=-1)
        return self.final_linear(outputs).mean(dim=1)


class BotSpot(torch.nn.Module):
    def __init__(self, edge_index_train,edge_index_test, channel_feats, device_feats, channel_split_val=-1, device_split_val=1,
                 gbm_best_model=None, agg='mean',
                 split_bots_normal=True, use_hierarchy=False, device_package=None,
                 channel_feats_unnormed=None, device_feats_unnormed=None,component = None):
        super(BotSpot, self).__init__()
        leaf_dim = 20
        self.num_gbm_trees = 200
        try:
            edge_index = edge_index.numpy()
        except:
            pass
        self.self_attn_module = Self_Attention(32, 2)
#         self.edge_index = edge_index
#         test_device_indices = self.get_test_devices_per_date(device_time_list)
#         mask = np.isin(edge_index[:, 1], np.asarray(test_device_indices))
        self.edge_index_train = edge_index_train
        self.edge_index_test = edge_index_test
        # -------------------------------------------------------------------
        self.gbm_best_model = gbm_best_model
        self.channel_feats = channel_feats
        self.device_feats = device_feats
        self.channel_feats_unnormed = channel_feats_unnormed
        self.device_feats_unnormed = device_feats_unnormed

        # construct embedding matrices for leaf embedding
        if gbm_best_model is not None:
            self.leaf_emb_models = nn.ModuleList()
            for n in range(gbm_best_model.n_estimators):
                self.leaf_emb_models.append(nn.Embedding(31, leaf_dim))

        # dims: [local_dims,remote_dims,local_hidden_dims,remote_hidden_dims]
        # graph_model_rev is the
        # dims[0]--> channel-campaign node features:20 node features + 32 for channel-id embeedings. device node-->22 node features + 3 embedding features, each of 32 dimensions and 1 embedding feature of 8 dimensions.
        self.graph_model = Graph_Conv(channel_feats, device_feats, [130 + 16, 1 + 7 * 16, 80, 64],
                                      self.edge_index_train, self.edge_index_test, agg='mean',
                                      channel_split_val=channel_split_val, device_split_val=device_split_val,
                                      split_bots_normal=split_bots_normal,
                                      use_hierachy=use_hierarchy, device_info_list=device_package,component = component)

        if self.gbm_best_model is not None:
            if not use_hierarchy:
                self.fusion_layer1 = nn.Linear(64 + 80 + 64 + 20, 128)  # channel_repre + device_repre + leaf_emb_size
                self.fusion_relu1 = nn.ReLU()
                self.fusion_dropout1 = nn.Dropout(0.1)
                self.fusion_layer2 = nn.Linear(128, 64)
                self.fusion_relu2 = nn.ReLU()
                self.fusion_layer3 = nn.Linear(64, 1)
            else:
                self.fusion_layer1 = nn.Linear(64 + 80 + 64 + 80 + 20,
                                               184)  # channel_repre + device_repre(with super node convolving) + leaf_emb_size
                self.fusion_relu1 = nn.ReLU()
                self.fusion_dropout1 = nn.Dropout(0.1)
                self.fusion_layer2 = nn.Linear(184, 96)
                self.fusion_relu2 = nn.ReLU()
                self.fusion_layer3 = nn.Linear(96, 1)
        else:
            if not use_hierarchy:
                self.fusion_layer1 = nn.Linear(64 + 80 + 64, 128)  # channel_repre + device_repre
                self.fusion_relu1 = nn.ReLU()
                self.fusion_dropout1 = nn.Dropout(0.1)
                self.fusion_layer2 = nn.Linear(128, 64)
                self.fusion_relu2 = nn.ReLU()
                self.fusion_layer3 = nn.Linear(64, 1)
            else:
                self.fusion_layer1 = nn.Linear(64 + 80 + 64 + 80,
                                               184)  # channel_repre + device_repre(with super node convolving)
                self.fusion_relu1 = nn.ReLU()
                self.fusion_dropout1 = nn.Dropout(0.1)
                self.fusion_layer2 = nn.Linear(184, 96)
                self.fusion_relu2 = nn.ReLU()
                self.fusion_layer3 = nn.Linear(96, 1)

        # self.left_linear = nn.Linear(16 + 28, 24)
        # self.linear_top = nn.Linear(24+24+24, 48)

    def unixtime_to_date_time(self, unix_time_str):
        import datetime
        unix_time = int(unix_time_str)
        value = datetime.datetime.fromtimestamp(unix_time)
        return value

    def get_test_devices_per_date(self, device_time_list):
        ret = sorted([(index, self.unixtime_to_date_time(int(i))) for index, i in enumerate(device_time_list)],
                     key=lambda x: x[1])[::-1]
        last_date = ret[0][1].day
        device_indices_test = []
        for i, j in ret:
            if j.day == last_date:
                device_indices_test.append(i)
            else:
                break
        return device_indices_test

    def forward(self, edges, evaluate=False):
        channel_vertices = edges[:, 0]
        device_vertices = edges[:, 1]
        labels = edges[:, 2]
        channel_repre, device_repre = self.graph_model(channel_vertices, device_vertices)

        # if no gbm models, use botspot-local only
        if self.gbm_best_model is None:  # if no gbm models, it's botspot-local
            h = torch.cat((channel_repre, device_repre), dim=1)
            h = torch.sigmoid(
                self.fusion_layer3(self.fusion_relu2(self.fusion_layer2(self.fusion_relu1(self.fusion_layer1(h))))))
            return h, labels
        else:
            # use gbm_mdoels for probability of a sample as meta information
            gbm_outputs = self.get_leaf_from_light_gbm(channel_vertices, device_vertices)
            h = torch.cat((channel_repre, device_repre, gbm_outputs), dim=1)
            if not evaluate:
                h = torch.sigmoid(
                    self.fusion_layer3(self.fusion_relu2(
                        self.fusion_layer2(self.fusion_dropout1(self.fusion_relu1(self.fusion_layer1(h)))))))
                return h, labels
            else:
                h = self.fusion_layer2(self.fusion_dropout1(self.fusion_relu1(self.fusion_layer1(h))))
                return h, labels

    def get_leaf_from_light_gbm(self, left_vertices, right_vertices, use_self_attn=False):
        # get leaf indices from gbm model and embed into dense matrix
        output_leaf_emb = []
        chan_data = self.channel_feats_unnormed[left_vertices]
        dev_data = self.device_feats_unnormed[right_vertices]
        try:
            edge_data = np.hstack((chan_data, dev_data))
        except:
            edge_data = torch.cat((chan_data, dev_data),
                                  dim=1)  # edge feature is the concatenation of channel_node and device_node
            edge_data = edge_data.cpu().numpy()
        # N = len(left_vertices)
        if len(edge_data.shape)==1:
            edge_data = edge_data.reshape((1,-1))
        pred_leaf = self.gbm_best_model.predict_proba(edge_data, pred_leaf=True)
        pred_leaf = torch.from_numpy(pred_leaf).long().to(Device)

        for i in range(pred_leaf.shape[1]):
            # print (self.leaf_emb_models[i](pred_leaf[:, i]).shape)
            output_leaf_emb.append(self.leaf_emb_models[i](pred_leaf[:, i]).unsqueeze(1))
            # ret = torch.cat(output_leaf_emb, dim=1).to(Device)  # leaf node concatenation
        if not use_self_attn:
            ret = torch.cat(output_leaf_emb, dim=1).mean(axis=1).to(Device)  # leaf node mean pooling
            return ret
        else:
            ret = torch.cat(output_leaf_emb, dim=1).to(Device)
            out = self.self_attn_module(ret)
            return out
