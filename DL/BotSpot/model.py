import random
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np


# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")

class Self_Attention(nn.Module):
    # a classical self_attention module as depicted in BERT
    def __init__(self, embed_size, num_head):
        super(Self_Attention, self).__init__()
        self.Q = nn.ModuleList([])
        self.K = nn.ModuleList([])
        self.V = nn.ModuleList([])
        output_size = embed_size // num_head
        self.output_size = output_size
        self.num_head = num_head
        self.final_linear = nn.Linear(output_size*num_head, embed_size)


        for i in range(num_head):
            self.Q.append(nn.Linear(embed_size, output_size))
            self.K.append(nn.Linear(embed_size, output_size))
            self.V.append(nn.Linear(embed_size, output_size))

    def calc_attention(self, X, Q, K, V):
        # print(f"X size: {X.size()}")
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
        # outputs = self.final_linear(outputs).max(dim=1)[0]
        # return outputs


class BotSpot(nn.Module):
    def __init__(self, combin_feats, device_feats, super_device_neibrs_cache=None, cluster_values=None, use_gbm=False, use_gnn=True, use_stratified=False, use_botspot_plus=False, use_self_attn=False, gbm_model=None, device=None, embed_size=16, num_heads=1):
        super(BotSpot,self).__init__()
        if use_gbm:
            leaf_dim = 20
            assert gbm_model is not None
            self.leaf_emb_models = nn.ModuleList()
            for n in range(gbm_model.n_estimators):
                self.leaf_emb_models.append(nn.Embedding(31, leaf_dim)) # 31 is the max depth of decision tree
        self.gbm_best_model = gbm_model
        self.use_gbm = use_gbm
        self.use_gnn = use_gnn
        self.use_self_attn = use_self_attn
        self.use_stratified = use_stratified
        self.self_attn_module = Self_Attention(20, num_heads)
        self.use_botspot_plus = use_botspot_plus
        self.combin_feats = torch.from_numpy(combin_feats).float() # feature matrix for channel-campaign nodes
        self.device_feats = torch.from_numpy(device_feats).float()  # package_name already removed in the device feature matrix
        self.super_device_neibrs_cache = super_device_neibrs_cache
        # self.device_neibrs_cache = device_neibrs_cache
        self.cluster_values = cluster_values
        self.device_split_val = 1 # the first col is ctit for device
        self.combin_split_val = -1 # the last col is channel id
        self.device = device
        self.embed_size = embed_size
        self.combin_feats_dim = combin_feats.shape[1] - 1 + self.embed_size # remove one col and add embeeding_size of 16
        self.device_feats_dim = self.embed_size * (device_feats.shape[1] - 1) + 1 # add one col of ctit, others are embeddings

        # initialze embedding matrix

        combin_id_max = int(combin_feats[:, -1].max() + 1)
        temp = np.max(device_feats[:, 1:], axis=0) + 1
        temp = [int(i) for i in temp]
        lang, plat, os, country, carrier, device_brand, plat_os = temp
        self.channel_id_emb = nn.Embedding(combin_id_max, self.embed_size)
        self.carrier_emb = nn.Embedding(carrier, self.embed_size)
        self.language_emb = nn.Embedding(lang, self.embed_size)
        self.device_brand_emb = nn.Embedding(device_brand, self.embed_size)
        self.plat_os_emb = nn.Embedding(plat_os, self.embed_size)
        self.plat_emb = nn.Embedding(plat, self.embed_size)
        self.os_emb = nn.Embedding(os, self.embed_size)
        self.country_emb = nn.Embedding(country, self.embed_size)

        # device modules if there is no super device convolution
        if not self.use_botspot_plus:
            self.dev_linear1 = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim)) # NOT also used in channel side for convolving device feats
            self.dev_relu1 = nn.ReLU()
            self.dev_dropout1 = nn.Dropout(0.2)
            self.dev_linear2 = nn.Linear(int(0.6 * self.device_feats_dim),int(0.75 * 0.6 * self.device_feats_dim))
            self.dev_relu2 = nn.ReLU()

        self.dev_normal_linear = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim))
        self.dev_bot_linear = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim))
        self.normal_bot_linear1 = nn.Linear(self.combin_feats_dim + int(0.6 * self.device_feats_dim), 64)
        self.normal_bot_linear2 = nn.Linear(64, 1)
        self.normal_relu = nn.ReLU()
        self.bot_relu = nn.ReLU()

        # ideas
        # self.dev_attn_score = nn.Linear(self.device_feats_dim, 100)
        # self.unknown_normal_device_feats = nn.Parameter(torch.zeros(size=(1, self.device_feats_dim)))
        # nn.init.xavier_uniform_(self.unknown_normal_device_feats.data, gain=1.414)
        #
        # self.unknown_bot_device_feats = nn.Parameter(torch.zeros(size=(1, self.device_feats_dim)))
        # nn.init.xavier_uniform_(self.unknown_bot_device_feats.data, gain=1.414)

        # self.dev_normal_attn_linear1 = nn.Linear(self.device_feats_dim, 64)
        # self.dev_normal_attn_linear2 = nn.Linear(64, 1)
        #
        # self.dev_bot_attn_linear1 = nn.Linear(self.device_feats_dim, 64)
        # self.dev_bot_attn_linear2 = nn.Linear(64, 1)
        #
        # self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU()
        # self.dev_trans_linear = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim))
        # self.leaky_relu = nn.LeakyReLU(0.2)
        #

        # channel linear and message passing modules
        self.channel_linear1 = nn.Linear(self.combin_feats_dim, int(0.6 * self.combin_feats_dim))
        self.channel_msg_pass1 = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim))
        fusion_input = int(0.6 * self.combin_feats_dim) + int(0.6 * self.device_feats_dim)
        self.fusion_linear1 = nn.Linear(fusion_input,int(0.6 * fusion_input))
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(0.2)
        fusion_output_dim =  int(0.6 * fusion_input)
        device_output_dim = int(0.75 * 0.6 * self.device_feats_dim)

        if not self.use_gnn:
            concat_input_dim = self.device_feats_dim + self.combin_feats_dim if not self.use_gbm else self.device_feats_dim + self.combin_feats_dim + leaf_dim
            self.concat_linear1 = nn.Linear(concat_input_dim, int(0.6 * concat_input_dim))
            self.concat_relu1 = nn.ReLU()
            self.concat_linear2 = nn.Linear(int(0.6 * concat_input_dim), int(0.5 * 0.6 * concat_input_dim))
            self.concat_relu2 = nn.ReLU()
            self.concat_linear3 = nn.Linear(int(0.5 * 0.6 * concat_input_dim), 1)
        else:
            if not self.use_botspot_plus:
                # concat modules if no botspot++
                concat_input_dim = fusion_output_dim + device_output_dim if not self.use_gbm else fusion_output_dim + device_output_dim + leaf_dim
                self.concat_linear1 = nn.Linear(concat_input_dim, int(0.6 * concat_input_dim))
                self.concat_relu1 = nn.ReLU()
                self.concat_linear2 = nn.Linear(int(0.6 * concat_input_dim), int(0.5 * 0.6 * concat_input_dim))
                self.concat_relu2 = nn.ReLU()
                self.concat_linear3 = nn.Linear(int(0.5 * 0.6 * concat_input_dim), 1)
            else:
                # device side gnn if botspot++ is used
                self.dev_linear1 = nn.Linear(self.device_feats_dim, int(0.6 * self.device_feats_dim)) # NOT also used in channel side for convolving device feats
    #             self.device_msg_passing = nn.Linear(combin_feats_dim,int(0.6*combin_feats_dim))
                in_dim = int(0.6 * self.device_feats_dim) + int(0.6 * self.combin_feats_dim)
                self.sup_dev_fusion_linear1 = nn.Linear(in_dim, int(0.6 * in_dim))
                self.sup_dev_fusion_relu1 = nn.ReLU()
                self.sup_dev_fusion_dropout1 = nn.Dropout(0.2)

                # concat layer for botspot++
                sup_dev_fusion_output_dim = int(0.6 * in_dim)
                concat_input_dim = fusion_output_dim + sup_dev_fusion_output_dim if not self.use_gbm else fusion_output_dim + sup_dev_fusion_output_dim + leaf_dim
                self.concat_linear1 = nn.Linear(concat_input_dim, int(0.6 * concat_input_dim))
                self.concat_relu1 = nn.ReLU()
                self.concat_linear2 = nn.Linear(int(0.6 * concat_input_dim), int(0.5 * 0.6 * concat_input_dim))
                self.concat_relu2 = nn.ReLU()
                self.concat_linear3 = nn.Linear(int(0.5 * 0.6 * concat_input_dim), 1)




    def to_emb(self, arr, *models):
        '''
        :param arr: matrix for holding high-cardinality features, without one-hot encoding
        :param left: channel node if left is True else device node
        :param models: a list of embedding matrices to embed each high-cardinality feature to dense embeddings
        :return: 2-d tensor with dense embeddings for all the high-cardinality features.
        '''

        out_arr = []
        arr = arr.long().to(self.device)
        # device node sparse features
        num_models = len(models)
        for i in range(len(models)):
            tmp = models[i](arr[:, i])
            out_arr.append(tmp)
        return torch.cat(out_arr, dim=1)

    def concat_device_feats(self, dev_feats):  # NEED TO MODIFY IT
        '''
        this method invokes to_emb to embed device categorical features into dense embeddings
        :param dev_feats: normalized device features
        :param more_dev_feats: feature matrix with high-cardinality features
        :return: feature matrix with dense embeddings
        '''
        dev_feats = dev_feats.to(self.device)
        cat_dev_feats = dev_feats[:, self.device_split_val:]
        emb_tensor = self.to_emb(cat_dev_feats, self.language_emb,
                                 self.plat_emb, self.os_emb, self.country_emb, self.carrier_emb,
                                 self.device_brand_emb, self.plat_os_emb)


        dev_emb_feats = torch.cat((dev_feats[:, :self.device_split_val], emb_tensor), dim=1).float().to(self.device)
        return dev_emb_feats

    def concat_combin_feats(self, combin_feats):
        '''
        this method invokes to_emb to embed channel_campaign node's categorical feature into dense embeddings
        similar to concat_device_feats, to add dense embeddings
        '''
        combin_feats = combin_feats.to(self.device)
        emb_tensor = self.to_emb(combin_feats[:, self.combin_split_val:], self.channel_id_emb)
        return torch.cat((combin_feats[:, :self.combin_split_val], emb_tensor), dim=1).float().to(self.device)


    def sample_neibrs_feats(self, edges, sampled_neibrs, train_stage=True):
        """
        input:
        for a minitach of edges, channel_vertices is edge[:,0], device_vertices is edge[:,1]
        this method takes a minibatch of edges outputs:
        1) features for channel_campaign nodes
        2) features for device nodes
        3) neighboring device features for each channel_campaign node
        4) neighboring channel_campaign features for each super device node
        """
        #set number of neighbors for channel and device for different stages
        sample_size = 100 if train_stage else 100
        # sup_dev_sample_size = 20 if train_stage else 50
        # sup_dev_sample_size = 20
        combin_idxes = edges[:, 0]
        device_idxes = edges[:, 1]

        # channel_vertices and device_vertices must be numpy array
        # original features

        sup_dev_neibrs_feats = []

        minibatch_combin_feats = self.concat_combin_feats(self.combin_feats[combin_idxes]).to(self.device)  # shape of (minibatch, feats_num_channel)
        minibatch_device_feats = self.concat_device_feats(self.device_feats[device_idxes]).to(self.device) # shape of (minibatch, feats_device_feats_dim)
        # combin_idxes = combin_idxes.cpu().numpy()
        batch_size = len(sampled_neibrs)
        neibrs_feats = self.device_feats[sampled_neibrs.view(-1)].to(self.device)
        neibrs_feats = self.concat_device_feats(neibrs_feats).view(batch_size, sample_size, -1)

        # if use botspot++, for each device node, retrieve its super device index and get its neighboring channel_campaign node features
        # sup_dev_neibrs_feats = []
        # if self.use_botspot_plus:
        #     cluster_values = edges[:, 2].cpu().numpy()
        #     for combin_idx, device_idx, cluster_value in zip(combin_idxes, device_idxes, cluster_values):
        #         cur_sup_dev_neibrs_feats = self.adj_indice_to_feat_mat_super_device(combin_idx, device_idx, cluster_value, sup_dev_sample_size, train_stage)
        #         sup_dev_neibrs_feats.append(cur_sup_dev_neibrs_feats)
        #     sup_dev_neibrs_feats = torch.cat(sup_dev_neibrs_feats, dim=0).to(self.device)
        #
        #     return minibatch_combin_feats, minibatch_device_feats, neibrs_feats, sup_dev_neibrs_feats

        return minibatch_combin_feats, minibatch_device_feats, neibrs_feats, -1

    def sample_neibrs_feats_stratified(self, edges, sampled_neibrs, train_stage=True):
        """
        input:
        for a minitach of edges, channel_vertices is edge[:,0], device_vertices is edge[:,1]
        this method takes a minibatch of edges outputs:
        1) features for channel_campaign nodes
        2) features for device nodes
        3) neighboring device features for each channel_campaign node
        4) neighboring channel_campaign features for each super device node
        """
        #set number of neighbors for channel and device for different stages
        sample_size = 100 if train_stage else 100
        sup_dev_sample_size = 20 if train_stage else 50
        combin_idxes = edges[:, 0]
        device_idxes = edges[:, 1]

        # channel_vertices and device_vertices must be numpy array
        # original features
        batch_size = len(combin_idxes)
        neibrs_normal_feats = torch.zeros(batch_size * int(sample_size/2), self.device_feats_dim).to(self.device)
        neibrs_bot_feats = torch.zeros(batch_size * int(sample_size/2), self.device_feats_dim).to(self.device)
        sup_dev_neibrs_feats = []

        minibatch_combin_feats = self.concat_combin_feats(self.combin_feats[combin_idxes]).to(self.device)  # shape of (minibatch, feats_num_channel)
        minibatch_device_feats = self.concat_device_feats(self.device_feats[device_idxes]).to(self.device) # shape of (minibatch, feats_device_feats_dim)

        normal_sampled_neibrs = sampled_neibrs[:, :int(sample_size/2)].reshape(-1)
        bot_sampled_neibrs = sampled_neibrs[:, int(sample_size/2):].reshape(-1)
        tmp_normal_sampled_neibrs = normal_sampled_neibrs[normal_sampled_neibrs != -1]
        tmp_feats = self.device_feats[tmp_normal_sampled_neibrs].to(self.device).squeeze(0)
        tmp_bot_sampled_neibrs = bot_sampled_neibrs[bot_sampled_neibrs != -1]
        neibrs_normal_feats[normal_sampled_neibrs != -1] = self.concat_device_feats(tmp_feats)
        #
        # neibrs_normal_feats[normal_sampled_neibrs == -1] = self.unknown_normal_device_feats
        #
        neibrs_normal_feats = neibrs_normal_feats.view(batch_size, int(sample_size/2), -1)

        tmp_feats = self.device_feats[tmp_bot_sampled_neibrs].to(self.device).squeeze(0)
        neibrs_bot_feats[bot_sampled_neibrs != -1] = self.concat_device_feats(tmp_feats)
        #
        # neibrs_bot_feats[bot_sampled_neibrs == -1] = self.unknown_bot_device_feats
        #
        neibrs_bot_feats = neibrs_bot_feats.view(batch_size, int(sample_size/2), -1)

        # if use botspot++, for each device node, retrieve its super device index and get its neighboring channel_campaign node features
        # sup_dev_neibrs_feats = []
        # if self.use_botspot_plus:
        #     cluster_values = edges[:, 2].cpu().numpy()
        #     for combin_idx, device_idx, cluster_value in zip(combin_idxes, device_idxes, cluster_values):
        #         cur_sup_dev_neibrs_feats = self.adj_indice_to_feat_mat_super_device(combin_idx, device_idx, cluster_value, sup_dev_sample_size, train_stage)
        #         sup_dev_neibrs_feats.append(cur_sup_dev_neibrs_feats)
        #     sup_dev_neibrs_feats = torch.cat(sup_dev_neibrs_feats, dim=0).to(self.device)
        #     return minibatch_combin_feats, minibatch_device_feats, neibrs_normal_feats, neibrs_bot_feats, sup_dev_neibrs_feats
        sup_dev_neibrs_feats = []
        if self.use_botspot_plus:
            cluster_values = edges[:, 2].cpu().numpy()
            for device_idx, cluster_value in zip(device_idxes, cluster_values):
                cur_sup_dev_neibrs_feats = self.adj_indice_to_feat_mat_super_device(cluster_value, sup_dev_sample_size)
                sup_dev_neibrs_feats.append(cur_sup_dev_neibrs_feats)
            sup_dev_neibrs_feats = torch.cat(sup_dev_neibrs_feats, dim=0).to(self.device)
            return minibatch_combin_feats, minibatch_device_feats, neibrs_normal_feats, neibrs_bot_feats, sup_dev_neibrs_feats


        return minibatch_combin_feats, minibatch_device_feats, neibrs_normal_feats, neibrs_bot_feats, -1



    def adj_indice_to_feat_mat_super_device(self, cluster_value, sample_size):
        sup_dev_neibr = list(self.super_device_neibrs_cache[cluster_value])

        if len(sup_dev_neibr) > sample_size:
            random.shuffle(sup_dev_neibr)
            c = self.concat_combin_feats(self.combin_feats[sup_dev_neibr[:sample_size]].to(self.device))
        else:
            sup_dev_neibr = np.random.choice(np.asarray(sup_dev_neibr),size = sample_size,replace = True)
            c = self.concat_combin_feats(self.combin_feats[sup_dev_neibr].to(self.device))
        return c.unsqueeze(0)

    # def adj_indice_to_feat_mat_super_device(self, combin_idx, device_idx, cluster_value, sample_size, train_stage):
    #     if train_stage:
    #         ori_dev_neibrs = self.device_neibrs_cache[device_idx.item()]
    #     else:
    #         ori_dev_neibrs = set([combin_idx.item()])
    #     new_dev_neibrs = list(self.super_device_neibrs_cache[cluster_value] - ori_dev_neibrs)
    #     ori_dev_neibrs = list(ori_dev_neibrs)
    #
    #     if len(ori_dev_neibrs) >= sample_size:
    #         random.shuffle(ori_dev_neibrs)
    #         c = self.concat_combin_feats(self.combin_feats[ori_dev_neibrs[:sample_size]].to(self.device))
    #     else:
    #         tmp = sample_size - len(ori_dev_neibrs)
    #         if len(new_dev_neibrs) == 0:
    #             dev_neibrs = np.random.choice(np.asarray(ori_dev_neibrs), size=sample_size, replace=True)
    #         elif len(new_dev_neibrs) < tmp:
    #             dev_neibrs = ori_dev_neibrs + list(np.random.choice(np.asarray(new_dev_neibrs), size=tmp, replace=True))
    #         else:
    #             random.shuffle(new_dev_neibrs)
    #             dev_neibrs = ori_dev_neibrs + new_dev_neibrs[:tmp]
    #         c = self.concat_combin_feats(self.combin_feats[dev_neibrs]).to(self.device)
    #
    #     return c.unsqueeze(0)


    def get_leaf_from_light_gbm(self, left_vertices, right_vertices, use_self_attn=False):
        # get leaf indices from gbm model and embed into dense matrix
        output_leaf_emb = []
        chan_data = self.combin_feats[left_vertices]
        dev_data = self.device_feats[right_vertices]
        try:
            edge_data = np.hstack((chan_data, dev_data))
        except:
            edge_data = torch.cat((chan_data, dev_data),
                                  dim=1)  # edge feature is the concatenation of channel_node and device_node
            edge_data = edge_data.cpu().numpy()
        # N = len(left_vertices)
        if len(edge_data.shape)==1:
            edge_data = edge_data.reshape((1, -1))
        pred_leaf = self.gbm_best_model.predict_proba(edge_data, pred_leaf=True)
        pred_leaf = torch.from_numpy(pred_leaf).long().to(self.device)

        for i in range(pred_leaf.shape[1]):
            # print (self.leaf_emb_models[i](pred_leaf[:, i]).shape)
            output_leaf_emb.append(self.leaf_emb_models[i](pred_leaf[:, i]).unsqueeze(1))
            # ret = torch.cat(output_leaf_emb, dim=1).to(self.device)  # leaf node concatenation
        if not use_self_attn:
            ret = torch.cat(output_leaf_emb, dim=1).mean(axis=1).to(self.device)  # leaf node mean pooling
            return ret
        else:
            ret = torch.cat(output_leaf_emb, dim=1).to(self.device)
            out = self.self_attn_module(ret)
            return out

    def forward(self, edges, sampled_neibrs, train_stage=True):
        if not self.use_gnn:
            combin_idxes = edges[:, 0]
            device_idxes = edges[:, 1]
            minibatch_combin_feats = self.concat_combin_feats(self.combin_feats[combin_idxes]).to(self.device)  # shape of (minibatch, feats_num_channel)
            minibatch_device_feats = self.concat_device_feats(self.device_feats[device_idxes]).to(self.device) # shape of (minibatch, feats_device_feats_dim)
            if not self.use_gbm:
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((minibatch_combin_feats, minibatch_device_feats), dim=1))))))
            else:
                leaf_out = self.get_leaf_from_light_gbm(edges[:,0], edges[:,1], self.use_self_attn)
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((minibatch_combin_feats, minibatch_device_feats, leaf_out), dim=1))))))
            return torch.sigmoid(h)


        if not self.use_stratified:
            minibatch_combin_feats, minibatch_device_feats, neibr_feats_tensor, sup_dev_neibrs_feats = self.sample_neibrs_feats(edges, sampled_neibrs, train_stage)
            # print(f"neibr_feats_tensor size: {neibr_feats_tensor.size()}")
            dev_conv = self.dev_linear1(neibr_feats_tensor).mean(dim=1) # share dev_linear1
        else:
            minibatch_combin_feats, minibatch_device_feats, neibr_normal_feats_tensor, neibr_bot_feats_tensor, sup_dev_neibrs_feats = self.sample_neibrs_feats_stratified(edges, sampled_neibrs, train_stage)
            dev_normal_conv = self.dev_normal_linear(torch.mean(neibr_normal_feats_tensor, dim=1, keepdim=True).squeeze(1))
            dev_bot_conv = self.dev_bot_linear(torch.mean(neibr_bot_feats_tensor, dim=1, keepdim=True).squeeze(1))

            normal_concat_feats = torch.cat((minibatch_combin_feats, dev_normal_conv), dim=1)
            bot_concat_feats = torch.cat((minibatch_combin_feats, dev_bot_conv), dim=1)
            normal_score = self.normal_bot_linear2(self.normal_relu(self.normal_bot_linear1(normal_concat_feats)))
            bot_score = self.normal_bot_linear2(self.bot_relu(self.normal_bot_linear1(bot_concat_feats)))
            normal_bot_score = F.softmax(torch.cat((normal_score, bot_score), dim=1), dim=1)
            normal_bot_score = normal_bot_score.unsqueeze(1)
            dev_bot_conv = dev_bot_conv.unsqueeze(-1)
            dev_normal_conv = dev_normal_conv.unsqueeze(-1)
            stack_feats = torch.cat((dev_normal_conv, dev_bot_conv), dim=-1)
            dev_conv = torch.mean(stack_feats * normal_bot_score, dim=-1)  # this is the device feats after linear combination


        # forward device feats
        if not self.use_botspot_plus:
            device_out = self.dev_relu2(self.dev_linear2(self.dev_dropout1(self.dev_relu1(self.dev_linear1(minibatch_device_feats)))))

            channel_conv = self.channel_linear1(minibatch_combin_feats)

            # dev_conv = self.dev_linear1(neibr_feats_tensor).mean(dim=1) # share dev_linear1
            fuse_conv = self.fusion_dropout1(self.fusion_relu1(self.fusion_linear1(torch.cat((channel_conv, dev_conv), dim=1))))
            if not self.use_gbm:
                h1 = self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv, device_out), dim=1))))
                h = self.concat_linear3(self.concat_relu2(h1))
            else:
                leaf_out = self.get_leaf_from_light_gbm(edges[:,0], edges[:,1])
                h1 = self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv, device_out, leaf_out), dim=1))))
                h = self.concat_linear3(self.concat_relu2(h1))
            return torch.sigmoid(h)
        else:
            channel_conv = self.channel_linear1(minibatch_combin_feats)
            # dev_conv = self.dev_linear1(neibr_feats_tensor).mean(dim=1) # share dev_linear1
            fuse_conv = self.fusion_dropout1(self.fusion_relu1(self.fusion_linear1(torch.cat((channel_conv, dev_conv),dim=1))))
            # device side conv:
            sup_dev_conv = self.dev_linear1(minibatch_device_feats)
            sup_channel_conv = self.channel_linear1(sup_dev_neibrs_feats).mean(dim=1)
            sup_fuse_conv = self.sup_dev_fusion_dropout1(self.sup_dev_fusion_relu1(self.sup_dev_fusion_linear1(torch.cat((sup_channel_conv, sup_dev_conv), dim=1))))

            if not self.use_gbm:
                h1 = self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv, sup_fuse_conv), dim=1))))
                h = self.concat_linear3(self.concat_relu2(h1))
            else:
                leaf_out = self.get_leaf_from_light_gbm(edges[:, 0], edges[:, 1], self.use_self_attn)
                h1 = self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv, sup_fuse_conv, leaf_out), dim=1))))
                h = self.concat_linear3(self.concat_relu2(h1))

            return torch.sigmoid(h)
