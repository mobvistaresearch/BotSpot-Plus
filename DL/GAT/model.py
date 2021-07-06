import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import time



class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, combin_feat_dim, device_feat_dim, out_dim, alpha, is_last_layer=False):
        super(GraphAttentionLayer, self).__init__()
        self.combin_feat_dim = combin_feat_dim
        self.device_feat_dim = device_feat_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.is_last_layer = is_last_layer
        self.combin_attn_fc = nn.Linear(self.combin_feat_dim, out_dim)
        self.device_attn_fc = nn.Linear(self.device_feat_dim, out_dim)

        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.fc = nn.Linear(2 * out_dim, 1)
        self.elu = nn.ELU()

    def forward(self, combin_feats, combin_neibrs_feats):
        # apply for loop due to insufficient of gpu memory
        outs = []
        cnt = 0
        for cur_combin_feats, cur_combin_neibrs_feats in zip(combin_feats, combin_neibrs_feats):
            num_neibrs = cur_combin_neibrs_feats.size()[0]
            h_combin = self.combin_attn_fc(cur_combin_feats)
            h_neibrs = self.device_attn_fc(cur_combin_neibrs_feats)
            h = torch.cat([h_combin.repeat(1, num_neibrs).view(num_neibrs, -1), h_neibrs]).view(-1, 2 * self.out_dim)
            e = self.leaky_relu(self.fc(h)).view(1, num_neibrs)
            normalized_scores = self.softmax(e)
            out = torch.matmul(normalized_scores, h_neibrs)
            if self.is_last_layer:
                outs.append(out)
            else:
                outs.append(self.elu(out))
            cnt += 1
            # torch.cuda.empty_cache()
        outs = torch.cat(outs, dim=0)
        return outs




class GAT(torch.nn.Module):
    def __init__(self, edge_index_train, combin_feats, device_feats, combin_category_embeds_desc, device_category_embeds_desc, num_classes, num_heads, device):
        super(GAT, self).__init__()
        self.combin_feats = torch.from_numpy(combin_feats).float()
        self.device_feats = torch.from_numpy(device_feats).float().to(device)
        # self.edge_matrix = torch.from_numpy(self.gen_edge_matrix(edge_index_train))
        # print(f"edge_index num: {len(edge_index_train)}")

        self.num_combin_categories = len(combin_category_embeds_desc)
        self.num_device_categories = len(device_category_embeds_desc)
        self.num_heads = num_heads
        self.device = device


        self.combin_neibrs_cache = {}
        for edge_index in edge_index_train:
            combin_idx = edge_index[0]
            device_idx = edge_index[1]
            if combin_idx not in self.combin_neibrs_cache:
                self.combin_neibrs_cache[combin_idx] = set()

            self.combin_neibrs_cache[combin_idx].add(device_idx)

        for key in self.combin_neibrs_cache:
            self.combin_neibrs_cache[key] = list(self.combin_neibrs_cache[key])


        device_input_size = self.device_feats.shape[1] - self.num_device_categories
        combin_input_size = self.combin_feats.shape[1] - self.num_combin_categories

        self.device_embeds = torch.nn.ModuleList()
        for device_category_embed_desc in device_category_embeds_desc:
            self.device_embeds.append(nn.Embedding(device_category_embed_desc[0], device_category_embed_desc[1]))
            device_input_size += device_category_embed_desc[1]

        self.combin_embeds = torch.nn.ModuleList()
        for combin_category_embed_desc in combin_category_embeds_desc:
            self.combin_embeds.append(nn.Embedding(combin_category_embed_desc[0], combin_category_embed_desc[1]))
            combin_input_size += combin_category_embed_desc[1]

        # print(f"combin_input_size: {combin_input_size}")
        # print(f"device_input_size: {device_input_size}")


        self.attn_layers = torch.nn.ModuleList()
        out_dim = 16
        alpha = 0.2
        for i in range(self.num_heads):
            self.attn_layers.append(GraphAttentionLayer(combin_input_size, device_input_size, out_dim, alpha))

        self.fc1 = nn.Linear(num_heads * out_dim, out_dim)

        self.fc2 = nn.Linear(out_dim + device_input_size + combin_input_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, edge_index):
        combin_idxes = edge_index[:, 0]
        device_idxes = edge_index[:, 1]

        combin_feats_batch = self.combin_feats[combin_idxes].to(self.device)
        # device_feats_batch = self.device_feats[device_idxes].to(self.device)

        device_feats = self.concat_embed_feats(self.device_feats, self.device_embeds, self.num_device_categories)
        device_feats_batch = device_feats[device_idxes]
        combin_feats_batch = self.concat_embed_feats(combin_feats_batch, self.combin_embeds, self.num_combin_categories)

        combin_neibrs_feats = self.get_combin_neibrs_feats(combin_idxes, device_feats)
        combin_neibrs_attn_feats = torch.cat([attn_layer(combin_feats_batch, combin_neibrs_feats) for attn_layer in self.attn_layers], dim=1)

        combin_neibrs_attn_feats = self.fc1(combin_neibrs_attn_feats)


        fusion_feats = torch.cat([combin_feats_batch, device_feats_batch, combin_neibrs_attn_feats], dim=1)
        fusion_feats = self.fc2(fusion_feats)
        fusion_feats = self.relu(fusion_feats)
        fusion_feats = self.dropout(fusion_feats)
        fusion_feats = self.fc3(fusion_feats)
        fusion_feats = self.relu(fusion_feats)
        fusion_feats = self.dropout(fusion_feats)
        fusion_feats = self.fc4(fusion_feats)

        out = self.sigmoid(fusion_feats)
        end = time.time()
        return out

    def concat_embed_feats(self, feats, embeds, num_categories):
        embed_inputs = feats[:, -num_categories:].long().t()
        outputs = [feats[:, :-num_categories]]
        for embed_input, embed in zip(embed_inputs, embeds):
            embed_output = embed(embed_input)
            outputs.append(embed_output)
        final_feats = torch.cat(outputs, 1)
        return final_feats


    def gen_edge_matrix(self, edge_index):
        num_combin = np.max(edge_index[:, 0]) + 1
        num_dev = np.max(edge_index[:, 1]) + 1
        edge_matrix = np.zeros((num_combin, num_dev), dtype="int")
        for i, j, _ in edge_index:
            edge_matrix[i, j] = 1
        return edge_matrix

    def get_combin_neibrs_feats(self, combin_idxes, device_feats):
        combin_idxes = combin_idxes.cpu().numpy()
        combin_neibrs_feats = []
        for combin_idx in combin_idxes:
            device_idxes = self.combin_neibrs_cache[combin_idx]
            if len(device_idxes) >= 1000:
                np.random.shuffle(device_idxes)
                device_idxes = device_idxes[:1000]
            cur_combin_neibrs_feats = device_feats[device_idxes]
            # else:
            #     cur_combin_neibrs_feats = torch.rand(10000, 113).to(self.device)


            combin_neibrs_feats.append(cur_combin_neibrs_feats)

        return combin_neibrs_feats
