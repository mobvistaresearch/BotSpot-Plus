import time

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")

class GraphConsis(torch.nn.Module):
    def __init__(self, edge_index_train, combin_feats, device_feats, combin_category_embeds_desc, device_category_embeds_desc, hidden_size, num_classes, batch_size, p):
        super(GraphConsis, self).__init__()
        context_dim = 16
        self.context_embed = nn.Embedding(len(device_feats), context_dim)

        self.combin_feats = torch.from_numpy(combin_feats).float()
        self.device_feats = torch.from_numpy(device_feats).float().to(device)
        # self.device_feats = torch.cat([self.context_vec, torch.from_numpy(device_feats).float()], dim=1)

        self.edge_matrix = torch.from_numpy(self.gen_edge_matrix(edge_index_train))

        self.num_combin_categories = len(combin_category_embeds_desc)
        self.num_device_categories = len(device_category_embeds_desc)

        # self.combin_neibrs_cache = {}
        #
        # for edge_index in edge_index_train:
        #     combin_idx = edge_index[0]
        #     device_idx = edge_index[1]
        #     if combin_idx in self.combin_neibrs_cache:
        #         self.combin_neibrs_cache[combin_idx].add(device_idx)
        #     else:
        #         self.combin_neibrs_cache[combin_idx] = set([device_idx])

        self.combin_neibrs_normal_cache = {}
        self.combin_neibrs_bot_cache = {}
        for edge_index in edge_index_train:
            combin_idx = edge_index[0]
            device_idx = edge_index[1]
            label = edge_index[2]
            if combin_idx not in self.combin_neibrs_normal_cache:
                self.combin_neibrs_normal_cache[combin_idx] = set()

            if combin_idx not in self.combin_neibrs_bot_cache:
                self.combin_neibrs_bot_cache[combin_idx] = set()

            if label == 0:
                self.combin_neibrs_normal_cache[combin_idx].add(device_idx)
            else:
                self.combin_neibrs_bot_cache[combin_idx].add(device_idx)

        for key in self.combin_neibrs_normal_cache:
            self.combin_neibrs_normal_cache[key] = list(self.combin_neibrs_normal_cache[key])

        for key in self.combin_neibrs_bot_cache:
            self.combin_neibrs_bot_cache[key] = list(self.combin_neibrs_bot_cache[key])

        device_input_size = self.device_feats.shape[1] - self.num_device_categories + context_dim
        combin_input_size = self.combin_feats.shape[1] - self.num_combin_categories

        self.device_embeds = torch.nn.ModuleList()
        for device_category_embed_desc in device_category_embeds_desc:
            self.device_embeds.append(nn.Embedding(device_category_embed_desc[0], device_category_embed_desc[1]))
            device_input_size += device_category_embed_desc[1]

        self.combin_embeds = torch.nn.ModuleList()
        for combin_category_embed_desc in combin_category_embeds_desc:
            self.combin_embeds.append(nn.Embedding(combin_category_embed_desc[0], combin_category_embed_desc[1]))
            combin_input_size += combin_category_embed_desc[1]

        self.combin_trans_fc = nn.Linear(combin_input_size, 64)
        self.device_trans_fc = nn.Linear(device_input_size, 64)

        # self.fc1 = nn.Linear(64, 32)
        self.fc1 = nn.Linear(64 + device_input_size, hidden_size)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.sigmoid = nn.Sigmoid()


    def forward(self, edge_index):
        combin_idxes = edge_index[:, 0]
        device_idxes = edge_index[:, 1].to(device)

        combin_feats_batch = self.combin_feats[combin_idxes].to(device)
        device_feats_batch = self.device_feats[device_idxes].to(device)
        context_feats_batch = self.context_embed(device_idxes.view(-1, 1)).squeeze(1)
        device_feats_batch = self.concat_embed_feats(device_feats_batch, self.device_embeds, self.num_device_categories)
        device_feats_batch = torch.cat([context_feats_batch, device_feats_batch], dim=1)

        combin_feats_batch = self.concat_embed_feats(combin_feats_batch, self.combin_embeds, self.num_combin_categories)
        combin_feats_batch = self.mean_agg(combin_idxes, combin_feats_batch)
        fusion_feats = torch.cat([device_feats_batch, combin_feats_batch], dim=1)
        x = self.fc1(fusion_feats)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        out = self.sigmoid(x)
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


    # def mean_agg(self, combin_idxes, combin_feats):
    #     combin_idxes = combin_idxes.cpu().numpy()
    #     device_idxes = set()
    #     for combin_idx in combin_idxes:
    #         device_idxes = device_idxes.union(self.combin_neibrs_cache[combin_idx])
    #     device_idxes = list(device_idxes)
    #     device_idxes.sort()
    #     combin_neibrs_feats = self.device_feats[device_idxes].to(device)
    #     combin_neibrs_feats = self.concat_embed_feats(combin_neibrs_feats, self.device_embeds, self.num_device_categories)
    #     tmp_device_idxes = torch.tensor(device_idxes, dtype=torch.long).view(-1, 1).to(device)
    #     combin_neibrs_context_feats = self.context_embed(tmp_device_idxes).squeeze(1)
    #     combin_neibrs_feats = torch.cat([combin_neibrs_context_feats, combin_neibrs_feats], dim=1)
    #     edge_matrix_batch = self.edge_matrix[combin_idxes][:, device_idxes]
    #     combin_neibrs_idxes = [np.where(edges == 1)[0] for edges in edge_matrix_batch]
    #
    #     neibrs_agg_feats = []
    #
    #     for cur_combin_idx, cur_combin_feats, cur_combin_neibrs_idxes in zip(combin_idxes, combin_feats, combin_neibrs_idxes):
    #         cur_combin_neibrs_feats = combin_neibrs_feats[cur_combin_neibrs_idxes]
    #         cur_neibrs_agg_feats = self.sample_combin_neibrs_feats(cur_combin_feats, cur_combin_neibrs_feats, 50, 0.001)
    #         cur_neibrs_agg_feats = cur_neibrs_agg_feats.mean(dim=0).view(1, -1)
    #         neibrs_agg_feats.append(cur_neibrs_agg_feats)
    #
    #     neibrs_agg_feats = torch.cat(neibrs_agg_feats, dim=0)
    #     return neibrs_agg_feats

    def mean_agg(self, combin_idxes, combin_feats):
        device_feats = self.concat_embed_feats(self.device_feats, self.device_embeds, self.num_device_categories)
        tmp_device_idxes = torch.tensor([i for i in range(len(self.device_feats))], dtype=torch.long).view(-1, 1).to(device)
        device_context_feats = self.context_embed(tmp_device_idxes).squeeze(1)
        device_feats = torch.cat([device_context_feats, device_feats], dim=1)

        neibrs_agg_feats = []
        for combin_idx, cur_combin_feats in zip(combin_idxes, combin_feats):
            normal_device_idxes = self.combin_neibrs_normal_cache[combin_idx.item()]
            bot_device_idxes = self.combin_neibrs_bot_cache[combin_idx.item()]
            if len(normal_device_idxes) > 9000:
                # may be need to shuffle before select
                # np.random.shuffle(normal_device_idxes)
                normal_device_idxes = normal_device_idxes[:9000]
            if len(bot_device_idxes) > 2000:
                bot_device_idxes = bot_device_idxes[:2000]

            device_idxes = np.concatenate([normal_device_idxes, bot_device_idxes])
            combin_neibrs_feats = device_feats[device_idxes]
            cur_neibrs_agg_feats = self.sample_combin_neibrs_feats(cur_combin_feats, combin_neibrs_feats, 100, 0.001)
            cur_neibrs_agg_feats = cur_neibrs_agg_feats.mean(dim=0).view(1, -1)
            neibrs_agg_feats.append(cur_neibrs_agg_feats)

        neibrs_agg_feats = torch.cat(neibrs_agg_feats, dim=0)
        return neibrs_agg_feats


    def sample_combin_neibrs_feats(self, combin_feats, combin_neibrs_feats, num_samples, consis_threshold=0.001):
        num_neibrs = combin_neibrs_feats.size()[0]
        combin_feats = combin_feats.repeat(num_neibrs, 1)
        combin_feats = self.combin_trans_fc(combin_feats)
        combin_neibrs_feats = self.device_trans_fc(combin_neibrs_feats)

        consis_scores = self.calc_feat_consis_score(combin_feats, combin_neibrs_feats)
        consis_scores[consis_scores <= consis_threshold] = 0

        if consis_scores.sum().item() == 0:
            all_idxes = [i for i in range(num_neibrs)]
            if num_neibrs >= num_samples:
                np.random.shuffle(all_idxes)
                idxes = all_idxes[: num_samples]
            else:
                idxes = np.random.choice(all_idxes, size=num_samples, replace=True)
        else:
            idxes = list(WeightedRandomSampler(consis_scores, num_samples, replacement=True))
        combin_neibrs_feats = combin_neibrs_feats[idxes]
        return combin_neibrs_feats


    def calc_feat_consis_score(self, u, v):
        dist = F.pairwise_distance(u, v, p=2)
        return torch.exp(-dist)
