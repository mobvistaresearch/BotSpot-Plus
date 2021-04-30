#!usr/bin/python3
import pickle
import datetime
import time
import os
import subprocess
import random

import numpy as np
import pandas as pd
from urllib.parse import unquote
import torch
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


def get_gbm_model(edge_index_train, combin_feats, device_feats, num_trees):
    train_dataset = np.hstack((combin_feats[edge_index_train[:, 0]], device_feats[edge_index_train[:, 1]]))
    train_labels = edge_index_train[:, 2]
    print(f"train_labels set: {set(train_labels)}")
    print ('training GBM')
    params = {'boosting_type': ['gbdt'], 'max_depth': [4], 'n_estimators': [num_trees]}
    lgbm = lgb.LGBMClassifier(objective='binary')
    clf = GridSearchCV(lgbm, params, cv=4, scoring='roc_auc')
    clf.fit(train_dataset, train_labels)
    gbm_model = clf.best_estimator_
    print ('best model is:', gbm_model)
    return gbm_model

# def gen_super_device_neibrs(edge_index_train, cluster_values):
#     edge_matrix = gen_edge_matrix(edge_index_train)
#     super_device_neibrs_cache = {}
#     # There are different cluster values on the same device
#     if len(edge_index_train) == len(cluster_values):
#         for device_idx, value in zip(edge_index_train[:, 1], cluster_values):
#             if value not in super_device_neibrs_cache:
#                 super_device_neibrs_cache[value] = set()
#             combin_idxes_tmp = np.where(edge_matrix[device_idx] == 1)[0]
#             for v in combin_idxes_tmp:
#                 super_device_neibrs_cache[value].add(v)
#     # There is only one cluster value on the same device
#     elif len(cluster_values) == edge_matrix.shape[0]:
#         for idx, value in enumerate(cluster_values):
#             if value not in super_device_neibrs_cache:
#                 super_device_neibrs_cache[value] = set()
#             combin_idxes_tmp = np.where(edge_matrix[idx] == 1)[0]
#             for v in combin_idxes_tmp:
#                 super_device_neibrs_cache[value].add(v)
#     else:
#         raise RuntimeError("Wrong cluster_values!")
#
#     for key in super_device_neibrs_cache:
#         super_device_neibrs_cache[key] = list(super_device_neibrs_cache[key])
#     keys = super_device_neibrs_cache.keys()
#     print(f"all clusters len: {len(keys)}")
#     return super_device_neibrs_cache

def gen_device_neibrs(edge_index_train):
    device_neibrs_cache = {}
    edge_matrix = gen_edge_matrix(edge_index_train)
    for device_idx in edge_index_train[:, 1]:
        if device_idx not in device_neibrs_cache:
            device_neibrs_cache[device_idx] = set()
        combin_idxes_tmp = np.where(edge_matrix[device_idx] == 1)[0]
        for v in combin_idxes_tmp:
            device_neibrs_cache[device_idx].add(v)
    return device_neibrs_cache

def gen_super_device_neibrs(edge_index_train, cluster_values):
    super_device_neibrs_cache = {}

    edge_matrix = gen_edge_matrix(edge_index_train)

    # There are different cluster values on the same device
    if len(edge_index_train) == len(cluster_values):
        for device_idx, value in zip(edge_index_train[:, 1], cluster_values):
            if value not in super_device_neibrs_cache:
                super_device_neibrs_cache[value] = set()
            combin_idxes_tmp = np.where(edge_matrix[device_idx] == 1)[0]
            for v in combin_idxes_tmp:
                super_device_neibrs_cache[value].add(v)
    else:
        raise RuntimeError("Wrong cluster_values!")

    return super_device_neibrs_cache

def stat_new_edges(edge_index_train, cluster_values, device_neibrs_cache, super_device_neibrs_cache):
    data_df = pd.DataFrame({"device_idx": edge_index_train[:, 1], "cluster_values": cluster_values})
    print(f"before data_df size: {len(data_df)}")
    data_df.drop_duplicates(subset=["device_idx", "cluster_values"], keep="first", inplace=True)
    print(f"after data_df size: {len(data_df)}")
    data_df["origin_edges_num"] = data_df.apply(lambda row: ori_edges_count(row, device_neibrs_cache), axis=1)

    data_df["new_edges_num"] = data_df.apply(lambda row: new_edges_count(row, device_neibrs_cache, super_device_neibrs_cache), axis=1)
    data_df.to_csv("stat_edges.csv", index=False)

def ori_edges_count(row, device_neibrs_cache):
    device_idx = row.device_idx

    ori_dev_neibrs = list(device_neibrs_cache[device_idx])
    ori_dev_neibrs_size = len(ori_dev_neibrs)
    return ori_dev_neibrs_size

def new_edges_count(row, device_neibrs_cache, super_device_neibrs_cache):
    device_idx = row.device_idx
    cluster_values = row.cluster_values
    ori_dev_neibrs = device_neibrs_cache[device_idx]
    new_dev_neibrs = list(super_device_neibrs_cache[cluster_values] - ori_dev_neibrs)
    new_dev_neibrs_size = len(new_dev_neibrs)
    return new_dev_neibrs_size








def gen_edge_matrix(edge_index):

    # e = np.vstack((self.edge_index_train, self.edge_index_test))
    device_feats_dim = np.max(edge_index[:, 1]) + 1
    combin_feats_dim = np.max(edge_index[:, 0]) + 1
    edge_matrix = np.zeros((device_feats_dim, combin_feats_dim), dtype=bool)
    for cur_edge_index in edge_index:
        combin_idx = cur_edge_index[0]
        device_idx = cur_edge_index[1]

        edge_matrix[device_idx, combin_idx] = True

    return edge_matrix

# def neibrs_sampling_mp(edge_index, combin_neibrs, sample_size, num_threads):
#     p = Pool(num_threads)
#     # edge_index = edge_index.cpu().numpy()
#     num_samples = int(len(edge_index) / num_threads)
#     print(f"num_samples: {num_samples}")
#     neibrs_sampled = []
#     thread_result_dict = {}
#     for i in range(num_threads):
#         print(f"cur_thread: {i}")
#         thread_result_dict[i] = p.apply_async(neibrs_sampling, args=(edge_index[i*num_samples: (i+1)*num_samples], combin_neibrs, sample_size))
#     p.close()
#     p.join()
#
#     for i in range(num_threads):
#         cur_neibrs_sampled = thread_result_dict[i].get()
#         neibrs_sampled.append(cur_neibrs_sampled)
#
#     if num_samples * num_threads < len(edge_index):
#         neibrs_sampled.append(neibrs_sampling(edge_index[(i+1)*num_samples:], combin_neibrs, sample_size))
#     neibrs_sampled = np.concatenate(neibrs_sampled, axis=0)
#     print(f"neibrs_sampled shape: {neibrs_sampled.shape}")
#     return neibrs_sampled

def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def setup_seeds(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(input_str):
    input_str = input_str.lower()
    if input_str == "true":
        return True
    elif input_str == "false":
        return False
    else:
        raise RuntimeError("Wrong parameter value!")
