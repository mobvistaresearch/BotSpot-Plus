from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import *
import numpy as np
import pandas as pd
import sys
import os
import os.path as osp


class BotSpotTrans(object):
    def __init__(self ,train_filepath,test_filepath,root_dir=None):
        ROOT_DIR = root_dir
        train_file = train_filepath
        test_file = test_filepath
        print("Loading train & test file...")
        train_df = pd.read_csv(train_file) # 训练集
        test_df = pd.read_csv(test_file) # 测试集
        total_df = pd.concat([train_df, test_df], axis=0)

        print("Graph Generating...")
        self.edge_index = total_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_train = train_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_test = test_df[["combin_index", "device_index", "target"]].astype(int).values


        # stat_columns表示的是统计相关的特征列list
        # category_columns表示的是category的特征列list
        # stat_columns最后两列是ctit与cvr_total, category_columns第一列是label encoder后的channel_id

        stat_columns_file = osp.join(ROOT_DIR, "stat_columns.txt")
        category_columns_file = osp.join(ROOT_DIR, "category_columns.txt")
        stat_columns = self.pickle_load(stat_columns_file)
        category_columns = self.pickle_load(category_columns_file)

        # 所有特征列
        feature_columns = stat_columns + category_columns

        # 这里假设只对ctit进行normalization
        normalized_columns = [stat_columns[-2]]
        except_normalized_columns = [column for column in stat_columns if column not in normalized_columns]


        # channel-campagin相关的用于训练的特征列, 包括除ctit的统计列与channel_id(最后一列)
        combin_feature_columns = except_normalized_columns + [category_columns[0]]
        # device节点相关的用于训练的特征列，包括ctit + 剩下的category列
        device_feature_columns = normalized_columns + category_columns[1:]
        # 所有的channel-campaign列
        device_columns = ["device_index"] + device_feature_columns
        # 所有的device的列
        combin_columns = ["combin_index"] + combin_feature_columns

        # 这里假设是从total_df划分出device_df,根据你的需要，如果是train_df也一样
        device_df = total_df[device_columns].sort_values(["device_index"])
        device_df.drop_duplicates(subset="device_index", keep="first", inplace=True)

        # 同理，也是从total_df划分出来combin_df,根据你的需要进行调整
        combin_df = total_df[combin_columns].sort_values(["combin_index"])
        combin_df.drop_duplicates(subset="combin_index", keep="first", inplace=True)

        # 对ctit列进行最大最小归一化
        norm_data = RobustScaler().fit_transform(device_df.loc[: ,normalized_columns[0]].values.reshape((-1 ,1)))
        device_df.loc[: ,normalized_columns[0]] = norm_data.reshape(-1 ,)

        print("feature matrix generating...")
        self.device_matrix = device_df[device_feature_columns].values
        # 这里对folat的精度转变成了float16，可以根据你的需要调整
        self.combin_matrix = combin_df[combin_feature_columns].astype('float16').values

    def pickle_dump(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)