import os
import os.path as osp

import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import pickle

INPUT_PATH = "../../datasets"

class Preprocess(object):
    def __init__(self, dataset_name):

        train_file = osp.join(INPUT_PATH, dataset_name, "train.csv")
        test_file = osp.join(INPUT_PATH, dataset_name, "test.csv")
        print("Loading train & test file...")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        self.total_df = pd.concat([train_df, test_df], axis=0)

        print("Graph Generating...")
        self.edge_index = self.total_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_train = train_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_test = test_df[["combin_index", "device_index", "target"]].astype(int).values


        stat_columns_file = osp.join(INPUT_PATH, "stat_columns.txt")
        category_columns_file = osp.join(INPUT_PATH, "category_columns.txt")
        stat_columns = self.pickle_load(stat_columns_file)
        category_columns = self.pickle_load(category_columns_file)[:-1]

        # feature_columns = stat_columns + category_columns

        normalized_columns = [stat_columns[-2]]
        except_normalized_columns = [column for column in stat_columns if column not in normalized_columns]

        self.combin_category_columns = [category_columns[0]]
        self.device_category_columns = category_columns[1: ]

        self.combin_feature_columns = except_normalized_columns + [category_columns[0]]
        self.device_feature_columns = normalized_columns + category_columns[1: ]
        print(f"combin_feature_columns: {self.combin_feature_columns}")
        print(f"device_feature_columns: {self.device_feature_columns}")

        device_columns = ["device_index"] + self.device_feature_columns
        combin_columns = ["combin_index"] + self.combin_feature_columns


        device_df = self.total_df[device_columns].sort_values(["device_index"])
        device_df.drop_duplicates(subset="device_index", keep="first", inplace=True)


        combin_df = self.total_df[combin_columns].sort_values(["combin_index"])
        combin_df.drop_duplicates(subset="combin_index", keep="first", inplace=True)


        norm_data = RobustScaler().fit_transform(device_df.loc[:, normalized_columns[0]].values.reshape((-1, 1)))
        device_df.loc[:, normalized_columns[0]] = norm_data.reshape(-1, )
        device_tmp = device_df["device_index"]
        combin_tmp = combin_df["combin_index"]



        print("feature matrix generating...")
        self.device_feats = device_df[self.device_feature_columns].values
        self.combin_feats = combin_df[self.combin_feature_columns].astype('float16').values

    def pickle_dump(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
