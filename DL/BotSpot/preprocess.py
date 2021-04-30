import os
import os.path as osp
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import pickle
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture as GMM


INPUT_PATH = "../../datasets"
CLUSTER_DATA_PATH = "./cluster_data"

class Preprocess(object):
    def __init__(self, dataset, cluster_method):
        self.dataset = dataset
        self.cluster_method = cluster_method
        encoder_method = "target"
        
        train_file = osp.join(INPUT_PATH, self.dataset, "train.csv")
        test_file = osp.join(INPUT_PATH, self.dataset, "test.csv")

        print("Loading train & test file...")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)


        total_df = pd.concat([train_df, test_df], axis=0)


        print("Graph Generating...")
        self.edge_index = total_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_train = train_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_test = test_df[["combin_index", "device_index", "target"]].astype(int).values

        stat_columns_file = osp.join(INPUT_PATH, "stat_columns.txt")
        category_columns_file = osp.join(INPUT_PATH, "category_columns.txt")
        stat_columns = self.pickle_load(stat_columns_file)
        category_columns = self.pickle_load(category_columns_file)
        # category_columns.remove("install_ip")

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


        device_df = total_df[device_columns].sort_values(["device_index"])
        device_df.drop_duplicates(subset="device_index", keep="first", inplace=True)

        combin_df = total_df[combin_columns].sort_values(["combin_index"])
        combin_df.drop_duplicates(subset="combin_index", keep="first", inplace=True)


        norm_data = RobustScaler().fit_transform(device_df.loc[:, normalized_columns[0]].values.reshape((-1, 1)))
        device_df.loc[:, normalized_columns[0]] = norm_data.reshape(-1, )

        print("feature matrix generating...")
        self.device_feats = device_df[self.device_feature_columns].values
        self.combin_feats = combin_df[self.combin_feature_columns].astype('float16').values

        if self.cluster_method is None:
            self.cluster_values_train = train_df["package_name"].values
            self.cluster_values_test = test_df["package_name"].values
        else:
            if not osp.exists(CLUSTER_DATA_PATH):
                os.makedirs(CLUSTER_DATA_PATH)
            cluster_file = osp.join(CLUSTER_DATA_PATH, f"{encoder_method}_{self.cluster_method}_constraint_{self.dataset}.csv")

            if not osp.exists(cluster_file):
                print("Genenerating cluster file...")
                start = time.time()
                if encoder_method == "target":
                    feats_train_df = train_df[self.device_feature_columns]
                    feats_test_df = test_df[self.device_feature_columns]

                    y_train = train_df["target"]
                    device_feats_df = device_df[self.device_feature_columns]
                    new_feats = self.target_transform(feats_train_df, device_feats_df, y_train)

                else:
                    raise RuntimeError("Wrong encoder method!")

                new_feats_df = pd.DataFrame(new_feats, columns=self.device_feature_columns)
                is_train_device = np.array([0]* len(device_feats_df))
                train_devices = self.edge_index_train[:, 1]
                is_train_device[train_devices] = 1
                new_feats_df["is_train_device"] = is_train_device

                labels = []

                # NOTE: the implementation of DBSCAN in sklearn will cause memory out error, deprecated!
                if self.cluster_method == "kmeans":
                    labels = new_feats_df.groupby("package_name").apply(lambda group_df: self.kmeans_cluster_constraint(group_df)).sort_values(by="idx")["labels"].values
                elif self.cluster_method == "mean_shift":
                    labels = new_feats_df.groupby("package_name").apply(lambda group_df: self.mean_shift_cluster_constraint(group_df, bandwidth=None)).sort_values(by="idx")["labels"].values
                elif self.cluster_method == "gmm":
                    labels = new_feats_df.groupby("package_name").apply(lambda group_df: self.gmm_cluster_constraint(group_df)).sort_values(by="idx")["labels"].values
                elif self.cluster_method == "gmm_mean_shift":
                    mean_shift_file = osp.join(CLUSTER_DATA_PATH, f"{encoder_method}_mean_shift_constraint_{self.dataset}.csv")
                    mean_shift_df = pd.read_csv(mean_shift_file)
                    new_feats_df["cluster_labels"] = mean_shift_df["cluster_labels"].values
                    labels = new_feats_df.groupby("package_name").apply(lambda group_df: self.gmm_mean_shift_cluster_constraint(group_df)).sort_values(by="idx")["labels"].values
                else:
                    raise ValueError("Wrong cluster method!")

                labels = [str(label) for label in labels]
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(labels)
                device_idxes = list(range(len(device_feats_df)))
                cluster_df = pd.DataFrame({"device_index": device_idxes, "cluster_labels": labels})
                cluster_df.to_csv(cluster_file, index=False)
            else:
                print("Cluster file exists, loading...")
                cluster_df = pd.read_csv(cluster_file)
            self.cluster_values_train = pd.merge(train_df, cluster_df, on="device_index", how="left")["cluster_labels"].values
            self.cluster_values_test = pd.merge(test_df, cluster_df, on="device_index", how="left")["cluster_labels"].values
            # raise RuntimeError("TEST!")

    def target_transform(self, feats_train_df, device_feats_df, y_train):
        device_category_columns = self.device_category_columns.copy()
        device_category_columns.remove("install_ip")
        device_category_columns.remove("package_name")
        ce_target = ce.TargetEncoder(cols = device_category_columns)
        ce_target.fit(feats_train_df, y_train)
        new_df = ce_target.transform(device_feats_df)
        return new_df.values

    def kmeans_cluster_constraint(self, group_df, threshold=2000):
        train_group_df = group_df[group_df["is_train_device"]==1]
        total_size = len(group_df)
        train_size = len(train_group_df)
        device_feature_columns = self.device_feature_columns.copy()
        packages = group_df["package_name"].values

        device_feature_columns.remove("install_ip")
        device_feature_columns.remove("package_name")
        if train_size <= threshold:
            labels = packages
        else:
            if train_size < 5000:
                k = 4
            elif train_size > 5000 and train_size <= 100000:
                k = int(train_size / 1000)
            else:
                k = 300
            kmeans = self.kmeans_cluster(train_group_df[device_feature_columns].values, k)
            labels = kmeans.predict(group_df[device_feature_columns].values)
            labels = [str(packages[i]) + "_" + str(labels[i]) for i in range(total_size)]

        df = pd.DataFrame({"idx": group_df.index.values, "labels": labels})
        return df

    def mean_shift_cluster_constraint(self, group_df, threshold=2000, bandwidth=None):
        train_group_df = group_df[group_df["is_train_device"]==1]
        total_size = len(group_df)
        train_size = len(train_group_df)
        print(f"total size: {total_size}")
        print(f"train size: {train_size}")
        cur_package_name = group_df["package_name"].values[0]

        device_feature_columns = self.device_feature_columns.copy()
        packages = group_df["package_name"].values

        device_feature_columns.remove("install_ip")
        device_feature_columns.remove("package_name")
        if train_size <= threshold:
            labels = packages
            print(f"package_name: {cur_package_name}, no need to cluster")
        else:
            mean_shift = self.mean_shift_cluster(train_group_df[device_feature_columns].values, bandwidth)
            labels = mean_shift.predict(group_df[device_feature_columns].values)
            labels = [str(packages[i]) + "_" + str(labels[i]) for i in range(total_size)]
            num_centers = len(list(set(labels)))
            print(f"package_name: {cur_package_name}, num_centers: {num_centers}")

        df = pd.DataFrame({"idx": group_df.index.values, "labels": labels})
        print(f"df:{df}")
        return df

    def gmm_cluster_constraint(self, group_df, threshold=2000):
        train_group_df = group_df[group_df["is_train_device"]==1]
        total_size = len(group_df)
        train_size = len(train_group_df)
        print(f"total size: {total_size}")
        print(f"train size: {train_size}")

        device_feature_columns = self.device_feature_columns.copy()
        packages = group_df["package_name"].values
        print(f"device_feature_columns: {device_feature_columns}")

        device_feature_columns.remove("install_ip")
        device_feature_columns.remove("package_name")
        if train_size <= threshold:
            labels = packages
        else:
            if train_size < 5000:
                k = 4
            elif train_size > 5000 and train_size <= 100000:
                k = int(train_size / 1000)
            else:
                k = 300
            gmm = self.gmm_cluster(train_group_df[device_feature_columns].values, k)
            labels = gmm.predict(group_df[device_feature_columns].values)
            print(f"predict num clusters: {len(set(labels))}")
            labels = [str(packages[i]) + "_" + str(labels[i]) for i in range(total_size)]

        df = pd.DataFrame({"idx": group_df.index.values, "labels": labels})
        print(f"df:{df}")
        return df

    def gmm_mean_shift_cluster_constraint(self, group_df, threshold=2000):
        train_group_df = group_df[group_df["is_train_device"]==1]
        total_size = len(group_df)
        train_size = len(train_group_df)
        print(f"total size: {total_size}")
        print(f"train size: {train_size}")

        device_feature_columns = self.device_feature_columns.copy()
        packages = group_df["package_name"].values

        device_feature_columns.remove("install_ip")
        device_feature_columns.remove("package_name")
        if train_size <= threshold:
            labels = packages
        else:
            k = len(list(set(group_df["cluster_labels"].values)))
            gmm = self.gmm_cluster(train_group_df[device_feature_columns].values, k)
            labels = gmm.predict(group_df[device_feature_columns].values)
            labels = [str(packages[i]) + "_" + str(labels[i]) for i in range(total_size)]

        df = pd.DataFrame({"idx": group_df.index.values, "labels": labels})
        print(f"df:{df}")
        return df

    def groupby_count(self, group_df, col):
        values = list(set(group_df[col].values))
        return len(values)

    def kmeans_cluster(self, X, k):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=1234, verbose=1).fit(X)
        return kmeans

    def mean_shift_cluster(self, X, bandwidth):
        mean_shift = MeanShift(bandwidth=bandwidth, min_bin_freq=3, n_jobs=-1, max_iter=150).fit(X)
        return mean_shift

    def dbscan_cluster(self, X):
        dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
        return dbscan

    def gmm_cluster(self, X, k):
        gmm = GMM(n_components=k, verbose=1).fit(X)
        return gmm

    def pickle_dump(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    def ip_seg(self, ip_str, num_seg):
        ip_arr = ip_str.split(".")
        if len(ip_arr) != 4:
            raise RuntimeError("Wrong ip value!")
        return ".".join(ip_arr[:num_seg])

if __name__ == '__main__':
    dataset = "dataset1"
    cluster_method = None
    Preprocess(dataset, cluster_method)
