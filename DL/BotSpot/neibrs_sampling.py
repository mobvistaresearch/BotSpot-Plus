#!usr/bin/python3
import os
import os.path as osp
import sys
import argparse
import gc
sys.path.append(r'./utils')
import random

import numpy as np
import pandas as pd

from utils import pickle_load, pickle_dump
from log import Logger

INPUT_PATH = "./datasets"
SAVE_PATH = "./neibrs_sampled"

RANDOM_STATE = 1234

logger = Logger("./sampling_logs")


class NeibrsSampling(object):
    def __init__(self, dataset, sample_size, sample_mode):
        self.dataset = dataset
        self.sample_size = sample_size

    def neibrs_sampling_random(self, edge_index, combin_neibrs):
        neibrs_sampled = np.array([-2] * len(edge_index) * self.sample_size).reshape(len(edge_index), -1)
        for idx, cur_edge_index in enumerate(edge_index):
            combin_idx = cur_edge_index[0]
            device_idxes = combin_neibrs[combin_idx]
            neibrs_sampled[idx] = self.neibrs_sampling_(idx, device_idxes, self.sample_size)
        return neibrs_sampled


    def neibrs_sampling_stratified(self, edge_index, combin_normal_neibrs, combin_bot_neibrs):

        neibrs_sampled = np.array([-2] * len(edge_index) * self.sample_size).reshape(len(edge_index), -1)
        for idx, cur_edge_index in enumerate(edge_index):
            combin_idx = cur_edge_index[0]
            device_idx = cur_edge_index[1]
            normal_device_idxes = combin_normal_neibrs[combin_idx]
            bot_device_idxes = combin_bot_neibrs[combin_idx]

            cur_normal_neibrs_sampled = self.neibrs_sampling_(idx, normal_device_idxes, int(self.sample_size/2), device_idx)
            cur_bot_neibrs_sampled = self.neibrs_sampling_(idx, bot_device_idxes, int(self.sample_size/2), device_idx)
            neibrs_sampled[idx] = np.concatenate([cur_normal_neibrs_sampled, cur_bot_neibrs_sampled], axis=1)

        return neibrs_sampled

    def neibrs_sampling_(self, idx, device_idxes, sample_size, self_device_idx=None):
        if idx % 10000 == 0 and idx != 0:
            logger.write(f"has already processed {idx} samples...")

        if self_device_idx is not None:
            device_idxes = device_idxes - set([self_device_idx])

        device_idxes = np.array(list(device_idxes))
        total_neibrs = len(device_idxes)
        if total_neibrs >= sample_size:
            np.random.shuffle(device_idxes)
            neibrs = device_idxes[: sample_size]
        elif total_neibrs > 0 and total_neibrs < sample_size:
            neibrs = np.random.choice(device_idxes, size=sample_size, replace=True)
        else:
            neibrs = np.array([-1] * sample_size)
        return neibrs.reshape(1, -1)

    def gen_combin_neibrs(self, edge_index):
        combin_neibrs = {}
        # generate the all neibrs of combins
        for edge_index in edge_index:
            combin_idx = edge_index[0]
            device_idx = edge_index[1]
            if combin_idx not in combin_neibrs:
                combin_neibrs[combin_idx] = set()
            combin_neibrs[combin_idx].add(device_idx)
        return combin_neibrs

    def gen_combin_normal_bot_neibrs(self, edge_index):
        combin_normal_neibrs = {}
        combin_bot_neibrs = {}
        # generate the all neibrs of combins
        for cur_edge_index in edge_index:
            combin_idx = cur_edge_index[0]
            device_idx = cur_edge_index[1]
            label = cur_edge_index[2]
            if combin_idx not in combin_normal_neibrs:
                combin_normal_neibrs[combin_idx] = set()
            if combin_idx not in combin_bot_neibrs:
                combin_bot_neibrs[combin_idx] = set()
            # Note: label can only be 0 or 1, otherwise will cause exception.
            if label:
                combin_normal_neibrs[combin_idx].add(device_idx)
            else:
                combin_bot_neibrs[combin_idx].add(device_idx)
        return combin_normal_neibrs, combin_bot_neibrs


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='choose dataset')
    parser.add_argument('--num', type=int, help='number of neibrs to be sampled')
    parser.add_argument('--mode', type=str, help='sample mode, random or stratified')
    parser.add_argument('--epoch', type=str, help='the sampled neibrs used for which epoch')

    args = parser.parse_args()
    dataset = args.dataset
    num = args.num
    mode = args.mode
    epoch = args.epoch
    logger.write(f"dataset: {dataset}")
    logger.write(f"num: {num}")
    logger.write(f"mode: {mode}")
    logger.write(f"epoch: {epoch}")

    train_file = osp.join(INPUT_PATH, dataset, "train.csv")
    print("Loading train file...")
    train_df = pd.read_csv(train_file)
    edge_index_train = train_df[["combin_index", "device_index", "target"]].astype(int).values
    del train_df
    gc.collect()

    neibrs_sampling = NeibrsSampling(dataset, num, mode)
    dataset_sampled_path = osp.join(SAVE_PATH, dataset)
    if not osp.exists(dataset_sampled_path):
        os.makedirs(dataset_sampled_path)

    neibrs_sampled_file = osp.join(dataset_sampled_path, f"neibrs_sampled_{mode}_{epoch}.pkl")
    if not osp.exists(neibrs_sampled_file):
        if mode == "random":
            combin_neibrs = neibrs_sampling.gen_combin_neibrs(edge_index_train)
            neibrs_sampled = neibrs_sampling.neibrs_sampling_random(edge_index_train, combin_neibrs)
        elif mode == "stratified":
            combin_normal_neibrs, combin_bot_neibrs = neibrs_sampling.gen_combin_normal_bot_neibrs(edge_index_train)
            neibrs_sampled = neibrs_sampling.neibrs_sampling_stratified(edge_index_train, combin_normal_neibrs, combin_bot_neibrs)

        else:
            raise ValueError("Wrong sample mode!")

        pickle_dump(neibrs_sampled, neibrs_sampled_file)
        print(f"neibrs sampled: {neibrs_sampled.shape}")
        print(f"neibrs samples: {neibrs_sampled[:5]}")

    logger.write("Finished!")
