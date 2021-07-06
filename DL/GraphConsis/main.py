#!usr/bin/python3
import os
import os.path as osp
import sys
import argparse
sys.path.append(r'./utils')
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn.utils as utils
import torch.multiprocessing as mp

from preprocess import Preprocess
from utils import pickle_load, pickle_dump
from model import GraphConsis
from log import Logger

INPUT_PATH = "../../datasets"
MODEL_PATH = "./models"

RANDOM_STATE = 1234

logger = Logger("./logs")
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose dataset')
parser.add_argument('--device_num', type=int, help='choose gpu device')

args = parser.parse_args()

dataset = args.dataset
device_num = args.device_num
logger.write("")
logger.write(f"Current model: GraphConsis")
logger.write(f"Current dataset: {dataset}")
device = torch.device(f'cuda:{device_num}') if torch.cuda.is_available() else torch.device('cpu')

print(f"use device: {device}")

def main():
    num_epoches = 10
    batch_size = 256
    num_classes = 1
    category_embed_size = 16
    hidden_size = 64
    p = 0.1
    resume = False
    resume_ckpt_file_name = f"model_{dataset}_6.pt"
    setup_seeds(RANDOM_STATE)

    # Preprocess process
    data = Preprocess(dataset)

    combin_category_embeds_desc = []
    device_category_embeds_desc = []

    for column in data.combin_category_columns:
        temp_values = data.total_df[column].values
        num_embed = int(max(temp_values) + 1)
        combin_category_embeds_desc.append([num_embed, category_embed_size])

    for column in data.device_category_columns:
        temp_values = data.total_df[column].values
        num_embed = int(max(temp_values) + 1)
        device_category_embeds_desc.append([num_embed, category_embed_size])


    # Train process
    logger.write("Training...")
    setup_seeds(RANDOM_STATE)
    if not osp.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_file_prefix = f"model_{dataset}"

    edge_index = torch.from_numpy(data.edge_index_train)
    train_dataset = TensorDataset(edge_index[:, :2], edge_index[:, 2])
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)


    model = GraphConsis(data.edge_index_train, data.combin_feats, data.device_feats, combin_category_embeds_desc, device_category_embeds_desc, hidden_size, num_classes, batch_size, p, device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-6)
    if resume:
        print(f"Resuming State...")
        resume_model_file = osp.join(MODEL_PATH, resume_ckpt_file_name)
        checkpoint = torch.load(resume_model_file)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        num_epoches = num_epoches - epoch - 1
    else:
        model.to(device)

    logger.write(model)

    train(train_data_loader, model, optimizer, criterion, num_epoches, model_file_prefix, batch_size=batch_size)

    # Predict process
    logger.write("Predicting...")
    edge_index = torch.from_numpy(data.edge_index_test)
    test_dataset = TensorDataset(edge_index[:, :2])
    test_data_loader = DataLoader(test_dataset, 1024, shuffle=False)
    y_test = data.edge_index_test[:, 2]
    model_file = osp.join(MODEL_PATH, f"{model_file_prefix}.pt")
    model.load_state_dict(torch.load(model_file)["model"])
    # model.to(device)
    y_test_prob = predict(test_data_loader, model)

    # calc metrics
    logger.write(f"test max prob:{y_test_prob.max()}")
    logger.write(f"test min prob:{y_test_prob.min()}")

    logger.write("Metrics calculation...")
    recall_precision_score(y_test_prob, y_test)

    print("")


def train(data_loader, model, optimizer, criterion, num_epoches, model_file_prefix, batch_size=512):
    setup_seeds(RANDOM_STATE)

    for epoch in range(num_epoches):
        losses = []
        model.train()
        for idx, data_batch in enumerate(data_loader):
            if idx % 1000 == 0:
                logger.write(f"Minibatch: {idx}")

            edges_batch = data_batch[0]
            y_batch = data_batch[1].float().to(device)

            optimizer.zero_grad()
            y_prob_batch = model(edges_batch)
            loss = criterion(y_prob_batch.squeeze(), y_batch)
            losses.append(loss)

            loss.backward()
            utils.clip_grad_value_(model.parameters(), 4)
            optimizer.step()
            torch.cuda.empty_cache()

        train_loss = sum(losses) / len(losses)
        combin_neibrs_normal_cache = model.combin_neibrs_normal_cache
        for key, value in combin_neibrs_normal_cache.items():
            np.random.shuffle(value)
            combin_neibrs_normal_cache[key] = value
        model.combin_neibrs_normal_cache = combin_neibrs_normal_cache

        combin_neibrs_bot_cache = model.combin_neibrs_bot_cache
        for key, value in combin_neibrs_bot_cache.items():
            np.random.shuffle(value)
            combin_neibrs_bot_cache[key] = value
        model.combin_neibrs_bot_cache = combin_neibrs_bot_cache

        logger.write(f"Epoch: {epoch}: train loss:{train_loss}")
    logger.write("Saving model...")
    model_file = osp.join(MODEL_PATH, f"{model_file_prefix}.pt")
    checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    torch.save(checkpoint, model_file)


    # return model


@torch.no_grad()
def predict(data_loader, model):
    model.eval()
    y_prob = []
    for idx, data_batch in enumerate(data_loader):
        edges_batch = data_batch[0]
        # print(f"data batch: {data_batch}")
        if idx % 100 == 0:
            print(f"has already processed {idx} batches")
        y_prob.extend(model(edges_batch).cpu().numpy())

    y_prob = np.array(y_prob)
    print(f"y_prob shape: {y_prob.shape}")
    return y_prob

def recall_precision_score(y_prob, y_true):
    P_R_scores = []
    for i in np.arange(0.01, 0.98, 0.0015):
        y_pred = y_prob > i
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        P_R_scores.append((i, precision, recall))

    recall_90 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.9))[0]
    recall_85 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.85))[0]
    recall_80 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.8))[0]

    logger.write(f"precision: {recall_90[1]}, recall: {recall_90[2]} at split: {recall_90[0]} when precision is 0.90")
    logger.write(f"precision: {recall_85[1]}, recall: {recall_85[2]} at split: {recall_85[0]} when precision is 0.85")
    logger.write(f"precision: {recall_80[1]}, recall: {recall_80[2]} at split: {recall_80[0]} when precision is 0.80")

def setup_seeds(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ =="__main__":
    main()
