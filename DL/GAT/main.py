#!usr/bin/python3
import os
import os.path as osp
import sys
import argparse
sys.path.append(r'./utils')
import random
import time

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

from preprocess import Preprocess
from utils import pickle_load, pickle_dump
from model import GAT
from log import Logger

INPUT_PATH = "../../datasets"
MODEL_PATH = "./models"

RANDOM_STATE = 1234

logger = Logger("./logs")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose dataset')
parser.add_argument('--device_num', type=str, help='choose gpu device')

args = parser.parse_args()
dataset = args.dataset
device_num = args.device_num
device = torch.device(f'cuda:{device_num}') if torch.cuda.is_available() else torch.device('cpu')
logger.write("")
logger.write(f"Current dataset: {dataset}")
logger.write(f"Current model: GAT")
logger.write(f"Use device: {device}")


def main():
    setup_seeds(RANDOM_STATE)
    # Preprocess process
    data = Preprocess(dataset)

    category_embed_size = 16
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

    num_classes = 1
    num_heads = 1
    logger.write("Training...")
    model = GAT(data.edge_index_train, data.combin_feats, data.device_feats, combin_category_embeds_desc, device_category_embeds_desc, num_classes, num_heads, device)
    model.to(device)


    logger.write(model)
    if not osp.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    train(data.edge_index_train, model, batch_size=256)

    # Predict process
    logger.write("Predicting...")
    model_file = osp.join(MODEL_PATH, f"model_{dataset}.pt")
    model.load_state_dict(torch.load(model_file)["model"])
    model.to(device)
    y_test = data.edge_index_test[:, 2]
    y_test_prob = predict(data.edge_index_test, model)

    # calc metrics

    logger.write(f"test max prob:{y_test_prob.max()}")
    logger.write(f"test min prob:{y_test_prob.min()}")


    logger.write("Metrics calculation...")
    recall_precision_score(y_test_prob, y_test)

    print("")


def train(edge_index, model, batch_size=512):
    setup_seeds(RANDOM_STATE)

    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle = False, random_state=RANDOM_STATE)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    train_dataset = TensorDataset(edge_index[:, :2], edge_index[:, 2])
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(edge_index[:, :1], edge_index[:, 2])
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=2e-6)

    epoch = 5
    for epoch in range(epoch):
        losses = []
        model.train()
        for idx, data_batch in enumerate(train_data_loader):
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

        train_loss = sum(losses)/len(losses)

        logger.write(f"Epoch: {epoch}: train loss:{train_loss}")
    logger.write("Saving model...")
    model_file = osp.join(MODEL_PATH, f"model_{dataset}.pt")
    checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    torch.save(checkpoint, model_file)

    return model



@torch.no_grad()
def predict(edge_index_test, model):
    edge_index_test = torch.from_numpy(edge_index_test)
    test_dataset = TensorDataset(edge_index_test[:, :2])
    test_data_loader = DataLoader(test_dataset, 1024, shuffle=False)
    model.eval()
    y_prob = []
    for idx, data_batch in enumerate(test_data_loader):
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
