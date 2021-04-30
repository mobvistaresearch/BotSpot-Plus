#!usr/bin/python3
import os
import os.path as osp
import sys
import argparse
import gc
import math
sys.path.append(r'./utils')
import random
from multiprocessing import Pool

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn.utils as utils

from preprocess import Preprocess
from utils import *
from model import BotSpot
from neibrs_sampling import NeibrsSampling
from log import Logger

INPUT_PATH = "../../datasets"
MODEL_PATH = "./models"

RANDOM_STATE = 1234

logger = Logger("./logs")

# device = torch.device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose dataset')
parser.add_argument('--use_gbm', type=str2bool, help='whether use gbm model')
# parser.add_argument('--use_gnn', type=str2bool, help='whether use gnn mechanism')
parser.add_argument('--use_stratified', type=str2bool, help='whether use split method')
parser.add_argument('--use_botspot_plus', type=str2bool, help='whether use botspot plus')
parser.add_argument('--use_self_attn', type=str2bool, help='whether apply self attention mechanism on leaf_embedding')
parser.add_argument('--device_num', type=str, help='choose gpu device')
parser.add_argument('--cluster_method', type=str, help='use which cluster method')
# parser.add_argument('--num_trees', type=int, help='how many trees used int botspot++')
# parser.add_argument('--num_heads', type=int, help='how many heads used in self attention')
# parser.add_argument('--alpha', type=float, help='the super parameter alpha of label smoothing')

args = parser.parse_args()
dataset = args.dataset
use_gbm = args.use_gbm
# use_gnn = args.use_gnn
use_gnn = True
use_stratified = args.use_stratified
use_botspot_plus = args.use_botspot_plus
use_self_attn = args.use_self_attn
device_num = args.device_num
cluster_method = args.cluster_method
# num_trees = args.num_trees
# num_heads = args.num_heads
num_trees = 200
num_heads = 2

# alpha = args.alpha
alpha = 0.95
device = torch.device(f'cuda:{device_num}') if torch.cuda.is_available() else torch.device('cpu')

if use_gbm is True and use_botspot_plus is True and use_stratified is True:
    model_name = "BotSpot_plus"
elif use_gbm is True and use_botspot_plus is False and use_stratified is True:
    model_name = "BotSpot"
elif use_gbm is False and use_botspot_plus is True and use_stratified is True:
    model_name = "BotSpot_plus_local"
elif use_gbm is False and use_botspot_plus is False and use_stratified is False:
    model_name = "GraphSAGE"
else:
    model_name = "Others"

logger.write("")
logger.write(f"Current model: {model_name}")
logger.write(f"Current dataset: {dataset}")
logger.write(f"GBM: {use_gbm}")
logger.write(f"use_gnn: {use_gnn}")
logger.write(f"Stratified: {use_stratified}")
logger.write(f"Botspot_plus: {use_botspot_plus}")
logger.write(f"use_self_attn: {use_self_attn}")
logger.write(f"use device: {device}")
logger.write(f"cluster_method: {cluster_method}")
logger.write(f"num_trees: {num_trees}")
logger.write(f"num_heads: {num_heads}")
# logger.write(f"alpha of label smoothing: {alpha}")



def main():
    num_epoches = 10
    batch_size = 256
    num_classes = 1
    sample_size = 100
    category_embed_size = 16
    hidden_size = 64
    p = 0.1
    resume = False
    resume_ckpt_file_name = f"{model_name}_{dataset}_6.pt"
    setup_seeds(RANDOM_STATE)

    # Preprocess process
    data = Preprocess(dataset, cluster_method)

    combin_category_embeds_desc = []
    device_category_embeds_desc = []
    edge_index_train = data.edge_index_train
    edge_index_test = data.edge_index_test
    combin_feats = data.combin_feats
    device_feats = data.device_feats

    super_device_neibrs_cache = None
    if use_botspot_plus:
        cluster_values_train = data.cluster_values_train.reshape(-1, 1)
        cluster_values_test = data.cluster_values_test.reshape(-1, 1)
        print(f"cluster_values_train shape: {cluster_values_train.shape}")
        print(f"edge_index_train shape: {edge_index_train.shape}")
        edge_index_train = np.concatenate([edge_index_train, cluster_values_train], axis=1)
        edge_index_test = np.concatenate([edge_index_test, cluster_values_test], axis=1)
        cluster_values = edge_index_train[:, -1]
        print("generating super device neibrs...")
        # device_neibrs_cache = gen_device_neibrs(edge_index_train)
        super_device_neibrs_cache = gen_super_device_neibrs(edge_index_train, cluster_values)
        # stat_new_edges(edge_index_train, cluster_values, device_neibrs_cache, super_device_neibrs_cache)

    del data
    gc.collect()

    device_feats = device_feats[:, :-1].astype("float")

    if use_gbm:
        gbm_model = get_gbm_model(edge_index_train, combin_feats, device_feats, num_trees)
    else:
        gbm_model  = None

    # Train process
    logger.write("Training...")
    setup_seeds(RANDOM_STATE)
    if not osp.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)


    model = BotSpot(combin_feats, device_feats, super_device_neibrs_cache, use_gbm=use_gbm, use_gnn=use_gnn, use_stratified=use_stratified, use_botspot_plus=use_botspot_plus, use_self_attn=use_self_attn, gbm_model=gbm_model, device=device, embed_size=category_embed_size, num_heads=num_heads)
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
    # logger.write(model)

    train(edge_index_train, edge_index_test, model, optimizer, criterion, num_epoches, batch_size=batch_size)


def train(edge_index_train, edge_index_test, model, optimizer, criterion, num_epoches, batch_size=512):
    setup_seeds(RANDOM_STATE)

    criterion = nn.BCELoss()
    edge_index_train = torch.from_numpy(edge_index_train)

    for epoch in range(num_epoches):

        if use_stratified:
            neibrs_sampled_file = osp.join("./neibrs_sampled", dataset, f"neibrs_sampled_stratified_{epoch}.pkl")
        else:
            neibrs_sampled_file = osp.join("./neibrs_sampled", dataset, f"neibrs_sampled_random_{epoch}.pkl")
        neibrs_sampled = pickle_load(neibrs_sampled_file)
        neibrs_sampled = torch.from_numpy(neibrs_sampled)
        # print(f"neibrs sampled size: {neibrs_sampled.size()}")
        if use_botspot_plus:
            train_dataset = TensorDataset(edge_index_train[:, [0,1,3]], neibrs_sampled, edge_index_train[:, 2])
        else:
            train_dataset = TensorDataset(edge_index_train[:, :2], neibrs_sampled, edge_index_train[:, 2])

        train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        losses = []
        model.train()
        for idx, data_batch in enumerate(train_data_loader):
            if idx % 1000 == 0:
                logger.write(f"Minibatch: {idx}")
            edges_batch = data_batch[0]
            neibrs_sampled_batch = data_batch[1]

            y_batch = data_batch[2].float().to(device)

            optimizer.zero_grad()
            y_prob_batch = model(edges_batch, neibrs_sampled_batch).view(-1)
            if use_botspot_plus:
                positive_loss = torch.sum(-1. * alpha * torch.log(y_prob_batch[y_batch == 1]) -1. * (1 - alpha) * torch.log(1. - y_prob_batch[y_batch == 1]))
                positive_sample_num = torch.sum(y_batch == 1).item()
                tmp = -1. * alpha * torch.log(1 - y_prob_batch[y_batch == 0]) - 1. * (1 - alpha) * torch.log(y_prob_batch[y_batch == 0])
                negative_loss = torch.sort(tmp, descending=True)[0][: int(5 * positive_sample_num)]
                negative_sample_num = len(negative_loss)
                negative_loss = torch.sum(negative_loss)
                loss = negative_loss + positive_loss
                loss = loss / (negative_sample_num + positive_sample_num)
            else:
                loss = criterion(y_prob_batch, y_batch)


            if math.isnan(loss.item()) or math.isinf(loss.item()):
                continue

            # smoothing_label = smooth_one_hot(y_batch.long(), 2, alpha)[:, -1]
            # loss = criterion(y_prob_batch, smoothing_label)

            losses.append(loss)

            loss.backward()
            utils.clip_grad_value_(model.parameters(), 4)
            optimizer.step()
            torch.cuda.empty_cache()

        train_loss = sum(losses) / len(losses)

        logger.write(f"Epoch: {epoch}: train loss:{train_loss}")

        if epoch >= 2:
            logger.write("Saving model...")
            if cluster_method is None:
                model_file_name = f"{model_name}_{dataset}_{epoch}.pt"
            else:
                model_file_name = f"{model_name}_{cluster_method}_{num_trees}_{dataset}_{epoch}.pt"

            model_file = osp.join(MODEL_PATH, model_file_name)
            checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer":optimizer.state_dict()}
            torch.save(checkpoint, model_file)

            if use_botspot_plus:
                logger.write("Predicting...")
                predict(edge_index_train.cpu().numpy(), edge_index_test, model, epoch)
            else:
                if epoch >= 8:
                    logger.write("Predicting...")
                    predict(edge_index_train.cpu().numpy(), edge_index_test, model, epoch)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, confidence=1.0):
    """
    if confidence == 1, it's one-hot method
    if 0 < confidence < 1, it's smooth method

    """
    assert 0 < confidence <= 1
    smoothing = 1.0 - confidence
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

@torch.no_grad()
def predict(edge_index_train, edge_index_test, model, epoch):
    # final_layer_embeddings = []
    y_test = edge_index_test[:, 2]
    sample_size_test = 100
    if use_stratified:
        neibrs_sampling = NeibrsSampling(dataset, sample_size_test, "stratified")
        combin_normal_neibrs, combin_bot_neibrs = neibrs_sampling.gen_combin_normal_bot_neibrs(edge_index_train)
    else:
        neibrs_sampling = NeibrsSampling(dataset, sample_size_test, "random")
        combin_neibrs = neibrs_sampling.gen_combin_neibrs(edge_index_train)
    edge_index_test = torch.from_numpy(edge_index_test)
    if use_botspot_plus:
        test_dataset = TensorDataset(edge_index_test[:, [0,1,3]])
    else:
        test_dataset = TensorDataset(edge_index_test[:, :2])
    test_data_loader = DataLoader(test_dataset, 1024, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        preds= []
        for idx, data_batch in enumerate(test_data_loader):
            edges = data_batch[0]

            if np.random.uniform() > 0.98:
                print (f'current progress: {len(preds)}, out of total of {len(test_data_loader.dataset)}')
            if use_stratified:
                neibrs_sampled_batch = neibrs_sampling.neibrs_sampling_stratified(edges.cpu().numpy(), combin_normal_neibrs, combin_bot_neibrs)
            else:
                neibrs_sampled_batch = neibrs_sampling.neibrs_sampling_random(edges.cpu().numpy(), combin_neibrs)
            neibrs_sampled_batch = torch.from_numpy(neibrs_sampled_batch)
            y_prob_batch = model(edges, neibrs_sampled_batch, train_stage=False)
            # final_layer_embeddings.extend(h.cpu().numpy())

            preds.extend(list(y_prob_batch.view(-1, ).cpu().numpy()))
    y_prob = np.array(preds)
    logger.write(f"test min prob:{y_prob.min()}, test max prob: {y_prob.max()}")
    recall_precision_score(y_prob, y_test, epoch)

    return y_prob


def recall_precision_score(y_prob, y_true, epoch):
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





if __name__ =="__main__":
    main()
