#!usr/bin/python3
import os
import os.path as osp
import sys
import argparse
sys.path.append(r'./utils')
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
import torch.nn.utils as utils


from utils import pickle_load, pickle_dump
from model import MLP
from earlyStopping import EarlyStopping
from log import Logger




INPUT_PATH = "../../input"




RANDOM_STATE = 1234

logger = Logger("./logs")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='choose dataset')
    args = parser.parse_args()
    dataset = args.dataset
    logger.write("")
    logger.write(f"Current model: MLP")
    logger.write(f"Current dataset: {dataset}")

    setup_seeds(RANDOM_STATE)

    # Data loading
    train_file = osp.join(INPUT_PATH, dataset, "train.csv")
    test_file = osp.join(INPUT_PATH, dataset, "test.csv")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    columns = train_df.columns
    stat_columns_file = osp.join(INPUT_PATH, "stat_columns.txt")
    category_columns_file = osp.join(INPUT_PATH, "category_columns.txt")
    stat_columns = pickle_load(stat_columns_file)
    category_columns = pickle_load(category_columns_file)
    feature_columns =  stat_columns + category_columns
    normalized_columns = [stat_columns[-2]]
    except_normalized_columns = [column for column in feature_columns if column not in normalized_columns]

    standard_scaler = StandardScaler()
    standard_scaler.fit(train_df[normalized_columns].values)
    train_normalized = standard_scaler.transform(train_df[normalized_columns].values)
    test_normalized = standard_scaler.transform(test_df[normalized_columns].values)


    X_train = np.concatenate((train_normalized, train_df[except_normalized_columns].values), axis=1)
    y_train = train_df["target"].values
    X_test = np.concatenate((test_normalized, test_df[except_normalized_columns].values), axis=1)
    y_test = test_df["target"].values

    logger.write("x_train.shape: " + str(X_train.shape))
    logger.write("y_train.shape: " + str(y_train.shape))
    logger.write("x_test.shape: " + str(X_test.shape))


    n_features = len(feature_columns)
    n_category_features = len(category_columns)
    n_stat_features = len(stat_columns)


    n_classes = 1
    embeds_desc = []

    X_total = np.concatenate((X_train, X_test), axis=0)
    for i in range(n_stat_features, n_features):
        cur_column = X_total[:, i]
        num_embed = int(max(cur_column) + 1)
        embeds_desc.append([num_embed, 16])

    # Train process

    logger.write("Training...")
    model = MLP(n_stat_features, 64, embeds_desc, n_classes, 0.1)
    model_path = osp.join(INPUT_PATH, "mlp", dataset)
    if not osp.exists(model_path):
        os.makedirs(model_path)
    model_file = osp.join(model_path, "checkpoint.pt")
    patience = 6
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_file)

    logger.write(model)

    train(X_train, y_train, model, n_stat_features, n_features, early_stopping)



    # Predict process
    logger.write("Predicting...")
    stat_matrix_test = X_test[:, :n_stat_features]
    embeds_input_test = []
    for i in range(n_stat_features, n_features):
        embeds_input_test.append(X_test[:, i])
    stat_matrix_test = torch.tensor(stat_matrix_test, dtype=torch.float)
    embeds_input_test = torch.tensor(embeds_input_test, dtype=torch.long)
    test_data = [stat_matrix_test, embeds_input_test]

    model.load_state_dict(torch.load(model_file))

    y_test_prob = predict(test_data, model)

    # calc metrics

    logger.write(f"test max prob:{y_test_prob.max()}")
    logger.write(f"test min prob:{y_test_prob.min()}")


    logger.write("Metrics calculation...")
    recall_precision_score(y_test_prob, y_test)

    print("")


def train(X_train, y_train, model, n_stat_features, n_features, early_stopping):
    setup_seeds(RANDOM_STATE)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle = False, random_state=RANDOM_STATE)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_valid = torch.tensor(y_valid, dtype=torch.float)

    train_dataset = TensorDataset(X_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle = True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=2e-6)

    epoch = 40
    for epoch in range(epoch):
        losses = []
        model.train()
        for idx, minibatch_data in enumerate(train_data_loader):
            if idx % 1000 == 0:
                logger.write(f"Minibatch: {idx}")
            X_minibatch = minibatch_data[0]
            y_minibatch = minibatch_data[1]

            embeds_input_minibatch = []
            for i in range(n_stat_features, n_features):
                embeds_input_minibatch.append(torch.tensor(X_minibatch[:, i],dtype=torch.long))

            stat_matrix_minibatch = X_minibatch[:, :n_stat_features]
            train_data_minibatch = [stat_matrix_minibatch, embeds_input_minibatch]

            optimizer.zero_grad()
            y_prob_minibatch = model(train_data_minibatch)
            loss = criterion(y_prob_minibatch.squeeze(), y_minibatch)
            losses.append(loss)

            loss.backward()
            utils.clip_grad_value_(model.parameters(), 4)
            optimizer.step()

        train_loss = sum(losses)/len(losses)

        model.eval()
        stat_matrix_valid = X_valid[:, :n_stat_features]
        embeds_input_valid = []
        for i in range(n_stat_features, n_features):
            embeds_input_valid.append(torch.tensor(X_valid[:, i], dtype=torch.long))

        stat_matrix_valid = torch.tensor(stat_matrix_valid, dtype=torch.float)

        valid_data = [stat_matrix_valid, embeds_input_valid]

        y_valid_prob = model(valid_data)
        valid_loss = criterion(y_valid_prob.squeeze(), y_valid)

        logger.write(f"Epoch: {epoch}: train loss:{train_loss}, valid loss:{valid_loss}")


        if epoch > 10:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                logger.write("Early stopping")
                break

    if not early_stopping.early_stop:
        print("Not early stop, save the last epoch model...")
        torch.save(model.state_dict(), early_stopping.path)



@torch.no_grad()
def predict(data, model):
    model.eval()
    y_prob = model(data)
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
