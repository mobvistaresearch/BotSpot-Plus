#!usr/bin/python3
import os
import os.path as osp
import sys
sys.path.append(r'./utils')

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_svmlight_file
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
import pandas as pd


from train import lgb_classifier
# from train import lightgbm_feature_importance
from utils import pickle_load, pickle_dump
from log import Logger


INPUT_PATH = "/data/xiliang.wang/BotSpot-Plus/input"

RANDOM_STATE = 1234

logger = Logger("./logs")


def main():
    max_depth = 5
    num_iterations = 500

    lgb_params = {'num_leaves': 18,
        'min_data_in_leaf': 30,
        'tree_learner': 'serial',
        'objective': 'binary',
        'max_depth': max_depth,
        'learning_rate': 0.01,
        'boosting': 'gbdt',
        'bagging_freq': 2,
        'bagging_fraction': 0.4,
        'feature_fraction': 0.6,
        'bagging_seed': 11,
        'reg_alpha': 1.728910519108444,
        'reg_lambda': 4.9847051755586085,
        'random_state': RANDOM_STATE,
        'num_iterations' : num_iterations,
        'verbosity': -1,
        'min_gain_to_split': 7,
        'num_threads': -1,
        'min_sum_hessian_in_leaf': 10.0,
        'boost_from_average':'false'}

    dataset = "dataset1"
    logger.write("")
    logger.write(f"Current model: LightGBM")
    logger.write(f"Current dataset: {dataset}")

    # Data loading mode
    train_file = osp.join(INPUT_PATH, dataset, "train.csv")
    test_file = osp.join(INPUT_PATH, dataset, "test.csv")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    stat_columns_file = osp.join(INPUT_PATH, "stat_columns.txt")
    category_columns_file = osp.join(INPUT_PATH, "category_columns.txt")
    stat_columns = pickle_load(stat_columns_file)
    category_columns = pickle_load(category_columns_file)
    feature_columns = stat_columns + category_columns

    X_train = train_df[feature_columns].values
    y_train = train_df["target"].values
    X_test = test_df[feature_columns].values
    y_test = test_df["target"].values


    logger.write("x_train.shape: " + str(X_train.shape))
    logger.write("y_train.shape: " + str(y_train.shape))
    logger.write("x_test.shape: " + str(X_test.shape))

    # Train process
    logger.write("Training...")

    category_idx = [i for i in range(len(feature_columns)) if feature_columns[i] in category_columns]
    model = lgb_classifier(X_train, y_train, X_test, y_test, category_idx, lgb_params)

    # Predict process
    logger.write("Predicting...")

    test_prob = model.predict(X_test)

    logger.write(f"max test prob: {np.max(test_prob)}")
    logger.write(f"min test prob: {np.min(test_prob)}")
    logger.write(f"origin test box num:{y_test.sum()}")


    # calc metrics
    logger.write("Metrics calculation...")
    recall_precision_score(test_prob, y_test)



def recall_precision_score(y_prob, y_true):
    P_R_scores = []
    for i in np.arange(0.01, 0.98, 0.0015):
        y_pred = y_prob > i
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        P_R_scores.append((i, precision, recall))

        if precision >= 0.95:
            break

    recall_90 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.9))
    recall_85 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.85))
    recall_80 = sorted(P_R_scores, key=lambda x: abs(x[1] - 0.8))

    logger.write(f"precision: {recall_90[0][1]}, recall: {recall_90[0][2]} at split: {recall_90[0][0]} when precision is 0.90")
    logger.write(f"precision: {recall_85[0][1]}, recall: {recall_85[0][2]} at split: {recall_85[0][0]} when precision is 0.85")
    logger.write(f"precision: {recall_80[0][1]}, recall: {recall_80[0][2]} at split: {recall_80[0][0]} when precision is 0.80")


if __name__ =="__main__":
    main()
