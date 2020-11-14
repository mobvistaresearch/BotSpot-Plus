#!usr/bin/python3s
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
from fastFM import sgd, mcmc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import shap

from metric import calc_precision, calc_recall
from sklearn.metrics import recall_score

# import matplotlib.pyplot as plt

SEED = 1234


def lgb_classifier(X_train, y_train, X_valid, y_valid, categorical_feature, lgb_params):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_model = lgb.train(lgb_params,
                      train_data,
                      valid_sets=[valid_data],
                      early_stopping_rounds=500,
                      verbose_eval=1,
                      categorical_feature=categorical_feature,
                      feval = lambda preds, train_data: [calc_precision(preds, train_data),
                                                  calc_recall(preds, train_data)],
                      evals_result=evals_result)
    return lgb_model
