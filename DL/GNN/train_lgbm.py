from sklearn.feature_selection import *
from sklearn.ensemble import *
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sys import argv

import pickle
import numpy as np
import pandas as pd
from sys import argv


def recall_at_90_precision(hidden_vec, labels):
    try:
        hidden_vec = hidden_vec.view(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    except:
        pass
    rec = []
    for j in np.arange(0.005, 0.999, 0.001):
        # print (j)
        y_pred = hidden_vec > j
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        # print (p,r)
        rec.append((j,p,r))
        if p >= 0.999:
            break
    rec_90 = sorted(rec, key=lambda x: abs(x[1] - 0.9))
    rec_85 = sorted(rec, key=lambda x: abs(x[1] - 0.85))
    rec_80 = sorted(rec, key=lambda x: abs(x[1] - 0.8))
    print ('90% precision:', rec_90[0])
    print ('85% precision:', rec_85[0])
    print('80% precision:', rec_80[0])
    return rec


def make_dataset(chan_feats,device_feats,channel_split_val,device_split_val,edge_index_train,edge_index_test):
    train_dataset = np.hstack((chan_feats[edge_index_train[:,0]],device_feats[edge_index_train[:,1]]))
    test_dataset = np.hstack((chan_feats[edge_index_test[:,0]],device_feats[edge_index_test[:,1]]))
    train_labels = edge_index_train[:,-1]
    test_labels = edge_index_test[:,-1]
    print ('training GBM')
    params = {'boosting_type': ['gbdt'], 'max_depth': [3,4], 'n_estimators': [50,100,200]}
    lgbm = lgb.LGBMClassifier(objective='binary')
    clf = GridSearchCV(lgbm, params, cv=4, scoring='roc_auc')
    clf.fit(train_dataset, train_labels)
    gbm_model = clf.best_estimator_
    print ('best model is:',gbm_model)
    probs = gbm_model.predict_proba(test_dataset)
    recall_at_90_precision(probs.reshape(-1,),test_labels)
    return gbm_model
