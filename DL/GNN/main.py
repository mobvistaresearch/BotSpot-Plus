
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from  torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.preprocessing import *
from model_train_utils import *
from model_utils import *
from apex import amp
import sys
from sys import argv
import os
import os.path as osp
import time
import datetime
from itertools import repeat
from typing import Optional
import gc
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from preprocess import BotSpotTrans
from train_lgbm import *

if __name__ == '__main__':
    inputs = argv[1] # e.g., dataset1 or dataset2
    use_gbm = argv[2] # True or False
    use_hierarchy = argv[3]  # True or False
    split_normal_bot = argv[4] # True or False
    bot_preprocess = BotSpotTrans(f'../../input/{inputs}/train.csv',f'../../input/{inputs}/test.csv',inputs)
    edge_index_train = bot_preprocess.edge_index_train
    edge_index_test = bot_preprocess.edge_index_test
    chan_feats = bot_preprocess.combin_matrix
    device_feats = bot_preprocess.device_matrix
    device_package = list(device_feats[:,-1])
    device_feats = device_feats[:,:-1]
    if use_gbm:
        gbm_model = make_dataset(chan_feats, device_feats, -1, 1, edge_index_train, edge_index_test)
    else:
        gbm_model = None

    botspot_model = BotSpot(edge_index_train, edge_index_test, chan_feats, device_feats, gbm_best_model=gbm_model,
                            agg='mean', split_bots_normal=split_normal_bot, use_hierarchy=use_hierarchy,
                            device_package=device_package, channel_feats_unnormed=chan_feats,
                            device_feats_unnormed=device_feats)
    tr_dset = TensorDataset(torch.from_numpy(edge_index_train).long())
    tr_dloader = DataLoader(tr_dset, batch_size=50, shuffle=True)
    test_dset = TensorDataset(torch.from_numpy(edge_index_test).long())
    test_dloader = DataLoader(test_dset, batch_size=5000, shuffle=False)
    optimizer = optim.Adam(botspot_model.parameters(), lr=2e-4, weight_decay=3e-6)
    botspot_model.to(Device)
    # apex
    opt_level = 'O1'
    botspot_model, optimizer = amp.initialize(botspot_model, optimizer, opt_level=opt_level)
    try:
        os.mkdir(inputs+'/')
    except:
        pass
    _ = train(botspot_model,tr_dloader,test_dloader,optimizer,10,save_name = inputs)


