from sklearn.feature_selection import *
from sklearn.ensemble import *
from sklearn.metrics import precision_score,confusion_matrix,f1_score,auc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import torch.nn.utils as utils
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
import torch
import sys
import os
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightgbm as lgb
from  torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.preprocessing import *
# from apex import amp
import time


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def recall_at_90_precision(hidden_vec, labels):
    try:
        hidden_vec = hidden_vec.view(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    except:
        pass
    rec = []
    for j in np.arange(0.001, 0.999, 0.002):
        # print (j)
        y_pred = hidden_vec > j
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        # print (p,r)
        rec.append((j,p,r))
        if p >= 1.91:
            break
    rec_90 = sorted(rec, key=lambda x: abs(x[1] - 0.9))
    rec_85 = sorted(rec, key=lambda x: abs(x[1] - 0.85))
    rec_80 = sorted(rec, key=lambda x: abs(x[1] - 0.8))
    print ('90% precision:', rec_90[0])
    print ('85% precision:', rec_85[0])
    print('80% precision:', rec_80[0])
    return rec


def eval_model(model,data_loader):
    model.eval()
    preds= []
    labels=[]
    with torch.no_grad():
        model.graph_model.super_device_cache = {}
        model.graph_model.channel_cache = {}
        for edges in data_loader:
            if np.random.uniform() >1:
                print ('break')
                break
            if np.random.uniform() > 0.95:
                print (f'current progress: {len(preds)}, out of total of {len(data_loader.dataset)}')
            try:
                hidden_vec, label = model(edges[0])
                preds.extend(list(hidden_vec.view(-1,).cpu().numpy()))
                labels.extend(list(label.view(-1,).cpu().numpy()))
            except:
                continue
    preds, labels = np.asarray(preds),np.asarray(labels)
    rec = recall_at_90_precision(preds, labels)
    return rec

def train(model, tr_data_loader, test_data_loader,optimizer, epoch=5, save_name='test'):
    for i in range(epoch):
        min_loss = 50.
        alpha = 0.95
        total_loss = 0
        model.train()
        print ('epoch: ', i)
        start_time = time.time()
        for index, d in enumerate(tr_data_loader):
            try:
                model.to(Device)
                hidden_vec = model(d[0])
                hidden_vec = hidden_vec.view(-1,)
                label = d[0][:,-1]
                # print ('hidden vec: ',hidden_vec)
                # print ('label: ',label)
                loss = torch.sum(-1.* alpha* torch.log(hidden_vec[label == 1])-1.*(1-alpha)*torch.log(1.-hidden_vec[label == 1]))
                # print ('loss:',loss)
                N = torch.sum(label == 1).item()
                # hard negative mining for loss back prop
                val = -1.*alpha* torch.log(1 - hidden_vec[label == 0]) -1.*(1-alpha)*torch.log(1 - hidden_vec[label == 0])
                neg_loss = torch.sort(val, descending=True)[0][:int(5*N)]
                neg_N = len(neg_loss)
                loss += torch.sum(neg_loss)
                loss = loss / (N+neg_N)
                # print ('loss item: ',loss.item())
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    continue
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # loss.backward()
                utils.clip_grad_value_(model.parameters(), 4)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                current_loss = total_loss/(index + 1)
                if index % 100 == 0:
                    print (f'current loss for index:{index} is: {current_loss}')
                            # num_eval += 1
                            # print ('num_eval:', num_eval)
                            # print (f'training loss for epoch:{i} is: {total_loss/(index +1)}')
            except:
                print ('exception')
                continue
        end_time = time.time()
        print (f'elasped time for epoch {i} is: {end_time - start_time} \n')
        path = f'{save_name}/model_checkpoints_{i}.pt'
        torch.save(model.state_dict(),path)
        if i>=1:
            print ('eval model')
            pr_rec = eval_model(model,test_data_loader)
            path = f'{save_name}/pr_rec_epoch_{i}'
            with open(path,'wb') as f:
                pickle.dump(pr_rec,f)



