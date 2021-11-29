# Botspot++: A Hierarchical Deep Ensemble Model for Bots Install Fraud Detection in Mobile Advertising
## Datasets
To evaluate our proposed model more comprehensively, we built three datasets for different time periods, which are avaliable from https://drive.google.com/drive/folders/1CBIOxCtI5Ztx-E5Ua7nO0UjdEabJM2nC?usp=sharing. And The statistics data of the four offline datasets are detailed as below.

| Dataset | #Dev | #Chan-Camp | #Normal Install(Train, Test) | #Bots Install(Train, Test) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| dataset-1 | 1676101 | 1347 | 1245650, 162960 | 270815, 20560 |
| dataset-2 | 1313073 | 1190 | 1049610, 195792  | 139349, 9596 |
| dataset-3 | 1299895 | 1139 | 1153705, 181437 | 77708, 12016  |

## Requirements
* Pytorch 1.6.0
* LightGBM 3.0.0
* Python 3.6
* scikit-learn 0.23.2
* Numpy 1.19.1

## Usage
1. ```git clone https://github.com/mobvistaresearch/BotSpot-Plus.git```
2. ```cd BotSpot-Plus```
3. Download datasets  
download them from this link(https://drive.google.com/drive/folders/1CBIOxCtI5Ztx-E5Ua7nO0UjdEabJM2nC?usp=sharing) and put the datasets folder on root folder of current project.
4. Model training
LightGBM:  
&nbsp;&nbsp;&nbsp;&nbsp;```cd ML```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset is used for training and the parameters of LightGBM```  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --num_trees 500 --max_depth 5```  
MLP:   
&nbsp;&nbsp;&nbsp;&nbsp;```cd DL/MLP```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset to use for training```  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1```  
GAT:   
&nbsp;&nbsp;&nbsp;&nbsp;```cd DL/GAT```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset to use for training```  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --device_num 0```  
GraphConsis:   
&nbsp;&nbsp;&nbsp;&nbsp;```cd DL/GraphConsis```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset and which gpu device to use for training```  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --device_num 0```  
GraphSAGE、BotSpot、BotSpot++:  
&nbsp;&nbsp;&nbsp;&nbsp;```cd DL/BotSpot```  
&nbsp;&nbsp;&nbsp;&nbsp;```--dataset: the dataset specified, e.g., dataset1, dataset2, etc.```  
&nbsp;&nbsp;&nbsp;&nbsp;```--use_gbm: whether to use gbm model for global context. e.g., take True or False.```   
&nbsp;&nbsp;&nbsp;&nbsp;```--use_stratified: whether to use stratified during message passing, take True or False.```  
&nbsp;&nbsp;&nbsp;&nbsp;```--use_botspot_plus: whether to use botspot_plus```  
&nbsp;&nbsp;&nbsp;&nbsp;```--use_self_attn: whether to use self attention for leaf embeddings```  
&nbsp;&nbsp;&nbsp;&nbsp;```--device_num: set which gpu device to use for training```  
&nbsp;&nbsp;&nbsp;&nbsp;GraphSAGE usage:  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --use_gbm false  --use_stratified false```  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```--use_botspot_plus false --use_self_attn false --device_num 0```  
&nbsp;&nbsp;&nbsp;&nbsp;BotSpot usage:  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --use_gbm true --use_stratified true```  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```--use_botspot_plus false --use_self_attn false --device_num 1```  
&nbsp;&nbsp;&nbsp;&nbsp;BotSpot++ usage:  
&nbsp;&nbsp;&nbsp;&nbsp;```python main.py --dataset dataset1 --use_gbm true --use_stratified true```  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```--use_botspot_plus true --use_self_attn true --device_num 2```

@article{10.1145/3476107,
author = {Zhu, Yadong and Wang, Xiliang and Li, Qing and Yao, Tianjun and Liang, Shangsong},
title = {BotSpot++: A Hierarchical Deep Ensemble Model for Bots Install Fraud Detection in Mobile Advertising},
year = {2021},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {3},
issn = {1046-8188},
url = {https://doi.org/10.1145/3476107},
doi = {10.1145/3476107},
abstract = {Mobile advertising has undoubtedly become one of the fastest-growing industries in the world. The influx of capital attracts increasing fraudsters to defraud money from advertisers. Fraudsters can leverage many techniques, where bots install fraud is the most difficult to detect due to its ability to emulate normal users by implementing sophisticated behavioral patterns to evade from detection rules defined by human experts. Therefore, we proposed BotSpot1 for bots install fraud detection previously. However, there are some drawbacks in BotSpot, such as the sparsity of the devices’ neighbors, weak interactive information of leaf nodes, and noisy labels. In this work, we propose BotSpot++ to improve these drawbacks: (1) for the sparsity of the devices’ neighbors, we propose to construct a super device node to enrich the graph structure and information flow utilizing domain knowledge and a clustering algorithm; (2) for the weak interactive information, we propose to incorporate a self-attention mechanism to enhance the interaction of various leaf nodes; and (3) for the noisy labels, we apply a label smoothing mechanism to alleviate it. Comprehensive experimental results show that BotSpot++ yields the best performance compared with six state-of-the-art baselines. Furthermore, we deploy our model to the advertising platform of Mobvista,2 a leading global mobile advertising company. The online experiments also demonstrate the effectiveness of our proposed method.},
journal = {ACM Trans. Inf. Syst.},
month = {nov},
articleno = {50},
numpages = {28},
keywords = {Mobile ad fraud, graph neural network, bots fraud detection, app install fraud}
}
