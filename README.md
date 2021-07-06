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
