# Botspot++: A Hierarchical Deep Ensemble Model for Bots Install Fraud Detection in Mobile Advertising
## Datasets
To evaluate our proposed model more comprehensively, we built four datasets for different time periods, which are avaliable from https://drive.google.com/drive/folders/1SJfWzhqnKcfF1aUh9z9fBfBFtEGhyd0r?usp=sharing. And The statistics data of the four offline datasets are detailed as below.

| Dataset | #Dev | #Chan-Camp | #Normal Install(Train, Test) | #Bots Install(Train, Test) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| dataset-1 | 1676101 | 2428 | 1249059, 163413 | 270827, 20561 |
| dataset-2 | 1216796 | 2135 | 884970, 141546  | 205567, 13865 |
| dataset-3 | 1313073 | 2073 | 1051838, 196119 | 139358, 9598  |
| dataset-4 | 1299895 | 1995 | 1156202, 181953 | 77717, 12018  |

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
download them from this link(https://drive.google.com/drive/folders/1SJfWzhqnKcfF1aUh9z9fBfBFtEGhyd0r?usp=sharing) and put the input folder on current folder.
4. Model training
LightGBM:  
&nbsp;&nbsp;&nbsp;&nbsp;```cd ML```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset is used for training and the parameters of LightGBM```  
&nbsp;&nbsp;&nbsp;&nbsp;```python3 main.py --dataset dataset1 --num_trees 500 --max_depth 5```  
MLP:   
&nbsp;&nbsp;&nbsp;&nbsp;```cd DL/MLP```  
&nbsp;&nbsp;&nbsp;&nbsp;```# set which dataset is used for training``` 
&nbsp;&nbsp;&nbsp;&nbsp;```python3 main.py --dataset dataset1```  
