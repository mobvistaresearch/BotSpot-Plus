#!usr/bin/python3
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# Using for lightgbm
def calc_precision(preds, train_data):
    threshold = 0.5
    y_true = train_data.get_label()
    y_pred = [int(pred > threshold) for pred in preds]

    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    # print("hahah", cm)
    denominator = float(cm[1][0] + cm[0][0])
    if denominator == 0:
        precision = 0
    else:
        precision =  float(cm[0][0]) / denominator
    return ("precision score: ", precision, True)

def calc_recall(preds, train_data):
    threshold = 0.5
    y_true = train_data.get_label()
    y_pred = [int(pred > threshold) for pred in preds]

    recall = recall_score(y_true, y_pred)


    return ("recall score: ", recall, True)
