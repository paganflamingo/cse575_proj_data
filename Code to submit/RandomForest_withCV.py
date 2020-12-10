# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:43:13 2020

@author: SuRaga
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


#Load data
d_path = 'output.csv'
data = pd.read_csv(d_path, header=0)

labels = data['koi_disposition'].to_numpy()
data = data.drop('koi_disposition', axis=1).to_numpy()

# generate kfolds
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# for each kfold
accuracy = []
macrof1 = []
microf1 = []
weightedf1 = []

tp = []
tn = []
fp = []
fn = []

for train_index, test_index in kf.split(data, labels):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    
    #Build the Random Forest classifier on the training data
    RFClassifier = RandomForestClassifier()
    RFClassifier.fit(x_train,y_train)

    #Predict the class labels for the test_data using the Random foreset model
    y_pred = RFClassifier.predict(x_test)

    accuracy.append(accuracy_score(y_pred, y_test))
    macrof1.append(f1_score(y_test, y_pred, average="macro"))
    microf1.append(f1_score(y_test, y_pred, average="micro"))
    weightedf1.append(f1_score(y_test, y_pred, average="weighted"))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()

    fp.append(false_pos)
    fn.append(false_neg)
    tp.append(true_pos)
    tn.append(true_neg)

print(f'Accuracy: {np.mean(accuracy)}')
print(f'Macro F1: {np.mean(macrof1)}')
print(f'Micro F1: {np.mean(microf1)}')
print(f'Weighted F1: {np.mean(weightedf1)}')
print()
print(f'True +: {np.mean(tp)}')
print(f'True -: {np.mean(tn)}')
print(f'False +: {np.mean(fp)}')
print(f'False -: {np.mean(fn)}')