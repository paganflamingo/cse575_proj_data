import sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



data = pd.read_csv("output.csv") #filename
labels = data['koi_disposition'].to_numpy()
data = data.drop('koi_disposition', axis=1).to_numpy()
#x = data.drop('koi_disposition', axis = 1)
#y = data['koi_disposition']

kf = KFold(n_splits=5,shuffle = True, random_state =0)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
accArr = []
macroArr = []
microArr = []
weightArr = []
tpArr = []
fpArr = []
fnArr = []
tnArr = []

for train_index, test_index in kf.split(data, labels):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print("length of xtrain: ", len(x_train))
    print("length of xtest: ", len(x_test))
    abTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=1000)
    abTree = abTree.fit(x_train,y_train)
    abtPred = abTree.predict(x_test)
    #y_pred = model.predict(x_test_std)
    cm = confusion_matrix(y_test, abtPred)
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    recall = TP/(TP+FP)
    precision = TP/(TP+FN)
    accur = metrics.accuracy_score(y_test,abtPred)
    #print("Boosted Tree Accuracy:",accur)
    accArr.append(accur)
    
    mac =f1_score(y_test, abtPred, average='macro')
    #print("Macro F1_Score: ", mac)
    macroArr.append(mac)
    
    micro = f1_score(y_test, abtPred, average='micro')
    #print("Micro F1_Score: ", micro)
    microArr.append(micro)
    
    weight =f1_score(y_test, abtPred, average='weighted')
    #print("Weighted F1_Score: ", weight)
    weightArr.append(weight)
    
    tpArr.append(TP)
    fpArr.append(FP)
    fnArr.append(FN)
    tnArr.append(TN)


avgAcc = np.average(accArr)
avgMac = np.average(macroArr)
avgMic = np.average(microArr)
avgWeight = np.average(weightArr)
avgTP = np.average(tpArr)
avgFP = np.average(fpArr)
avgTN = np.average(tnArr)
avgFN = np.average(fnArr)

print("Accuracy " ,avgAcc)
print("Macro F1 " ,avgMac)
print("Micro F1 ",avgMic)
print("Weighted F1 ",avgWeight)
print("TP ", avgTP)
print("FP ", avgFP)
print("FN ", avgFN)
print("TN ", avgTN)
'''
xtrain,xtest,ytrain,ytest = train_test_split(x,y, train_size = 0.7, test_size = 0.3, random_state = 0)



treeCLF = DecisionTreeClassifier(max_depth = 1)
treeCLF = treeCLF.fit(xtrain,ytrain)
treePredict = treeCLF.predict(xtest)
cmtree = confusion_matrix(ytest, treePredict)
print("Tree Accuracy:",metrics.accuracy_score(ytest,treePredict))
print(cmtree)

ab = AdaBoostClassifier(n_estimators=1000,learning_rate=1,algorithm='SAMME')
ab = ab.fit(xtrain, ytrain)
boostPred = ab.predict(xtest)
cmboost = confusion_matrix(ytest, boostPred)
print("Boost Accuracy:",metrics.accuracy_score(ytest,boostPred))
print(cmboost)

sumabtPred = 0;


abTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=1000)
abTree = abTree.fit(xtrain,ytrain)
abtPred = abTree.predict(xtest)
cmABT = confusion_matrix(ytest, abtPred)
cmPlot = plot_confusion_matrix(abTree, xtest, ytest,cmap=plt.cm.Blues)
plt.show()
TP = cmABT[0][0]
FP = cmABT[0][1]
FN = cmABT[1][0]
TN = cmABT[1][1]

recall = TP/(TP+FP)
precision = TP/(TP+FN)

print("Boosted Tree Accuracy:",metrics.accuracy_score(ytest,abtPred))

print("Macro F1_Score: ", f1_score(ytest, abtPred, average='macro'))

print("Micro F1_Score: ", f1_score(ytest, abtPred, average='micro'))


print("Weighted F1_Score: ", f1_score(ytest, abtPred, average='weighted'))


print(cmPlot)
'''
