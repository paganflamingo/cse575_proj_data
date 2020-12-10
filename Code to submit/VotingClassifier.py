import pandas as pd
import numpy as np

# Import the data
RAW = pd.read_csv("output.csv")

# prep the data
ylist = RAW['koi_disposition'].to_list()
y = np.array(ylist)

Xdf = RAW.drop(columns=['koi_disposition'])
xnplist = [np.array(xi) for xi in Xdf.values]
X = np.stack(xnplist)



# Intialize the classifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm

clf1 = LogisticRegression(max_iter = 1000,random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = svm.SVC(probability = True)

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)], voting='soft')



# Train and evaluate the classifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=0, shuffle = True)

avgs = []
macrof1s = []
microf1s = []
weightedf1s = []
tps = []
fps = []
fns = []
tns = []


for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    eclf1 = eclf1.fit(X_train, y_train)
    
    y_pred = eclf1.predict(X_test)
    avgs.append(accuracy_score(y_test, y_pred))
    macrof1s.append(f1_score(y_test, y_pred, average='macro'))
    microf1s.append(f1_score(y_test, y_pred, average='micro'))
    weightedf1s.append(f1_score(y_test, y_pred, average='weighted'))
    tps.append(confusion_matrix(y_test, y_pred)[0][0])
    fps.append(confusion_matrix(y_test, y_pred)[0][1])
    fns.append(confusion_matrix(y_test, y_pred)[1][0])
    tns.append(confusion_matrix(y_test, y_pred)[1][1])
    
    

print("average Accuracy: " + str(sum(avgs)/len(avgs)))
print("average Macro F1 Score: " + str(sum(macrof1s)/len(avgs)))
print("average Micro F1 Score: " + str(sum(microf1s)/len(avgs)))
print("average weighted F1 Score: " + str(sum(weightedf1s)/len(avgs)))
print("average number of true positive rate: " + str(sum(tps)/len(avgs)))
print("average number of false positives: " + str(sum(fps)/len(avgs)))
print("average number of true negatives: " + str(sum(fns)/len(avgs)))
print("average number of true negatives: " + str(sum(tns)/len(avgs)))