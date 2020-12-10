from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('output.csv')
data = pd.read_csv("output.csv") #filename
labels = data['koi_disposition'].to_numpy()
data = data.drop('koi_disposition', axis=1).to_numpy()

#generate kfolds
kf = KFold(n_splits=5,shuffle = True,random_state=0)



# X = df[[col for col in df.columns if col != 'koi_disposition']]
# y = df['koi_disposition']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# amount_correct = len([x for x,y in zip(list(y_test),list(y_pred)) if y == x])
# f1_macro = f1_score(y_test, y_pred, average='macro')
# f1_micro = f1_score(y_test, y_pred, average='micro')
# f1_weighted = f1_score(y_test, y_pred, average='weighted')
# print(f'accuracy: {amount_correct/len(y_test)}')
# print(f'macro f1: {f1_macro}')
# print(f'micro f1: {f1_micro}')
# print(f'weighted f1: {f1_weighted}')
# # confusion matrix
# cm = confusion_matrix(list(y_test),list(y_pred))
# plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)

# X = df1[[col for col in df1.columns if col != 'koi_disposition']]
# y = df1['koi_disposition']
f1_macro = []
f1_micro = []
f1_weighted = []
accuracy = []
cm = []


for train_index, test_index in kf.split(data, labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print(len(y_test))
    print(len(y_train))

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    amount_correct = len([x for x,y in zip(list(y_test),list(y_pred)) if y == x])
    accuracy.append(amount_correct/len(y_test))
    # f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_macro.append(f1_score(y_test, y_pred, average='macro'))
    # f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_micro.append(f1_score(y_test, y_pred, average='micro'))
    # f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
    # print(f'accuracy: {amount_correct/len(y_test)}')
    # print(f'macro f1: {f1_macro}')
    # print(f'micro f1: {f1_micro}')
    # print(f'weighted f1: {f1_weighted}')
    # confusion matrix
    # cm = confusion_matrix(list(y_test),list(y_pred))
    cm.append(confusion_matrix(list(y_test),list(y_pred)))
    # print(cm)
# plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()

print(sum(accuracy)/len(accuracy))
print(sum(f1_macro)/len(f1_macro))
print(sum(f1_micro)/len(f1_micro))
print(sum(f1_weighted)/len(f1_weighted))
true_pos = []
false_pos = []
false_neg = []
true_neg = []
# print(cm)
for x in cm:
    print(x)
    true_pos.append(x[0][0])
    false_pos.append(x[0][1])
    false_neg.append(x[1][0])
    true_neg.append(x[1][1])
print(sum(true_pos)/len(true_pos))
print(sum(false_pos)/len(false_pos))
print(sum(false_neg)/len(false_neg))
print(sum(true_neg)/len(true_neg))
