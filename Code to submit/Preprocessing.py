# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:05:38 2020

@author: Suprada Ganesh Nadiger
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

#Reading the CSV file into a dataframe
#d_path = 'cumulative_2020.09.04_15.26.41.csv'
#df = pd.read_csv(d_path, header=0)
#df = pd.read_csv('cumulative_2020.09.04_15.26.41.csv')
df = pd.read_csv(r"C:\Users\SuRaga\Documents\Classes - Suprada\Fall 2020\CSE 575 - Machine Learning\Project\My code\cumulative_2020.09.04_15.26.41.csv")

#Deleting the columns that are not required
df = df.drop(['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_vet_stat', 
              'koi_vet_date','koi_pdisposition','koi_disp_prov', 'koi_comment', 
              'koi_eccen', 'koi_longp', 'koi_fittype', 'koi_ingress',
              'koi_limbdark_mod', 'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_parm_prov', 
              'koi_tce_delivname', 'koi_trans_mod', 'koi_model_dof', 'koi_model_chisq',
              'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sage', 'koi_sparprov',
              'koi_quarters','koi_score','koi_fpflag_nt', 'koi_fpflag_ss', 
              'koi_fpflag_co', 'koi_fpflag_ec'], axis=1)



#In koi_disposition column convert 'Confirmed' to 1 and 'False Positive', 'Candidate' to 0
df = df[df['koi_disposition'] != 'CANDIDATE']
df.koi_disposition.replace(['FALSE POSITIVE','CONFIRMED'],[0,1], inplace = True)
#df.koi_disposition.replace(['FALSE POSITIVE','CANDIDATE','CONFIRMED'],[0,0,1], inplace = True)

#drop all the rows with missing values. Deletes 2282 rows
df.dropna(how='any', inplace=True)


#Generates the correlation matrix and extracts all the features with high correlation into corr_features set
corr_features = set()
corr_matrix = df.drop('koi_disposition', axis=1).corr()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)


#Drop the highly correlated columns
df.drop(corr_features, axis = 1, inplace=True)

#Normalize the data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

#Run the Recursive Feature Elimination method to identify the most important features
features = df.drop('koi_disposition', axis=1)
target = df['koi_disposition']

class_model = RandomForestClassifier(random_state=7001)
#class_model = DecisionTreeClassifier(random_state=7001)
rfe_model = RFECV(estimator=class_model, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfe_model.fit(features, target)

#plot of accuracy versus number of features
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of selected features', fontsize=14, labelpad=20)
plt.ylabel('% Classification accuracy', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfe_model.grid_scores_) + 1), rfe_model.grid_scores_, color='#303F9F', linewidth=3)

plt.show()

#Drop the least important features
features.drop(features.columns[np.where(rfe_model.support_ == False)[0]], axis=1, inplace=True)

#To print the selected features with their importance
imp_matrix = pd.DataFrame()
imp_matrix['Features'] = features.columns
imp_matrix['importance'] = rfe_model.estimator_.feature_importances_

imp_matrix = imp_matrix.sort_values(by='importance', ascending=False)
print(imp_matrix)

#This is the final dataframe with just selected features and target feature
df = features.merge(target, left_index=True, right_index=True)

#Extracting the processed data into a CSV file
df.to_csv(r'output.csv', index = False)
