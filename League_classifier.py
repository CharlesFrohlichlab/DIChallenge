# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:22:12 2019

@author: Zhe
"""


### Import toolboxes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

### Load data

data=pd.read_csv("C:\\Users\\Zhe\\Documents\\DataScienceProjects\\NYPD_Collisions\\totalSup.csv")

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
print(data.columns)

# separate category and feature data
dataX_all=data.drop('win',axis=1)
dataY=data['win']

# define columns to analyze
columns2Keep = ['champion_name','match_rank_score','companion_score','gold_earned','wards_placed','damage_dealt_to_objectives','damage_dealt_to_turrets','kda','total_damage_dealt_to_champions']
dataX = dataX_all[columns2Keep]

###### Logistic regression Data Preprocessing

# Define which columns should be encoded vs scaled
columns_to_encode = ['champion_name']
columns_to_scale  = ['match_rank_score','companion_score','gold_earned','wards_placed','damage_dealt_to_objectives','damage_dealt_to_turrets','kda','total_damage_dealt_to_champions']
# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)
# Scale and Encode Separate Columns
scaled_columns  = scaler.fit_transform(dataX[columns_to_scale]) 
encoded_columns =    ohe.fit_transform(dataX[columns_to_encode])
# Concatenate (Column-Bind) Processed Columns Back Together
processedX = np.concatenate([scaled_columns, encoded_columns], axis=1)

# from scikitlearn: split data into test and training sets
xTrain,xTest,yTrain,yTest=train_test_split(processedX,dataY,test_size=0.2,random_state=42)

###### Logistic regression

parameters=[
{
    'penalty':['l1','l2'],
    'C':[0.1,0.4,0.5,1],
    'random_state':[0]
    },
]

logOptimal = GridSearchCV(LogisticRegression(), parameters, scoring='accuracy')
logOptimal.fit(xTrain, yTrain)
print('Best parameters set:')
print(logOptimal.best_params_)

pred = logOptimal.predict(xTest)

from sklearn.metrics import accuracy_score
print('Optimized logistic regression performance: ',
      round(accuracy_score(yTest,pred),5)*100,'%')


