# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:22:12 2019

Objective: Load and clean League of Legends game data to identify important 
    game features that contribute and predict game outcome. This is important 
    because the goal of many players is to identify aspects of the game to improve
    on and the analyses outlined below is generalizable to all skill-based games
    and sports.
    
Data from: https://github.com/DoransLab/data/tree/master/champion_clustering

@author: Zhe Charles Zhou
"""

#### Import toolboxes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from riotAPI_matchInfo import dfPlayer, dataYPlayer # IMPORTANT: separate script to pull data from RiotAPI for specific player data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

#### Load data

data=pd.read_csv("C:\\Users\\The Iron Maiden\\Documents\\DataScienceProjects\\totalSup.csv")

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
print(data.columns)

# separate category and feature data
dataX_all=data.drop('win',axis=1)
dataY=data['win']

# define columns to analyze
columns2Keep = ['champion_name','match_rank_score','max_time','gold_earned','wards_placed','damage_dealt_to_objectives','damage_dealt_to_turrets','kda','total_damage_dealt_to_champions']
dataX = dataX_all[columns2Keep]

# append player data for on hot encoding and scaling
numPlayerSamps = dfPlayer.shape[0]
appendXPlayer = dataX.append(dfPlayer)

###### Logistic regression Data Preprocessing

# Define which columns should be encoded vs scaled
columns_to_encode = ['champion_name']
columns_to_scale  = ['match_rank_score','max_time','gold_earned','wards_placed','damage_dealt_to_objectives','damage_dealt_to_turrets','kda','total_damage_dealt_to_champions']
# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)
# Scale and Encode Separate Columns
scaled_columns  = scaler.fit_transform(appendXPlayer[columns_to_scale]) 
encoded_columns =    ohe.fit_transform(appendXPlayer[columns_to_encode])
# Concatenate (Column-Bind) Processed Columns Back Together
processedX_appendPlayer = np.concatenate([scaled_columns, encoded_columns], axis=1)

# IMPORTANT: split appended player data off after one hot encoding and scaling
processedX = processedX_appendPlayer[:-numPlayerSamps,:]
processedPlayerX = processedX_appendPlayer[-numPlayerSamps:,:]

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

#### examine contribution of variables to win

bestLR=LogisticRegression(C=1,penalty='l1',random_state=0)
bestLR.fit(xTrain, yTrain)

logCoefs = bestLR.coef_

x_labels = ['Rank','MaxTime','Gold','Wards','ObjDmg','TurretDmg','KDA','ChampDmg']
plt.bar(columns_to_scale[0:8],logCoefs[0,0:8])
plt.ylabel('Coef Score')
plt.xticks(np.arange(8), x_labels)
plt.title('Log Reg Coef Scores')

# plot absolute value of coeff and sort by highest coeff

logCoefs_abs = abs(bestLR.coef_)
logCoefs_absSort = sorted(logCoefs_abs[0,0:8],reverse=True)
sortedInds = np.argsort(-logCoefs_abs[0,0:8])

plt.figure(figsize=(10,5))
plt.bar(columns_to_scale[0:8],logCoefs_absSort)
plt.ylabel('Coefficient Score (Impact)', fontsize=14)
plt.xticks(np.arange(8), [x_labels[i] for i in sortedInds], fontsize=13 ) # need to reorder x labels according to sorting of coeffs
plt.yticks(fontsize=13)
plt.title('Player Metrics Sorted by Impact on Win/Loss', fontsize=14)

#### calculate model performance

# calculate predicted probability
prob = logOptimal.predict_proba(xTest)[:,1]
# calculate true and false pos 
falsePos,truePos,thresh = roc_curve(yTest,prob)
#Calculate area under the curve
AUCscore = roc_auc_score(yTest,prob)

# ROC plot
sns.set_style('whitegrid')
plt.figure(figsize=(8,5))

plt.plot(falsePos,truePos)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')

plt.title('ROC Curve; AUC = ' + str(round(AUCscore,5)) + '; Model Test Accuracy = ' + str(round(accuracy_score(yTest,pred),3)*100) + '%')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

#### Now predict game outcome for player data pulled from riot API

pred = logOptimal.predict(processedPlayerX)

print('Optimized logistic regression performance: ',
      round(accuracy_score(dataYPlayer,pred),5)*100,'%')

# calculate predicted probability
prob = logOptimal.predict_proba(processedPlayerX)[:,1]
# calculate true and false pos 
falsePos,truePos,thresh = roc_curve(dataYPlayer,prob)
#Calculate area under the curve
AUCscore = roc_auc_score(dataYPlayer,prob)

# ROC plot
sns.set_style('whitegrid')
plt.figure(figsize=(8,5))

plt.plot(falsePos,truePos)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')

plt.title('ROC Curve; AUC = ' + str(round(AUCscore,5)) + '; Model Test Accuracy = ' + str(round(accuracy_score(dataYPlayer,pred),3)*100) + '%')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()
