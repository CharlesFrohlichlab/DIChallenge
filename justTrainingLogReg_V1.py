# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:19:22 2019

@author: The Iron Maiden
"""

#### Import toolboxes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
#print(data.columns)

# separate category and feature data
dataX_all=data.drop('win',axis=1)
dataY=data['win']

# define columns to analyze
columns2Keep = columns2Keep = ['champion_name', 'participant_id', 
       'match_rank_score', 'account_id', 'champion_id', 
       'companion_score', 'split_score', 'rotation_score', 'gold_earned',
       'kills', 'deaths', 'assists', 'total_minions_killed',
       'neutral_minions_killed', 'neutral_minions_killed_team_jungle',
       'neutral_minions_killed_enemy_jungle', 'wards_placed', 'wards_killed',
       'damage_self_mitigated', 'total_damage_taken',
       'damage_dealt_to_objectives', 'damage_dealt_to_turrets',
       'magic_damage_dealt_to_champions', 'physical_damage_dealt_to_champions',
       'true_damage_dealt_do_dhampions', 'total_heal', 'time_ccing_others',
       'percent_taken', 'total_damage_dealt_to_champions', 'percent_magic',
       'kda', 'max_time']


dataX = dataX_all[columns2Keep]


###### Logistic regression Data Preprocessing

# Define which columns should be encoded vs scaled
columns_to_encode = ['champion_name']

columns_to_scale  = ['participant_id', 
       'match_rank_score', 'account_id', 'champion_id', 
       'companion_score',  'gold_earned',
       'kills', 'deaths', 'assists', 'total_minions_killed',
       'neutral_minions_killed', 'neutral_minions_killed_team_jungle',
       'neutral_minions_killed_enemy_jungle', 'wards_placed', 'wards_killed',
       'damage_self_mitigated', 'total_damage_taken',
       'damage_dealt_to_objectives', 'damage_dealt_to_turrets',
       'magic_damage_dealt_to_champions', 'physical_damage_dealt_to_champions',
       'true_damage_dealt_do_dhampions', 'total_heal', 'time_ccing_others',
       'percent_taken', 'total_damage_dealt_to_champions',
       'kda', 'max_time']

colRename = ['partID', 
       'Rank', 'Acct', 'Champ', 
       'CompanScore',  'Gold',
       'K', 'D', 'A', 'minions',
       'nMinions', 'nMinions_Us',
       'nMinions_Enem', 'wards', 'wardsKilled',
       'dmgMit', 'totDmg',
       'dmgObj', 'dmgTurr',
       'magic2Champ', 'phys2Champ',
       'true2Champ', 'heal', 'cc',
       'perTaken', 'dmg2Champ',
       'kda', 'maxTime']

# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)
# Scale and Encode the continuous and categorical data separately
scaled_columnsX  = scaler.fit_transform(dataX[columns_to_scale]) 
encodedColumnsX =    ohe.fit_transform( dataX[columns_to_encode])

# Concatenate (Column-Bind) Processed Columns Back Together
processedX = np.concatenate([scaled_columnsX, encodedColumnsX], axis=1)

# from scikitlearn: split data into test and training sets
xTrain,xTest,yTrain,yTest=train_test_split(processedX,dataY,test_size=0.2,random_state=42)

## for test data, take mean across games

testMeanGames = 0
if testMeanGames == 1:
    
    # calculate mean across matches for post-game metrics
    dfPlayer_shape = dfPlayer.shape    
    
    # take mean across continuous columns        
    column_list = ['max_time','gold_earned','wards_placed','damage_dealt_to_objectives',
                                'damage_dealt_to_turrets','kda',
                                'total_damage_dealt_to_champions']
    row_index_list = range(0,dfPlayer_shape[0])        
    matchMean = dfPlayer[column_list].iloc[row_index_list].mean(axis=0)
    
    # replace all rows of dfPlayer with mean
    for iRow in range(0,dfPlayer_shape[0]) : # 
        dfPlayer.iloc[ iRow,2:] = matchMean
    

###### Logistic regression

parameters=[
{
    'penalty':['l1','l2'],
    'C':[0.01, 0.1, 1, 10, 100],
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

numVars = 28
logCoefs = bestLR.coef_

x_labels = colRename
plt.figure(figsize=(20,5))
plt.bar(columns_to_scale[0:numVars],logCoefs[0,0:numVars])
plt.ylabel('Coef Score')
plt.xticks(np.arange(numVars), x_labels, rotation = 45, fontsize=13 )
plt.title('Log Reg Coef Scores')

# plot absolute value of coeff and sort by highest coeff

logCoefs_abs = abs(bestLR.coef_)
logCoefs_absSort = sorted(logCoefs_abs[0,0:numVars],reverse=True)
sortedInds = np.argsort(-logCoefs_abs[0,0:numVars])

plt.figure(figsize=(20,5))
plt.bar(columns_to_scale[0:numVars],logCoefs_absSort)
plt.ylabel('Coefficient Score (Impact)', fontsize=14)
plt.xticks(np.arange(numVars), [x_labels[i] for i in sortedInds], rotation = 45, fontsize=13 ) # need to reorder x labels according to sorting of coeffs
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
