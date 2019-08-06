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

#params
role = 'supp'

#### Import toolboxes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pickle

# use riotAPI_matchInfo for individual match info; riotAPI_avgMatchInfo to take avg across all matches for each sample
from riotAPI_avgMatchInfo import dfPlayer, dataYPlayer # IMPORTANT: separate script to pull data from RiotAPI for specific player data


from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

#### Load data

data=pd.read_csv("C:\\Users\\The Iron Maiden\\Documents\\DataScienceProjects\\supp_playerDB_cleaned.csv")

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
print(data.columns)

# separate category and feature data
dataX_all=data.drop('win',axis=1)
dataY=data['win']

# define columns to analyze
columns2Keep = ['champion_name','match_rank_score','max_time','goldearned','wardsplaced','damagedealttoobjectives',
                'damagedealttoturrets','kda','totaldamagedealttochampions', 'totaldamagetaken', 'totalminionskilled',
                'opp'+role]
dataX = dataX_all[columns2Keep]

# append player data for on hot encoding and scaling
numPlayerSamps = dfPlayer.shape[0]
dfPlayer = dfPlayer.drop('playersupp',axis=1)
###### Logistic regression Data Preprocessing

# Define which columns should be encoded vs scaled
columns_to_encode_champName = ['champion_name']
columns_to_encode_oppChamp = ['opp'+role]
columns_to_scale  = columns2Keep[1:-1]
# we're going to encode the categorical data together (dataX + player) since we might find new champions in the player data
toEncode_champName_plusPlayer = dataX[columns_to_encode_champName].append(dfPlayer[columns_to_encode_champName])
toEncode_oppChamp_plusPlayer = dataX[columns_to_encode_oppChamp].append(dfPlayer[columns_to_encode_oppChamp])

# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)
ohe_opp    = OneHotEncoder(sparse=False)
# Scale and Encode the continuous and categorical data separately
scaled_columnsX  = scaler.fit_transform(dataX[columns_to_scale]) 
filename = 'scaler_lolpredict.sav'
#pickle.dump(scaler, open(filename, 'wb'))

encoded_playerChamp =    ohe.fit_transform(toEncode_champName_plusPlayer)
filename = 'ohe_lolpredict.sav'
#pickle.dump(ohe, open(filename, 'wb'))

encoded_oppChamp =    ohe_opp.fit_transform(toEncode_oppChamp_plusPlayer)
filename = 'oheOpp_lolpredict.sav'
#pickle.dump(ohe_opp, open(filename, 'wb'))

scaled_columns_player  = scaler.transform(dfPlayer[columns_to_scale]) 

# IMPORTANT: split appended player data off after one hot encoding 
encoded_champName_X = encoded_playerChamp[:-numPlayerSamps,:]
encoded_champName_player = encoded_playerChamp[-numPlayerSamps:,:]

encoded_oppChamp_X = encoded_oppChamp[:-numPlayerSamps,:]
encoded_oppChamp_player = encoded_oppChamp[-numPlayerSamps:,:]

# Concatenate (Column-Bind) Processed Columns Back Together
processedX = np.concatenate([scaled_columnsX, encoded_champName_X,encoded_oppChamp_X], axis=1)
processedPlayerX = np.concatenate([scaled_columns_player, encoded_champName_player,encoded_oppChamp_player], axis=1)


# from scikitlearn: split data into test and training sets
xTrain,xTest,yTrain,yTest=train_test_split(processedX,dataY,test_size=0.2,random_state=42)

###### Logistic regression

params_lrc=[
{
    'penalty':['l1','l2'],
    'C':[ 0.1,0.5,1,1.5, 2, 3, 4],
    'random_state':[0]
    },
]

lrc=LogisticRegression()

gs_model = GridSearchCV(lrc, params_lrc,  cv= 5, scoring='accuracy') 
gs_model.fit(xTrain, yTrain)
print('Best parameters set:')
print(gs_model.best_params_)

pred = gs_model.predict(xTest)

from sklearn.metrics import accuracy_score
print('Optimized logistic regression performance: ',
      round(accuracy_score(yTest,pred),5)*100,'%')

# save the model to disk
#filename = 'final_logRegLoL.sav'
#pickle.dump(gs_model, open(filename, 'wb'))
#gs_model = pickle.load(open(filename, 'rb'))

#### examine contribution of variables to win

numVars = len(columns_to_scale)

bestLR=LogisticRegression(C=1,penalty='l1',random_state=0)
bestLR.fit(xTrain, yTrain)

logCoefs = bestLR.coef_

x_labels = ['Rank','MaxTime','Gold','Wards','ObjDmg','TurretDmg','KDA','ChampDmg', 'dmgTaken', 'minion#']
plt.bar(columns_to_scale[0:numVars],logCoefs[0,0:numVars])
plt.ylabel('Coef Score')
plt.xticks(np.arange(numVars), x_labels, rotation = 45, fontsize=13 )
plt.title('Log Reg Coef Scores')

# plot absolute value of coeff and sort by highest coeff

logCoefs_abs = abs(bestLR.coef_)
logCoefs_absSort = sorted(logCoefs_abs[0,0:numVars],reverse=True)
sortedInds = np.argsort(-logCoefs_abs[0,0:numVars])

plt.figure(figsize=(10,5))
plt.bar(columns_to_scale[0:numVars],logCoefs_absSort)
plt.ylabel('Coefficient Score (Impact)', fontsize=14)
plt.xticks(np.arange(numVars), [x_labels[i] for i in sortedInds], rotation = 45, fontsize=13 ) # need to reorder x labels according to sorting of coeffs
plt.yticks(fontsize=13)
plt.title('Player Metrics Sorted by Impact on Win/Loss', fontsize=14)

#### calculate model performance for test data

## calculate predicted probability
#prob = gs_model.predict_proba(xTest)[:,1]
## calculate true and false pos 
#falsePos,truePos,thresh = roc_curve(yTest,prob)
##Calculate area under the curve
#AUCscore = roc_auc_score(yTest,prob)
#
## ROC plot
#sns.set_style('whitegrid')
#plt.figure(figsize=(8,5))
#
#plt.plot(falsePos,truePos)
#plt.plot([0,1],ls='--')
#plt.plot([0,0],[1,0],c='.5')
#plt.plot([1,1],c='.5')
#
#plt.title('ROC Curve; AUC = ' + str(round(AUCscore,5)) + '; Model Test Accuracy = ' + str(round(accuracy_score(yTest,pred),3)*100) + '%')
#plt.ylabel('True positive rate')
#plt.xlabel('False positive rate')
#plt.show()

#### Now predict game outcome for player data pulled from riot API

pred = gs_model.predict(processedPlayerX)

print('Optimized logistic regression performance: ',
      round(accuracy_score(dataYPlayer,pred),5)*100,'%')

# calculate predicted probability
prob = gs_model.predict_proba(processedPlayerX)[:,1]
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

plt.title('ROC Curve; AUC = ' + str(round(AUCscore,5)) + '; Model Test Accuracy = ' + str(round(accuracy_score(dataYPlayer,pred),3)*100) + '%',fontsize = 20)
plt.ylabel('True positive rate',fontsize = 20)
plt.xlabel('False positive rate',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()

################### plot learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Check: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html for details

    """
    plt.figure()
    plt.title(title, fontsize = 20)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize = 20)
    plt.ylabel("Score", fontsize = 20)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    return plt

from sklearn.model_selection import ShuffleSplit

title = "Learning Curves (Logistic Regression Classifier)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(lrc, title, xTrain, yTrain, ylim=(0.75, 0.95), cv=cv, n_jobs=4)

plt.show()

################### process data and plot radarplot

import plotly.graph_objects as go
from plotly.offline import plot

# append column for data group
tmpData = dataX.drop('champion_name',axis=1).drop('oppsupp',axis=1).assign(Group='data')
tmpDataPlayer = dfPlayer.drop('champion_name',axis=1).drop('oppsupp',axis=1).assign(Group='player')
allDataWithPlayer = tmpData.append(tmpDataPlayer, ignore_index=True)

# normalize (0-1) ccontinuous data and add back on group
df_2norm = allDataWithPlayer.iloc[:,1:-1] # .drop('oppsupp',axis=1).drop('playersupp',axis=1).drop('Group',axis=1)
normalized_df=( df_2norm-df_2norm.min() )/( df_2norm.max()-df_2norm.min() )
normalized_df['Group']=allDataWithPlayer['Group']

col2Group = columns2Keep[1:-1]
justDataData = normalized_df.loc[normalized_df['Group'] == 'data']
norm_dataMean = justDataData[0:1000].mean(axis=0)
norm_playerMean = normalized_df.loc[normalized_df['Group'] == 'player'].mean(axis=0)

######

# for displaying plotly in spyder: https://community.plot.ly/t/plotly-for-spyder/10527/3
categories = columns2Keep[1:-1]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=norm_dataMean,
      theta=categories,
      fill='toself',
      name='Average Player Performance'
))

fig.add_trace(go.Scatterpolar(
      r=norm_playerMean,
      theta=categories,
      fill='toself',
      name='Your Performance'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
     
    )),
  showlegend=True
)
# fig.show()
plot(fig, auto_open=True)


######### plotly bar chart

topThreeFeatures = [columns_to_scale[i] for i in sortedInds][:3]

topThreePlayer = dfPlayer[topThreeFeatures].mean(axis=0)
topThreeData = dataX[topThreeFeatures].mean(axis=0)

topThreeAll = allDataWithPlayer[topThreeFeatures].mean(axis=0)
stdDevThree_player = allDataWithPlayer[topThreeFeatures].std(axis=0)/2
ylimLow = topThreeAll - stdDevThree_player
ylimHigh = topThreeAll + stdDevThree_player 

xLabels=['Your Performance', 'Average Player']

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3)

for i in range(3):

    fig.add_trace(
        go.Bar(name=topThreeFeatures[i], x=xLabels, y=[topThreePlayer[i],topThreeData[i]]),
        row=1, col=i+1
    )
    
    
fig.update_layout(height=600, width=800, title_text="Subplots")
fig.show()
plot(fig, auto_open=True)

############ 

import matplotlib.pyplot as plt


tmpTitles = ['Kill/Death/Assist Ratio', 'Gold Earned', 'Damage to Turrets']
tmpYlab = ['KDA Ratio','Gold','Damage Units']
tmpYlim = [ [0,5.5],[6000,9000],[1000,1700] ]
fig2, axs = plt.subplots(1, 3,figsize=(16,4))

x_pos = np.arange(len(xLabels))

for i in range(3):

    axs[i].bar(x_pos[0], topThreePlayer[i], align='center', alpha=0.8)
    axs[i].bar(x_pos[1], topThreeData[i], align='center', alpha=0.8)
    axs[i].set_title(tmpTitles[i], fontsize = 20)
    axs[i].set_ylabel(tmpYlab[i], fontsize = 15)
    axs[i].set_ylim(tmpYlim[i])
    
    axs[i].set_xticks(x_pos)
    axs[i].set_xticklabels(xLabels, rotation=0, fontsize=15)
   

    
fig2.show()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO,StringIO

buff = BytesIO()
plt.savefig(buff, format='png', dpi=180)
buff.seek(0)


