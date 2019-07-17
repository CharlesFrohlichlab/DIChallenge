# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:33:05 2019

@author: The Iron Maiden
"""

import time
import numpy as np
import requests
import json
import cassiopeia as cass
from cassiopeia import Champion, Champions
import pandas as pd
import os

## Parameters
summonerName = "Duvet Cover"
APIKey = os.environ.get('League_API')
region = 'euw1'

rankNames = ['BRONZE',  'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTERS', 'CHALLENGER']

columnNames = ['account_id', 'assists', 'champion_id','champion_name',
                    'damageDealtToObjectives','damageDealtToTurrets',
                    'damageSelfMitigated','deaths','game_version', 'goldEarned','kda','kills',
                    'magicDamageDealtToChampions','match_id','match_rank_score',
                    'max_time','neutralMinionsKilled','neutralMinionsKilledEnemyJungle',
                    'neutralMinionsKilledTeamJungle','participantId','physicalDamageDealtToChampions',
                    'timeCCingOthers','totalDamageDealtToChampions',
                    'totalDamageTaken','totalHeal','totalMinionsKilled',
                    'trueDamageDealtToChampions','true_role','wardsKilled','wardsPlaced','visionScore',
                    'win'
                    ]
        
dfTop = pd.DataFrame(columns=columnNames)
dfJungle = pd.DataFrame(columns=columnNames)
dfMid = pd.DataFrame(columns=columnNames)
dfBot = pd.DataFrame(columns=columnNames)
dfSupp = pd.DataFrame(columns=columnNames)


# load champion names and IDs
dfChampNames = pd.DataFrame(columns=['champion_name','champion_ID'])
champions = Champions(region="NA")
index = 0
for champion in champions:
        dfChampNames.loc[index] = [champion.name, champion.id ]
        index+=1


## Get account details by providing the account name
def requestSummonerData(region,summonerName, APIKey):
    URL = "https://{}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}".format(region,summonerName,APIKey)
    response = requests.get(URL)
    return response.json()

## Get an account's ranked match data by account ID
def requestRankedData(region,summonerID, APIKey):
    URL = "https://{}.api.riotgames.com/lol/summoner/v4/positions/by-summoner/{}?api_key={}".format(region,summonerID,APIKey)
    response = requests.get(URL)
    return response.json()

def requestMatchList(region,acctID, APIKey):
    URL = "https://{}.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?api_key={}".format(region,acctID,APIKey)
    response = requests.get(URL)
    return response.json()

def requestMatchInfo(region,matchID, APIKey):
    URL = "https://{}.api.riotgames.com/lol/match/v4/matches/{}?api_key={}".format(region,matchID,APIKey)
    response = requests.get(URL)
    return response.json()

def requestLeagueInfo(region,leagueID, APIKey):
    URL = "https://{}.api.riotgames.com/lol/league/v4/leagues/{}?api_key={}".format(region,leagueID,APIKey)
    response = requests.get(URL)
    return response.json()

leagueList=pd.read_csv("C:\\Users\\The Iron Maiden\\Documents\\DataScienceProjects\\league_euw1.csv")

leagueList_toAnalyze = leagueList # CZ__

topCounter = 0
jungCounter = 0
midCounter = 0
botCounter = 0
suppCounter = 0
for index, row in leagueList_toAnalyze.iterrows():
 
    ## Pull the ID field from the response data, cast it to an int
    thisLeagueID = row ['leagueId']
    
    # entries is a dict 
    thisLeagueList  = requestLeagueInfo(region,thisLeagueID, APIKey)['entries']
    #thisLeagueList = thisLeagueList[0:2] # CZ__
    for summoner in thisLeagueList:
        
        print(summoner['summonerName'])
    
        acctID = requestSummonerData(region,summoner['summonerName'], APIKey)['accountId']
        
        matchList  = requestMatchList(region,acctID, APIKey)   
        numMatches = len(matchList ['matches'])
        
        for iMatch in range(numMatches-1):
         
            # need to pause bc of rate limits for riotAPI
            if iMatch%99 == 0 and iMatch != 0: 
                time.sleep(121)
                
            print( 'Get match'+ str(iMatch) )
            
            matchID = matchList ['matches'][iMatch]['gameId'] # get this match's ID
            
            matchInfo = requestMatchInfo(region,matchID, APIKey) # pull this game's info from riotAPI
            # get metrics from match info
            max_time = matchInfo['gameDuration']
            game_version = matchInfo['gameVersion']
            
            # find index of player in player list
            for i in range( len(matchInfo['participantIdentities'])-1 ):
                if matchInfo['participantIdentities'][i]['player']['accountId'] == acctID:
                    playerKey = i
  
            try: 
                statsDict = matchInfo['participants'][playerKey]['stats'] # get stats dict from this game
                playerTeam = matchInfo['participants'][playerKey]['teamId']
                
                ############ preprocess data
                
                # calculate KDA
                if statsDict['deaths'] == 0:
                    kda = statsDict['kills'] + statsDict['assists']
                else:
                    kda = ( statsDict['kills'] + statsDict['assists'] ) / statsDict['deaths']
                    
                # figure out champion name from ID
                thisChampionID = matchInfo['participants'][playerKey]['championId']
                tfIndex = dfChampNames['champion_ID'] == thisChampionID
                champName =  dfChampNames[tfIndex]['champion_name'].item()
                
                # figure out player's match rank
                thisRank = matchInfo['participants'][playerKey]['highestAchievedSeasonTier']
                matchRank = rankNames.index(thisRank) + 1
                
                ############ preprocess data end
                
                role = matchInfo['participants'][playerKey]['timeline']['role']
                lane = matchInfo['participants'][playerKey]['timeline']['lane']
                
                if role in ['SOLO','NONE']:
                    true_role = lane
                else:
                    true_role = role
                # create a vector of data to append for this match
                addVector =  [acctID, statsDict['kills'], thisChampionID,champName,
                    statsDict['damageDealtToObjectives'],statsDict['damageDealtToTurrets'],
                    statsDict['damageSelfMitigated'],statsDict['deaths'], game_version, statsDict['goldEarned'],kda,statsDict['kills'],
                    statsDict['magicDamageDealtToChampions'],matchID,matchRank,
                    max_time,statsDict['neutralMinionsKilled'],statsDict['neutralMinionsKilledEnemyJungle'],
                    statsDict['neutralMinionsKilledTeamJungle'],statsDict['participantId'],statsDict['physicalDamageDealtToChampions'],
                    statsDict['timeCCingOthers'],statsDict['totalDamageDealtToChampions'],
                    statsDict['totalDamageTaken'],statsDict['totalHeal'],statsDict['totalMinionsKilled'],
                    statsDict['trueDamageDealtToChampions'],true_role,statsDict['wardsKilled'],statsDict['wardsPlaced'],statsDict['visionScore'],
                    int(statsDict['win'])
                    ]
                
                ######### add data to dataframes to save

                if  role == 'SOLO' and lane == 'TOP':   
                    dfTop.loc[topCounter] = addVector
                    topCounter += 1
                    print('Top#: ' + str(topCounter))
                elif role == 'NONE' and lane == 'JUNGLE':    
                    dfJungle.loc[jungCounter] = addVector
                    jungCounter += 1
                    print('Jung#: ' + str(jungCounter))
                elif role == 'SOLO' and lane == 'MIDDLE':    
                    dfMid.loc[midCounter] = addVector
                    midCounter += 1
                    print('Mid#: ' + str(midCounter))
                elif role == 'DUO_CARRY':    
                    dfBot.loc[botCounter] = addVector
                    botCounter += 1
                    print('Bot#: ' + str(botCounter))
                elif role == 'DUO_SUPPORT':    
                    dfSupp.loc[suppCounter] = addVector 
                    suppCounter += 1
                    print('Supp#: ' + str(suppCounter))

            except:
                print ('Missing fields')

                
# dfPlayer.to_csv('output.csv', header=columnNames)                    