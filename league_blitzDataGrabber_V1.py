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

columnNames = ['champion_name','match_rank_score','max_time',
                            'gold_earned','wards_placed','damage_dealt_to_objectives',
                            'damage_dealt_to_turrets','kda',
                            'total_damage_dealt_to_champions']
dfPlayer = pd.DataFrame(columns=columnNames)
dataYPlayer = pd.DataFrame(columns=['win'])
dataYPlayer = pd.Series(name="win")

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

leagueList_toAnalyze = leagueList[0:1]

counter_match = 0
for index, row in leagueList_toAnalyze.iterrows():
 
    ## Pull the ID field from the response data, cast it to an int
    thisLeagueID = row ['leagueId']
    
    # entries is a dict 
    thisLeagueList  = requestLeagueInfo(region,thisLeagueID, APIKey)['entries']
    thisLeagueList = thisLeagueList[0:2]
    for summoner in thisLeagueList:
        
        print(summoner['summonerName'])
    
        acctID = requestSummonerData(region,summoner['summonerName'], APIKey)['accountId']
        
        matchList  = requestMatchList(region,acctID, APIKey)   
        numMatches = len(matchList ['matches'])
        
        for iMatch in range(numMatches-1):
         
            # need to pause bc of rate limits for riotAPI
            if iMatch%49 == 0 and iMatch != 0: 
                time.sleep(121)
                
            print( 'Get match'+ str(iMatch) )
            
            matchID = matchList ['matches'][iMatch]['gameId'] # get this match's ID
            
            matchInfo = requestMatchInfo(region,matchID, APIKey) # pull this game's info from riotAPI
            
            # find index of player in player list
            for i in range( len(matchInfo['participantIdentities'])-1 ):
                if matchInfo['participantIdentities'][i]['player']['accountId'] == acctID:
                    playerKey = i
            
            
            if matchInfo['participants'][playerKey]['timeline']['role'] == 'DUO_SUPPORT':
                    
                #try: 
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
                    
                    # create a vector of data to append for this match
                    addVector = [ champName, matchRank, matchInfo['gameDuration'], statsDict['goldEarned'], 
                                 statsDict['wardsPlaced'], statsDict['damageDealtToObjectives'], 
                                 statsDict['damageDealtToTurrets'], kda, 
                                 statsDict['totalDamageDealtToChampions']
                            ]
                    
                    dfPlayer.loc[counter_match] = addVector
                    if statsDict['win'] == True: 
                        dataYPlayer.loc[counter_match] = 1
                    else:
                        dataYPlayer.loc[counter_match] = 0
                    counter_match += 1
                    print(counter_match)
                #except:
                #    print ('Missing fields')
        
# dfPlayer.to_csv('output.csv', header=columnNames)                    