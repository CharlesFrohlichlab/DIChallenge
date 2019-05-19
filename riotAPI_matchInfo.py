# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:14:13 2019

@author: Zhe
"""
import time
import numpy as np
import requests
import json
import cassiopeia as cass
from cassiopeia import Champion, Champions
import pandas as pd

## Parameters
summonerName = "Artificial Anus"
APIKey = "RGAPI-d9a303bf-dc95-4535-ae5e-366d46c4b984"

rankNames = ['BRONZE',  'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTERS', 'CHALLENGER']
dfPlayer = pd.DataFrame(columns=['champion_name','match_rank_score','max_time',
                            'gold_earned','wards_placed','damage_dealt_to_objectives',
                            'damage_dealt_to_turrets','kda',
                            'total_damage_dealt_to_champions'])
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
def requestSummonerData(summonerName, APIKey):
    URL = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/" + summonerName + "?api_key=" + APIKey
    response = requests.get(URL)
    return response.json()

## Get an account's ranked match data by account ID
def requestRankedData(ID, APIKey):
    URL = "https://na1.api.riotgames.com/lol/summoner/v4/positions/by-summoner/" + str(ID) + "?api_key=" + APIKey
    response = requests.get(URL)
    return response.json()

def requestMatchList(ID, APIKey):
    URL = "https://na1.api.riotgames.com/lol/match/v4/matchlists/by-account/" + str(ID) + "?api_key=" + APIKey
    response = requests.get(URL)
    return response.json()

def requestMatchInfo(matchID, APIKey):
    URL = "https://na1.api.riotgames.com//lol/match/v4/matches/" + str(matchID) + "?api_key=" + APIKey
    response = requests.get(URL)
    return response.json()

summonerData  = requestSummonerData(summonerName, APIKey)

# Uncomment this line if you want a pretty JSON data dump
#print(json.dumps(summonerData, sort_keys=True, indent=2))

## Print to the console some basic account information
#print("\n\nSummoner Name:\t" + str(summonerData ['name']))
#print("Level:\t\t" + str(summonerData ['summonerLevel']))

## Pull the ID field from the response data, cast it to an int
ID = summonerData ['id']
accountID = summonerData ['accountId']   
    
matchList  = requestMatchList(accountID, APIKey)
   
numMatches = len(matchList ['matches'])

for iMatch in range(numMatches-1):

    # need to pause bc of rate limits for riotAPI
    if iMatch == 50: 
        time.sleep(121)
        
    print( 'match'+ str(iMatch) )
    
    matchID = matchList ['matches'][iMatch]['gameId'] # get this match's ID
    
    matchInfo = requestMatchInfo(matchID, APIKey) # pull this game's info from riotAPI
    
    # find index of player in player list
    for i in range( len(matchInfo['participantIdentities'])-1 ):
        if matchInfo['participantIdentities'][i]['player']['accountId'] == accountID:
            playerKey = i
    
    
    if matchInfo['participants'][playerKey]['timeline']['role'] == 'DUO_SUPPORT':
    
        try: 
            statsDict = matchInfo['participants'][playerKey]['stats'] # get stats dict from this game
            
            
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
            
            dfPlayer.loc[iMatch] = addVector
            if statsDict['win'] == True: 
                dataYPlayer.loc[iMatch] = 1
            else:
                dataYPlayer.loc[iMatch] = 0
        except:
            print ('Missing fields')
      