# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:14:13 2019

@author: Zhe
"""

import requests
import json
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

## Parameters
summonerName = "Artificial Anus"
APIKey = "RGAPI-a8eb35fe-392b-4730-a51f-ba35fc002e28"

summonerData  = requestSummonerData(summonerName, APIKey)

# Uncomment this line if you want a pretty JSON data dump
#print(json.dumps(summonerData, sort_keys=True, indent=2))

## Print to the console some basic account information
print("\n\nSummoner Name:\t" + str(summonerData ['name']))
print("Level:\t\t" + str(summonerData ['summonerLevel']))

## Pull the ID field from the response data, cast it to an int
ID = summonerData ['id']
accountID = summonerData ['accountId']   
    
matchList  = requestMatchList(accountID, APIKey)
   
numMatches = len(matchList ['matches'])

for iMatch in range(numMatches):

    matchID = matchList ['matches'][iMatch]['gameId']
    
    matchInfo = requestMatchInfo(matchID, APIKey)
    
    playerKey = participantIdentities
    
    matchInfo['participants'][playerKey] #  'championId' 'win' 'wardsPlaced' 'goldEarned' 'visionScore' 'damageDealtToObjectives' 'damageDealtToTurrets' 'totalDamageDealtToChampions'
    
    'kills' 'deaths' 'assists'