# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:36:11 2019

 Dataset from: https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95

 Objective: Load, clean, preprocess NYPD collision reports to examine summary 
     and specific data analytics

 Author: Zhe Charles Zhou
 
 Note: Wow, this project is really fun to work out. Especially using the geopy module

"""

### Import toolboxes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopy as geopy
from geopy.geocoders import Nominatim
import datetime as datetime

data=pd.read_csv("C:\\Users\\Zhe\\Documents\\DataScienceProjects\\NYPD_Collisions\\NYPD_Motor_Vehicle_Collisions.csv")

#We will list all the columns for all data. We check all columns.
#print('Data Show Columns:\n')
#print(data.columns)

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# zips are in floats and strings; make them homogenous - all floats
data['zip_code'] = pd.to_numeric(data['zip_code'], errors='coerce')

#Now, our data is loaded and cleaned. We're writing the following snippet to see the loaded data. The purpose here is to see the top five of the loaded data.

print('Data First 5 Rows Show\n')
print(data.head(5))

dataDims = data.shape
numSamps = dataDims[0]
numColumns = dataDims[1]
print(numSamps)

print('Are there NaN/Empty entries in the CSV?')
print( data.isnull().values.any() )

# Show summary analytics of data (eg. mean, std, min, max, etc)
#print('Data Show Describe\n')
#print(data.describe())

##### Problem 1: Number of injuries up until 12/31/2018

# convert date strings to datetime type ; then find entries before the date
endDate = '12/31/2018'
dateTimeData = pd.to_datetime(data['date']) # convert date strings to datetime
#mask =  ( dateTimeData <= endDate ) 
#print( data.number_of_persons_injured[mask].sum() )
#print(mask.head()) # DELETE__
# answer is 368034

###### Problem 2: Proportion of all collisions in 2016 occured in Brooklyn (exclude Null Boroughs)
startDate = '01/01/2016'
endDate = '12/31/2016'

# generate boolean vectors based on how we want to filter the data
indexValidBorough = data.borough.notnull() 
indexBrooklyn = data.borough == 'BROOKLYN'
index2016 = (dateTimeData >= startDate) & ( dateTimeData <= endDate ) 
mergedBool = indexBrooklyn & indexValidBorough & index2016


print( 'Num Valid Boroughs: ' + str(np.sum(indexValidBorough)) )
print( 'Num Collisons in Brooklyn: ' + str(np.sum(indexBrooklyn)) )
print( 'Num Collisons in 2016: ' + str(np.sum(index2016)) )
print( 'Num Collisions in Brooklyn in 2016: ' + str(np.sum(mergedBool)) )
propBrooklyn2016 = np.sum(mergedBool)/np.sum(index2016) # calculate proportion
print( 'Proportion of Collisions in Brooklyn in 2016: ' + str(propBrooklyn2016) )

###### We know there are empty values - let's try our best to fill them out

# fill missing features by imputing
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
data.iloc[:,2:9] = imp.fit_transform(data.iloc[:,2:9])

### First start out with finding missing Borough names

# Use geopy to find full address based on coordinates
# geocode sometimes times out - this solves this problem
from geopy.exc import GeocoderTimedOut
def do_geocode(coords):
    try:
        return geolocator.reverse(coords)
    except GeocoderTimedOut:
        return do_geocode(coords)

geolocator = Nominatim(user_agent="specify_your_app_name_here")
pd.options.mode.chained_assignment = None  # default='warn' - gets rid of warning from overwriting nans in dataframe

# generate boolean vectors for entries without boroughs and that have coordinates
emptyBorough = data.borough.isnull() 
hasCoord = data.location.notnull() 

# print(data.borough.head(5)) __DELETE

for i in range(0,numSamps-1):
    if (emptyBorough[i] == True) & (hasCoord[i] == True): # only fill in borough if missing and coordinates are present
        #print(i)
        thisLoc = str( data.location[i] )
        #print(thisLoc[1:-1]) __DELETE
        location = do_geocode(thisLoc[1:-1]) # grab location data from coords
        splitString = location.address.split(",") # split the address name
        countySplit = splitString[3].split() # get rid of "county" part of string
        #print(countySplit[0].upper()) # __DELETE
        data.borough[i] = countySplit[0].upper() #  caps the county name 
        #print(data.borough[i])  __DELETE

### Problem 3 : 

# generate boolean vectors based on how we want to filter the data
# we need index2016 from problem 1
boolCycleInjured = data.number_of_cyclist_injured >= 1
boolCycleKilled = data.number_of_cyclist_killed >= 1

boolCycleInjuredKilled = boolCycleInjured | boolCycleKilled

boolCycleInjuredKilled_2016 = boolCycleInjuredKilled & index2016
print( np.sum(boolCycleInjuredKilled_2016) / np.sum(index2016) )

### Problem 4 : 

startDate = '01/01/2017'
endDate = '12/31/2017'
index2017 = (dateTimeData >= startDate) & ( dateTimeData <= endDate ) 
accPerCap = []

# make a data frame of each borough's population at 2017 - from wikipedia
toDf = [['BRONX',1471160],['BROOKLYN',2648771],['MANHATTAN',1664727],
        ['QUEENS',2358582],['STATEN ISLAND',479458]]
population = pd.DataFrame(toDf,columns=['borough', 'pop'])

for iBorough in population.borough:
    
    thisBoroughDat = data.loc[ (data['borough'] == iBorough) &  index2017 ]
    vehAlc2017 = thisBoroughDat.loc[ (data['contributing_factor_vehicle_1'] == 'Alcohol Involvement') | (data['contributing_factor_vehicle_2'] == 'Alcohol Involvement') ]
    accPerCap +=  [ vehAlc2017.shape[0]  / int(population.loc[population.borough == iBorough,'pop']) ]
    
max(accPerCap)*100000

### Problem 5 : 
# needs datetime data from problem 1 and index2016 from problem 2

zipTotVeh = []

uniqueZips = data['zip_code'].unique()

for iZip in uniqueZips:
    
    boolZip2016 = (data['zip_code'] == iZip) & index2016
    
    zipVehInfo = data.loc[boolZip2016].iloc[:,24:29]
    numVeh = zipVehInfo.notnull().sum(axis=1)
    zipTotVeh += [sum(numVeh)]

print(max(zipTotVeh))

# problem 6:
from sklearn.linear_model import LinearRegression

collByYear = []

uniqueYears = dateTimeData.dt.year.unique()[1:] # get rid of 2019

for iYear in uniqueYears:
    collByYear += [sum(dateTimeData.dt.year == iYear)]

reg = LinearRegression().fit(uniqueYears.reshape(-1, 1),  collByYear ) # (X,y)
slope = reg.coef_

# Problem 7:

multiVehColl = []
thisMonthTotalColl = []
for iMonth in range(1,13):
   
    # find entries for 2017 and for this month and calc total collisions
    bool2017 = dateTimeData.dt.year == 2017 
    boolMonth = dateTimeData.dt.month == iMonth
    thisMonthTotalColl += [sum( bool2017 & boolMonth )]
    
    vehInfo = data.loc[bool2017 & boolMonth].iloc[:,24:29]
    numVeh = vehInfo.notnull().sum(axis=1)
    multiVehColl += [sum( numVeh >= 3 ) ] # first make bool vector of >2 coll; then sum trues

from scipy.stats import chi2_contingency
stat,p,dof,expected = chi2_contingency( [[multiVehColl[0],thisMonthTotalColl[0]  ] ,[multiVehColl[4], thisMonthTotalColl[4]]] )
print('stat=%.10f p=%.5f' % (stat,p))

# Problem 8:

from geopy import distance
import math

zips2017 = data.loc[bool2017].iloc[:,3]
coords2017 = data.loc[bool2017].iloc[:,4:6]

#plt.scatter(coords2017['latitude'],coords2017['longitude'])

# get rid of outliers via std dev
latStd = coords2017['latitude'].std()
longStd = coords2017['longitude'].std()

latMean = coords2017['latitude'].mean()
longMean = coords2017['longitude'].mean()

latThresh = latStd * 3
longThresh = longStd * 3

dropLat = abs( coords2017['latitude'] - latMean ) > latThresh
dropLong = abs( coords2017['longitude'] - longMean ) > longThresh

 
coords2017 = coords2017.drop(coords2017[dropLat | dropLong].index)

#plt.scatter(coords2017['latitude'],coords2017['longitude'])

# find unique zips and loop through

collPerKm2 = []
uniqueZips = zips2017.unique()
uniqueZips = uniqueZips[~np.isnan(uniqueZips)]

for iZip in uniqueZips:
    
    thisZip2017 = zips2017 == iZip
    
    if sum(thisZip2017) >= 1000:
        
        latStd = coords2017.loc[thisZip2017,['latitude']].std()
        longStd = coords2017.loc[thisZip2017,['longitude']].std()
        latMean = coords2017.loc[thisZip2017,['latitude']].mean()
        longMean = coords2017.loc[thisZip2017,['longitude']].mean()
    
        r1 = distance.distance(latMean, latMean + latStd).km
        r2 = distance.distance(longMean, longMean + longStd).km
        area = math.pi * r1 * r2

        collPerKm2 += [ sum(thisZip2017) / area ]
    else:
        collPerKm2 +=  [] 

print(max(collPerKm2))