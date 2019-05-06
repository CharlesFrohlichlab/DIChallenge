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
import geopy as geopy
from geopy.geocoders import Nominatim
import datetime as datetime

# load data
data=pd.read_csv("C:\\Users\\Zhe\\Documents\\DataScienceProjects\\NYPD_Collisions\\NYPD_Motor_Vehicle_Collisions.csv")

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# zips are in floats and strings; make them homogenous - all floats
data['zip_code'] = pd.to_numeric(data['zip_code'], errors='coerce')

# Show summary analytics of data (eg. mean, std, min, max, etc)
print('Show descriptive analytics of data:\n')
print(data.describe())

##### Problem 1: 

# convert date strings to datetime type ; then find entries before the date
endDate = '12/31/2018'
dateTimeData = pd.to_datetime(data['date']) # convert date strings to datetime
mask =  ( dateTimeData <= endDate ) 
print( data.number_of_persons_injured[mask].sum() )

###### Problem 2: 
startDate = '01/01/2016'
endDate = '12/31/2016'

# generate boolean vectors based on how we want to filter the data
indexValidBorough = data.borough.notnull() 
indexBrooklyn = data.borough == 'BROOKLYN'
index2016 = (dateTimeData >= startDate) & ( dateTimeData <= endDate ) 
mergedBool = indexBrooklyn & indexValidBorough & index2016

propBrooklyn2016 = np.sum(mergedBool)/np.sum(index2016) # calculate proportion
print( 'Proportion of Collisions in Brooklyn in 2016: ' + str(propBrooklyn2016) )

###### We know there are empty values - let's try our best to fill them out

# fill missing features by imputing
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
data.iloc[:,2:9] = imp.fit_transform(data.iloc[:,2:9])

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

# loop through boroughs and calculate accidents per capita
for iBorough in population.borough:
    
    thisBoroughDat = data.loc[ (data['borough'] == iBorough) &  index2017 ]
    vehAlc2017 = thisBoroughDat.loc[ (data['contributing_factor_vehicle_1'] == 'Alcohol Involvement') | (data['contributing_factor_vehicle_2'] == 'Alcohol Involvement') ]
    accPerCap +=  [ vehAlc2017.shape[0]  / int(population.loc[population.borough == iBorough,'pop']) ]
    
print( max(accPerCap)*100000 )

### Problem 5 : 
# needs datetime data from problem 1 and index2016 from problem 2

zipTotVeh = []

uniqueZips = data['zip_code'].unique()

# loop through zips and calculate number of vehicles in collisions
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

# loop through years and grab number of collisions
for iYear in uniqueYears:
    collByYear += [sum(dateTimeData.dt.year == iYear)]

# fit linear regression of number of collisions as function of year
reg = LinearRegression().fit(uniqueYears.reshape(-1, 1),  collByYear ) # (X,y)

print(reg.coef_)

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

# chi square between Jan and May
from scipy.stats import chi2_contingency
stat,p,dof,expected = chi2_contingency( [[multiVehColl[0],thisMonthTotalColl[0]  ] ,[multiVehColl[4], thisMonthTotalColl[4]]] )
print('stat=%.10f p=%.5f' % (stat,p))

# Problem 8:

from geopy import distance
import math

zips2017 = data.loc[bool2017].iloc[:,3]
coords2017 = data.loc[bool2017].iloc[:,4:6]

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