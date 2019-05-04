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

#Now, our data is loaded. We're writing the following snippet to see the loaded data. The purpose here is to see the top five of the loaded data.

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

#We will list all the columns for all data. We check all columns.
print('Data Show Columns:\n')
#print(data.columns)

# make the column names reference-friendly
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

##### Problem 1: Number of injuries up until 12/31/2018

# convert date strings to datetime type ; then find entries before the date
#endDate = '12/31/2018'
#dateTimeData = pd.to_datetime(data['date'])
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

## First start out with finding missing Borough names
geolocator = Nominatim(user_agent="specify_your_app_name_here")

emptyBorough = data.borough.isnull() 

print(data.borough.head(5))

for i in range(3): #(0,numSamps):
    if emptyBorough[i] == True:
        print(i)
        thisLoc = str( data.location[i] )
        #print(thisLoc)
        location = geolocator.reverse(thisLoc) # grab location data from coords
        splitString = location.address.split(",") # split the address name
        countySplit = splitString[3].split() # get rid of "county" part of string
        #print(countySplit[0].upper()) # caps the county name
        data.borough[i] = countySplit[0].upper()
        #print(data.borough[i])

### Problem 3 : 

