# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:36:11 2019

 Dataset from: https://www.kaggle.com/ronitf/heart-disease-uci

 Objective: 

 Author: Zhe Charles Zhou

"""

### Import toolboxes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopy as geopy
import datetime as datetime

data=pd.read_csv("C:\\Users\\Zhe\\Documents\\DataScienceProjects\\NYPD_Collisions\\NYPD_Motor_Vehicle_Collisions.csv")

#Now, our data is loaded. We're writing the following snippet to see the loaded data. The purpose here is to see the top five of the loaded data.

print('Data First 5 Rows Show\n')
print(data.head(5))

print(data.shape)

print('Are there NaN/Empty entries in the CSV?')
print( data.isnull().values.any() )

# Show summary analytics of data (eg. mean, std, min, max, etc)
#print('Data Show Describe\n')
#print(data.describe())

#We will list all the columns for all data. We check all columns.
print('Data Show Columns:\n')
print(data.columns)

### We know there are empty values - let's try our best to fill them out

### Problem 1: Number of injuries up until 12/31/2018

# convert date strings to datetime type ; then find entries before the date
endDate = '12/31/2018'
mask =  ( pd.to_datetime(data['DATE']) <= endDate ) 
print(mask.sum())

