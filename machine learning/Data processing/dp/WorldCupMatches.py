# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:18:21 2018

@author: dell
"""
# step 1 importing modules 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# step  2 importing data set
'''
dataset= pd.read_csv('WorldCupMatches.csv')

dataset.head(5)
dataset.shape
dataset.index
dataset.columns
dataset.shape

#  checking the missing values 
dataset.isnull().sum()

# handle the missing values methode1 

dataset.dropna(inplace=True)
dataset.isnull().sum()
dataset.shape



# replace the NaN value with mean ,median or mode methode 2
dataset['Year'].mean()
dataset['Year'].tail()
dataset['Year'].replace(np.NaN,dataset['Year'].mean()).tail()
'''
dataset = pd.read_csv('Data.csv')
dataset.dropna(inplace=True)
dataset.isnull().sum()
X = dataset.iloc[:, :-1].values
X

# working on categorical Data

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:,0] =label_encoder.fit_transform(X[:,0])
X

# Working on Dummy Variable 
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X)
dummy =pd.get_dummies(dataset['Country'])
dummy
# concatnating dummy variables to data set
dataset = pd.concat([dataset,dummy],axis=1)
dataset.Purchased.replace(('Yes', 'No'), (1, 0), inplace=True)
dataset.drop(['Country'],axis=1)
y=dataset.iloc[:,3].values
#dataset['Purchased']=dataset['Purchased'].map({'yes':1,'no':0})
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

# feature scaling 
from sklearn.preprocessing import StandardScaler
standard_X = StandardScaler()
X_train =standard_X.fit_transform(X)
#X_test =standard_X.fit_transform(X_test)
