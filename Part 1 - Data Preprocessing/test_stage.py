# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:10:53 2019

@author: Harada
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


dataset = pd.read_csv("web_scraping.csv")
X = dataset.iloc[:,0]
y = dataset.iloc[:,1]

from sklearn.preprocessing import Imputer
#permet de completer les valeurs manquantes ex en untilisattn les maoyennes

from sklearn.prepocessing import LabelEncoder, One

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc()


Imputer, LabelEncoder 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


