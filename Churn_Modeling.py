#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:55:10 2019

@author: Arvinthkumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(["RowNumber","CustomerId","Surname"],axis=1)

#to find null 
df.isnull().sum()

# box plots to check outliers
df.plot(kind='box', subplots=True, layout=(4,4), fontsize=8, figsize=(14,14))
plt.show()

df = pd.get_dummies(df)
df = df.drop(["Gender_Male","Geography_Spain"],axis=1)

X = df.loc[:, df.columns!='Exited'].values
y = df.loc[:, df.columns=='Exited'].values
print(X)
print(y)

#Spliting into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(X_train,y_train)
pred=lor.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test,pred))
print(confusion_matrix)

from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
pred=nb.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test,pred))
print(confusion_matrix)

from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
