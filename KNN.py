# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:58:39 2024

@author: Shikhar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import random
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
print(data.describe())
print(data.info())
dummies = pd.get_dummies(data,columns=['RestingECG', 'ChestPainType', 'ExerciseAngina', 'ST_Slope'],dtype='int64')
print(dummies)
data1 = dummies.drop(columns=['Sex'])
print(data1)
print(data1.columns)
print(data1.corr().to_string())
X = data1 [[ 'Cholesterol', 'FastingBS', 'MaxHR', 'RestingBP']]
Y = data1 ['HeartDisease']
print(X)
print(Y)
sns.scatterplot(data=data1)
plt.plot(X,Y)
random.seed(1)
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= .20)
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")