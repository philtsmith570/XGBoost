# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:01:09 2021

@author: philt
"""

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load data
data = read_csv('.\Datasets\iris.csv', header=None)
dataset = data.values
print(data.head())

# split data into X and Y
X = dataset[:, 0:4]
Y = dataset[:, 4]

# encode string class values as intergers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoder_y = label_encoder.transform(Y)
seed = 7
test_size = 0.33
X_train, X_test, y_train,y_test = train_test_split(X, 
                                                      label_encoder_y,
                                                      test_size=test_size,
                                                      random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

# make predictions
predictions = model.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: ', accuracy * 100)


