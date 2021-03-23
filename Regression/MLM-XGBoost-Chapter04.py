# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:34:50 2021

Regression model using pima-indians-diabetes data.

XGBoost implementation.

@author: philt
"""


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = loadtxt(".\Datasets\pima-indians-diabetes.csv", delimiter=",")

# Define X, Y - Input and Output
X = dataset[:, 0:8]  # Use all data first 7 columns
Y = dataset[:, 8]  # Last column is 0 no diabetes/1 diabetes

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=test_size,
                                                    random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# check model
print(model)

''' Parameters used in XGBoost Classifier

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
'''

# make predictions for test data
predictions = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: ', accuracy * 100)
