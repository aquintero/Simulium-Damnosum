# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:58:25 2016

@author: Alex
"""

import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

def regress(x, y, cv):
    linear_estimator = make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    score = cross_validation.cross_val_score(linear_estimator, x, y, cv=cv, n_jobs = 4).mean()
    linear_estimator.fit(x, y)
    prediction = linear_estimator.predict(x)
    
    return prediction, score
    
def get_learning_curve(x, y, cv, train_space = np.linspace(.5, 1, 10)):
    linear_estimator = make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    train_sizes, train_scores, valid_scores = learning_curve(linear_estimator, x, y, train_sizes = train_space, cv = cv, n_jobs = 4)
    
    return train_sizes, train_scores, valid_scores