# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:45:21 2016

@author: Alex
"""

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

C_tag = "svr__C"
gamma_tag = "svr__gamma"
epsilon_tag = "svr__epsilon"

def regress(x, y, cv, C_space = np.logspace(-3, 3, 10), gamma_space = np.logspace(-3, 3, 10), epsilon_space = np.logspace(-3, 3, 10)):
    svr_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("svr", svm.SVR(kernel = "rbf")),
    ])
    
    search_params = dict(
        svr__C = C_space,
        svr__gamma = gamma_space,
        svr__epsilon = epsilon_space
    )
    
    svr_search = GridSearchCV(svr_estimator, param_grid = search_params, cv = cv)
    svr_search.fit(x, y)
    
    prediction = svr_search.best_estimator_.predict(x)
    score = cross_validation.cross_val_score(svr_search, x, y, cv=cv, n_jobs = 4).mean()
    grid_scores = svr_search.grid_scores_
    
    return prediction, score, grid_scores
    
def get_learning_curve(x, y, cv, train_space = np.linspace(.5, 1, 10), C_space = np.logspace(-3, 3, 10), gamma_space = np.logspace(-3, 3, 10), epsilon_space = np.logspace(-3, 3, 10)):
    svr_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("svr", svm.SVR(kernel = "rbf")),
    ])
    
    search_params = dict(
        svr__C = C_space,
        svr__gamma = gamma_space,
        svr__epsilon = epsilon_space
    )
    
    svr_search = GridSearchCV(svr_estimator, param_grid = search_params, cv = cv)
    
    train_sizes, train_scores, valid_scores = learning_curve(svr_search, x, y, train_sizes = train_space, cv = cv, n_jobs = 4)
    
    return train_sizes, train_scores, valid_scores