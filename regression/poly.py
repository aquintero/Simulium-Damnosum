# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:58:25 2016

@author: Alex
"""

import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

degree_tag = "poly__degree"
alpha_tag = "linear__alpha"

def regress(x, y, cv, degree_space = range(2, 6), alpha_space = np.logspace(-3, 3, 12)):
    polynomial_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("poly", preprocessing.PolynomialFeatures()),
        ("linear", linear_model.Lasso(tol = .01))
    ])
    
    search_params = dict(
        poly__degree = degree_space,
        linear__alpha = alpha_space
    )    
    
    poly_search = GridSearchCV(polynomial_estimator, param_grid = search_params, cv = cv)
    poly_search.fit(x, y)
    poly_search.best_estimator_.fit(x, y)
    
    prediction = poly_search.best_estimator_.predict(x)
    grid_scores = poly_search.grid_scores_
    
    score = cross_validation.cross_val_score(poly_search, x, y, cv=cv, n_jobs = 4).mean()
    
    return prediction, score, grid_scores
    
def get_learning_curve(x, y, cv, train_space = np.linspace(.5, 1, 10), degree_space = range(2, 6), alpha_space = np.logspace(-3, 3, 12)):
    polynomial_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("poly", preprocessing.PolynomialFeatures()),
        ("linear", linear_model.Lasso(tol = .01))
    ])
    
    search_params = dict(
        poly__degree = degree_space,
        linear__alpha = alpha_space
    )
    
    poly_search = GridSearchCV(polynomial_estimator, param_grid = search_params, cv = cv)
    
    train_sizes, train_scores, valid_scores = learning_curve(poly_search, x, y, train_sizes = train_space, cv = cv, n_jobs = 4)
    
    return train_sizes, train_scores, valid_scores