# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:44:27 2016

@author: Alex
"""

import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from matplotlib import pyplot as plot
import os

def main():
    f = open("Data/simulium.csv")
    labels = f.readline().split(",")
    data = np.loadtxt(f, delimiter = ",")
    
    x = data[:,:-1]
    y = data[:,-1]
    
    m = x.shape[0]
    n = x.shape[1]
    
    x_labels = labels[:-1]
    y_labels = labels[-1]
    
    cv_folds = 10
    
    linear_estimator = make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    log_learning_curve(linear_estimator, x, y, cv_folds, "logs/linear/")
    log_polynomial_features(x, y, cv_folds)
    
def log_learning_curve(estimator, x, y, cv, path):
    train_sizes, train_scores, valid_scores = learning_curve(estimator, x, y, train_sizes = np.linspace(.5, 1, 10), cv = cv)
    
    plot.plot(train_sizes, train_scores.mean(axis=1), 'ro-', lw = 2, label = "training")
    plot.plot(train_sizes, valid_scores.mean(axis=1), 'go-', lw=2, label = "validation")
    plot.xlabel("# of Training Samples")
    plot.ylabel("Score")
    plot.suptitle("Learning Curve with %d-Fold Validation" % cv)
    plot.legend(bbox_to_anchor=(.99, .26), fancybox=True)
    directory = path
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    plot.savefig(directory + "learning_curve_cv_%d.png" % cv, bbox_inches="tight")
    plot.clf()
    
def log_polynomial_features(x, y, cv):
    polynomial_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("poly", preprocessing.PolynomialFeatures()),
        ("linear", linear_model.Lasso(tol = .01))
    ])
    
    max_degree = 5    
    
    search_params = dict(
        poly__degree = range(1, max_degree + 1),
        linear__alpha = np.logspace(-3, 3, 12)
    )
    
    poly_search = GridSearchCV(polynomial_estimator, param_grid = search_params, cv = cv)
    poly_search.fit(x, y)    
    
    degree_scores = []
    for i in range(1, max_degree + 1):
        degree_scores.append(max([score for score in poly_search.grid_scores_ if score.parameters["poly__degree"] == i], key = lambda score: score.mean_validation_score))
    degrees = [score.parameters["poly__degree"] for score in degree_scores]
    scores = [score.mean_validation_score for score in degree_scores]
    plot.plot(degrees, scores, 'bo-', lw = 2)
    plot.xlabel("Polynomial Degree")
    plot.ylabel("Score")
    plot.suptitle("Degree Comparison with %d-Fold Validation" % cv)
    directory = "logs/poly/"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    plot.savefig(directory + "poly_score_cv_%d.png" % cv, bbox_inches="tight")
    plot.clf()
    
    log_learning_curve(poly_search.best_estimator_, x, y, cv, directory)
    
if __name__ == "__main__":
    main()