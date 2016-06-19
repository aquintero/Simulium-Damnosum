# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:44:27 2016

@author: Alex
"""

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from matplotlib import pyplot as plot
import os

def main():
    data_name = "simulium_plus_elevation"
    f = open("Data/" + data_name + ".csv")
    labels = f.readline().split(",")
    data = np.loadtxt(f, delimiter = ",")
    
    x = data[:,:-1]
    y = data[:,-1]
    
    m = x.shape[0]
    n = x.shape[1]
    
    x_labels = labels[:-1]
    y_labels = labels[-1]
    
    cv = 5
    
    regress(x, y, cv, "logs/" + data_name + "/cv-%d/" % cv)
    
def regress(x, y, cv, path):
    linear_estimator = make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    print "Linear Regression ..."
    log_learning_curve(linear_estimator, x, y, cv, path + "linear/")
    print "Polynomial Regression ..."
    log_polynomial_features(x, y, cv, path)
    print "Support Vector Regression ..."
    log_svr(x, y, cv, path)
    
def log_learning_curve(estimator, x, y, cv, path):
    train_sizes, train_scores, valid_scores = learning_curve(estimator, x, y, train_sizes = np.linspace(.5, 1, 10), cv = cv)
    
    plot.plot(train_sizes, train_scores.mean(axis=1), 'ro-', lw = 2, label = "training")
    plot.plot(train_sizes, valid_scores.mean(axis=1), 'go-', lw = 2, label = "validation")
    plot.xlabel("# of Training Samples")
    plot.ylabel("Score")
    plot.ylim([-1, 1])
    plot.suptitle("Learning Curve with %d-Fold Validation" % cv)
    plot.legend(bbox_to_anchor=(.81, 1.08), fancybox=True, ncol = 2)
    directory = path
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    plot.savefig(directory + "learning_curve.png", bbox_inches="tight")
    plot.clf()
    
def log_polynomial_features(x, y, cv, path):
    polynomial_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("poly", preprocessing.PolynomialFeatures()),
        ("linear", linear_model.Lasso(tol = .01))
    ])
    
    max_degree = 5    
    
    search_params = dict(
        poly__degree = range(2, max_degree + 1),
        linear__alpha = np.logspace(-3, 3, 12)
    )
    
    poly_search = GridSearchCV(polynomial_estimator, param_grid = search_params, cv = cv)
    poly_search.fit(x, y)    
    
    degrees, scores = extract_feature(poly_search.grid_scores_, "poly__degree")
    
    plot.plot(degrees, scores, 'bo-', lw = 2)
    plot.xlabel("Polynomial Degree")
    plot.ylabel("Score")
    plot.ylim([-1, 1])
    plot.suptitle("Degree Comparison with %d-Fold Validation" % cv)
    directory = path + "poly/"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    plot.savefig(directory + "poly_score.png", bbox_inches="tight")
    plot.clf()
    log_learning_curve(poly_search.best_estimator_, x, y, cv, directory)
    
def log_svr(x, y, cv, path):  
    
    svr_estimator = Pipeline([
        ("scale", preprocessing.StandardScaler()),
        ("svr", svm.SVR(kernel = "rbf")),
    ])
    
    search_params = dict(
        svr__C = np.logspace(-2, 2, 10),
        svr__gamma = np.logspace(-2, 2, 10),
        svr__epsilon = np.logspace(-2, 2, 10)
    )
    
    svr_search = GridSearchCV(svr_estimator, param_grid = search_params, cv = cv)
    svr_search.fit(train_x, train_y)
    
    directory = path + "svr/"
    if not os.path.isdir(directory):
        os.makedirs(directory)
        
    C_values, C_scores = extract_feature(svr_search.grid_scores_, "svr__C")
    plot.plot(C_values, C_scores, 'ko-', lw = 2)
    plot.xscale("log")
    plot.xlabel("Error Parameter(C)")
    plot.ylabel("Score")
    plot.ylim([-1, 1])
    plot.suptitle("C Parameter Comparison with %d-Fold Validation" % cv)
    plot.savefig(directory + "svr_C_score.png", bbox_inches="tight")
    plot.clf()

    gamma_values, gamma_scores = extract_feature(svr_search.grid_scores_, "svr__gamma")
    plot.plot(gamma_values, gamma_scores, 'co-', lw = 2)
    plot.xscale("log")
    plot.xlabel("Kernel Coefficient(gamma)")
    plot.ylabel("Score")
    plot.ylim([-1, 1])
    plot.suptitle("Kernel Coefficient Comparison with %d-Fold Validation" % cv)
    plot.savefig(directory + "svr_gamma_score.png", bbox_inches="tight")
    plot.clf()
    
    epsilon_values, epsilon_scores = extract_feature(svr_search.grid_scores_, "svr__epsilon")
    plot.plot(epsilon_values, epsilon_scores, 'co-', lw = 2)
    plot.xscale("log")
    plot.xlabel("Epsilon")
    plot.ylabel("Score")
    plot.ylim([-1, 1])
    plot.suptitle("Epsilon Comparison with %d-Fold Validation" % cv)
    plot.savefig(directory + "svr_epsilon_score.png", bbox_inches="tight")
    plot.clf()
    
    log_learning_curve(svr_search.best_estimator_, x, y, cv, directory)
    
    print svr_search.best_score_, svr_search.best_estimator_.score(x, y)

def extract_feature(scores, feature):
    feature_values = set()
    for score in scores:
        if(not score.parameters[feature] in feature_values):
            feature_values.add(score.parameters[feature])
            
    filtered_scores = []
    for value in feature_values:
        feature_scores = [score for score in scores if score.parameters[feature] == value]
        filtered_scores.append(max(feature_scores, key = lambda score: score.mean_validation_score))
        
    filtered_scores.sort(key = lambda score: score.parameters[feature])        
        
    feature_list = [score.parameters[feature] for score in filtered_scores]
    score_list = [score.mean_validation_score for score in filtered_scores]
        
    
    
    return feature_list, score_list
    
if __name__ == "__main__":
    main()