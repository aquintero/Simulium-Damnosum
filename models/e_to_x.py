# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:35:44 2016

@author: Alex
"""

from util import logger
from regression import linear, poly, svr, grid_search
import numpy as np

log_dir = "../logs/e_to_x/"    

def main():
    n = 100
    
    x = np.linspace(-2, 2, n)
    y = np.exp(x)
    x = x.reshape(n, 1)
    
    cv = 5
    
    #linear_regression(x, y, cv)
    #polynomial_regression(x, y, cv)
    support_vector_regression(x, y, cv)
   
def linear_regression(x, y, cv):
    prediction, score = linear.regress(x, y, cv)
    
    print "Linear Regression Score: %f" % score
    
    logger.plot_prediction(x, y, prediction, ["X"], "Y", "Linear Regression of y = e^x", log_dir + "linear/")
    
    train_sizes, train_scores, valid_scores = linear.get_learning_curve(x, y, cv)
    logger.plot_learning_curve(train_sizes, train_scores, valid_scores, "Linear Regression Learning Curve", log_dir + "linear/")
    
def polynomial_regression(x, y, cv):
    prediction, score, grid_scores = poly.regress(x, y, cv)
    
    print "Polynomial Regression Score: %f" % score 
    
    logger.plot_prediction(x, y, prediction, ["X"], "Y", "Polynomial Regression of y = e^x", log_dir + "poly/")
    
    degrees, degree_scores = grid_search.extract_hyper_parameter(grid_scores, poly.degree_tag)
    logger.plot_hyper_parameter(degrees, degree_scores, "Polynomial Degree", "Polynomial Degree Comparison", log_dir + "poly/")
    
    train_sizes, train_scores, valid_scores = poly.get_learning_curve(x, y, cv)  
    logger.plot_learning_curve(train_sizes, train_scores, valid_scores, "Polynomial Regression Learning Curve", log_dir + "poly/")

def support_vector_regression(x, y, cv):
    prediction, score, grid_scores = svr.regress(x, y, cv)
    
    print "SVR Score: %f" % score
    
    logger.plot_prediction(x, y, prediction, ["X"], "Y", "SVR of y = e^x", log_dir + "svr/")
    
    Cs, C_scores = grid_search.extract_hyper_parameter(grid_scores, svr.C_tag)
    logger.plot_hyper_parameter(Cs, C_scores, "Error Parameter (C)", "C Parameter Comparison", log_dir + "svr/", log_space = True)
    
    gammas, gamma_scores = grid_search.extract_hyper_parameter(grid_scores, svr.gamma_tag)
    logger.plot_hyper_parameter(gammas, gamma_scores, "Kernel Coefficient (gamma)", "Gamma Parameter Comparison", log_dir + "svr/", log_space = True)

    epsilons, epsilon_scores = grid_search.extract_hyper_parameter(grid_scores, svr.epsilon_tag)
    logger.plot_hyper_parameter(epsilons, epsilon_scores, "Margin Parameter (epsilon)", "Epsilon Parameter Comparison", log_dir + "svr/", log_space = True)
    
    train_sizes, train_scores, valid_scores = svr.get_learning_curve(x, y, cv)  
    logger.plot_learning_curve(train_sizes, train_scores, valid_scores, "SVR Learning Curve", log_dir + "svr/")


if __name__ == "__main__":
    main()