# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:44:27 2016

@author: Alex
"""

import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing

if __name__ == "__main__":
    f = open("Data/simulium.csv")
    labels = f.readline().split(",")
    data = np.loadtxt(f, delimiter = ",")
    
    x = data[:,:-1]
    y = data[:,-1]
    
    x_labels = labels[:-1]
    y_labels = labels[-1]
    
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = .3, random_state = 0)
    
    scaler = preprocessing.StandardScaler().fit(x_train)
    
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    lr = linear_model.LinearRegression()
    lr.fit(x_train_scaled, y_train)
    
    print "Intercept: ", lr.intercept_
    print "Coefficients: ", lr.coef_
    print "R Squared: ", lr.score(x_test_scaled, y_test)