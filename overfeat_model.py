# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:59:57 2016

@author: alex
"""

import os
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.cross_validation import LabelKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def main():
    input_dir = "data/overfeat/06/"
    out_dir = "data/model/06/"
    cv = 5
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)   
    
    ft_dirs = os.listdir(input_dir)
    x = []
    y = []
    labels = []
    for ft_i, ft_dir in enumerate(ft_dirs):
        in_dir = input_dir + ft_dir + "/"
        x_part = np.loadtxt(in_dir + "features.csv", delimiter = ",")
        x.append(x_part)
        
        if len(x_part.shape) == 1:
            y_part = np.loadtxt(in_dir + "value.csv", delimiter = ",")
            y.append(y_part)
            
            labels.append(ft_i)
        else:
            y_part = np.empty(x_part.shape[0], dtype = np.float)
            y_part.fill(np.loadtxt(in_dir + "value.csv", delimiter = ","))
            y.append(y_part)
        
            for i in range(x_part.shape[0]):
                labels.append(ft_i)
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
        
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    y = y.reshape(y.shape[0] * y.shape[1])
        
    x, y, labels = shuffle(x, y, labels, random_state = 666)
    
    features = np.zeros((x.shape[0], x.shape[1] + 1))
    features[:,:-1] = x
    features[:,-1] = y
    np.savetxt(out_dir + "features.csv", features, delimiter = ",")    
    np.savetxt(out_dir + "feature_labels.csv", np.asarray(labels, dtype = np.int32), delimiter = ",")
    
    estimator = LinearSVR()
    
    scores = []
    train_scores = []
    fold = 1
    
    for train_index, test_index in LabelKFold(labels, cv):
        print "Fold: %d" % fold
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        estimator.fit(x_train, y_train)
        train_scores.append(estimator.score(x_train, y_train))
        scores.append(estimator.score(x_test, y_test))
        
        fold += 1
    
    print scores
    print np.asarray(scores).mean()
    
    print train_scores
    print np.asarray(train_scores).mean()
if __name__ == "__main__":
    main()