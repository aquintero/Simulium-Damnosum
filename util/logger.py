# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:15:52 2016

@author: Alex
"""

import os
import re
from matplotlib import pyplot as plot

def plot_prediction(x, y_true, y_predict, xlabels, ylabel, title, log_dir):
    n = x.shape[1]
    
    assert(len(xlabels) == n)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        
    for i in range(n):
        plot.suptitle(title)
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabel)
        
        xi, yi_true, yi_predict = zip(*sorted(zip(x[:,i], y_true, y_predict), key = lambda t: t[0]))
        plot.plot(xi, yi_true, "k-", lw = 2, label = "actual data")
        plot.plot(xi, yi_predict, "b-", lw = 2, label = "predicted data")
        plot.margins(.05, .2)
        plot.legend(loc = "upper center", bbox_to_anchor = (.5, 1.08), fancybox = True, ncol = 2)
            
        plot.savefig(os.path.join(log_dir, "%s_prediction.png" % xlabels[i]), bbox_inches="tight")
        plot.clf()
        
def plot_learning_curve(train_sizes, train_scores, valid_scores, title, log_dir):
    plot.plot(train_sizes, train_scores.mean(axis=1), 'ro-', lw = 2, label = "training")
    plot.plot(train_sizes, valid_scores.mean(axis=1), 'go-', lw = 2, label = "validation")
    plot.xlabel("# of Training Samples")
    plot.ylabel("Score (r2)")
    plot.margins(.05, .2)
    plot.suptitle(title)
    plot.legend(loc = "upper center", bbox_to_anchor = (.5, 1.08), fancybox = True, ncol = 2)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
            
    plot.savefig(os.path.join(log_dir, "%s.png" % _to_file_name(title)), bbox_inches="tight")
    plot.clf()
    
def plot_hyper_parameter(x, y, xlabel, title, log_dir, log_space = False, limit = False):
    plot.plot(x, y, 'co-', lw = 2)
    plot.xlabel(xlabel)
    plot.ylabel("Score (r2)")
    plot.suptitle(title)
    if(log_space):
        plot.xscale("log")
    if(limit):
        plot.ylim([-1, 1.2])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    plot.savefig(log_dir + _to_file_name(title), bbox_inches = "tight")
    plot.clf()
    
def _to_file_name(s):
    return re.sub("\s+", "_", s).lower()