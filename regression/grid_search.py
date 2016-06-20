# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:51:48 2016

@author: Alex
"""

def extract_hyper_parameter(scores, feature):
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