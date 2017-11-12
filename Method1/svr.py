import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from scipy.stats import expon

def main():
    input_dir = "data/features/06/"
    
    x = np.loadtxt(input_dir + "features.csv", delimiter = ",")
    y = np.loadtxt(input_dir + "values.csv", delimiter = ",")
    
    est = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearSVR(C = 1e2))
    ])
    
    results = pd.DataFrame(cross_validate(est, x, y, scoring = 'r2', cv = 5, return_train_score = True))
    results = results.drop(['score_time', 'fit_time'], 1)
    print(results)
    results.to_html('results.html')
    
if __name__ == "__main__":
    main()