import numpy as np
from sklearn.svm import SVR
from sklearn.cross_validation import LabelKFold
from sklearn.preprocessing import StandardScaler

def main():
    in_dir = "data/features/06/"
    cv = 5
    
    features = np.loadtxt(in_dir + "features.csv", delimiter = ",")
    labels = np.loadtxt(in_dir + "feature_labels.csv", delimiter = ",")
    
    x = features[:,:-1]
    y = features[:,-1]
    estimator = SVR()
    
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