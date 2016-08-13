import os
import numpy as np
from sklearn.utils import shuffle

import misc

def main():
    resolution = misc.getResolution()
    input_dir = "data/caffe/%s/" % resolution
    out_dir = "data/features/%s/" % resolution
    
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
    labels = np.array(labels, dtype = int)
        
    #x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    #y = y.reshape(y.shape[0] * y.shape[1])
        
    x, y, labels = shuffle(x, y, labels, random_state = 666)
    
    features = np.zeros((x.shape[0], x.shape[1] + 1))
    features[:,:-1] = x
    features[:,-1] = y
    np.savetxt(out_dir + "features.csv", features, delimiter = ",")    
    np.savetxt(out_dir + "feature_labels.csv", labels, delimiter = ",", fmt = "%d")
    
if __name__ == "__main__":
    main()