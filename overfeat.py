# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:04:21 2016

@author: alex
"""

import os
import glob
import cv2
import numpy as np

from sklearn_theano.feature_extraction import OverfeatTransformer

def main():
    input_dir = "data/transform/06/"
    output_dir = "data/overfeat/06/"
    tf = OverfeatTransformer(output_layers=[-3], large_network = True)
    
    im_dirs = os.listdir(input_dir)
    
    print "Generating Features ..."
    for im_i, im_dir in enumerate(im_dirs):
        print "%d/%d" % (im_i + 1, len(im_dirs))
        
        in_dir = input_dir + im_dir + "/"
        
        im_data = []
        for im_file in glob.glob(in_dir + "*.png"):
            im_data.append(cv2.imread(im_file))
        features = tf.transform(im_data)

        out_dir = output_dir + im_dir + "/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        np.savetxt(out_dir + "features.csv", features, delimiter = ",")
        
        value = np.loadtxt(in_dir + "value.csv", delimiter = ",")
        np.savetxt(out_dir + "value.csv", [value], delimiter = ",")

if __name__ == "__main__":
    main()