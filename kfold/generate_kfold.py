# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:06:15 2016

@author: alex
"""

import glob
import os
from sklearn.cross_validation import KFold
import numpy as np
import cv2

def main():
    input_dir = "../data/gis/output/0/"
    out_dir = "../data/kfold/0/"
    cv = 5
    resize_rot = (364, 364)
    resize = (256, 256)
    out_size = (221, 221)
    n_rot = 10
    n_crop = 10
    
    files = glob.glob(input_dir + "*.png")
    n = len(files)
    
    values = np.loadtxt(input_dir + "values.csv", delimiter = ",")
    
    folds = KFold(n, n_folds = cv)
    fold = 0
    for train_idx, test_idx in folds:
        fold_dir = out_dir + "%d/" % fold
        for train_i in train_idx:
            train_dir = fold_dir + "train/%d/" % train_i
            train_file = files[train_i]
            rot = np.zeros(n_rot + 1)
            rot[1:] = np.random.random(n_rot) * 360
            crop = np.zeros(n_crop + 1)
            crop[1:] = np.random.random(n_crop) * (resize[0] - out_size[0]) * (resize[1] - out_size[1])
            
            im = cv2.imread(train_file)
            im = cv2.resize(im, resize_rot, interpolation = cv2.INTER_AREA)
            center = (resize_rot[0] / 2, resize_rot[1] / 2)
            for r_i, degree in enumerate(rot):
                rot_m = cv2.getRotationMatrix2D(center, degree, 1.0)
                rot_im = cv2.warpAffine(im, rot_m, resize_rot)
                rot_crop = ((resize_rot[0] - resize[0]) / 2, (resize_rot[1] - resize[1]) / 2)
                rot_im = rot_im[rot_crop[0]: resize_rot[0] - rot_crop[0], rot_crop[1]: resize_rot[1] - rot_crop[1]]
                for p_i, point in enumerate(crop):
                    cx = int((point % (resize[0] - out_size[0])) / 2)
                    cy = int((point / (resize[0] - out_size[0])) / 2)
                    crop_rot_im = rot_im[cx: out_size[0] - cx, cy: out_size[1] - cy]
                    final_im = cv2.resize(crop_rot_im, out_size, interpolation = cv2.INTER_AREA)
                    
                    if not os.path.isdir(train_dir):
                        os.makedirs(train_dir)
                    cv2.imwrite(train_dir + "%d_%d.png" % (r_i, p_i), final_im)
                    
            np.savetxt(train_dir + "value.csv", [values[train_i]], delimiter = ",")
        
        for test_i in test_idx:
            test_dir = fold_dir + "test/%d/" % test_i
            test_file = files[test_i]
            im = cv2.imread(test_file)
            im = cv2.resize(im, out_size, interpolation = cv2.INTER_AREA) 
            
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)
            cv2.imwrite(test_dir + "%d.png" % test_i, im)
            np.savetxt(test_dir + "value.csv", [values[test_i]], delimiter = ",")
        fold += 1
            
if __name__ == "__main__":
    main()