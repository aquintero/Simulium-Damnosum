# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 19:11:12 2016

@author: alex
"""

import glob
import os
import numpy as np
import cv2

def main():
    input_dir = "data/sites/06/"
    out_dir = "data/transform/06/"
    resize_rot = (400, 400)
    resize = (280, 280)
    out_size = (227, 227)
    n_rot = 0
    n_crop = 0
    
    im_files = glob.glob(input_dir + "*.png")
    values = np.loadtxt(input_dir + "values.csv", delimiter = ",")
    
    for im_i, im_file in enumerate(im_files):
        im_dir = out_dir + "%d/" % im_i
        if not os.path.isdir(im_dir):
            os.makedirs(im_dir)
            
        rot = np.linspace(0, 360, n_rot + 1, endpoint = False)
        crop = np.linspace(0, 1, n_crop + 1, endpoint = False) * (resize[0] - out_size[0]) * (resize[1] - out_size[1])
        
        im = cv2.imread(im_file)
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
        
                cv2.imwrite(im_dir + "%d_%d.png" % (r_i, p_i), final_im)
                
        np.savetxt(im_dir + "value.csv", [values[im_i]], delimiter = ",")
            
if __name__ == "__main__":
    main()