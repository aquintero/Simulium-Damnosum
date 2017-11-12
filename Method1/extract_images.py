# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:26:18 2016

@author: alex
"""

import os
import math
import glob
import gdal
from gdalconst import GA_ReadOnly
import osr
import numpy as np
from scipy.misc import imsave

gdal.UseExceptions()

def main():
    data_dir = "../data/sat/06/"
    out_dir = "data/sites/06/"
    radius = .0005
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)    
    
    values = []
    coordinates = []
    data = np.loadtxt(glob.glob(data_dir + "*.csv")[0], delimiter = ",", skiprows = 1)
    values = data[:, -1]
    coordinates = data[:, 1:3]
    
    
    tiff = gdal.Open(glob.glob(data_dir + "*.TIF")[0], GA_ReadOnly)
    gt = tiff.GetGeoTransform()
    
    cs = osr.SpatialReference()
    cs.ImportFromWkt(tiff.GetProjectionRef())
    cs_latlong = osr.SpatialReference()
    cs_latlong.SetWellKnownGeogCS("WGS84");
    
    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(cs_latlong, cs)
    
    coord = np.array(transform.TransformPoint(coordinates[0][0], coordinates[0][1]))
    radius_coord = np.array(transform.TransformPoint(coordinates[0][0] + radius, coordinates[0][1]))

    coord[0] = int((coord[0] - gt[0]) / gt[1]) #x pixel
    radius_coord[0] = int((radius_coord[0] - gt[0]) / gt[1]) #x pixel

    pixel_radius = int(math.ceil(radius_coord[0] - coord[0]))
    
    pixels = []
    for band in range(1, 4):
        raster = tiff.GetRasterBand(band)
        raster_pixels = []
        for geo_coord in coordinates:
            coord = np.array(transform.TransformPoint(geo_coord[0], geo_coord[1]))
    
            coord[0] = int((coord[0] - gt[0]) / gt[1]) #x pixel
            coord[1] = int((coord[1] - gt[3]) / gt[5]) #y pixel
            
            min_coord = (coord - [pixel_radius, pixel_radius, 0]).astype(np.int)
            dimensions = np.array([2 * pixel_radius, 2 * pixel_radius]).astype(np.int)
            img = raster.ReadAsArray(min_coord[0], min_coord[1], dimensions[0], dimensions[1])
            raster_pixels.append(img) 
        
            
        pixels.append(raster_pixels)        
 
    pixels = np.array(pixels)
    rgb = np.zeros((pixels.shape[1], pixels.shape[2], pixels.shape[3], pixels.shape[0]))   
    
    for i in range(pixels.shape[0]):
        rgb[:,:,:,i] = pixels[i]
        
    for i in range(rgb.shape[0]):
        f = open(out_dir + "%d.png" % i, "wb")
        imsave(f, rgb[i].astype(np.uint8))
        
    np.savetxt(out_dir + "values.csv", values, delimiter = ",")
    
if __name__ == "__main__":
    main()