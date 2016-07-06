# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:26:18 2016

@author: alex
"""

import os
import glob
from pykml import parser as kml
import gdal
from gdalconst import GA_ReadOnly
import osr
import numpy as np
from scipy.misc import imsave
from natsort import natsorted

gdal.UseExceptions()

def main():
    data_dir = "data/gis/input/1/"
    out_dir = "data/gis/output/1/"
    pixel_radius = 3
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)    
    
    values = []
    coordinates = []
    for file in glob.glob(data_dir + "*.kml"):
        tree = kml.parse(file)
        root = tree.getroot()
        nsmap = {k:v for k,v in root.nsmap.iteritems() if k}
        values.extend(tree.xpath("//kml:Placemark/kml:description", namespaces=nsmap))
        coordinates.extend((tree.xpath("//kml:Placemark/kml:LookAt/kml:latitude", namespaces = nsmap), tree.xpath("//kml:Placemark/kml:LookAt/kml:longitude", namespaces = nsmap)))
        
    coordinates[0] = [float(co) for co in coordinates[0]]
    coordinates[1] = [float(co) for co in coordinates[1]]
    values = np.mat([float(value) for value in values]).transpose()
    coordinates = np.array(np.matrix(coordinates).transpose())
    
    
    tiffs = []
    for file in natsorted(glob.glob(data_dir + "*.TIF")):
        tiffs.append(gdal.Open(file, GA_ReadOnly))
        
    gt = tiffs[0].GetGeoTransform()
    
    cs = osr.SpatialReference()
    cs.ImportFromWkt(tiffs[0].GetProjectionRef())
    cs_latlong = osr.SpatialReference()
    cs_latlong.SetWellKnownGeogCS("WGS84");
    
    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(cs_latlong, cs)
    
    pixels = []
    for tiff in tiffs:
        raster = tiff.GetRasterBand(1)
        raster_pixels = []
        for geo_coord in coordinates:
            coord = transform.TransformPoint(geo_coord[1], geo_coord[0])

            coord = np.array(coord)

            coord[0] = int((coord[0] - gt[0]) / gt[1]) #x pixel
            coord[1] = int((coord[1] - gt[3]) / gt[5]) #y pixel


            min_coord = (coord - [pixel_radius, pixel_radius, 0]).astype(np.int)
            dimensions = np.array([2 * pixel_radius, 2 * pixel_radius])

            raster_pixels.append(raster.ReadAsArray(min_coord[0], min_coord[1], dimensions[0], dimensions[1]))

        pixels.append(raster_pixels)
    
    pixels = np.array(pixels)
    rgb = np.zeros((pixels.shape[1], pixels.shape[2], pixels.shape[3], 3))   
    
    #rgb[:,:,:,0] = np.sum(pixels[0:red_end], axis = (0)) / (red_end) / 256
    #rgb[:,:,:,1] = np.sum(pixels[red_end:green_end]) / (green_end - red_end) / 256
    #rgb[:,:,:,2] = np.sum(pixels[green_end:blue_end]) / (blue_end - green_end) / 256
    rgb[:,:,:,0] = pixels[2] / 256.0
    rgb[:,:,:,1] = pixels[1] / 256.0
    rgb[:,:,:,2] = pixels[0] / 256.0
        
    for i in range(rgb.shape[0]):
        f = open(out_dir + "%d.png" % i, "wb")
        imsave(f, rgb[i].astype(np.uint8))
        
    np.savetxt(out_dir + "values.csv", values, delimiter = ",")
    
if __name__ == "__main__":
    main()