from rpy2.robjects import r
import os

import misc

def main():
    resolution = misc.getResolution()
    feature_path = "data/features/%s/" % resolution
    data_path = "data/r/%s/" % resolution
    
    r('feature_path <- "%s"' % feature_path)
    r('data_path <- "%s"' % data_path)
    
    r.source("predict.r")
    
    print r['model']
    print r['prediction']
    print r['y']
    

    
if __name__ == "__main__":
    main()