from rpy2.robjects import r

def main():
    r.source("predict.r")
    
    print r['model']
    print r['prediction']
    print r['y']
    

    
if __name__ == "__main__":
    main()