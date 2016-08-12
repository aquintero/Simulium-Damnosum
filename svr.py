from rpy2.robjects import r

def main():
    r.source("svr.r")
    
if __name__ == "__main__":
    main()