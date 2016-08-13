import ConfigParser
def getResolution():
    config = ConfigParser.ConfigParser()
    config.readfp(open("config.cfg"))
    return config.get("config", "resolution")