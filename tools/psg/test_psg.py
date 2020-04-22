# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:11:36 2019

@author: iant

curl --data-urlencode file@config.txt https://psg.gsfc.nasa.gov/api.php

"""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

BASE_DIRECTORY = os.path.join(r"C:\Users\iant\Dropbox\NOMAD\Python")



def runPsg(config_name):
    url = r"https://psg.gsfc.nasa.gov/api.php"
    #url = r"http://localhost:3000/api.php"
    stringIn = str(subprocess.check_output("curl --data-urlencode file@%s %s" %(config_name, url))).split(r"\n")
    
    if stringIn == ["b''"]:
        print("Error: no data received")
        
        return {"error":True}
    
    else:
        stringIn.pop(0)
        stringIn.pop(-1)
        
        headers = [string[2:] for string in stringIn if ((string[0]=="#") and (string[2:4]!="--"))]
        
        dataIn = np.asfarray([string.split() for string in stringIn if string[0]!="#"])

        wavenumbers = dataIn[:, 0]
        total = dataIn[:, 1]
        noise = dataIn[:, 2]
        mars = dataIn[:, 3]
                              
        return headers, {"wavenumbers":wavenumbers, "total":total, "noise":noise, "mars":mars}                                 


plt.figure()
headers, psgDict = runPsg("config.txt")        
atmos = psgDict["total"]
headers, psgDict = runPsg("config2.txt")        
toa = psgDict["total"]

plt.plot(psgDict["wavenumbers"], atmos/toa)

