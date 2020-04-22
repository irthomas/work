# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:03:40 2018

@author: iant

TERMINATOR POINTS

"""

import numpy as np
#import numpy.linalg as la
import os
from datetime import datetime
import matplotlib.pyplot as plt

import spiceypy as sp



BASE_DIRECTORY = os.path.normcase(os.getcwd())
KERNEL_DIRECTORY = os.path.normcase(r"D:\kernels\mk")
METAKERNEL_NAME = "em16_ops.tm"

OUTPUT_PATH = os.path.normcase(BASE_DIRECTORY+os.sep+"test.csv")

def writeOutput(lineToWrite):
    global OUTPUT_PATH
    logFile=open(OUTPUT_PATH, 'a')
    logFile.write(lineToWrite+'\n')
    logFile.close()




abcorr="None"
tolerance="1"
method="Intercept: ellipsoid"
formatstr="C"
prec=3
shape="Ellipsoid"

def et2utc(et):
    return sp.et2utc(et, formatstr, 0)

def lsubs(et):
    
    return sp.lspcn("MARS",et,abcorr) * sp.dpr()


os.chdir(KERNEL_DIRECTORY)
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(BASE_DIRECTORY)


trmtyp = "PENUMBRAL"
source = "SUN"
target = "MARS"
npts = 360

ref="IAU_MARS"
observer="-143"
typein = "PLANETOCENTRIC"
body = 499

etStart = sp.utc2et("2018MAR24-11:50:00 UTC")
etEnd = sp.utc2et("2020MAR24-11:50:00 UTC")

ets = np.linspace(etStart,etEnd,num=180)

writeOutput("UTC Time, LSubS, Longitude, Latitude, Local Solar Time")
for et in list(ets):

    lSubS = lsubs(et)
    terminatorPoints = sp.edterm(trmtyp, source, target, et, ref, abcorr, observer, npts)
    
    terminatorLatLonsRad = np.asfarray([sp.reclat(terminatorPoint)[1:3] for terminatorPoint in terminatorPoints[2]])
    
    terminatorLatLonsDeg = terminatorLatLonsRad * sp.dpr()
    
    
    terminatorLSTs = [sp.et2lst(et, body, terminatorLatLonRad[0], typein)[3] for terminatorLatLonRad in list(terminatorLatLonsRad)]
    
    for index,terminatorLST in enumerate(terminatorLSTs):
        writeOutput("%s,%0.2f,%0.2f,%0.2f,%s" %(et2utc(et),lSubS,terminatorLatLonsDeg[index,0],terminatorLatLonsDeg[index,1],terminatorLST))