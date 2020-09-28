# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:24:51 2020

@author: iant

CHECK OCCULTATION MISPOINTING, SAVE TO PDF

"""
import re
import matplotlib.pyplot as plt
import numpy as np
import os

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames

def plotSignalPointing(hdf5_file, hdf5_filename):
    tangentAlt = hdf5_file["Geometry/Point0/TangentAltAreoid"][:,0]
    tangentAltIndices = np.where(tangentAlt > 0.0)[0]
    tangentAlts = tangentAlt[tangentAltIndices]
    bins = hdf5_file["Science/Bins"][tangentAltIndices, 0]
    y = hdf5_file["Science/Y"][tangentAltIndices, :]
    
    pixelIndices = [200]
    
    uniqueBins = sorted(list(set(bins)))
    binIndices = [np.where(bins == binStart)[0] for binStart in uniqueBins]
    
    
    fig, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
    ax1.set_title(hdf5_filename)
    ax1.set_xlabel("Tangent Altitude km")
    ax1.set_ylabel("Approx. transmittance")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Solar Pointing Deviation arcmins")
    for binIndexList in binIndices:
        for pixelIndex in pixelIndices:
            toaIndices = np.where(tangentAlts[binIndexList]>200.0)[0]
            yPixel = y[binIndexList, pixelIndex]
            yToa = np.mean(yPixel[toaIndices])
            yTrans = yPixel/yToa
            
            ax1.plot(tangentAlts[binIndexList], yTrans)
    ax2.plot(tangentAlts[binIndexList], sunWobble[binIndexList], "k--")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIRECTORY, "Solar_Pointing_Deviation_%s.png" %hdf5_filename))
    plt.close()


fileLevel = "hdf5_level_0p3k"
regexDates = [
    "201804", "201805", "201806", "201807", "201808", #"201809", "201810", "201811", "201812", 
    "201905", "201906", "201907", "201908", "201909", "201910", "201911", "201912"
]


badFilelist = []
nFiles = 0
for regexDate in regexDates:       
#    regex = re.compile("%s[0-9][0-9]_.*_SO_A_[IE]_(134|136)" %regexDate)
    regex = re.compile("%s[0-9][0-9]_.*_SO_A_[IE]_134" %regexDate)
    
    hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
    
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(len(hdf5Filenames))/len(hdf5Filenames)]
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.xlabel("Tangent Altitude km")
    plt.ylabel("Solar Pointing Deviation arcmins")
    plt.title("Solar Pointing Deviation %s-%s" %(regexDate[0:4], regexDate[4:6]))
    for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    #hdf5_file = hdf5Files[0]
    #hdf5_filename = hdf5Filenames[0]
    
        tangentAlt = hdf5_file["Geometry/Point0/TangentAltAreoid"][:,0]
        tangentAltIndices = np.where(tangentAlt > 0.0)[0]
        tangentAlts = tangentAlt[tangentAltIndices]
        sunWobble = hdf5_file["Geometry/FOVSunCentreAngle"][tangentAltIndices, 0]
    
        bins = hdf5_file["Science/Bins"][tangentAltIndices, 0]
        uniqueBins = sorted(list(set(bins)))
        binIndices = np.where(bins == uniqueBins[0])[0]
        
        plt.plot(tangentAlts[binIndices], sunWobble[binIndices], color=colours[fileIndex], label=hdf5_filename[:15])
        
        if max(sunWobble[binIndices]) > 0.7:
            badFilelist.append(hdf5_filename)
            plotSignalPointing(hdf5_file, hdf5_filename)
        nFiles += 1
        
#    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIRECTORY, "Solar_Pointing_Deviation_%s.png" %regexDate))

print("Bad occultations:")
for badOcc in badFilelist:
    print(badOcc)
print("%i of %i files analysed are bad" %(len(badFilelist), nFiles))