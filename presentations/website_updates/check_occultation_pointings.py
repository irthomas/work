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
# from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths, FIG_X_PDF, FIG_Y_PDF


#for writing code for website
IMG_WIDTH = 400
IMG_HEIGHT = 400

PLOT = False

def plotSignalPointing(hdf5_file, hdf5_filename):
    tangentAlt = hdf5_file["Geometry/Point0/TangentAltAreoid"][:,0]
    tangentAltIndices = np.where(tangentAlt > 0.0)[0]
    tangentAlts = tangentAlt[tangentAltIndices]
    bins = hdf5_file["Science/Bins"][tangentAltIndices, 0]
    y = hdf5_file["Science/Y"][tangentAltIndices, :]
    
    pixelIndices = [200]
    
    uniqueBins = sorted(list(set(bins)))
    binIndices = [np.where(bins == binStart)[0] for binStart in uniqueBins]
    
    
    fig, ax1 = plt.subplots(figsize=(FIG_X_PDF, FIG_Y_PDF))
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
    plt.savefig(os.path.join(paths["BASE_DIRECTORY"], "Solar_Pointing_Deviation_%s.png" %hdf5_filename))
    plt.close()


fileLevel = "hdf5_level_0p3k"


badFilelist = []
nFiles = 0
regex = re.compile("20[0-9][0-9][0-9][0-9][0-9][0-9]_.*_SO_A_[IE]_134")

hdf5Files, hdf5Filenames, titles = make_filelist(regex, fileLevel)

cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(len(hdf5Filenames))/len(hdf5Filenames)]

if PLOT:
    plt.figure(figsize=(FIG_X_PDF, FIG_Y_PDF))
    plt.xlabel("Tangent Altitude (km)")
    plt.ylabel("Solar Pointing Deviation (arcmins)")
    plt.title("Solar Pointing Deviation")
for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

    if np.mod(fileIndex, 100) == 0:
        print("Analysing %i/%i" %(fileIndex, len(hdf5Filenames)))

    tangentAlt = hdf5_file["Geometry/Point0/TangentAltAreoid"][:,0]
    tangentAltIndices = np.where(tangentAlt > 0.0)[0]
    tangentAlts = tangentAlt[tangentAltIndices]
    sunWobble = hdf5_file["Geometry/FOVSunCentreAngle"][tangentAltIndices, 0]

    bins = hdf5_file["Science/Bins"][tangentAltIndices, 0]
    uniqueBins = sorted(list(set(bins)))
    binIndices = np.where(bins == uniqueBins[0])[0]
    
    if PLOT:
        plt.plot(tangentAlts[binIndices], sunWobble[binIndices], color=colours[fileIndex], label=hdf5_filename[:15])
    
    if max(sunWobble[binIndices]) > 0.7:
        badFilelist.append(hdf5_filename)
        if PLOT:
            plotSignalPointing(hdf5_file, hdf5_filename)
    nFiles += 1

if PLOT:    
    plt.tight_layout()
    plt.savefig(os.path.join(paths["BASE_DIRECTORY"], "Solar_Pointing_Deviation.png"))

print("Bad occultations:")
for badOcc in badFilelist:
    print(badOcc[:15])
print("%i of %i files analysed are bad" %(len(badFilelist), nFiles))

for badOcc in badFilelist:
    print("<a href=\"ProjectDir/images/bad_observations/Solar_Pointing_Deviation_%s.png\" target=\"_blank\"> <img src=\"ProjectDir/images/bad_observations/Solar_Pointing_Deviation_%s.png\" alt=\"\" width=\"%s\" height=\"%s\" /></a>" \
          %(badOcc, badOcc, IMG_WIDTH, IMG_HEIGHT))


