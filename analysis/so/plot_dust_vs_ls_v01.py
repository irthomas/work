# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:50:47 2019

@author: iant
"""

#from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
#import h5py
import re
from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, getFile, makeFileList#, printFileNames

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    print("Running on windows")
    import spiceypy as sp
    #load spiceypy kernels if required
    KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
    #KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
    METAKERNEL_NAME = "em16_ops_win.tm"
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)




#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
regex = re.compile("201[89][01][0-9][0-9][0-9]_.*_SO_.*_134")
#regex = re.compile("2018[01][0-9][0-9][0-9]_.*_SO_.*_134")
#regex = re.compile("201805[0-9][0-9]_.*_SO_.*_134")
fileLevel = "hdf5_level_1p0a"







BIN_INDEX = 1
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel, open_files=False, silent=True)


with open(os.path.join(BASE_DIRECTORY, "reference_files", "mcd_dust_column_od_lat_vs_ls.txt"), "r") as f:
    lines = f.readlines()

mcdLats = np.asfarray([float(value) for value in lines[11].replace("----", "").replace("|", "").strip().split()])

lineNumbers = list(range(13,38))
mcdLs = np.asfarray([float(lines[lineNumber].split("|")[0].replace("+","").strip()) for lineNumber in lineNumbers])
mcdDust = np.asfarray([[float(value) for value in lines[lineNumber].split("|")[-1].strip().split()] for lineNumber in lineNumbers])
mcdDustNorm = mcdDust / np.max(mcdDust)

    
fig, ax = plt.subplots(figsize=(FIG_X + 2, FIG_Y + 1))
#cp = plt.contourf(mcdLs, mcdLats, mcdDustNorm.T, alpha=0.7)



for hdf5_index, hdf5_filename in enumerate(hdf5Filenames):
    
    if len(hdf5Filenames) > 100:
        if np.mod(hdf5_index, 100) == 0:
            print("Processing files %i/%i" %(hdf5_index, len(hdf5Filenames)))
            
    
    hdf5_filename_split = hdf5_filename.split("_")
    if len(hdf5_filename_split) == 7 and hdf5_filename_split[5] != "G":
        diffractionOrder = hdf5_filename.split("_")[6]
        
        if diffractionOrder != "134":
            print(diffractionOrder)
            
        name, hdf5_file = getFile(hdf5_filename, fileLevel, 0, silent=True)
        bins = hdf5_file["Science/Bins"][:, 0]
        uniqueBins = sorted(list(set(bins)))
        
        binIndex = np.where(bins == uniqueBins[BIN_INDEX])[0]
        
        transmittanceMean = np.mean(hdf5_file["Science/Y"][binIndex, 160:240], axis=1)
        lons = hdf5_file["Geometry/Point0/Lon"][binIndex, 0]
        lats = hdf5_file["Geometry/Point0/Lat"][binIndex, 0]
        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][binIndex, 0]
        ls = hdf5_file["Geometry/LSubS"][0, 0]
        
        opticalDepthMean = -1.0 * np.log(np.abs(transmittanceMean))
        closestIndex = np.where(opticalDepthMean < 1.0)[0][0]
        closestIndices = [closestIndex-2, closestIndex-1, closestIndex, closestIndex+1, closestIndex+2]
        altInterpolated = np.polyval(np.polyfit(opticalDepthMean[closestIndices], alts[closestIndices], 2), 1.0)
        
        indicesAtmos = np.where((0.05 < transmittanceMean) & (transmittanceMean < 0.95))[0]
        indicesSelected = range(indicesAtmos[0], indicesAtmos[-1])
#        lonsSelected = lons[indicesSelected]
        latsSelected = lats[indicesSelected]
        altsSelected = np.zeros_like(latsSelected) + altInterpolated
        lsSelected = np.zeros_like(latsSelected) + ls

        plot = ax.scatter(lsSelected, latsSelected, c=altsSelected, cmap=plt.cm.jet, marker='o', linewidth=0, vmin=0, vmax=60)

cbar = fig.colorbar(plot)
colorbarLabel = "Lowest altitude where optical depth < 1.0"
cbar.set_label(colorbarLabel, rotation=270, labelpad=20)

ax.set_ylim([-90, 90])
#ax.set_xlim([min(lsSelected)-1, max(lsSelected)+1])
ax.set_xlabel("Ls (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("SO diffraction order 134: continuum line-of-sight optical depths versus Ls and observation latitude")
        
#months = np.arange(4, 13, 1)
#lsMonths = [sp.lspcn("MARS", sp.utc2et(datetime(2018, month, 1).strftime("%Y-%m-%d")), "NONE") * sp.dpr() for month in months]




