# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:04:36 2019

@author: iant

COMPARE LNO FILES OF DIFFERENT LEVELS

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, FIG_X, FIG_Y

DATA_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
BIN_START = 80
BIN_END = 223

levels = ["0p1d", "0p1e", "0p2a", "0p3a", "1p0a"]
#linestyles = [":", "-.", "--", "-"]
linestyles = ["-"] * 5
#levels = ["0p1d"]


for hdf5_filename_base in [
    "20180501_063544_1p0a_LNO_1_D_168",
#    "20180611_131514_1p0a_LNO_1_D_189",
#    "20180704_013820_1p0a_LNO_1_D_189",
#    "20180819_043902_1p0a_LNO_1_D_131",
#    "20181111_014134_1p0a_LNO_1_D_169",
#    "20181216_045527_1p0a_LNO_1_D_167",
#    "20190307_044518_1p0a_LNO_1_D_136",
#    "20190414_045842_1p0a_LNO_1_D_189",
#    "20190715_050315_1p0a_LNO_1_D_168",
#    "20190921_045106_1p0a_LNO_1_D_168",
        ]:



    yMeanAll = []
    for level in levels:
        
        hdf5_filename = hdf5_filename_base[:16] + level + hdf5_filename_base[20:]
        
        hdf5_file_level = "hdf5_level_" + level
        year_in = hdf5_filename[0:4]
        month_in = hdf5_filename[4:6]
        day_in = hdf5_filename[6:8]
        
        hdf5file_path = os.path.join(DATA_DIRECTORY, hdf5_file_level, year_in, month_in, day_in, hdf5_filename+".h5")
        
        
        with h5py.File(hdf5file_path, "r") as hdf5FileIn:
            if level == "1p0a":
                yIn = hdf5FileIn["Science/YUnmodified"][...]
            else:
                yIn = hdf5FileIn["Science/Y"][...]
            binsStart = hdf5FileIn["Science/Bins"][:, 0]
            binsEnd = hdf5FileIn["Science/Bins"][:, 1]
    
#            integrationTimes = hdf5FileIn["Channel/IntegrationTime"][...]
#            bins = hdf5FileIn["Science/Bins"][...]
#            nAccumulations = hdf5FileIn["Channel/NumberOfAccumulations"][...]
#    
#        integrationTime = np.float(integrationTimes[0]) / 1.0e3 #milliseconds to seconds
#        nAccumulation = np.float(nAccumulations[0])/2.0 #assume LNO nadir background subtraction is on
#        binning = np.float(bins[0,1] - bins[0,0]) + 1.0 #binning starts at zero
#        nBins = 1.0 #Science/Bins reflects the real binning
#        
#        measurementSeconds = integrationTime * nAccumulation
#        measurementPixels = binning * nBins
#        
#        measurementPixelSeconds = measurementPixels * measurementSeconds
#        
#        print("integrationTime = %0.3f, light nAccumulation = %i, binning = %i, measurementSeconds = %0.1f" %(integrationTime, nAccumulation, binning, measurementSeconds))
#        print("measurementPixelSeconds = %0.3f" %(measurementPixelSeconds))
            
        indicesStart = np.where(binsStart == BIN_START)[0]
        indicesEnd = np.where(binsEnd == BIN_END)[0]
        
        yMean = []
        for indexStart, indexEnd in zip(indicesStart, indicesEnd):
    #        print(indexStart, indexEnd)
            ySelection = yIn[indexStart:indexEnd+1, :]
            yMean.append(np.sum(ySelection, axis=0))
    
        yMeanAll.append(yMean)
    
    levels_to_compare = ["0p1d", "0p3a", "1p0a"]
    
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(ncols=3, figsize=(FIG_X+5, FIG_Y), sharey=True)
    for yMean, level, linestyle in zip(yMeanAll, levels, linestyles):
        for spectrum_number in range(1, int(len(binsStart)), 30):
            if level == levels_to_compare[0]:
                ax1a.plot(yMean[spectrum_number], label="i=%i" %(spectrum_number), linestyle=linestyle)
            elif level == levels_to_compare[1]:
                ax1b.plot(yMean[spectrum_number], label="i=%i" %(spectrum_number), linestyle=linestyle)
            elif level == levels_to_compare[2]:
                ax1c.plot(yMean[spectrum_number], label="i=%i" %(spectrum_number), linestyle=linestyle)
    ax1a.legend()
    ax1b.legend()
    ax1c.legend()
    ax1a.set_title("%s summed" %levels_to_compare[0])
    ax1b.set_title("%s summed" %levels_to_compare[1])
    ax1c.set_title("%s summed" %levels_to_compare[2])
    ax1a.set_ylim((-500, 6500))

    hdf5_filename_title = hdf5_filename_base[:16] + "####" + hdf5_filename_base[20:]
    fig1.suptitle(hdf5_filename_title)
    
        
    fig1.savefig(os.path.join(BASE_DIRECTORY, "%s_counts_comparison.png" %hdf5_filename_title))

#for spectrum_number in range(1, int(len(binsStart)), 30):
#    plt.figure(figsize=(FIG_X, FIG_Y))
#    for yMean, level, linestyle in zip(yMeanAll, levels, linestyles):
#        plt.plot(yMean[spectrum_number], label="%s-%i" %(level, spectrum_number), linestyle=linestyle)
#    plt.legend()





