# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:08:39 2019

@author: iant
"""

#import os
#import h5py
import numpy as np
#import numpy.linalg as la
#import gc
#from scipy import stats
#import scipy.optimize

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

#from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, getFile, makeFileList, printFileNames
#from hdf5_functions_v03 import getFilesFromDatastore
#from analysis_functions_v01b import write_log
#from filename_lists_v01 import getFilenameList



#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = [
        "20190107_012128_0p1a_SO_1",
        "20190107_012128_0p1a_SO_2",
        "20190107_015635_0p1a_SO_1",
        "20190107_015635_0p1a_SO_2",
        ]
fileLevel = "hdf5_level_0p1a"



def applyFilter(data, index_start, plot=False):
    from scipy.signal import butter, lfilter
    
    RESOLUTION = 10.0
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    order = 3
    fs = 15.0       # sample rate, Hz
    cutoff = 0.5#3.667  # desired cutoff frequency of the filter, Hz
    
    
    
    pixel_in = np.arange(len(data))

    dataFit = butter_lowpass_filter(data, cutoff, fs, order)
    
    pixelInterp = np.arange(pixel_in[0], pixel_in[-1]+1.0, 1.0/RESOLUTION)
    dataInterp = np.interp(pixelInterp, pixel_in, data)
    dataFitInterp = np.interp(pixelInterp, pixel_in, dataFit)
    
    firstPoint = int(index_start * RESOLUTION)
    pixelInterp = pixelInterp[firstPoint:]
    dataInterp = dataInterp[firstPoint:]
    dataFitInterp = dataFitInterp[firstPoint:]
    
    nPoints = len(dataInterp)
    
    
    chi = np.asfarray([np.sum((dataInterp[0:(nPoints-index)] - dataFitInterp[index:(nPoints)])**2) / (nPoints - index) \
                       for index in np.arange(0, 1000, 1)])
    minIndex = np.argmin(chi)-1

    if plot:
        plt.subplots(figsize=(14,10), sharex=True)
        plt.subplot(2, 1, 1)
        plt.plot(pixelInterp, dataInterp, 'b-', label='data')
        plt.plot(pixelInterp[0:(nPoints-minIndex)], dataFitInterp[minIndex:(nPoints)], 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    
    x = pixelInterp[0:(nPoints-minIndex)]
    yfit = dataFitInterp[minIndex:(nPoints)]
    y = dataInterp[0:(nPoints-minIndex)] / yfit
    
    
    
    if plot:
        plt.subplot(2, 1, 2)
        plt.plot(x, y, label="residual")
        plt.ylim([0.95, 1.02])
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return x, yfit, y



hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)

#from plot_animations_v01 import animateLines

plt.figure()
#for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):    
for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files[3:4], hdf5Filenames[3:4])):    
    detectorData = hdf5_file["Science/Y"][...]
    aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
    exponent = hdf5_file["Channel/Exponent"][...]

    detectorDataMean = np.mean(detectorData[:,12:13], axis=1)
#    a = animateLines([detectorDataMean])
    
#    plotIndices = [0,1,2,3,256,257]
#    cmap = plt.get_cmap('jet')
#    colours = [cmap(i) for i in aotfFrequency[np.asarray(plotIndices)]]
    
#    for plotIndex in plotIndices:
##        plt.plot(detectorDataMean[plotIndex,:], label=aotfFrequency[plotIndex])
    plt.plot(aotfFrequency, detectorDataMean[:,100]/np.mean(detectorDataMean, axis=1))
    plt.plot(aotfFrequency, detectorDataMean[:,200]/np.mean(detectorDataMean, axis=1))

#    filteredData = [applyFilter(spectrum, 30) for spectrum in detectorDataMean[256:512,:]]
#    residuals = np.asfarray([data[2][0:2800] for data in filteredData])
#    a = animateLines([residuals])
    
#    plt.legend()
    