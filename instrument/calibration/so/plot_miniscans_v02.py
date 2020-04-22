# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:08:39 2019

@author: iant
"""

import os
#import h5py
import numpy as np
#import numpy.linalg as la
#import gc
#from scipy import stats
#import scipy.optimize
import re
#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

from scipy.signal import savgol_filter
#from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, getFile, makeFileList
#from hdf5_functions_v03 import getFilesFromDatastore
#from analysis_functions_v01b import write_log
#from filename_lists_v01 import getFilenameList



#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
"""plot miniscan transmittances"""
#regex = re.compile("20190.*_SO_.*_S")
regex = re.compile("20190117_061101.*_SO_.*_S")
fileLevel = "hdf5_level_0p3a"



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


def plotMiniscan0p1a(obspaths, fileLevel):
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
    
    #from plot_animations_v01 import animateLines
    
    plt.figure()
    #for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):    
    for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files[3:4], hdf5Filenames[3:4])):    
        detectorData = hdf5_file["Science/Y"][...]
        aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
#        exponent = hdf5_file["Channel/Exponent"][...]
    
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
        
    


def searchForMiniscans(regex, fileLevel):
    
    hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel, open_files=False, silent=True)
    
    for hdf5_filename in hdf5Filenames:
        hdf5_filename_split = hdf5_filename.split("_")
        if len(hdf5_filename_split) == 6:
            if hdf5_filename_split[-1] == "S":
                name, hdf5_file = getFile(hdf5_filename, fileLevel, 0, silent=True)
                detectorData = hdf5_file["Science/Y"][...]
                nSpectra = detectorData.shape[0]
                alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
                if alts[0] > 250.0 and alts[-1] > 250.0:
                    print("Warning: merged occultation")
                print("nSpectra=%i, hdf5_filename=%s, orders=" %(nSpectra, hdf5_filename))
                print(list(set(list(hdf5_file["Channel/DiffractionOrder"][...]))))

"""search for specific miniscans in dataset"""
#searchForMiniscans(regex, fileLevel)   
    
"""plot miniscans"""
hdf5Files, hdf5Filenames, _ = makeFileList(regex, fileLevel, silent=True)

hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]

plt.figure(figsize=(FIG_X, FIG_Y))
alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]

if alts[0] > 250.0 and alts[-1] > 250.0:
    isMerged = True
    halfLength = int(len(alts)/2)

if isMerged:
    alts = alts[0:halfLength]
    detectorData = hdf5_file["Science/Y"][0:halfLength, 50:]
    
diffractionOrder = hdf5_file["Channel/DiffractionOrder"][0:halfLength]
x = hdf5_file["Science/X"][0:halfLength, 50:]
bins = hdf5_file["Science/Bins"][0:halfLength, 0]

nPixels = len(x[0, :])
chosenBinTop = 124
chosenAltitudeRange = [30, 40]

uniqueDiffractionOrders = sorted(list(set(diffractionOrder)))[5:95]
for chosenDiffractionOrder in uniqueDiffractionOrders:
    if chosenDiffractionOrder < 130:
        sunAltitude = 75
    elif chosenDiffractionOrder < 155:
        sunAltitude = 90
    elif chosenDiffractionOrder < 160:
        sunAltitude = 100
    elif chosenDiffractionOrder < 170:
        sunAltitude = 150
    else:
        sunAltitude = 80

    if chosenDiffractionOrder in [119, 120, 121]:
        colour = "r"
    elif chosenDiffractionOrder in [133, 134, 135, 136]:
        colour = "gold"
    elif chosenDiffractionOrder in [147, 148, 149]:
        colour = "b"
    elif chosenDiffractionOrder in list(range(157, 168)):
        colour = "green"
    elif chosenDiffractionOrder in [168, 169, 170, 171]:
        colour = "fuchsia"
    elif chosenDiffractionOrder in [186, 187, 192, 193]:
        colour = "c"
    elif chosenDiffractionOrder in [188, 189, 190, 191]:
        colour = "purple"
    else:
        colour = "grey"
        
    
    
    orderBinIndices = np.where((alts>-100) & (bins==chosenBinTop) & (diffractionOrder==chosenDiffractionOrder))[0]
#    plt.plot(alts[orderBinIndices], detectorData[orderBinIndices, 200], label=chosenDiffractionOrder)
#    plt.plot(alts[orderBinIndices], detectorData[orderBinIndices, 195], linestyle="--", label=chosenDiffractionOrder)
    sunIndices = np.where((alts>sunAltitude) & (bins==chosenBinTop) & (diffractionOrder==chosenDiffractionOrder))[0]
    detectorDataSun = np.mean(detectorData[sunIndices[0:3], :], axis=0)
#    wavenumbers = np.arange(nPixels)
    wavenumbers = x[sunIndices[0], :]

    atmosIndices = np.where((alts>chosenAltitudeRange[0]) & (alts<chosenAltitudeRange[1]) & (bins==chosenBinTop) & (diffractionOrder==chosenDiffractionOrder))[0]
    print("%i spectra found order %i" %(len(atmosIndices), chosenDiffractionOrder))

    detectorDataAtmos = detectorData[atmosIndices[0], :]
    transmittance = detectorDataAtmos / detectorDataSun

    transSmooth = savgol_filter(transmittance, 9, 3)
    transSmooth2 = np.polyval(np.polyfit(wavenumbers, transmittance, 3), wavenumbers)
    
    tempNorm = transSmooth / transSmooth2
    
    locMaxPixels = (np.diff(np.sign(np.diff(tempNorm))) < 0).nonzero()[0] + 1
    locMaxPixels = np.intersect1d(np.where(tempNorm>1)[0], locMaxPixels)
#    locMaxPixels = np.where(tempNorm -1 > 0.8*np.std(tempNorm))[0]
    locMaxPixels = np.insert(locMaxPixels, 0, 0)
    locMaxPixels = np.append(locMaxPixels, nPixels-1)
    
    
#    locMaxPixels = range(nPixels)
    polyfit = np.polyval(np.polyfit(wavenumbers[locMaxPixels], transSmooth[locMaxPixels], 4), wavenumbers)
    transNorm = transSmooth/polyfit
    
#    transSmooth3 = savgol_filter(transNorm, 9, 3)
    locMaxPixels2 = (np.diff(np.sign(np.diff(tempNorm))) < 0).nonzero()[0] + 1
    locMaxPixels2 = np.insert(locMaxPixels2, 0, 0)
    locMaxPixels2 = np.append(locMaxPixels2, nPixels-1)
#    transSmooth4 = savgol_filter(transSmooth3[locMaxPixels2], 5, 3)
    interpfit = np.interp(wavenumbers, wavenumbers[locMaxPixels2], transNorm[locMaxPixels2])
    
#    plt.plot(wavenumbers, )
    
    
#    plt.scatter(wavenumbers[locMaxPixels], transSmooth[locMaxPixels])
#    plt.plot(wavenumbers, transSmooth)
#    plt.plot(wavenumbers, transSmooth2)
#    plt.plot(wavenumbers, polyfit, linestyle="--")

    if chosenDiffractionOrder > 185:
        plt.plot(wavenumbers[50:], transNorm[50:], color=colour, alpha=0.8)
    else:
        plt.plot(wavenumbers, transNorm, color=colour, alpha=0.8)
#    plt.plot(wavenumbers[locMaxPixels2], tempNorm[locMaxPixels2])
#    plt.plot(wavenumbers, interpfit)
#    
        
#    plt.plot(wavenumbers, tempNorm)
#    plt.plot(wavenumbers, transSmooth/transSmooth2)
#    plt.plot(wavenumbers, transmittance, label="order %s index %i" %(chosenDiffractionOrder, atmosIndices[0]))
#plt.plot(wavenumbers, detectorDataAtmos, label="order %s index %i" %(chosenDiffractionOrder, atmosIndices[0]))
#plt.plot(wavenumbers, detectorDataSun, label="order %s index %i" %(chosenDiffractionOrder, atmosIndices[0]))
#plt.legend()
plt.ylabel("Normalised transmittance")
plt.xlabel("Wavenumbers (cm-1)")
plt.title("SO fullscan %s" %hdf5_filename)

if SAVE_FIGS:
    plt.savefig(os.path.join(BASE_DIRECTORY, "so_fullscan_%s" %hdf5_filename))

