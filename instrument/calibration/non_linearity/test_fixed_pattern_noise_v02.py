# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Thu Feb  7 14:43:38 2019

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

from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

from plot_simulations_v01 import plotSimulation, getSimulation, getSOAbsorptionPixels, getSimulationDataNew
from plot_solar_line_simulations import getSolarLineShiftPixels

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames
from get_hdf5_data_v01 import getLevel1Data



SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

APPLY_CORRECTION = False
#APPLY_CORRECTION = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


#regex = re.compile("201906.*SO.*_(133|134|135|136).*")
#regex = re.compile("201906.*SO.*_[IE]_.*136.*")
#regex = re.compile("(20190618_105903|20190621_015027|20190623_180452).*SO.*_(133|134|135|136).*")
#regex = re.compile("20190618_105903.*SO.*_(133|134|135|136).*")
#regex = re.compile("20190618_105903.*SO.*_(133|134|135|136).*")
#regex = re.compile("(20190625_233600|20190622_230012|20190620_195653|20190617_223039|20190615_180901).*SO.*")
#regex = re.compile("20190625_233600.*SO.*")
#regex = re.compile("(20190615_180901|20190620_195653|20190625_233600).*SO_A_E_136.*") # 5 x 136 files
#regex = re.compile("(20190625_233600|20190622_230012).*SO_A_.*") # one 5 x 134 and one 5 x 136 file
regex = re.compile("20190625_233600.*SO_A_.*") # one 5 x 134 and one 5 x 136 file
fileLevel = "hdf5_level_1p0a"





def polynomialFit(array_in, order_in):
    arrayShape = array_in.shape
    if len(arrayShape) == 1:
        nElements = array_in.shape[0]
    
    return np.polyval(np.polyfit(range(nElements), array_in, order_in), range(nElements))


def writeOutput(filename, lines_to_write):
    """function to write output to a file"""
    outFile = open("%s.txt" %filename, 'w')
    for line_to_write in lines_to_write:
        outFile.write(line_to_write+'\n')
    outFile.close()









CENTRE_PIXEL = 180
POLYFIT_DEGREE = 5
bin_index = 1
#check 1 bin only, split into transmittance bins
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
#loop through, taking y spectra nearest T=0.3 from each order and storing pixel values where no water lines are present


#load correction
pixelFittingCorrectionData = np.loadtxt("pixel_fitting_all.csv", delimiter=",")
pixelFittingCorrectionPolynomials = pixelFittingCorrectionData[:, 1:3]
corrected_pixels = np.where(pixelFittingCorrectionData[:, 1] != 0.0)[0]
not_corrected_pixels = np.where(pixelFittingCorrectionData[:, 1] == 0.0)[0]





hdf5Filenames = ["20190625_233600_1p0a_SO_A_E_136"]
import h5py
hdf5Files = [h5py.File(os.path.join(BASE_DIRECTORY, hdf5Filenames[0]+".h5"), "r")]


for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):

    #hdf5_file = hdf5Files[0]
    #hdf5_filename = hdf5Filenames[0]
    
    obsDict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=65.0) #use mean method, returns dictionary
    
    
    wavenumbers = obsDict["x"]#[50:] #choose limited region of spectrum for analysis
    pixels = np.arange(0.0, len(wavenumbers))
    y = obsDict["y_mean"]
    y_raw = obsDict["y_raw"]
    order = obsDict["order"]
    temperature = np.round(obsDict["temperature"])
    
    binDelta = 0.2
    binStart = 0.05
    
#    transmittanceBins = np.zeros((19, 2))
#    transmittanceBins[:,0] = np.arange(19) / 20 + 0.05
#    transmittanceBins[:,1] = np.arange(19) / 20 + 0.1

    transmittanceBins = np.zeros((int(1.0/binDelta - 1), 2))
    transmittanceBins[:,0] = np.arange(0.0, 1.0-binStart-binDelta+0.001, binDelta) + binStart
    transmittanceBins[:,1] = np.arange(0.0, 1.0-binStart-binDelta+0.001, binDelta) + binDelta + binStart

    
    meanTransmittanceBins = np.mean(transmittanceBins, axis=1)
    
#    not_water_pixels, water_pixels, combined_spectrum = getSOAbsorptionPixels(order, temperature, molecules=["H2O"], delta_t=1.0, cutoff=0.999, plot=True)
    not_water_pixels, water_pixels, combined_spectrum = getSOAbsorptionPixels(order, temperature, molecules=["H2O"], delta_t=1.0, cutoff=0.999, plot=False)
    
    
    """check solar line shifts"""
#    solspecFile = os.path.join(BASE_DIRECTORY, "reference_files", "nomad_solar_spectrum_solspec.txt")
#    solarLineShift = getSolarLineShiftPixels(order, [temperature, temperature + 1.0], solspecFile)
#    np.savetxt(os.path.join(BASE_DIRECTORY, "solar_shift_order%i.txt" %order), solarLineShift)
#    plt.figure(figsize=(FIG_X + 6, FIG_Y))
#    plt.plot(solarLineShift)
#    plt.xlim([0,320])
#    plt.title("Solar line shifts order %i, T1=%0.1f, T2=%0.1f" %(order, temperature, temperature + 1.0))
#    plt.ylabel("Normalised transmittance")
#    plt.xlabel("Pixel number")
    solarLineShift = np.loadtxt(os.path.join(BASE_DIRECTORY, "solar_shift_order%i.txt" %order))
    
    
    yAll = np.zeros((len(transmittanceBins), len(pixels)))
    
    fig1, ax1 = plt.subplots(figsize=(FIG_X + 3, FIG_Y+3))
    ax1.set_xlim([0,320])
    for transmittanceBinIndex, transmittanceBin in enumerate(transmittanceBins):
        indicesInBin = np.where((y[:, CENTRE_PIXEL] > transmittanceBin[0]) & (y[:, CENTRE_PIXEL] < transmittanceBin[1]))[0]
    
        yNormalisedMean = np.zeros((len(indicesInBin), len(pixels)))
    
        for spectrumIndex, indexInBin in enumerate(indicesInBin):
            fit = np.polyval(np.polyfit(pixels[not_water_pixels], y[indexInBin, not_water_pixels], POLYFIT_DEGREE), np.arange(320))
            yNormalised = y[indexInBin, :] - fit #this is the residual
            pixels[water_pixels] = np.nan
            yNormalised[water_pixels] = np.nan
            
#            apply pixel correction
            if APPLY_CORRECTION:
                yCorrection = np.zeros(320)
                for pixel in range(320):
                    yCorrection[pixel] = np.polyval(pixelFittingCorrectionPolynomials[pixel, :], fit[pixel])
                ax1.plot(pixels, yNormalised - yCorrection + transmittanceBin[0]/10.0, alpha=0.5)
                yNormalisedMean[spectrumIndex, :] = yNormalised - yCorrection
            else:
                yNormalisedMean[spectrumIndex, :] = yNormalised
                ax1.plot(pixels, yNormalised + transmittanceBin[0]/10.0, alpha=0.5)
    #        ax1.plot(pixels, yNormalised, alpha=0.5)
    
        
    
        yAll[transmittanceBinIndex, :] = np.mean(yNormalisedMean, axis=0)
    
    #CHOSEN_PIXELS = [180, 182, 183, 186]
    CHOSEN_PIXELS = pixels[~np.isnan(pixels)] #all pixels not nan
    
    plt.figure(figsize=(FIG_X, FIG_Y));

    outputLines = ["%i, %0.5g, %0.5g" %(pixel, 0.0, 0.0) for pixel in range(320)]
    
    
    for pixelNumber in CHOSEN_PIXELS:
        plt.plot(meanTransmittanceBins, yAll[:, int(pixelNumber)]);
        polyfit = np.polyfit(meanTransmittanceBins, yAll[:, int(pixelNumber)], 1)
        polyval = np.polyval(polyfit, meanTransmittanceBins)
        plt.plot(meanTransmittanceBins, polyval, "--")
        
        outputLines[int(pixelNumber)] = "%i, %0.5g, %0.5g" %(pixelNumber, polyfit[0], polyfit[1])
    plt.title(hdf5_filename)
    plt.xlabel("Transmittance")
    plt.ylabel("Deviation from fitted mean")
    
    writeOutput("pixel_fitting_order%i" %order, outputLines)
    
    
    
    plotIndices = range(0, 1000, 1)

    fig1, (ax1a, ax1b) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(FIG_X, FIG_Y), gridspec_kw={'width_ratios': [1, 3]})
    yResidual = np.zeros_like(y)
    for spectrumIndex, spectrum in enumerate(y):
        polyfit = np.polyval(np.polyfit(np.arange(320)[not_water_pixels], spectrum[not_water_pixels], 5), range(320))
        residualLine = spectrum - polyfit
        yResidual[spectrumIndex, :] = residualLine
    ax1a.plot(y[plotIndices, 200], range(len(y[plotIndices,0])))
    ax1a.set_xlabel("Transmittance")
    ax1a.set_ylabel("Frame number")
    imshow1 = ax1b.imshow(yResidual[plotIndices, :], aspect=0.30, vmin=-0.01, vmax=0.01)
    fig1.colorbar(imshow1)
    print("Corrected pixels std before correction = ", np.std(yResidual[580, not_water_pixels]))


    fig2, (ax2a, ax2b) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(FIG_X, FIG_Y), gridspec_kw={'width_ratios': [1, 3]})
    yResidualCorrected = np.zeros_like(y)
    for spectrumIndex, spectrum in enumerate(y):
        polyfit = np.polyval(np.polyfit(np.arange(320)[not_water_pixels], spectrum[not_water_pixels], 5), range(320))
        residualLine = spectrum - polyfit
        
        # apply pixel correction
        yCorrection = np.zeros(320)
        for pixel in range(320):
            if 0.01< polyfit[pixel] < 0.95:
                yCorrection[pixel] = np.polyval(pixelFittingCorrectionPolynomials[pixel, :], polyfit[pixel])
        
        yResidualCorrected[spectrumIndex, :] = residualLine - yCorrection
    ax2a.plot(y[plotIndices, 200], range(len(y[plotIndices,0])))
#    yResidualCorrected[:, not_corrected_pixels] = np.nan
    ax2a.set_xlabel("Transmittance")
    ax2a.set_ylabel("Frame number")
    imshow2 = ax2b.imshow(yResidualCorrected[plotIndices, :], aspect=0.3, vmin=-0.01, vmax=0.01)
    fig2.colorbar(imshow2)
    print("Corrected pixels std after correction = ", np.std(yResidualCorrected[580, not_water_pixels]))

    #plot only non-water corrected pixels
    fig3, (ax3a, ax3b) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(FIG_X, FIG_Y), gridspec_kw={'width_ratios': [1, 3]})
    yResidualNotWater = np.zeros_like(y[:, not_water_pixels])
    for spectrumIndex, spectrum in enumerate(y):
        polyfit = np.polyval(np.polyfit(np.arange(320)[not_water_pixels], spectrum[not_water_pixels], 5), range(320))
        residualLine = spectrum - polyfit
        
        #  apply pixel correction
        yCorrection = np.zeros(320)
        for pixel in range(320):
            if 0.01< polyfit[pixel] < 0.95:
                yCorrection[pixel] = np.polyval(pixelFittingCorrectionPolynomials[pixel, :], polyfit[pixel])
        
        yResidualNotWater[spectrumIndex, :] = residualLine[not_water_pixels] - yCorrection[not_water_pixels]
    ax3a.plot(y[plotIndices, 200], range(len(y[plotIndices,0])))
#    yResidualCorrected[:, not_corrected_pixels] = np.nan
    ax3a.set_xlabel("Transmittance")
    ax3a.set_ylabel("Frame number")
    imshow3 = ax3b.imshow(yResidualNotWater[plotIndices, :], aspect=0.12, vmin=-0.01, vmax=0.01)
    fig3.colorbar(imshow3)



plt.figure()
plt.plot(yResidual[580, :]); 
plt.plot(yResidualCorrected[580, :])
plt.plot(yResidualCorrected[581, :])
plt.scatter(np.arange(320)[not_water_pixels], yResidualCorrected[580, not_water_pixels])

plt.figure()
plt.plot(solarLineShift)

