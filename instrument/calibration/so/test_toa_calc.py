# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:05:42 2019

@author: iant

TEST HEND
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate as interpolate


from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, makeFileList

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


"""plot solar lines in order"""
title = "toa"

fileLevel = "hdf5_level_0p3j"
obspaths = ["*201808*_0p3j_SO*_I_134"]




def splitIntoBins(data_in, n_bins):
    nSpectra = data_in.shape[0]
    data_out = []
    for index in range(n_bins):
        binRange = range(index, nSpectra, n_bins)
        if data_in.ndim ==2:
            data_out.append(data_in[binRange, :])
        else:
            data_out.append(data_in[binRange])
            
    return data_out


def poly(data_in, order):
    indices = np.arange(data_in.shape[0])
    return np.polyval(np.polyfit(indices, data_in, order), indices)


def normalise(spectrum_in):
    return spectrum_in / np.max(spectrum_in)


hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)

plot = False
silent = True
SPECTRAL_FITTING_POLYNOMIAL_DEGREE = 15

lats = []
toaAlts = []

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):


    x = hdf5_file["Science/X"][...]
    y = hdf5_file["Science/Y"][...]
    pixels = np.arange(320)
    alt = np.mean(hdf5_file["Geometry/Point0/TangentAltAreoid"][...], axis=1)
    lat = np.mean(hdf5_file["Geometry/Point0/Lat"][...], axis=1)
    latMean = np.mean(lat[lat>-100])
    order = hdf5_file["Channel/DiffractionOrder"][0]
    
    hdf5_file.close()
    
    
    
    nSpectaLow = np.histogram(alt, bins=[10.,40.])[0][0]
    nSpectraHigh = np.histogram(alt, bins=[60.,90.])[0][0]

    if nSpectraHigh > 0:
        yBin = splitIntoBins(y, 4)[1]
        altBin = splitIntoBins(alt, 4)[1]
        yIndices = list(range(yBin.shape[0]))
        
        smoothing_degree = int(len(yIndices) / 10.0 / 2.0) * 2 + 1
        nSpectraTop = np.sum(altBin > 200.0)

        #fit high degree polynomial to raw spectra
        yBinPoly = np.asfarray([poly(spectrum, SPECTRAL_FITTING_POLYNOMIAL_DEGREE) for spectrum in yBin])
        
        
        
        if plot: 
            fig1, ax1 = plt.subplots()
            plt.title(hdf5_filename)
        toaPixelIndices = []
        for pixel in [100, 150, 200, 250, 300]: #check various pixels
            pixelValue = yBinPoly[:, pixel]
            pixelValueSmoothed = signal.savgol_filter(pixelValue, smoothing_degree, 1) #apply smoothing filter
            pixelValueDeviation = pixelValue - pixelValueSmoothed #subtract filtered from raw value
            
            noisePeak = np.max(np.abs(pixelValueDeviation[0:nSpectraTop]))  #ingress
            
            
            if plot: plt.plot(pixelValueDeviation)
        
        
        
            toaIndex = (np.abs(pixelValueDeviation)>(noisePeak * 3.0)).argmax() #ingress
            
            if plot: plt.scatter(yIndices[toaIndex], pixelValueDeviation[toaIndex])
            
            value = pixelValueDeviation[toaIndex]
            while value > 0.0 and toaIndex > 50: #ingress
                toaIndex -= 1 #ingress
                value = pixelValueDeviation[toaIndex]
            
            toaPixelIndices.append(toaIndex)
            
            if plot: plt.scatter(yIndices[toaIndex], pixelValueDeviation[toaIndex], marker="x")
            if not silent: print(altBin[toaIndex])
        
        toaIndexDust = min(toaPixelIndices) #ingress
        toaAltDust = altBin[toaIndexDust]
        print("%s: Lat = %0.1f; Dust TOA = %0.1fkm" %(hdf5_filename, latMean, toaAltDust))
        lats.append(latMean)
        toaAlts.append(toaAltDust)
        
        if toaAltDust > 120:
            plt.figure()
            plt.title(hdf5_filename)
            plt.plot(altBin[altBin>-100], yBin[altBin>-100,200])

        #now check TOA spectra to search for absorption lines
        toaObsIndices = range(toaIndexDust)
        #normalise each spectrum
        highAltMean = np.mean(yBin[0:5, :], axis=0)
#        plt.figure()
#        
#        for spectrumIndex, spectrum in enumerate(yBin[toaObsIndices,:]):
#            if np.mod(spectrumIndex, 10) == 0:
#                transmittance = spectrum / highAltMean
#                normalisedTransmittance = transmittance / poly(transmittance, 7)
#                plt.plot(normalisedTransmittance)
            
    else:
        print("%s: no high altitude spectra" %hdf5_filename)
        
        
    
    
plt.figure()
plt.plot(lats, toaAlts)














