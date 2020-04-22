# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:43:27 2020

@author: iant

MIGUEL: AVERAGE TOGETHER ALL ORDER 134 AND 136 SO OBSERVATIONS BETWEEN 2 ALTITUDES
"""

import os
#import h5py
import numpy as np
#import numpy.linalg as la
#from scipy import stats
#import scipy.optimize
import re

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

#from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt


from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames
from get_hdf5_data_v01 import getLevel1Data
#from plot_simulations_v01 import getSimulationDataNew, getSOAbsorptionPixels
from retrievals.SO_retrieval_v02b import NOMAD_ret, forward_model

diffractionOrder = 134
#diffractionOrder = 136

#regex = re.compile("20(18|19|20).*_SO_A_[IE]_%s" %diffractionOrder)
regex = re.compile("20200101.*_SO_A_[IE]_%s" %diffractionOrder)
fileLevel = "hdf5_level_1p0a"

#regex = re.compile("20191215_050908.*_SO_A_[IE]_136")
#fileLevel = "hdf5_level_1p0a"

min_alt = 5.0
max_alt = 15.0


def baseline_als(y, lam=250.0, p=0.95, niter=10):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z


print("Opening files")
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
#hdf5_file = hdf5Files[0]
#hdf5_filename = hdf5Filenames[0]

wavenumberGrid = np.arange(3005.0, 3040.0, 0.01)
spectrumGrid = np.zeros_like(wavenumberGrid)

nIndicesGrid = np.zeros_like(wavenumberGrid)

plt.figure()

print("Getting data")
for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

    nIndices = 0
    summedSpectra = np.zeros(320)

    for bin_index in [0, 1, 2, 3]:
        obsDict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=100.0) #use mean method, returns dictionary
    
    
        indices = np.where((min_alt < obsDict["alt"]) & (obsDict["alt"] < max_alt))[0]
        for spectrum in obsDict["y_mean"][indices, :]:
            
            pixelCentre = np.arange(-160.0, 160.0)
            yBaseline = np.polyval(np.polyfit(pixelCentre, spectrum, 5), pixelCentre)
#            yBaseline = baseline_als(obsDict["y_mean"][index, :]) #doesn't work. Needs mean
            summedSpectra += (spectrum - yBaseline)
            plt.plot(spectrum - yBaseline)
            
#        summedSpectra += np.sum(obsDict["y_mean"][indices, :], axis=0)
        nIndices += len(indices)
#        plt.plot(summedSpectra/nIndices)

    x = obsDict["x"][0, :]
    xIndices = np.digitize(x, wavenumberGrid)
    spectrumGrid[xIndices] += summedSpectra
    nIndicesGrid[xIndices] += nIndices

spectrumGrid[nIndicesGrid == 0] = np.nan
spectrumGrid = spectrumGrid / nIndicesGrid
meanSpectrum = spectrumGrid
#xGrid = 

#meanBaseline = baseline_als(meanSpectrum)
#meanCorrected = meanSpectrum / meanBaseline
meanCorrected = meanSpectrum

plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(wavenumberGrid[np.isfinite(meanCorrected)], meanCorrected[np.isfinite(meanCorrected)])
plt.title("Mean of all spectra %i-%i km for regex %s" %(min_alt, max_alt, regex))
plt.ylabel("Mean transmittance with baseline removed")
plt.xlabel("Pixel number")


simulate = False
#simulate = True

if simulate:
    #the retrieval bit stolen from Justin + modified for IDE and using dictionaries rather than h5 file and classes
    if diffractionOrder == 134:
        regex2 = re.compile("20191224_235111.*_SO_A_[IE]_134")
    elif diffractionOrder == 136:
        regex2 = re.compile("20191215_050908.*_SO_A_[IE]_136")
    else:
        print("Error")
    hdf5Files, hdf5Filenames, titles = makeFileList(regex2, fileLevel)
    
    bin_index = 3
    obsDict = getLevel1Data(hdf5Files[0], hdf5Filenames[0], bin_index, silent=True, top_of_atmosphere=70.0) #use mean method, returns dictionary
    vals = obsDict["alt"] > -999
    vals = vals & (obsDict["alt"] >= min_alt)
    vals = vals & (obsDict["alt"] <= max_alt)
    zind = np.nonzero(vals)[0]
        
    print("Altitude range at tmin=", np.min(obsDict["alt"][zind]), "tmax=", np.max(obsDict["alt"][zind]))
    obsDict["pixels"] = np.arange(320.0)
        
            
    for dataset in ["x", "y", "y_mean", "y_error", "y_raw"]:
        obsDict[dataset] = obsDict[dataset][zind, :]
    for dataset in ["alt", "latitude", "longitude", "ls", "lst"]:
        obsDict[dataset] = obsDict[dataset][zind]
    
    
    obsDict["resolving_power"] = 14000.0
    obsDict["gaussian_scalar"] = None
    obsDict["gaussian_xoffset"] = None
    obsDict["gaussian_width"] = None
        
    obsDict["molecule"] = "CH4"
    
    retDict = NOMAD_ret(obsDict)
    xa_fact = retDict["xa_fact"]
    retDict = forward_model(retDict, xa_fact=xa_fact)
    
    for index in [0, -1]:
        absorption_line = retDict["Trans_p"][index, :]
        plt.plot(absorption_line, label="Simulation %0.1fppb at %0.1fkm" %(retDict["xa"][0]+xa_fact[0], obsDict["alt"][index]))
    plt.legend()
    plt.savefig(os.path.join(BASE_DIRECTORY, "mean_transmittance_order_%i.png" %diffractionOrder))
