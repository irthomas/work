# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:40:13 2019

@author: iant
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
from get_hdf5_data_v01 import getLevel1Data, getLevel0p2Data
#from plot_simulations_v01 import getSimulationDataNew, getSOAbsorptionPixels
from analysis.retrievals.SO_retrieval_v02b import NOMAD_ret, Rodgers_OEM, forward_model


#regex = re.compile("(20190619|20190618)_1.*_SO_A_[IE]_134")
#fileLevel = "hdf5_level_1p0a"
#diffractionOrder = 134

diffractionOrder = 136
#regex = re.compile("20191215_050908.*_SO_A_[IE]_136") #no water, high dust, mid lat
#regex = re.compile("20191017_144026.*_SO_A_[IE]_136")
regex = re.compile("20191224_235111.*_SO_A_[IE]_136") #lots of water
#regex = re.compile("20190626_145411.*_SO_A_[IE]_136")
#diffractionOrder = 134

fileLevel = "hdf5_level_1p0a"

REMOVE_WATER = True
#REMOVE_WATER = False


#pixels = np.arange(80, 130)
pixels = np.arange(0, 320)


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



hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
#for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]


"""all bins - doesn't work well as fixed pattern noise is different in each bin"""
#obsDicts = []
#for bin_index in [0,1,2,3]:
#    obsDict1 = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=50.0) #use mean method, returns dictionary
#    obsDicts.append(obsDict1)
#
#obsDict = {}
#for dataset in ['error', 'first_pixel', 'hdf5_filename', 'label', 'obs_datetime', 'order', 'temperature']:
#    obsDict[dataset] = obsDict1[dataset]
#for dataset in ['alt', 'latitude', 'longitude', 'ls', 'lst', 'x', 'y', 'y_error', 'y_mean', 'y_raw']:
#    data = []
#    for obsDict1 in obsDicts:
#        data.extend(obsDict1[dataset])
#    data = np.asfarray(data)
#    
#    if dataset == "alt":
#        sortIndices = np.argsort(data)
#    
#    if data.ndim == 1:
#        obsDict[dataset] = data[sortIndices]
#    if data.ndim == 2:
#        obsDict[dataset] = data[sortIndices, :]



"""bin 3 only"""
#bin_index = 3



"""all bins (data collected after residual removal)"""
y_all = []
residuals_all = []
tangent_alt_all = []

#for bin_index in [0,1,2,3]:
for bin_index in [0]:
    #obsDict = getLevel0p2Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=100.0) #use mean method, returns dictionary
#    obsDict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=50.0) #use mean method, returns dictionary
    obsDict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    
    
    #limit spectral range
    obsDict["pixels"] = pixels
    for dataset in ["x", "y", "y_mean", "y_error", "y_raw"]:
        obsDict[dataset] = obsDict[dataset][:, pixels]
    
    #select spectra
    tmin = 0.05
    tmax = 0.99
    #select based on number of pixels, as it can be variable now
    xLength = len(obsDict["x"][0, :])
    yCentre = np.mean(obsDict["y_mean"][:, (int(xLength/2)-int(xLength/4)):(int(xLength/2)+int(xLength/4))], axis=1)
    nSpectra = len(yCentre)
    vals = obsDict["alt"] > -999
    vals = vals & (yCentre >= tmin)
    vals = vals & (yCentre <= tmax)
    zind = np.nonzero(vals)[0]
    
    print("Altitude range at tmin=", np.min(obsDict["alt"][zind]), "tmax=", np.max(obsDict["alt"][zind]))
    
    if np.max(obsDict["alt"][zind]) < 80.0: #add higher altitude indices
        zEnd = np.min(np.where(obsDict["alt"] > 80.0)[0])
        zStart = max(zind)
        zStep = np.linspace(zStart, zEnd, num=30)
        zNewSteps = np.round(zStep)[1:].astype(np.int64)
        zind = np.concatenate((zind, zNewSteps))
    
        
    for dataset in ["x", "y", "y_mean", "y_error", "y_raw"]:
        obsDict[dataset] = obsDict[dataset][zind, :]
    for dataset in ["alt", "latitude", "longitude", "ls", "lst"]:
        obsDict[dataset] = obsDict[dataset][zind]
    
    nSpectra = len(zind)    
    yCentre = np.mean(obsDict["y_mean"][:, (int(xLength/2)-int(xLength/4)):(int(xLength/2)+int(xLength/4))], axis=1)
    obsDict["y_centre"] = yCentre
    
        
    #correct spectral cal
    #correct baseline first
    observation_wavenumbers = obsDict["x"][0, :]
    
    
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(nSpectra)/nSpectra]



    #remove water from spectra
    if REMOVE_WATER:
        obsDict["resolving_power"] = 14000.0
        obsDict["gaussian_scalar"] = None
        obsDict["gaussian_xoffset"] = None
        obsDict["gaussian_width"] = None
        obsDict["molecule"] = "H2O"
        
        obsDict["first_pixel"] = obsDict["first_pixel"] + 3.0
        
        retDict1 = NOMAD_ret(obsDict)
        retDict1 = Rodgers_OEM(retDict1, alpha=1.0)

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("Pixel")
        plt.ylabel("Transmittance")
        plt.plot(obsDict["y_mean"][15, :])
        plt.plot(retDict1["Y"][15, :])
#        stop()
        
        obsDict["y_mean"] = (obsDict["y_mean"] / retDict1["Y"]) * np.tile(np.mean(retDict1["Trans_background"], axis=1), [320, 1]).T



    
    #plt.figure(figsize=(FIG_X, FIG_Y))
    y_residual = np.zeros_like(obsDict["y_mean"])
    for spectrum_index in range(nSpectra):
    #    y_mean_baseline = baseline_als(obsDict["y_mean"][spectrum_index, :])
    #    y_mean_corrected = obsDict["y_mean"][spectrum_index, :] / y_mean_baseline
    
        y_mean_baseline = np.polyval(np.polyfit(pixels, obsDict["y_mean"][spectrum_index, :], 5), pixels)
        y_mean_corrected = obsDict["y_mean"][spectrum_index, :]  - y_mean_baseline
        
        y_residual[spectrum_index, :] = y_mean_corrected
    
    #    plt.plot(y_mean_corrected)
    
    
    if bin_index == 3:
        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Transmittance")
        plt.plot(obsDict["x"].T, obsDict["y_mean"].T)

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Transmittance Residual")
        for spectrum_index in range(nSpectra):
            plt.plot(obsDict["x"][0,:], y_residual[spectrum_index, :], color=colours[spectrum_index])
    
    
    y_residual_median = np.median(y_residual, axis=0)
    y_residual_2 = y_residual - y_residual_median
    y_std = np.std(y_residual_2, axis=1)
    
    if bin_index == 3:
        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Transmittance Residual 2")
        for spectrum_index in range(nSpectra):
            plt.plot(obsDict["x"][0,:], y_residual_2[spectrum_index, :], color=colours[spectrum_index])

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("Transmittance")
        plt.ylabel("Altitude (km)")
        plt.plot(obsDict["y_centre"], obsDict["alt"])
        plt.ylim([29, 60])
    
        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(obsDict["label"])
        plt.xlabel("SNR (mean transmittance / standard dev)")
        plt.ylabel("Altitude (km)")
        plt.plot(obsDict["y_centre"]/y_std, obsDict["alt"], label="Unbinned")
        plt.ylim([29, 60])

    
    y_all.extend(obsDict["y_mean"])
    """use original y_residual - std smaller??"""
#    residuals_all.extend(y_residual_2)
    residuals_all.extend(y_residual)
    tangent_alt_all.extend(obsDict["alt"])
    
    



y_all = np.asfarray(y_all)
residuals_all = np.asfarray(residuals_all)
tangent_alt_all = np.asfarray(tangent_alt_all)

sortIndices = np.argsort(tangent_alt_all)

tangent_alt_sorted = tangent_alt_all[sortIndices]
y_sorted = y_all[sortIndices, :]
residuals_sorted = residuals_all[sortIndices, :]

#bin into 2km altitudes
#binnedAltitudes = np.arange(31, 65, 2)
binnedAltitudes = np.arange(3, 65, 2)
#consider pixels around ch4 line
#chosen_px = np.arange(110, 190)
chosen_px = np.arange(150, 220)


#get indices of each bin where 1=first bin, 2=second bin, etc.
verticalBinIndices = np.digitize(tangent_alt_sorted, binnedAltitudes)
uniqueBinIndices = list(set(verticalBinIndices))

#get mean altitude of each bin
binnedCentreAltitudes = np.mean((binnedAltitudes, np.append(binnedAltitudes[1::], binnedAltitudes[-1]+(binnedAltitudes[1]-binnedAltitudes[0]))), axis=0)

#for each altitude bin, get mean data for y_mean, residual2
#y_residual_binned = []
y_binned = []
tangent_alt_binned = []
for uniqueBinIndex in uniqueBinIndices:
#    residual_binned = np.mean(residuals_sorted[verticalBinIndices == uniqueBinIndex, :], axis=0)
    spectrum_binned = np.mean(y_sorted[verticalBinIndices == uniqueBinIndex, :], axis=0)
#    y_residual_binned.append(residual_binned)
    y_binned.append(spectrum_binned)
    tangent_alt_binned.append(binnedCentreAltitudes[uniqueBinIndex-1])
    
#y_residual_binned = np.asfarray(y_residual_binned)
y_binned = np.asfarray(y_binned)
tangent_alt_binned = np.asfarray(tangent_alt_binned)



"""get residual of binned spectrum, calculate std"""
y_binned_residual = np.zeros_like(y_binned)
y_binned_baseline = np.zeros_like(y_binned)
for spectrum_index in range(len(tangent_alt_binned)):
    y_binned_baseline[spectrum_index, :] = np.polyval(np.polyfit(pixels, y_binned[spectrum_index, :], 5), pixels)
    y_binned_corrected = y_binned[spectrum_index, :] - y_binned_baseline[spectrum_index, :]
    
    y_binned_residual[spectrum_index, :] = y_binned_corrected

y_binned_residual_median = np.median(y_binned_residual, axis=0)
y_binned_residual_2 = y_binned_residual# - y_binned_residual_median
#y_binned_std = np.std(y_binned_residual_2, axis=1)
y_binned_residual_2_std = np.std(y_binned_residual_2[:, chosen_px], axis=1)
double_y_binned_residual_2_std = 2.0 * y_binned_residual_2_std


#plt.figure()
#plt.title(obsDict["label"])
#plt.xlabel("SNR 2km binned")
#plt.ylabel("Altitude (km)")
plt.plot(np.mean(y_binned, axis=1)/y_binned_residual_2_std, tangent_alt_binned, label="2km binned")
#plt.ylim([30, 60])
plt.legend()










"""v2: average together residuals instead of raw spectra"""
#reduced chi sq
#rms 









#the retrieval bit stolen from Justin + modified for IDE and using dictionaries rather than h5 file and classes
obsDict["resolving_power"] = 14000.0
obsDict["gaussian_scalar"] = None
obsDict["gaussian_xoffset"] = None
obsDict["gaussian_width"] = None
    
obsDict["molecule"] = "CH4"
retDict = NOMAD_ret(obsDict)

found_detection_limits = []
found_tangent_altitudes = []

"""loop through each binned spectrum"""
for spectrumIndex in range(len(tangent_alt_binned)):


    binned_spectrum = y_binned[spectrumIndex, :]
    binned_spectrum_mean = np.mean(binned_spectrum[chosen_px])
    binned_spectrum_baseline = y_binned_baseline[spectrumIndex, :]
    binned_spectrum_flat = binned_spectrum/binned_spectrum_baseline * binned_spectrum_mean
    
    
    plt.figure()
    plt.title("%0.1fkm" %(tangent_alt_binned[spectrumIndex]))
    plt.plot(binned_spectrum_flat)
    plt.plot(pixels[chosen_px], [binned_spectrum_mean + double_y_binned_residual_2_std[spectrumIndex]]*len(chosen_px))
    plt.plot(pixels[chosen_px], [binned_spectrum_mean - 1.0 * double_y_binned_residual_2_std[spectrumIndex]]*len(chosen_px))
    
    
    
    
    
    #find index where binned altitude matches obsDict
    binnedCentreAltitude = tangent_alt_binned[spectrumIndex]
    matchingIndex = (np.abs(obsDict["alt"] - binnedCentreAltitude)).argmin()
    print("Altitude=%0.3fkm, difference=%0.3fkm" %(binnedCentreAltitude, binnedCentreAltitude - obsDict["alt"][matchingIndex]))
    
    #now match minima of simulation to binned std value
    xa_fact = retDict["xa_fact"]
    relative_difference = 1.0
    while np.abs(relative_difference) > 0.05:
        retDict = forward_model(retDict, xa_fact=xa_fact)
    
        absorption_line = retDict["Trans_p"][matchingIndex, :]
        absorption_line_scaled = absorption_line * binned_spectrum_mean
        plt.plot(absorption_line_scaled, label="%0.1f" %(retDict["xa"][0]+xa_fact[0]))
    
        line_depth = np.max(absorption_line_scaled) - np.min(absorption_line_scaled)
        
        relative_difference = (line_depth - double_y_binned_residual_2_std[spectrumIndex]) / double_y_binned_residual_2_std[spectrumIndex]
        if relative_difference > 5.0:
            relative_difference = 2.0
        print(relative_difference, xa_fact[0])
        if np.abs(relative_difference) < 0.05:
            print("%i/%i found" %(spectrumIndex, len(tangent_alt_binned)))
            
            found_tangent_altitudes.append(binnedCentreAltitude)
            found_detection_limits.append(retDict["xa"][0]+xa_fact[0])
            
        else:
            xa_fact = xa_fact - relative_difference * 0.5
        
    plt.legend()

plt.figure(figsize=(FIG_X, FIG_Y))
plt.title(obsDict["label"])
plt.xlabel("Approximate detection limit (2-sigma) ppbv")
plt.ylabel("Altitude (km)")
plt.scatter(found_detection_limits, found_tangent_altitudes)
plt.ylim([0, 60])


#res = retDict["YObs"][30,:] - retDict["Trans_background"][30,:]
#chosen_px = np.arange(110, 190)
#res_std = np.std(res[chosen_px])
#plt.figure()
#plt.plot(res)
#plt.plot(pixels[chosen_px], [res_std]*len(chosen_px))
#plt.plot(pixels[chosen_px], [-1.0 * res_std]*len(chosen_px))
#plt.plot(retDict["Y"][30,:] - retDict["Trans_background"][30,:])





#ppbDetected = (retDict["xa"] * retDict["xa_fact"] * 1.0e9)
#print("ppbDetected=", ppbDetected)
#
#cmap = plt.get_cmap('jet')
#colours = [cmap(i) for i in np.arange(retDict["NbZ"])/retDict["NbZ"]]
#
#fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
#for iz in range(retDict["NbZ"]):
#    ax1a.plot(retDict["nu_p"], retDict["YObs"][iz,:]/retDict["Trans_background"][iz,:], color=colours[iz], label="Obs %0.1fkm" %obsDict["alt"][iz])
#    ax1a.plot(retDict["nu_p"], retDict["Trans_p"][iz,:], color=colours[iz], linestyle='--')
#    ax1b.plot(retDict["nu_hr"], retDict["sigma_hr"][iz,:], color=colours[iz])
#
#obsDict["alt"]
#
#
#plt.figure(figsize=(FIG_X, FIG_Y))
#plt.plot(obsDict["alt"], ppbDetected)
#plt.yscale("log")


#retDict["NbZ"]





#ret_index = 0
#vmr = retDict["xa"][ret_index] * retDict["xa_fact"][ret_index] * 1.0e6 #ppm
#
#
#plt.figure(figsize=(FIG_X, FIG_Y))
#for spectrum_index in range(nSpectra):
#    plt.plot(observation_wavenumbers, y_residual_2[spectrum_index, :], color=colours[spectrum_index], label="T=%0.2f, approx ~%0.1fkm" %(yCentre[spectrum_index], obsDict["alt"][spectrum_index]))
#y_res_std = np.std(y_residual_2, axis=0)
#plt.title("%s - first look" %hdf5_filename)
#plt.ylabel("Residual after baseline and median removal")
#plt.xlabel("Wavenumbers cm-1")
#plt.plot(retDict["nu_p"], (retDict["Trans_p"][ret_index,:]-1.0)/100.0, "k--", label="CH4 simulation")
#plt.legend(loc="lower right", prop={'size': 8})


#stop()





