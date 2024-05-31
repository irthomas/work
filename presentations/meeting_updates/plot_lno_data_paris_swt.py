# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:04:02 2020

@author: iant

PLOT LNO ABSORPTION DEPTHS

"""

import numpy as np
import os
import re
#from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from PIL import Image


from tools.file.hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, getFile, makeFileList
from tools.file.database_functions_v01 import obsDB, makeObsDict

fileLevel = "hdf5_level_1p0a"
regex = re.compile("20(18|19|20)[0-1][0-9][0-9][0-9].*_LNO_.*_189")
#regex = re.compile("2018042[0-9].*_LNO_.*_189")
#regex = re.compile("201[89][0-1][0-9][0-9][0-9]_.*_LNO_.*")



def runningMean(detector_data, n_spectra_to_mean):
    """make a running mean of n data points. detector data output has same length as input"""
    
    plus_minus = int((n_spectra_to_mean - 1) / 2)
    
    nSpectra = detector_data.shape[0]
    runningIndices = np.asarray([np.arange(runningIndexCentre-plus_minus, runningIndexCentre+(plus_minus+1), 1) for runningIndexCentre in np.arange(nSpectra)])
    runningIndices[runningIndices<0] = 0
    runningIndices[runningIndices>nSpectra-1] = nSpectra-1

    running_mean_data = np.zeros_like(detector_data)
    for rowIndex,indices in enumerate(runningIndices):
        running_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return running_mean_data


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



def fit_gaussian_absorption(x_in, y_in):
    def gaussian(x, a, b, c, d):
        return 1.0 - a * np.exp(-((x - b)/c)**2.0) + d
    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    def resfunc(params):
        return y_in-gaussian(x_centred,params[0],params[1],params[2],params[3])
    popt = least_squares(resfunc, [0.05, -0.1, 0.35, 0.0])

    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
    y_hr = gaussian(x_hr - x_mean, *popt.x)
    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    x_min_position = x_hr[min_index]
    y_depth = 1.0 - y_hr[min_index]
    #find simple chisq error
    chi_sq_fit = np.sum(popt.fun **2)
    return x_hr, y_hr, x_min_position, y_depth, chi_sq_fit


#def fit_gaussian_absorption(x_in, y_in):
#    def gaussian(x, a, c):
#        return 1.0 - a * np.exp(-(x/c)**2.0)
#    x_mean = np.mean(x_in)
#    x_centred = x_in - x_mean
#    def resfunc(params):
#        return y_in-gaussian(x_centred,params[0],params[1])
#    popt = least_squares(resfunc, [0.08, 0.25])
#
#    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
#    y_hr = gaussian(x_hr - x_mean, *popt.x)
#    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
#    x_min_position = x_hr[min_index]
#    y_depth = y_hr[0] - y_hr[min_index]
#    #find simple chisq error
#    chi_sq_fit = np.sum(popt.fun **2)
#    
#    return x_hr, y_hr, x_min_position, y_depth, chi_sq_fit



def fft_zerofilling(row, filling_amount):
    """apply fft, zero fill by a multiplier then reverse fft to give very high resolution spectrum"""
    n_pixels = len(row)
    
    rowrfft = np.fft.rfft(row, len(row))
    rowzeros = np.zeros(n_pixels * int(filling_amount), dtype=np.complex)
    rowfft = np.concatenate((rowrfft, rowzeros))
    row_hr = np.fft.irfft(rowfft).real #get real component for reversed fft
    row_hr *= len(row_hr)/len(row) #need to scale by number of extra points

    pixels_hr = np.linspace(0, n_pixels, num=len(row_hr))    
    return pixels_hr, row_hr


def fftHrSpectrum(wavenumbers, spectrum, zerofilling=10):
    
    px_hr, abs_hr = fft_zerofilling(spectrum, zerofilling)
    wavenumbers_hr = np.interp(px_hr, np.arange(len(spectrum)), wavenumbers)
    return wavenumbers_hr, abs_hr




N_SPECTRA_TO_MEAN = 5
MINIMUM_INCIDENCE_ANGLE = 5.0
PLOT_TYPE = [1]


##read in TES file
#im = Image.open(os.path.join(BASE_DIRECTORY,"reference_files","Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"))
#albedoMap = np.array(im)

if 0 in PLOT_TYPE:
    fig1, ax1 = plt.subplots(1, figsize=(FIG_X+8, FIG_Y+2))
#albedoPlot = plt.imshow(albedoMap, extent = [-180,180,-90,90])
#cbar = plt.colorbar(albedoPlot)
#colorbarLabel = "TES Albedo"
#cbar.set_label(colorbarLabel, rotation=270, labelpad=20)



"""read in each LNO file, plot radiance factor"""
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)

for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    print("%i/%i: Reading in file %s" %(fileIndex, len(hdf5Filenames), hdf5_filename))
    #get Y data
    yIn = hdf5_file["Science/Y"][...]
    xIn = hdf5_file["Science/X"][0, :]
    lonIn = hdf5_file["Geometry/Point0/Lon"][...]
    latIn = hdf5_file["Geometry/Point0/Lat"][...]
    lsIn = hdf5_file["Geometry/LSubS"][0, 0]
    
    incidenceAngleIn = hdf5_file["Geometry/Point0/IncidenceAngle"][:, 0]
    minIncidenceAngle = np.min(incidenceAngleIn)

    if minIncidenceAngle < MINIMUM_INCIDENCE_ANGLE:
    
        yIn[np.isnan(yIn)] = 0.0 #replace nans
        yInMean = np.nanmean(yIn[:, 160:240], axis=1)
        
        yRunningMean = runningMean(yIn, N_SPECTRA_TO_MEAN)
        
        lonMean = np.mean(lonIn, axis=1)
        latMean = np.mean(latIn, axis=1)

        if 1 in PLOT_TYPE:
            fig2, ax2 = plt.subplots(1, figsize=(FIG_X+8, FIG_Y+2))

       
        for spectrumIndex, spectrum in enumerate(yRunningMean):
            spectrum_baseline = baseline_als(spectrum) #find continuum of mean spectrum
            corrected_spectrum = spectrum / spectrum_baseline

                
            if incidenceAngleIn[spectrumIndex] < MINIMUM_INCIDENCE_ANGLE:
                smoothed_spectrum = savgol_filter(corrected_spectrum, 9, 2)
                x_hr, y_hr = fftHrSpectrum(xIn, smoothed_spectrum)

                nu_centres = [4267.542, 4271.178, 4274.741]
                y_depth_all = []
                for abs_nu_centre in nu_centres:
                    abs_nu_start = abs_nu_centre - 0.5
                    abs_nu_end = abs_nu_centre + 0.5
                    abs_pixels = np.where((abs_nu_start < xIn) & (xIn < abs_nu_end))[0]
    
            
    
                    abs_pixels_hr = np.where((abs_nu_start < x_hr) & (x_hr < abs_nu_end))[0]
                    y_depth = 1.0 - np.min(y_hr[abs_pixels_hr])
                    y_depth_all.append(y_depth)
        
#                    y_abs = smoothed_spectrum[abs_pixels]
#                    x_hr, y_hr, x_min_position, y_depth, chi_sq_fit = fit_gaussian_absorption(x_abs, y_abs)
#                    print(y_depth)
    
                    if 1 in PLOT_TYPE:
                        p = ax2.plot(xIn[abs_pixels], smoothed_spectrum[abs_pixels])
                        ax2.plot(x_hr, y_hr, color=p[0].get_color(), alpha=0.1)
#                        plt.axhline(y=y_depth, color=p[0].get_color())
                    if 0 in PLOT_TYPE:
                        sc = ax1.scatter(lonMean[spectrumIndex], latMean[spectrumIndex], c=np.mean(y_depth), vmin=0.0, vmax=0.2, cmap=plt.cm.jet, marker='o', linewidth=0)
                        sc = ax1.scatter(lsIn, latMean[spectrumIndex], c=y_depth, vmin=0.0, vmax=0.2, cmap=plt.cm.jet, marker='o', linewidth=0)

        if 1 in PLOT_TYPE:
            ax2.set_xlabel("Wavenumber cm-1")
            ax2.set_ylabel("Normalised radiance factor")
            ax2.set_title("CO spectra order 189 absorption band fit")
            fig2.savefig(os.path.join(BASE_DIRECTORY, "lno_CO_band_fitting_example.png"))
            stop()


if 0 in PLOT_TYPE:
    cbar = fig1.colorbar(sc)
    colorbarLabel = "CO band depth"
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
    
    ax1.set_xlabel("Ls")
    ax1.set_ylabel("Latitude")
    ax1.set_title("CO band depth order 189")
    fig1.savefig(os.path.join(BASE_DIRECTORY, "lno_CO_band_depth.png"), dpi=300)
    
