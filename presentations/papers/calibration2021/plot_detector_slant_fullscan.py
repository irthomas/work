# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:47:22 2021

@author: iant

plot detector slant v2
"""

import numpy as np
# import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import scipy.signal as ss

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist

from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.spectra.fit_polynomial import fit_polynomial
from tools.plotting.colours import get_colours

# 4 atmospheric slow fullscans
# 20191110_094349_0p1a_SO_1.h5
# 20191114_040633_0p1a_SO_1.h5
# 20191205_071132_0p1a_SO_1.h5
# 20200101_204247_0p1a_SO_1.h5


file_level = "hdf5_level_0p1a"
detector_rows = range(128-8, 128+8)

regex = re.compile("20191110_094349_.*_SO_.") #SO fullscan slow
good_indices_range = range(298)
temperature_offset = 0
absorption_indices = {
    # 142:np.arange(227,234),
    # # 156:np.arange(165,171),
    # # 168:np.arange(129,136),
    # 174:np.arange(242,249),
    # 178:np.arange(276,283),
    194:np.arange(214,221),
    # # 196:np.arange(135,141),
    # 197:np.arange(193,198),
    }

# regex = re.compile("20191114_040633_.*_SO_.") #SO fullscan slow
# good_indices_range = range(250)
# temperature_offset = 2
# absorption_indices = {
#     142:np.arange(228,234)+temperature_offset,
#     # 156:np.arange(165,171)+temperature_offset,
#     # 168:np.arange(129,136)+temperature_offset,
#     174:np.arange(242,249)+temperature_offset,
#     178:np.arange(276,283)+temperature_offset,
#     194:np.arange(214,221)+temperature_offset,
#     # 196:np.arange(135,141)+temperature_offset,
#     197:np.arange(193,198)+temperature_offset,
#     }



# for i in range(100,230):
#     if i not in absorption_indices.keys():
#         absorption_indices[i] = np.arange(227,234)


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    


    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1

    dim = detector_data_all.shape
    n_rows_raw = dim[1] #data rows
    n_rows_binned = dim[1] * binning #pixel detector rows
    frame_indices = np.arange(dim[0])
    n_u = len(list(set(window_top_all))) #number of unique window tops
    n_ff = int(np.floor(dim[0]/n_u)) #number of full frames
    
    colours = get_colours(dim[1])
    
    #if not window stepping
    row_no = np.arange(window_top_all[0], window_top_all[0]+n_rows_binned, binning)
    
    y_mean = np.mean(detector_data_all[:, 12, 160:240], axis=1)


    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])

    minima = {}
    
    for chosen_order in absorption_indices.keys():
        
        good_indices = [i for i, order in enumerate(orders) if order == chosen_order and i in good_indices_range]
        colours2 = get_colours(len(good_indices), cmap="brg")
    
        for index,i in enumerate(good_indices): #loop through good frames in file
    
        # if orders[i] in absorption_indices.keys():
    
            # plt.figure(figsize = (FIG_X, FIG_Y))
            fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize = (FIG_X+2, FIG_Y))
            ax2.set_title("Detector Slant, Inflight Fullscan Order %i" %(orders[i]))
            ax1.set_xlabel("Pixel")
            ax1.set_ylabel("Baseline removed transmittance")

            y_mean = np.mean(detector_data_all[i, 12, 160:240])
            # y_std = np.std(detector_data_all[i, 12, 160:240])
            
            minima[i] = {"row_no":[], "min_pixel":[], "chisq":[], "colour":colours2[index]}
            
            
            for j in range(n_rows_raw): #loop through detector rows
                
                if np.mean(detector_data_all[i, j, 160:240]) > 0.3*y_mean:
                    spectrum = detector_data_all[i, j, :]
                
                    continuum = baseline_als(spectrum)
                    absorption = spectrum/continuum
                    
                    ax1.plot(absorption, color=colours[j], label="i=%i, row=%i" %(i,row_no[j]))
                    
                    # sav_gol = savgol_filter(spectrum, 9, 2)
                    # plt.plot(sav_gol, color=colours[j], linestyle=":", label="i=%i, row=%i" %(i,row_no[j]))
                    
                    # oversampled = ss.resample(spectrum, 640)
                    # plt.plot(np.arange(0, 320, 0.5), oversampled, color=colours[j], linestyle="--", label="i=%i, row=%i" %(i,row_no[j]))
                    
                    rel_indices = absorption_indices[orders[i]] - np.mean(absorption_indices[orders[i]])
                    gaussian = fit_gaussian_absorption(rel_indices, absorption[absorption_indices[orders[i]]], error=True)
                    # print("i=%i, row=%i" %(i,row_no[j]), gaussian[3])
                    if len(gaussian[0])>0: #if not error
                        if gaussian[3] < 0.003:
                            ax1.plot(absorption_indices[orders[i]][0]-rel_indices[0]+gaussian[0], gaussian[1], color=colours[j], linestyle="--", label="i=%i, row=%i" %(i,row_no[j]))
                            minima[i]["row_no"].append(row_no[j])
                            minima[i]["min_pixel"].append(absorption_indices[orders[i]][0]-rel_indices[0]+gaussian[2])
                            minima[i]["chisq"].append(gaussian[3])
            minima[i]["row_no"] = np.asarray(minima[i]["row_no"])
            minima[i]["min_pixel"] = np.asarray(minima[i]["min_pixel"])
            minima[i]["chisq"] = np.asarray(minima[i]["chisq"])
                            
            ax1.set_ylim((0.8, 1.1))

        # plt.figure()
        # plt.title(hdf5_filename)
        ax2.set_xlabel("Pixel of absorption minimum")
        ax2.set_ylabel("Detector row")
        for i in minima.keys():
            if i in good_indices:
                if len(minima[i]["min_pixel"])>0:
                    
                    x_offset = 0.0#np.mean(minima[i]["min_pixel"])
                    
                    ax2.scatter(minima[i]["min_pixel"]-x_offset, minima[i]["row_no"], color=minima[i]["colour"], label="Frame %i (order %i)" %(i, orders[i]))
                    detector_row_indices = np.asarray([i for i, row in enumerate(minima[i]["row_no"]) if row in detector_rows])
                    linear_coeffs = fit_polynomial(minima[i]["min_pixel"][detector_row_indices], minima[i]["row_no"][detector_row_indices], coeffs=True)[1]
                    y_range = [detector_rows[0], detector_rows[-1]+1]
                    x_range = [(y_range[0]- linear_coeffs[1])/linear_coeffs[0], (y_range[-1]- linear_coeffs[1])/linear_coeffs[0]]
                    
                    
                    ax2.plot(x_range, y_range, color=minima[i]["colour"])
        ax2.axhline(y=128-8, color="k", linestyle="--")
        ax2.axhline(y=128+8, color="k", linestyle="--")
        ax2.legend()


