# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:58:12 2021

@author: iant

CORRECT SPECTRAL CALIBRATION

"""



import numpy as np
import h5py
# import os
# import sys
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
# from scipy.signal import savgol_filter
# import scipy.signal as ss
# import lmfit

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist
# from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict
# from tools.file.paths import paths

# from tools.sql.get_sql_spectrum_temperature import get_sql_temperatures_all_spectra

from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.general.get_minima_maxima import get_local_minima
from tools.general.get_nearest_index import get_nearest_index
from tools.file.read_write_hdf5 import read_hdf5_to_dict, write_hdf5_from_dict


spectral_lines_dict = {
    189:[4248.318, 4252.302, 4256.217, 4263.837, 4267.542, 4271.177, 4274.741, 4278.235]}



#read in h5 file
regex = re.compile("20180930_113957_1p0a_SO_A_I_189") #(approx. orders 188-202) in steps of 8kHz
chosen_alt = 30.0
order = int(regex.pattern.split("_")[-1])

file_level="hdf5_level_1p0a"

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)


for hdf5_file, hdf5_filename, hdf5_path in zip(hdf5_files, hdf5_filenames, hdf5_paths):
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
    y_all = hdf5_file["Science/Y"][:, :]
    x = hdf5_file["Science/X"][0, 50:]
    hdf5_file.close()

    pixels = np.arange(50, 320, 1)
    n_spectra = y_all.shape[0]


    
    frame_index = get_nearest_index(chosen_alt, alts)
    frame_indices = np.arange(frame_index-5, frame_index+6)
    
    y = np.mean(y_all[frame_indices, 50:], axis=0)
    y_cont = baseline_als(y)
    y_cr = y / y_cont
    
    ax.plot(x, y_cr, label="X and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
    ax.set_xlabel("Wavenumber cm-1")
    ax.set_ylabel("Transmittance")
    ax.set_title(hdf5_filename)
    
    spectral_lines_nu = spectral_lines_dict[order]
    for spectral_line_nu in spectral_lines_nu:
        ax.axvline(spectral_line_nu, c="k", linestyle="--")
    
    absorption_points = np.where(y_cr < 0.985)[0]
    all_local_minima = get_local_minima(y_cr)
    
    local_minima = [i for i in all_local_minima if i in absorption_points]
    # ax.scatter(x[local_minima], y_cr[local_minima], c="k", marker="+", label="Absorption Minima")
    
    delta_nus = []
    pixel_minima = []
    nu_minima = []
    
    for i, local_minimum in enumerate(local_minima):
        
        local_minimum_indices = np.arange(local_minimum-2, local_minimum+3, 1)
        
        x_hr, y_hr, x_min_position, chisq = fit_gaussian_absorption(x[local_minimum_indices], y_cr[local_minimum_indices], error=True)
        
        if i == 0:
            label = "Gaussian fit to absorption bands"
        else:
            label = ""
        ax.plot(x_hr, y_hr, "r", label=label)

        closest_spectral_line = get_nearest_index(x_min_position, spectral_lines_nu)
        delta_nu = x_min_position - spectral_lines_nu[closest_spectral_line]
        if np.abs(delta_nu) > 0.5:
            print("Ignoring %0.1f line, too far from expected value" %x_min_position)
        else:
            print(delta_nu)
            delta_nus.append(delta_nu)
            #find pixel of minimum
            x_hr, y_hr, x_min_position, chisq = fit_gaussian_absorption(pixels[local_minimum_indices], y_cr[local_minimum_indices], error=True)
            pixel_minima.append(x_min_position)
            nu_minima.append(spectral_lines_nu[closest_spectral_line])
            
    #remove bad points
    delta_min_max = [np.mean(delta_nus) - np.std(delta_nus)*1.5, np.mean(delta_nus) + np.std(delta_nus)*1.5]
    valid_indices = [i for i,v in enumerate(delta_nus) if v>delta_min_max[0] and v<delta_min_max[1]]
    
    valid_pixel_minima = [v for i,v in enumerate(pixel_minima) if i in valid_indices]
    valid_nu_minima = [v for i,v in enumerate(nu_minima) if i in valid_indices]
    
    
    # nu_shift = np.mean([v for i,v in enumerate(delta_nus) if i in valid_indices])

    # x_new = x - nu_shift
    
    # ax.plot(x_hr - nu_shift, y_hr, "g")
    
    
    polyfit = np.polyfit(valid_pixel_minima, valid_nu_minima, 3)
        
    x_new = np.polyval(polyfit, pixels)
    x_new_all_px = np.polyval(polyfit, np.arange(320))
    # ax.plot(x_new, y_cr, "g")

    #write spectra to new file
    replace_datasets = {"Science/X":np.tile(x_new_all_px, (n_spectra,1))}
    replace_attributes = {}
    hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(hdf5_path[:-3])
    write_hdf5_from_dict(hdf5_filename+"_corr", hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes)

    ax.legend(loc="lower right")
   
    fig.savefig("%s_uncorrected_x.png" %hdf5_filename)
 
    with h5py.File(hdf5_filename+"_corr.h5", "r") as f:
    
        x = f["Science/X"][0, :]
        y_all = f["Science/Y"][:, :]
        y = np.mean(y_all[frame_indices, :], axis=0)
        y_cont = baseline_als(y)
        y_cr = y / y_cont
    
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(x, y_cr, label="X and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
    ax2.set_xlabel("Wavenumber cm-1")
    ax2.set_ylabel("Transmittance")
    ax2.set_title(hdf5_filename+"_corr")
    
    spectral_lines_nu = spectral_lines_dict[order]
    for spectral_line_nu in spectral_lines_nu:
        ax2.axvline(spectral_line_nu, c="k", linestyle="--")
    fig2.savefig("%s_corrected_x.png" %hdf5_filename)
