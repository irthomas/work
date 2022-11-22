# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:08:02 2021

@author: iant

CORRECT TRANSMITTANCE CALIBRATION TO REDUCE NOISE

PLOT DELTA BETWEEN 0.3K SPECTRA

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


def poly(y, degree, x=None, extrap=None):
    """fit n degreee polynomial to a 1d array and then make fitted spectrum
    extrapolate x to a different range of x values (optional)
    extrapolate extrap[0] values backwards and extrap[1] values forwards"""
    
    if not x:
        x = np.arange(len(y))
    coeffs = np.polyfit(x, y, degree)
    
    if extrap:
        dx = x[1] - x[0]
        x_new = np.arange(x[0] - (dx * extrap[0]), x[-1] + (dx * extrap[1]), dx)
    else:
        x_new = x
    return np.polyval(coeffs, x_new)


#read in h5 file
regex = re.compile("20180930_113957_.p.._SO_A_I_189") #(approx. orders 188-202) in steps of 8kHz
chosen_alt = 30.0
order = int(regex.pattern.split("_")[-1])

bin_top = 132
px_range = [20, 320]

calibration_alt = 110. #km
surf_alt = 0. #km

file_level_3k="hdf5_level_0p3k"
file_level="hdf5_level_1p0a"

hdf5_files_3k, hdf5_filenames_3k, hdf5_paths_3k = make_filelist(regex, file_level_3k, full_path=True)
hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)



# for hdf5_file, hdf5_filename, hdf5_path in zip(hdf5_files, hdf5_filenames, hdf5_paths):
    
#     fig, ax = plt.subplots(figsize=(10,5))
    
#     alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
#     y_all = hdf5_file["Science/Y"][:, :]
#     x_old = hdf5_file["Science/X"][0, px_range[0]:]
    
#     hdf5_file.close()

#     pixels = np.arange(px_range[0], px_range[1], 1)
#     n_spectra = y_all.shape[0]

#     #shift by 0.22cm-1 at start
#     x_shifted = x_old + 0.32

    
#     frame_index = get_nearest_index(chosen_alt, alts)
#     frame_indices = np.arange(frame_index-15, frame_index+16)
    
#     y = np.mean(y_all[frame_indices, px_range[0]:], axis=0)
#     y_cont = baseline_als(y)
#     y_cr = y / y_cont
    
#     # ax.plot(x, y_cr, label="X and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
#     # ax.plot(x_shifted, y_cr, label="X+0.22cm-1 and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
#     ax.plot(x_shifted, y_cr, label="CO spectrum at 30km")
#     ax.set_xlabel("Wavenumber cm-1")
#     ax.set_ylabel("Normalised Transmittance")
#     ax.set_title(hdf5_filename)

for hdf5_file_3k, hdf5_filename_3k, hdf5_path_3k in zip(hdf5_files_3k, hdf5_filenames_3k, hdf5_paths_3k):
    
    if "SO_A_I_" not in hdf5_filename_3k:
        print("Error: must be ingress")
    
    alts = hdf5_file_3k["Geometry/Point0/TangentAltAreoid"][:, 0]
    y_all = hdf5_file_3k["Science/Y"][:, :]
    x_old = hdf5_file_3k["Science/X"][0, px_range[0]:]
    bins = hdf5_file_3k["Science/Bins"][:, 0]
    
    # exponent = hdf5_file_3k["Channel/Exponent"][0]
    
    hdf5_file_3k.close()

    pixels = np.arange(320)
    
    #get indices and spectra for 1 bin only
    indices = np.where(bins == bin_top)[0]
    #y data for 1 bin
    y_bin = y_all[indices, :]
    alt_bin = alts[indices]

    n_spectra = y_bin.shape[0]
    n_pixels = len(pixels)

    #get mean of all spectra above TOA
    toa_index = get_nearest_index(calibration_alt, alt_bin)
    toa_indices = np.arange(toa_index-10, toa_index)
    y_toa = np.mean(y_bin[toa_indices, :], axis=0)

    #get mean of all spectra hitting surface
    surf_index = get_nearest_index(surf_alt, alt_bin)
    surf_indices = np.arange(surf_index, n_spectra)
    y_surf = np.mean(y_bin[surf_indices, :], axis=0)
    
    
    #set -999s to NaN
    alt_bin[alt_bin < -100] = np.nan


    #calculate transmittance for each spectrum including TOA and surface
    y_trans = np.zeros_like(y_bin)
    for i in range(n_spectra):
        y_trans[i, :] = (y_bin[i, :] - y_surf) / (y_toa - y_surf)
    
    #continuum removal - make spectrum divided by continuum for each spectrum
    y_cr_poly = np.zeros_like(y_bin) 
    for i in range(n_spectra):
        y_poly = np.polyval(np.polyfit(pixels, y_trans[i, :], 5), pixels)
        y_cr_poly[i, :] = y_trans[i, :]/ y_poly


    #fit a polynomial to the continuum removed spectra for each pixel, from index 20 to top of atmosphere
    #extrapolate into atmospheric region
    y_cr_poly_nlr = np.zeros_like(y_bin)
    for j in range(n_pixels):
        nl_poly = poly(y_cr_poly[20:toa_index, j], 2, extrap=[20, n_spectra-toa_index+1])
        y_cr_poly_nlr[:, j] = y_cr_poly[:, j] / nl_poly


    #plot from max altitude down to this frame index
    frame_index_to_stop = 240
    pixel_index_to_start = 50
    chosen_alts = alt_bin[:frame_index_to_stop]
        
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    fig.suptitle("%s\nY mean transmittance above atmosphere from 0.3k" %hdf5_filename_3k)
    im = ax.imshow(y_trans[:frame_index_to_stop, pixel_index_to_start:], aspect="auto", extent=(pixel_index_to_start, 320, frame_index_to_stop, 0))
    fig.colorbar(im)
    ax.set_xlabel("Pixel number")
    ax.set_ylabel("Frame index")
    ax.text(pixel_index_to_start + 10, 10, "%0.1f km" %chosen_alts[0])
    ax.text(pixel_index_to_start + 10, frame_index_to_stop-10, "%0.1f km" %chosen_alts[-1])
    fig.savefig("%s_ymean_transmittance.png" %hdf5_filename_3k)

    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    fig.suptitle("%s\nY continuum removed transmittance from 0.3k" %hdf5_filename_3k)
    im = ax.imshow(y_cr_poly[:frame_index_to_stop, pixel_index_to_start:], aspect="auto", extent=(pixel_index_to_start, 320, frame_index_to_stop, 0))
    fig.colorbar(im)
    ax.set_xlabel("Pixel number")
    ax.set_ylabel("Frame index")
    ax.text(10, 10, "%0.1f km" %chosen_alts[0])
    ax.text(10, frame_index_to_stop-10, "%0.1f km" %chosen_alts[-1])

    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    fig.suptitle("%s\nPoly fit to each pixel of the Y continuum removed transmittance, extrapolated into atmosphere" %hdf5_filename_3k)
    im = ax.imshow(y_cr_poly_nlr[:frame_index_to_stop, pixel_index_to_start:], aspect="auto", extent=(pixel_index_to_start, 320, frame_index_to_stop, 0))
    fig.colorbar(im)
    ax.set_xlabel("Pixel number")
    ax.set_ylabel("Frame index")
    ax.text(10, 10, "%0.1f km" %chosen_alts[0])
    ax.text(10, frame_index_to_stop-10, "%0.1f km" %chosen_alts[-1])




    plt.figure()
    plt.title("%s\nY mean transmittance above atmosphere from 0.3k" %hdf5_filename_3k)
    
    pixel_ixs = [65, 66, 199, 201]
    
    for pixel_ix in pixel_ixs:
        plt.plot(alt_bin, y_trans[:, pixel_ix], label="Pixel %i" %pixel_ix)
    plt.xlabel("Tangent altitude (km)")
    plt.ylabel("Transmittance")
    plt.legend()
    plt.grid()
    plt.savefig("%s_ymean_transmittance_vs_alt.png" %hdf5_filename_3k)
    

    
    plt.figure()
    plt.title("Y continuum removed transmittance vs altitude")
    plt.plot(alt_bin, y_cr_poly[:, 65])
    plt.plot(alt_bin, y_cr_poly[:, 66])
    plt.plot(alt_bin, y_cr_poly[:, 199])
    plt.plot(alt_bin, y_cr_poly[:, 201])

    plt.plot(poly(y_cr_poly[:toa_index, 65], 2, extrap=[20, n_spectra-toa_index+1]))
    plt.plot(poly(y_cr_poly[:toa_index, 66], 2, extrap=[20, n_spectra-toa_index+1]))

    plt.figure()
    plt.plot(y_cr_poly_nlr[:, 65])
    plt.plot(y_cr_poly_nlr[:, 66])
    plt.plot(y_cr_poly_nlr[:, 199])

    
    plt.figure()
    plt.plot(y_cr_poly_nlr[toa_index:, 65])
    plt.plot(y_cr_poly_nlr[toa_index:, 66])
    plt.plot(y_cr_poly_nlr[toa_index:, 199])

    # plt.plot(y_bin[:, 65], y_cr_poly[:, 65])
    # plt.plot(y_bin[:, 66], y_cr_poly[:, 66])
    


    # bins_int = np.arange(min())
    # ind = np.digitize(all_points["A_nu0"], bins_int)
    # bins = bins_int + (bins_int[1] - bins_int[0])
    # F_aotf_binned = np.array([np.mean(all_points["F_aotf"][ind == j]) for j in range(0, len(bins_int))])



    # plt.plot(y_cr_poly[:, 310])
    
    # plt.figure()
    # plt.plot(np.mean(y_bin[:, 160:240], axis=1))
    # plt.plot(y_bin[:, 195])
    # plt.plot(y_bin[:, 199])

    # #shift by 0.22cm-1 at start
    # x_shifted = x_old + 0.32

    
    # frame_index = get_nearest_index(chosen_alt, alts)
    # frame_indices = np.arange(frame_index-15, frame_index+16)
    
    # y = np.mean(y_all[frame_indices, px_range[0]:], axis=0)
    # y_cont = baseline_als(y)
    # y_cr = y / y_cont
    
    # # ax.plot(x, y_cr, label="X and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
    # # ax.plot(x_shifted, y_cr, label="X+0.22cm-1 and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
    # ax.plot(x_shifted, y_cr, label="CO spectrum at 30km")
    # ax.set_xlabel("Wavenumber cm-1")
    # ax.set_ylabel("Normalised Transmittance")
    # ax.set_title(hdf5_filename_3k)
