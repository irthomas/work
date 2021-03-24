# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:46:00 2021

@author: iant

PLOT DETECTOR SMILE
"""

import numpy as np
import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.baseline_als import baseline_als

file_level = "hdf5_level_0p1a"
# regex = re.compile("(20160615_224950|20180428_023343|20180511_084630|20180522_221149|20180821_193241|20180828_223824)_0p1a_SO_1")
# regex = re.compile("20160615_224950_0p1a_SO_1") #best absorption line avoiding bad pixels

regex = re.compile("(20180619_020651|20190704_121530|20200724_125331|20200728_144718)_0p1a_LNO_1")
# regex = re.compile("20160615_233950_0p1a_LNO_1")


SAVE_FIGS = True



def findAbsorptionMinimum(spectrum, continuum_range, plot=False):
    
    continuum_centre = int((continuum_range[3] - continuum_range[0])/2)
    ABSORPTION_WIDTH_INDICES = list(range(continuum_centre - 3, continuum_centre + 3, 1))
        
    pixels = np.arange(320)
    
    continuum_pixels = pixels[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]    
    continuum_spectra = spectrum[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]
    
    #fit polynomial to continuum on either side of absorption band
    coefficients = np.polyfit(continuum_pixels, continuum_spectra, 2)
    continuum = np.polyval(coefficients, pixels[range(continuum_range[0], continuum_range[3])])
    #divide by continuum to get absorption
    absorption = spectrum[range(continuum_range[0], continuum_range[3])] / continuum
    
    #fit polynomial to centre of absorption
    abs_coefficients = np.polyfit(pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES], absorption[ABSORPTION_WIDTH_INDICES], 2)
#    detector_row = illuminated_window_tops[frame_index]+bin_index*binning
    
    if plot:
        plt.plot(pixels[list(range(continuum_range[0], continuum_range[3]))], absorption)
        fitted_absorption = np.polyval(abs_coefficients, pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES])
        plt.plot(pixels[list(range(continuum_range[0],continuum_range[3]))][ABSORPTION_WIDTH_INDICES], fitted_absorption)
    
    absorption_minima = (-1.0 * abs_coefficients[1]) / (2.0 * abs_coefficients[0])
    
    return absorption_minima


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
    channel = hdf5_filename.split("_")[3].lower()
    
    
    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    n_rows = detector_data_all.shape[1]
    frame_indices = range(len(window_top_all))

    dim = detector_data_all.shape
    nu = len(list(set(window_top_all))) #number of unique window tops
    nff = int(np.floor(dim[0]/nu)) #number of full frames
    
    #make list of detector rows in each frame
    row_numbers_all = []
    for window_top in window_top_all:
        row_numbers = np.arange(window_top, window_top + n_rows, binning)
        row_numbers_all.append(row_numbers)


    detector_data_reshaped = np.zeros((int(dim[0]/nu), int(dim[1]*nu), int(dim[2])))

    for u in range(nu):
        detector_data_reshaped[:, u*dim[1]:(u+1)*dim[1], :] = detector_data_all[range(u, nff*nu, nu), :, :]

    row_numbers = np.arange(window_top_all[0], max(window_top_all)+dim[1])

    if channel == "so":
        continuum_range = [109,113,117,122]
        signal_minimum = 200000
        centre_line = 128
        
        frame_ranges = {
            "20160615_224950_0p1a_SO_1":[50,75],
            "20180821_193241_0p1a_SO_1":[420, 700]
            }
        


    if channel == "lno":
        # continuum_range = [203,209,217,223]
        continuum_range = [65,68,74,76]
        signal_minimum = 300000
        centre_line = 152
 
        frame_ranges = {
            "20160615_233950_0p1a_LNO_1":[100,160],
            # "20180821_193241_0p1a_SO_1":[420, 700]
            }
        
    
    """make animation"""
    from tools.plotting.anim import make_frame_anim
    make_frame_anim(detector_data_reshaped, 350000, 0, hdf5_filename, ymax=detector_data_reshaped.shape[1])
    """search for best region"""
    plt.figure(figsize = (FIG_X, FIG_Y))
    plt.plot(detector_data_reshaped[:, int((centre_line-10)/binning), 180].T)
    plt.plot(detector_data_reshaped[:, int(centre_line/binning), 180].T)
    plt.plot(detector_data_reshaped[:, int((centre_line+10)/binning), 180].T)


    frame_range = frame_ranges[regex.pattern]

    detector_data_selected = detector_data_reshaped[frame_range[0]:frame_range[1], : , :]
    
    
    
    absorption_minima = []
    detector_rows = []
    indices = []
    spectra_cont_removed = []
    i = 0
    for frame in detector_data_selected:
        for frame_index, spectrum in enumerate(frame):
            if spectrum[180] > signal_minimum:
                i += 1
                spectrum_cont = baseline_als(spectrum)
                spectra_cont_removed.append(spectrum / spectrum_cont)
                
                absorption_minimum = findAbsorptionMinimum(spectrum / spectrum_cont, continuum_range)
                absorption_minima.append(absorption_minimum )
                detector_rows.append(frame_index)
                indices.append(i)
                
    ill_rows = sorted(list(set(detector_rows)))
    
    spectra_cont_removed = np.asfarray(spectra_cont_removed)
    
    plt.figure(figsize = (FIG_X - 3.5, FIG_Y))
    # plt.scatter(absorption_minima, detector_rows, marker="o", c=indices, linewidth=0, alpha=0.5)
    plt.scatter(absorption_minima, detector_rows, marker="o", c="C0", linewidth=0, alpha=0.8)

    fit_coefficients = np.polyfit(detector_rows, absorption_minima, 1)
    fit_line = np.polyval(fit_coefficients, detector_rows)

    plt.plot(fit_line, detector_rows, "k", label="Line of best fit, min=%0.1f, max=%0.1f" %(np.min(fit_line), np.max(fit_line)))
    plt.xlabel("Spectral pixel number at centre of absorption band")
    plt.ylabel("Vertial detector row")
    plt.title(channel.upper()+" Detector slant")
    plt.legend()
    plt.savefig(channel+"_detector_slant.png")
    
    
    plt.figure(figsize = (FIG_X, FIG_Y))
    cmap = plt.get_cmap("Spectral")
    colours = [cmap(i) for i in np.arange(len(ill_rows))/len(ill_rows)]
    
    rows_measured = []
    for i, spectrum in enumerate(spectra_cont_removed):
        if detector_rows[i] in rows_measured:
            plt.plot(spectrum, color=colours[detector_rows[i]-ill_rows[0]])
        else:
            plt.plot(spectrum, color=colours[detector_rows[i]-ill_rows[0]], label="Detector row %i" %detector_rows[i])
            rows_measured.append(detector_rows[i])
        
    plt.xlabel("Spectral pixel number at centre of absorption band")
    plt.ylabel("Vertial detector row")
    plt.title(channel.upper()+" Detector slant")
    plt.legend()

