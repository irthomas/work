# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:03:52 2021

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


file_level = "hdf5_level_0p1a"
regex = re.compile("20160615_224950_0p1a_SO_1")
# regex = re.compile("20160615_233950_0p1a_LNO_1")



"""make vertical detector plots where sun is seen to determine slit position and time when in centre"""
DETECTOR_V_CENTRE = 201
DETECTOR_CUTOFF = 20000
SAVE_FIGS = True


def splitWindowSteppingByWindow(hdf5_file):
    
    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    
    unique_window_tops = sorted(list(set(window_top_all)))
    window_rows = unique_window_tops[1] - unique_window_tops[0]
    
    window_data = []
    detector_row_numbers = []
    for unique_window_top in unique_window_tops:
        window_frame = detector_data_all[window_top_all == unique_window_top, :, :]
        window_data.append(window_frame)
        
        detector_row_numbers.append(list(range(unique_window_top, unique_window_top + window_rows, binning)))
        
    return window_data, unique_window_tops, detector_row_numbers




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

    cmap = plt.get_cmap('Set1')
    window_data, unique_window_tops, detector_row_numbers = splitWindowSteppingByWindow(hdf5_file)
    nColours = 0
    illuminated_window_tops = []
    for window_frames, unique_window_top in zip(window_data, unique_window_tops):
#            print(np.max(window_frames[:, :, DETECTOR_V_CENTRE]))
        if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
            nColours += 1
            illuminated_window_tops.append(unique_window_top)
    colours = [cmap(i) for i in np.arange(nColours)/nColours]


    row_max_value = []
    row_number = []
    for window_index, (window_frames, detector_row_number) in enumerate(zip(window_data, detector_row_numbers)):
        if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
            vertical_slices = window_frames[:, :, DETECTOR_V_CENTRE]

            row_number.extend(detector_row_number)
            row_max_value.extend(np.max(vertical_slices, axis=0))


    plt.figure(figsize=(FIG_X - 4, FIG_Y + 2))
    plt.xlabel("Relative instrument sensitivity for pixel number %i" %DETECTOR_V_CENTRE)
    plt.ylabel("Detector row number")
    plt.title(channel.upper()+" MCC line scan: sun detector illumination")
    plt.tight_layout()
    plt.grid(True)
    #normalise

    colour_index = -1
    for window_index, (window_frames, detector_row_number) in enumerate(zip(window_data, detector_row_numbers)):
        if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
            vertical_slices = window_frames[:, :, DETECTOR_V_CENTRE]

            colour_index += 1
            
#                plt.scatter(detector_row_number, np.max(vertical_slices, axis=0), color=colours[colour_index])
            for slice_index, vertical_slice in enumerate(vertical_slices):
                if slice_index == 0:
                    plt.plot(vertical_slice/np.max(row_max_value), detector_row_number, color=colours[colour_index], label="Vertical rows %i-%i" %(np.min(detector_row_number), np.max(detector_row_number)))
                else:
                    plt.plot(vertical_slice/np.max(row_max_value), detector_row_number, color=colours[colour_index])
    
    row_numbers = np.arange(np.min(row_number), np.max(row_number))
    row_max_values = np.interp(row_numbers, row_number, row_max_value)
    
    smooth = savgol_filter(row_max_values[row_max_values > DETECTOR_CUTOFF], 9, 5)
    smooth_rows = row_numbers[row_max_values > DETECTOR_CUTOFF]
#        plt.scatter(row_numbers, row_max_values)
    plt.plot(smooth/np.max(row_max_value), smooth_rows, linewidth=5, color="k", label="Instrument sensitivity")
    plt.legend(loc="upper right")
    
    if SAVE_FIGS: 
        plt.savefig(channel+"_MCC_line_scan_vertical_columns_on_detector_where_sun_is_seen.png")


#        """check detector smile"""
        
    if channel == "so":
        continuum_range = [210,215,223,228]
        signal_minimum = 200000
        bad_pixel_rows = [114]
        binning=1


    if channel == "lno":
        continuum_range = [203,209,217,223]
        signal_minimum = 300000
        bad_pixel_rows = [106, 108, 112, 218]
        binning=2

    
    #first loop through each window frame, finding minima of chosen absorption (if frame is illuminated)
    absorption_minima=[]
    detector_rows=[]
    for window_frames, unique_window_top, detector_row_list in zip(window_data, unique_window_tops, detector_row_numbers):
        if unique_window_top in illuminated_window_tops:
            for window_frame in window_frames:
                for spectrum, detector_row_number in zip(window_frame, detector_row_list):
                    if spectrum[200] > signal_minimum:
                        if detector_row_number not in bad_pixel_rows:
                            detector_rows.append(detector_row_number)
                            absorption_minimum = findAbsorptionMinimum(spectrum, continuum_range)
                            absorption_minima.append(absorption_minimum)
                            
    colour = np.arange(len(absorption_minima))
                        
    
    plt.figure(figsize = (FIG_X - 4, FIG_Y + 2))
    plt.scatter(absorption_minima, detector_rows, marker="o", c=colour, linewidth=0, alpha=0.5)

    fit_coefficients = np.polyfit(detector_rows,absorption_minima,1)
    fit_line = np.polyval(fit_coefficients,detector_rows)

    plt.plot(fit_line, detector_rows, "k", label="Line of best fit, min=%0.1f, max=%0.1f" %(np.min(fit_line), np.max(fit_line)))
    plt.legend()
    plt.xlabel("Peak absorption pixel number, determined from quadratic fit")
    plt.ylabel("Detector row")
    plt.title(channel.upper()+" MCC line scan Detector smile: Quadratic fits to absorption line")
    plt.tight_layout()
    plt.grid(True)
    if SAVE_FIGS: 
        plt.savefig(channel+"_MCC_line_scan_detector_smile.png")


