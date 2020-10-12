# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:10:51 2020

@author: iant

MAKE HCL CORRECTION AUXILIARY FILE
"""


# import os
import matplotlib.pyplot as plt
import numpy as np
import re
# import sys
# import h5py

from tools.file.hdf5_functions import make_filelist
from tools.file.save_dict_to_hdf5 import save_dict_to_hdf5

# from tools.file.hdf5_functions_v04 import getFile, makeFileList
from tools.file.get_hdf5_data_v01 import getLevel1Data
# from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_polynomial import fit_polynomial
# from tools.general.get_nearest_index import get_nearest_index
from tools.plotting.colours import get_colours
# from instrument.nomad_so_instrument import nu_mp



# SAVE_FILES = True
SAVE_FILES = False



#select obs for deriving correction
#be careful of which detector rows are used. Nom pointing after 2018 Aug 11

# diffraction_order = 129
diffraction_order = 130

chosen_window_top = 120

pixel_index = 200


if diffraction_order == 129:
# regex = re.compile("20180(615|804|813|816|818|820|823|827)_.*_SO_A_I_129") #bad
    regex = re.compile("20180(813|816|818|820|823|827)_.*_SO_A_I_129") #row120 used 
    toa_alt = 100.0

elif diffraction_order == 130:
    regex = re.compile("20180501_15.*_SO_A_[IE]_130")
    # regex = re.compile("20(180828|180830|180901|181125|181201|181207|190203|190311|190504|191211)_.*_SO_A_[IE]_130") #row120 used 
    toa_alt = 80.0


file_level = "hdf5_level_1p0a"
hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True)



fig1, ax1 = plt.subplots()
colours = get_colours(len(hdf5_filenames))


pixels = np.arange(320.0)
bin_indices = list(range(4))
window_tops = []

#calibrate transmittances for all 4 bins
correction_dict = {}
for bin_index in bin_indices:
    spectra_in_bin = [] #get all spectra 
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
        
        
        #use mean method, returns dictionary
        obs_dict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=toa_alt)
        
        window_tops_in_file = list(hdf5_file["Channel/WindowTop"][:])
        window_tops.extend(window_tops_in_file)
        
        good_indices = np.where((obs_dict["y_mean"][:, pixel_index] > 0.1) & (obs_dict["y_mean"][:, pixel_index] < 0.9))[0]
        
        for spectrum_index in good_indices:
            spectra_in_bin.append(obs_dict["y_mean"][spectrum_index, :])
            
        
        #check lowest altitude where transmittance > 0.1
        altitudes_with_signal = obs_dict["alt"][np.where(obs_dict["y_mean"][:, pixel_index] > 0.1)[0]]
        file_good = True

        if np.min(altitudes_with_signal) > 30.0:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has no signal below 30km")
        else:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has signal below 30km")
            file_good = False

        #check if max transmittance > 1.03
        if np.max(obs_dict["y_mean"][:, pixel_index]) < 1.03:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has no transmittance above 1.03")
        else:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has transmittance above 1.03")
            file_good = False
            
            
        #check if window top is consistent and matches chosen value
        if window_tops_in_file.count(window_tops_in_file[0]) == len(window_tops_in_file) and window_tops_in_file[0] == chosen_window_top:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has good window top")
        else:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "has wrong window top")
            file_good = False
            
        
        if file_good:
            print("%i/%i:" %(file_index, len(hdf5_filenames)), hdf5_filename, "is good")
            # if bin_index == 0:
            plt.plot(obs_dict["alt"], obs_dict["y_mean"][:, pixel_index], color=colours[file_index], label="%s" %hdf5_filename)
            # else:
            #     plt.plot(obs_dict["alt"], obs_dict["y_mean"][:, pixel_index], color=colours[file_index])
        
        
        
    ### derive correction
    spectra_in_bin = np.asfarray(spectra_in_bin)
    correction_dict[bin_index] = {}
    correction_dict[bin_index]["spectra"] = spectra_in_bin

    continuum = np.zeros_like(spectra_in_bin)
    deviation = np.zeros_like(spectra_in_bin)
    
    for spectrum_index, spectrum in enumerate(spectra_in_bin):
        polyfit = np.polyfit(pixels, spectrum, 5)
        continuum[spectrum_index, :] = np.polyval(polyfit, pixels)
        deviation[spectrum_index, :] = spectrum - continuum[spectrum_index, :]

    correction_dict[bin_index]["continuum"] = continuum
    correction_dict[bin_index]["deviation"] = deviation

    fit_coefficients = []
    for pixel in pixels:
        pixel = int(pixel)
        linear_fit, coefficients = fit_polynomial(continuum[:, pixel], deviation[:, pixel], coeffs=True)
        fit_coefficients.append(coefficients)

    fit_coefficients = np.asfarray(fit_coefficients).T
    correction_dict[bin_index]["coefficients"] = fit_coefficients


plt.legend()

#check if all window tops are the same
if window_tops.count(window_tops[0]) == len(window_tops):
    window_top = window_tops[0]



if SAVE_FILES:
    save_dict_to_hdf5(correction_dict, "px_correction_order%i_windowtop=%i" %(diffraction_order, window_top))






