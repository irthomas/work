# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:57:46 2021

@author: iant
"""

import numpy as np
import os
import re
# import json
from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
# import scipy.signal as ss
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths#, FIG_X, FIG_Y

from tools.plotting.colours import get_colours
from tools.plotting.anim import make_line_anim

def sin(x, A, B, C, D):  
    return A * np.sin(B * x + C) + D

def sin_slope(x, A, B, C, D, E):  
    return A * np.sin(B * x + C) + D + E * x


file_level = "hdf5_level_0p2a"


#1 khz 
# 20190223_054340_0p2a_SO_1_C # orders 125-127
# 20190223_054340_0p2a_SO_2_C
# 20190223_061847_0p2a_SO_1_C
# 20190223_061847_0p2a_SO_2_C
# regex = re.compile("20190223_(054340|061847)_0p2a_SO_._C")



#order 194:
#1 khz
# 20190416_020948_0p2a_SO_1_C

#2 khz
# 20181129_002850_0p2a_SO_2_C

#4 khz
# 20181010_084333_0p2a_SO_2_C
# 20190416_024455_0p2a_SO_1_C

#8 khz
# 20190107_015635_0p2a_SO_2_C
# 20190307_011600_0p2a_SO_1_C
#


# regex = re.compile("20190416_020948_0p2a_SO_1_C")
regex = re.compile("20181129_002850_0p2a_SO_2_C")

# regex = re.compile("(20190416_020948_0p2a_SO_1_C|20181129_002850_0p2a_SO_2_C|20181010_084333_0p2a_SO_2_C|20190416_024455_0p2a_SO_1_C|20190107_015635_0p2a_SO_2_C|20190307_011600_0p2a_SO_1_C)")


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


pixels = np.arange(320)
colours = get_colours(len(hdf5Filenames), cmap="viridis")

plt.subplots()



for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    

    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    unique_aotf = sorted(list(set(aotf_freq)))

    unique_indices_all = [[i for i,v in enumerate(aotf_freq) if v==aotf] for aotf in unique_aotf]
    min_elements = min([len(i) for i in unique_indices_all])
    unique_indices = [i[0:min_elements] for i in unique_indices_all]
    
    unique_aotf_freqs = np.asfarray([aotf_freq[i[0]] for i in unique_indices])
    aotf_stepping = unique_aotf_freqs[1] - unique_aotf_freqs[0]
    
    
   
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels


    d = {i:[] for i in unique_aotf_freqs}
    for frame_index, frame_nos in enumerate(unique_indices):
 
        out = {}
        y_max_frame = []
        for n, frame_no in enumerate(frame_nos):
            mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)

            d[aotf_freq[frame_no]].append(mean)
            
        
    # sg_window = {1.0:19, 2.0:19, 4.0:19, 8.0:9}[aotf_stepping]
    sg_window = {1.0:29, 2.0:19, 4.0:19, 8.0:9}[aotf_stepping]
        
        
    sum_spectra = []
    for i, key in enumerate(d.keys()):
        sum_spectra.append(d[key][0][180])
        # sum_spectra.append(np.sum(d[key][0]))
    sum_spectra = np.array(sum_spectra)
        
    sg = savgol_filter(sum_spectra, sg_window, 1)
    
    plt.plot(unique_aotf_freqs, sg)
    
    values_norm = sum_spectra#/sg
    plot_offset = float(file_index) * 0.05

    amp = 0.018
    freq = 0.38492
    temp_coeffs = np.array([-7.84038467, 55.42655564])
    first_peak = np.polyval(temp_coeffs, temperature)
    x_offset = first_peak - 6.0
    
    # sine = sin(unique_aotf_freqs, amp, freq, x_offset, 1.0)
    sine = sin_slope(unique_aotf_freqs, amp, freq, x_offset, 1.0, -0.2)

    values_norm_corr = values_norm - sine

    plt.scatter(unique_aotf_freqs, values_norm + plot_offset, label="%ikHz stepping, T=%0.1fC" %(aotf_stepping, temperature), color=colours[file_index], alpha=0.7)
    plt.plot(unique_aotf_freqs, values_norm + plot_offset, color=colours[file_index], alpha=0.7)
    plt.plot(unique_aotf_freqs, values_norm_corr + plot_offset + 1.05, color=colours[file_index], alpha=1.0)
    plt.plot(unique_aotf_freqs, sine + plot_offset, color="b", alpha=1.0)
            
# x_hr = np.arange(26590.0, 28630.0, 0.1)

