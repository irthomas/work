# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:47:33 2021

@author: iant

TEST AOTF FITTING

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



file_level = "hdf5_level_0p2a"

regex = re.compile("20210201_111011_0p2a_SO_2_C")  #order 129 CH4 T=-15
good_indices = range(0, 1800, 100)
max_chisq = 0.005
absorption_indices = np.arange(242,249)

hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


linestyles = {0:"-", 1:":", 2:"--", 3:"-."}
colours = get_colours(len(good_indices))
pixels = np.arange(320)

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    
    detector_rows = {"so":[128-8, 128+8], "lno":[152-72, 152+72]}[channel]
    


    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])
    

    dim = detector_data_all.shape
    
    detector_centre_data = detector_data_all[:, 10:14, :]
    dim_rows = detector_centre_data.shape


    fig1 = plt.figure(figsize=(FIG_X, FIG_Y))
    gs = fig1.add_gridspec(4,1)
    ax1a = fig1.add_subplot(gs[0:3, 0])
    ax2a = fig1.add_subplot(gs[3, 0], sharex=ax1a)
    
    for frame_index, frame_no in enumerate(good_indices):
        for row_index in range(dim_rows[1]):
            ax1a.plot(pixels, detector_centre_data[frame_no, row_index, :].T, linestyle=linestyles[row_index], color=colours[frame_index])
        std = np.std(detector_centre_data[frame_no, :, :], axis=0)
        mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)
        ax2a.plot(pixels[50:], std[50:]/mean[50:], color=colours[frame_index])





