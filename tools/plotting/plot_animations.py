# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:30:00 2021

@author: iant

MAKE ANIMATIONS OF SELECTED FILES
"""


import numpy as np
# import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
# from scipy.signal import savgol_filter

# import matplotlib.pyplot as plt

from tools.plotting.anim import make_frame_anim
from tools.file.hdf5_functions import make_filelist

from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

# from tools.file.paths import paths, FIG_X, FIG_Y
# from tools.spectra.baseline_als import baseline_als

file_level = "hdf5_level_0p1a"
# regex = re.compile("(20160615_224950|20180428_023343|20180511_084630|20180522_221149|20180821_193241|20180828_223824)_0p1a_SO_1")
# regex = re.compile("20160615_224950_0p1a_SO_1") #best absorption line avoiding bad pixels

# regex = re.compile("201504[0-9][0-9]_.*_SO_.") 


regex = re.compile("20150404_072956_.*_SO_.") 



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


    """print diffraction order"""
    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    if len(list(set(aotf_freq))) > 1:
        print("AOTF changing")
        # continue
    
    
    if channel == "so":
        order = m_aotf_so(aotf_freq[0])
    elif channel == "lno":
        order = m_aotf_lno(aotf_freq[0])
    print("Order=%i" %order)


    
    
    detector_ffs = np.zeros((dim[0], 256, 320))
    
    for i in frame_indices:
        wtop = window_top_all[i]
        d = detector_data_all[i, :, :]
        frame = np.repeat(d, repeats=binning, axis=0)
        
        detector_ffs[i, wtop:(wtop+n_rows_binned), :] = frame
    
    
    
    # #make list of detector rows in each frame
    # row_numbers_all = []
    # for window_top in window_top_all:
    #     row_numbers = np.arange(window_top, window_top + n_rows, binning)
    #     row_numbers_all.append(row_numbers)


    # detector_data_reshaped = np.zeros((int(dim[0]/nu), int(dim[1]*nu), int(dim[2])))

    # for u in range(nu):
    #     detector_data_reshaped[:, u*dim[1]:(u+1)*dim[1], :] = detector_data_all[range(u, nff*nu, nu), :, :]

    """make animation"""
    max_value = np.ma.masked_equal(detector_ffs, 0).mean() + np.ma.masked_equal(detector_ffs, 0).std()*4
    min_value = np.ma.masked_equal(detector_ffs, 0).mean() - np.ma.masked_equal(detector_ffs, 0).std()*4
    make_frame_anim(detector_ffs, max_value, min_value, hdf5_filename)


