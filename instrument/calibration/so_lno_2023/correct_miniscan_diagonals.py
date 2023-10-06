# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:52:02 2023

@author: iant

STEP 1: MINISCAN DIAGONAL CORRECTIONS:
    SEARCH THROUGH MINISCAN OBSERVATIONS
    APPLY FFT CORRECTIONS
    GENERATE H5 FILES WITH CORRECTED DIAGONALS
    
    TODO: GET IT WORKING FOR 1KHZ SCANS

"""

import os
import re
import numpy as np
# from scipy.interpolate import RegularGridInterpolator
import h5py

import matplotlib.pyplot as plt
# from scipy.interpolate import splrep, splev, BSpline

# from tools.file.hdf5_functions import make_filelist, open_hdf5_file
# from tools.general.cprint import cprint
# from tools.spectra.running_mean import running_mean_1d
from tools.general.progress_bar import progress

# from instrument.nomad_so_instrument_v03 import aotf_peak_nu, lt22_waven
# from instrument.nomad_lno_instrument_v02 import nu0_aotf, nu_mp

from instrument.calibration.so_lno_2023.derive_blaze_aotf_miniscans import list_miniscan_data_1p0a, \
    get_miniscan_data_1p0a, remove_oscillations, find_peak_aotf_pixel, get_diagonal_blaze_indices

from instrument.calibration.so_lno_2023.make_hr_array import make_hr_array

# inflight

channel = "SO"
# channel = "LNO"


file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel)

regex = re.compile("20.*_%s_.*_CM" %channel) #search all files
# regex = re.compile("20180716.*_%s_.*_CM" %channel) #search specific file

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


HR_SCALER = 10. #make HR grid with N times more points

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")



if channel == "SO":
    aotf_steppings = [4.0]
    # aotf_steppings = [2.0]
    binnings = [0]

    starting_orders = list(range(178, 210))
    # starting_orders = [188]

    # dictionary of fft_cutoff for each aotf_stepping
    fft_cutoff_dict = {
        1:4,
        2:15,
        4:15,
        8:40,
        }



elif channel == "LNO":
    # aotf_steppings = [8.0]
    aotf_steppings = [4.]
    binnings = [0]

    # starting_orders = [194]
    starting_orders = list(range(210))

    # dictionary of fft_cutoff for each aotf_stepping
    fft_cutoff_dict = {} #don't apply FFT - no ringing




illuminated_row_dict = {24:slice(6, 19)}


list_files = True
# list_files = False

if "d" not in globals():
    list_files = True



# plot_fft = True
plot_fft = False







"""get data"""
if __name__ == "__main__":
    if file_level == "hdf5_level_1p0a":
        if list_files:
            h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, starting_orders, aotf_steppings, binnings)
            if "d" not in globals():
                d = get_miniscan_data_1p0a(h5_filenames)
            if h5_prefixes != list(d.keys()):
                d = get_miniscan_data_1p0a(h5_filenames)
    
    
        
    if channel == "SO":
        d2 = remove_oscillations(d, fft_cutoff_dict, cut_off=["inner"], plot=plot_fft)


    elif channel == "LNO":
        #no FFT ringing correction
        d2 = {}
        for h5_prefix in h5_prefixes:
            
            n_reps = d[h5_prefix]["y_rep"].shape[0]

            d2[h5_prefix] = {"aotf":d[h5_prefix]["a_rep"], "t":np.mean(d[h5_prefix]["t_rep"])} #don't do anything
            d2[h5_prefix]["nreps"] = n_reps

            #bad pixel correction
            for rep_ix in range(n_reps):
                miniscan_array = d[h5_prefix]["y_rep"][rep_ix, :, :]  #get 2d array for 1st repetition in file
                # miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)
                d2[h5_prefix]["array%i" %(rep_ix)] = miniscan_array

    
    """make HR arrays"""
    for h5_prefix in progress(h5_prefixes):

        n_reps = d2[h5_prefix]["nreps"]
        #HR array spectra for all repetitions
        aotfs = d2[h5_prefix]["aotf"]
        for rep in range(n_reps):
            
            #interpolate onto high res grid
            array = d2[h5_prefix]["array%i" %rep]
            array_hr, aotf_hr = make_hr_array(array, aotfs, HR_SCALER)
            d2[h5_prefix]["array%i_hr" %rep] = array_hr
            
        d2[h5_prefix]["aotf_hr"] = aotf_hr


        d2[h5_prefix]["t"] = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
        d2[h5_prefix]["t_range"] = [[np.min(d[h5_prefix]["t_rep"][:, rep]), np.max(d[h5_prefix]["t_rep"][:, rep])] for rep in range(n_reps)]


        
        for rep in range(d2[h5_prefix]["nreps"]):
        #calc blaze diagonals
        
            t = d2[h5_prefix]["t"][rep]
            px_ixs = np.arange(d2[h5_prefix]["array%i_hr" %rep].shape[1])

            px_peaks, aotf_nus = find_peak_aotf_pixel(t, d2[h5_prefix]["aotf_hr"], px_ixs)
            px_peaks = np.asarray(px_peaks)  * int(HR_SCALER)
            aotf_nus = np.asarray(aotf_nus)
            blaze_diagonal_ixs_all = get_diagonal_blaze_indices(px_peaks, px_ixs)
        
        
            #make diagonally corrected array
            diagonals = []
            diagonals_aotf = []
             
            for row in range(d2[h5_prefix]["array%i_hr" %rep].shape[0]-5):
                #find closest diagonal pixel number (in first column)
                closest_ix = np.argmin(np.abs(blaze_diagonal_ixs_all[:, 0] - row))
                row_offset = blaze_diagonal_ixs_all[closest_ix, 0] - row
                
                
                #apply offset to diagonal indices
                blaze_diagonal_ixs = (blaze_diagonal_ixs_all[closest_ix, :] - row_offset)
                
                if np.all(blaze_diagonal_ixs < d2[h5_prefix]["array%i_hr" %rep].shape[0]):
                    diagonals.append(d2[h5_prefix]["array%i_hr" %rep][blaze_diagonal_ixs, px_ixs])
                    diagonals_aotf.append(d2[h5_prefix]["aotf_hr"][blaze_diagonal_ixs])
                    
            diagonals = np.asarray(diagonals)
            diagonals_aotf = np.asarray(diagonals_aotf)
            d2[h5_prefix]["array_diag%i_hr" %rep] = diagonals
            d2[h5_prefix]["aotf_diag%i_hr" %rep] = diagonals_aotf
            
            # print("Diagonal shape: ", diagonals.shape)
            # print("Diagonal AOTF shape: ", diagonals_aotf.shape)

    
        """Save figures and files"""
        #save diagonally-correct array and aot freqs to hdf5
        with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" %h5_prefix), "w") as f:
            for rep in range(d2[h5_prefix]["nreps"]):
                f.create_dataset("array%02i" %rep, dtype=np.float32, data=d2[h5_prefix]["array_diag%i_hr" %rep], \
                                 compression="gzip", shuffle=True)
                f.create_dataset("aotf%02i" %rep, dtype=np.float32, data=d2[h5_prefix]["aotf_diag%i_hr" %rep], \
                                 compression="gzip", shuffle=True)
                f.create_dataset("t%02i" %rep, dtype=np.float32, data=d2[h5_prefix]["t_range"][rep], \
                                 compression="gzip", shuffle=True)
                
        #save miniscan png
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.title(h5_prefix)
        plt.imshow(d2[h5_prefix]["array_diag%i_hr" %rep], aspect="auto")
        plt.savefig(os.path.join(MINISCAN_PATH, channel, "%s.png" %h5_prefix))
        plt.close()

