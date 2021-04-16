# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:15:17 2021

@author: iant

plot relative signal strengths of SO channel bins to check alignment
"""


import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime
# import re
#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap


#from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

from tools.file.hdf5_functions import make_filelist, get_file
from tools.file.paths import FIG_X, FIG_Y, paths
#from analysis_functions_v01b import spectralCalibration,write_log,get_filename_list,stop
#from filename_lists_v01 import getFilenameList

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    print("Running on windows")
    import spiceypy as sp
    
    from tools.spice.load_spice_kernels import load_spice_kernels
    
    load_spice_kernels()


#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True


fileLevel = "hdf5_level_0p1a"
obspaths = ["*201812*_0p1a_SO_1"]

fileLevel = "hdf5_level_0p1a"
obspaths = ["*201812*_0p1a_SO_1"]




def getOccultationReferenceCounts(hdf5File, chosen_aotf_frequency=-999.0):

#    obs_start_time = hdf5File["Timestamp"][0]
    obs_start_string = hdf5File["Geometry/ObservationDateTime"][0][0].decode("utf-8")
    
    aotf_freq = hdf5File["Channel/AOTFFrequency"][...]
    
    if chosen_aotf_frequency == -999.0:
        chosen_aotf_frequency = aotf_freq[0]
    matching_indices_boolean = aotf_freq == chosen_aotf_frequency
    matching_indices = np.where(matching_indices_boolean == True)[0]
    
    if len(matching_indices) == 0:
        print("AOTF frequency %0.0f not found in file; skipping" %chosen_aotf_frequency)
        return 0, "", 0, np.zeros(4)
    else:
#        print("AOTF frequency %0.0f found; getting info" %chosen_aotf_frequency)
        first_match = matching_indices[0]
        last_match = matching_indices[-1]
        binned_counts_start = np.mean(hdf5File["Science/Y"][first_match,:,160:240], axis=1)
        binned_counts_end = np.mean(hdf5File["Science/Y"][last_match,:,160:240], axis=1)
        
#        binned_counts_start = np.mean(hdf5File["Science/Y"][matching_indices[0:5],:,160:240], axis=(0,2))
#        binned_counts_end = np.mean(hdf5File["Science/Y"][matching_indices[-6:-1],:,160:240], axis=(0,2))
        
        if np.mean(binned_counts_start) > np.mean(binned_counts_end): #if ingress
            max_counts = np.max(binned_counts_start)
            counts_out = binned_counts_start / max_counts
        else:
            max_counts = np.max(binned_counts_end)
            counts_out = binned_counts_end / max_counts
    
        return obs_start_string, max_counts, counts_out



def plotBinStrengths(hdf5Files, hdf5Filenames, obspaths):
    et_string_all = []
    max_counts_all = []
    relative_counts_all = []
    for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        print("%i/%i: Reading in file %s" %(fileIndex, len(hdf5Filenames), hdf5Filename))

    #    et, et_string, max_counts, relative_counts = getOccultationReferenceCounts(hdf5File, hdf5Filename, 17859.0)
        et_string, max_counts, relative_counts = getOccultationReferenceCounts(hdf5File)
        
        if len(relative_counts) == 4: #just take nominal 6 order data
            et_string_all.append(et_string)
            max_counts_all.append(max_counts)
            relative_counts_all.append(relative_counts)
            
            if np.min(relative_counts) < 0.9:
                print("File %s has minimum relative counts of %0.2f (max counts = %0.0f)" %(hdf5Filename, np.min(relative_counts), max_counts))
    
    relative_counts_array = np.asfarray(relative_counts_all)
    et_array = np.asfarray([sp.utc2et(string) for string in et_string_all])
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title("SO channel relative counts for each bin\nSearch string: %s" %obspaths[0])
    for bin_index in range(4):
        plt.scatter(et_array, relative_counts_array[:,bin_index], label="Bin %i" %bin_index, marker=".")
    plt.xlabel("Ephemeris Time (s)")
    plt.ylabel("Relative counts for each bin")
    plt.legend()
    plt.grid(True)
    
    months = np.arange(4, 13, 1)
    monthStarts = [sp.utc2et(datetime(2018, month, 1).strftime("%Y-%m-%d")) for month in months]
    monthNames = [datetime(2018, month, 1).strftime("%B") for month in months]
    for monthStart, monthName in zip(monthStarts, monthNames):
        plt.axvline(x=monthStart, color='k', linestyle='--')
        plt.text(monthStart+100000, 0.7, monthName)
    


"""plot relative signal strengths of SO channel bins to check alignment"""
hdf5Files, hdf5Filenames, titles = make_filelist(obspaths, fileLevel)
plotBinStrengths(hdf5Files, hdf5Filenames, obspaths)

