# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:15:17 2021

@author: iant

plot relative signal strengths of SO channel bins to check alignment
"""


import os
import numpy as np
from datetime import datetime
import re


#from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from tools.file.hdf5_functions import make_filelist
from tools.file.paths import FIG_X, FIG_Y, paths



# if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
#     print("Running on windows")

# import spiceypy as sp
# from tools.spice.load_spice_kernels import load_spice_kernels
# load_spice_kernels()


#SAVE_FIGS = False
SAVE_FIGS = True

# SAVE_FILES = False
# #SAVE_FILES = True


file_level = "hdf5_level_0p3k"
regex = re.compile("20.*_0p3k_SO_A_[IE]_134")


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level)


#loop through files

obs_datetimes = []
relative_signals = []

for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    year = int(hdf5_filename[0:4])
    month = int(hdf5_filename[4:6])
    day = int(hdf5_filename[6:8])
    hour = int(hdf5_filename[9:11])  
    minute = int(hdf5_filename[11:13])  
    second = int(hdf5_filename[13:15]) 
    obs_datetime = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    
    y = hdf5_file["Science/Y"][...]
    y_len = y.shape[0]
    sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]

    if "_I_" in hdf5_filename:
        bin_indices_toa = [np.arange(0, 20, 4), np.arange(1, 20, 4), np.arange(2, 20, 4), np.arange(3, 20, 4)]
    elif "_E_" in hdf5_filename:
        bin_indices_toa = [np.arange(y_len-20, y_len, 4), np.arange(y_len-19, y_len, 4), np.arange(y_len-18, y_len, 4), np.arange(y_len-17, y_len, 4)]

    if sbsf == 0:
        continue
    

    bin_means = []
    for bin_indices in bin_indices_toa:
        bin_means.append(np.mean(y[bin_indices, 160:240]))
        
    if np.max(bin_means)<100000:
        print("%s: Error - max signal too low" %hdf5_filename)

    else:
        relative_means = bin_means / np.max(bin_means)
        # relative_means = np.asfarray(bin_means)
        obs_datetimes.append(obs_datetime)
        relative_signals.append(relative_means)
    
    
    
relative_signals = np.asfarray(relative_signals)

fig, ax = plt.subplots(figsize=(15, 5))
plt.title("SO channel relative counts for each bin\nSearch string: %s" %regex.pattern)
for bin_index in range(4):
    ax.plot_date(obs_datetimes, relative_signals[:,bin_index], label="Bin %i" %bin_index, marker=".")
ax.set_xlabel("Observation Date")
ax.set_ylabel("Relative counts for each bin")
ax.legend()
ax.grid(True)

ax.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=1, tz=None))    

if SAVE_FIGS:
    plt.savefig("so_relative_counts.png")
   
