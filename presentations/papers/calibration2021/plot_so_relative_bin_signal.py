# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:15:17 2021

@author: iant

plot relative signal strengths of SO channel bins to check alignment
"""


import os
import numpy as np
import numpy.linalg as la

from datetime import datetime, timedelta

import re
import h5py

#from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths

import spiceypy as sp
from tools.spice.load_spice_kernels import load_spice_kernels


# if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
#     print("Running on windows")

load_spice_kernels()


#SAVE_FIGS = False
SAVE_FIGS = True


file_level = "hdf5_level_0p3k"
# regex = re.compile("20(18|19|20)[0-9][0-9][0-9][0-9]_.*_0p3k_SO_A_[IE]_134")
regex = re.compile("20......_.*_0p3k_SO_A_[IE]_(134|136)")
# regex = re.compile("20180[456][0-9][0-9]_.*_0p3k_SO_A_[IE]_(134|136)")


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level, open_files=False)


#loop through files

obs_datetimes = []
relative_signals = []

for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    if np.mod(file_index, 100) == 0:
        print(file_index, hdf5_filename)
        
        
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]
    hour = hdf5_filename[9:11]
    minute = hdf5_filename[11:13]
    second = hdf5_filename[13:15]
    obs_datetime = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
    
    file_path = os.path.join(paths["DATA_DIRECTORY"], file_level, year, month, day, hdf5_filename+".h5")
    
    with h5py.File(file_path, "r") as f:
        y = f["Science/Y"][...]
        sbsf = f["Channel/BackgroundSubtraction"][0]
    y_len = y.shape[0]

    if "_I_" in hdf5_filename:
        bin_indices_toa = [np.arange(0, 20, 4), np.arange(1, 20, 4), np.arange(2, 20, 4), np.arange(3, 20, 4)]
    elif "_E_" in hdf5_filename:
        bin_indices_toa = [np.arange(y_len-20, y_len, 4), np.arange(y_len-19, y_len, 4), np.arange(y_len-18, y_len, 4), np.arange(y_len-17, y_len, 4)]

    # if sbsf == 0:
    #     continue
    

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

#set up subplots
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(3, 1)
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[1:3, 0], sharex=ax1a)




for bin_index in range(4):
    ax1b.plot_date(obs_datetimes, relative_signals[:,bin_index], label="Bin %i" %bin_index, ms=3)
ax1b.set_ylabel("Relative counts for each bin")
ax1b.legend()
ax1b.grid(True)





abcorr="None"
ref="J2000"
observer="-143" #observer
target = "SUN"


datetime_start = datetime(year=2018, month=4, day=21)
datetimes = [datetime_start + timedelta(days=x) for x in range((obs_datetimes[-1]-obs_datetimes[0]).days)]

date_strs = [datetime.strftime(x, "%Y-%m-%d") for x in datetimes]
date_ets = [sp.str2et(x) for x in date_strs]
tgo_pos = np.asfarray([sp.spkpos(target, time, ref, abcorr, observer)[0] for time in list(date_ets)])



tgo_dist = la.norm(tgo_pos,axis=1)
code = sp.bodn2c(target)
pradii = sp.bodvcd(code, 'RADII', 3) # 10 = Sun
sun_radius = pradii[1][0]
sun_diameter_arcmins = np.arctan(sun_radius/tgo_dist) * sp.dpr() * 60.0 * 2.0




ax1a.plot_date(datetimes, sun_diameter_arcmins, linestyle="-", ms=0)


ax1b.set_xlabel("Observation Date")
ax1a.set_ylabel("Solar diameter as seen\nfrom TGO (arcminutes)")

ax1a.set_title("Apparent diameter of Sun")
ax1b.set_title("SO channel relative counts for each bin")

ax1a.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=1, tz=None))    
ax1a.grid(True)


if SAVE_FIGS:
    plt.savefig("sun_diameter_so_relative_counts.png", dpi=300)
   
