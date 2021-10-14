# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:17:32 2021

@author: iant

PLOT ALTITUDE, LAT, LON RESOLUTION FOR DIFFERENT OCCULTATIONS
"""
import os
import numpy as np
from datetime import datetime, timedelta
import re
import h5py

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from tools.file.hdf5_functions import make_filelist
from tools.file.paths import FIG_X, FIG_Y, paths


d = {}


file_level = "hdf5_level_1p0a"
# regex = re.compile("20(18|19)[0-9][0-9][0-9][0-9]_.*_1p0a_SO_A_[IE]_(134|136)")

for ie in ["i","e"]:
    
    regex = re.compile("20......_.*_1p0a_SO_A_[%s]_(134|136|167|168|189|190)" %ie.upper())
    
    
    #get files
    hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level, open_files=False, silent=True)
    
    d[ie] = {"filename":[], "dt":[], "et":[], "lat_mean":[], "alt_min_d":[], "alt_max_d":[], "lat_min_d":[], "lat_max_d":[]}
    
    
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
            sbsf = f["Channel/BackgroundSubtraction"][0]
            
            #use non-bg subtracted for start of mission; use bg-subtracted later
            if sbsf == 1 and obs_datetime < datetime(2019, 7, 1):
                continue
            if sbsf == 0 and obs_datetime > datetime(2019, 7, 1):
                continue
            
            alt_all = f["Geometry/Point0/TangentAltAreoid"][...]
            
            indices = np.where((alt_all[:, 0] > 10) & (alt_all[:, 0] < 100))[0]
            
            alts = alt_all[indices, :]
            
            lons = f["Geometry/Point0/Lon"][indices, :]
            lats = f["Geometry/Point0/Lat"][indices, :]
            
            # et = f["Geometry/ObservationEphemerisTime"][:, 0]
            
            
        
    
        mean_lat = np.mean(lats)
        alt_diff = np.diff(alts)
        lat_diff = np.diff(lats)
        # et_diff = np.diff(et)
        
        if len(d[ie]["dt"]) > 0: #if not the first entry
            #check time delta between this and previous observation - add nans to split occultation-free zones
            if obs_datetime - d[ie]["dt"][-1] > timedelta(days=10): 
                d[ie]["filename"].append(hdf5_filename)
                d[ie]["dt"].append(obs_datetime)
                d[ie]["lat_mean"].append(np.nan)
                d[ie]["alt_min_d"].append(np.nan)
                d[ie]["alt_max_d"].append(np.nan)
                d[ie]["lat_min_d"].append(np.nan)
                d[ie]["lat_max_d"].append(np.nan)
                # d[ie]["et"].append([np.nan])
            elif obs_datetime - d[ie]["dt"][-1] < timedelta(minutes=1): #check if same occultation but different order, ignore
                # print("same occultation:", obs_datetime, obs_datetime - d[ie]["dt"][-1])
                continue

        error = False
        if len(d[ie]["dt"]) > 1:
            curr = np.min(alt_diff)*1000.
            prev = d[ie]["alt_min_d"][-1]
            prev_2 = d[ie]["alt_min_d"][-2]
            exp = 2.0 * prev - prev_2 #prev + (prev - prev_2)
            
            
            if np.abs(curr - exp) > 25:
                error = True
            
                print(file_index, obs_datetime, curr - exp)

        if not error:
            d[ie]["filename"].append(hdf5_filename)
            d[ie]["dt"].append(obs_datetime)
            d[ie]["lat_mean"].append(mean_lat)
            d[ie]["alt_min_d"].append(np.min(alt_diff)*1000.)
            d[ie]["alt_max_d"].append(np.max(alt_diff)*1000.)
            d[ie]["lat_min_d"].append(np.min(lat_diff))
            d[ie]["lat_max_d"].append(np.max(lat_diff))
            # d[ie]["et"].append(et_diff)
        
        
        

fig = plt.figure(figsize=(FIG_X+3, FIG_Y+4), constrained_layout=True)
gs = fig.add_gridspec(5, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1:3, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[3:5, 0], sharex=ax1)
        

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(FIG_X, FIG_Y))
# fig.suptitle("Solar Occultation Tangent Altitude and Latitudinal Sampling Resolution from 10-100km Altitude")
ax3.set_xlabel("Date")
ax1.set_ylabel("Mean latitude of\nTangent Point (deg)")
ax2.set_ylabel("Change in tangent altitude\nbetween consecutive spectra (m)")
ax3.set_ylabel("Change in latitude\nbetween consecutive spectra (deg)")
ax1.set_title("Solar Occultation Tangent Altitude and Latitudinal Sampling Resolution from 10-100km Altitude")
ax1.set_ylim((-90,90))
ax1.set_yticks([-90,-45,0,45,90])

ax1.grid()
ax2.grid()
ax3.grid()

linestyle = "-"
ms = 4

ie = "i"
colour = "tab:blue"
alpha = 1.0

ax1.plot_date(d[ie]["dt"], d[ie]["lat_mean"], linestyle=linestyle, ms=ms, color=colour, label="Ingress")
ax1.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=1, tz=None))

ax2.fill_between(d[ie]["dt"], y1=d[ie]["alt_min_d"], y2=d[ie]["alt_max_d"], color=colour, alpha=alpha, label="Ingress Altitude Change")
ax3.fill_between(d[ie]["dt"], y1=d[ie]["lat_min_d"], y2=d[ie]["lat_max_d"], color=colour, alpha=alpha, label="Ingress Latitude Change")


ie = "e"
colour = "tab:red"
ax1.plot_date(d[ie]["dt"], d[ie]["lat_mean"], linestyle=linestyle, ms=ms, color=colour, label="Egress")

ax2.fill_between(d[ie]["dt"], y1=d[ie]["alt_min_d"], y2=d[ie]["alt_max_d"], color=colour, alpha=alpha, label="Egress Altitude Change")
ax3.fill_between(d[ie]["dt"], y1=d[ie]["lat_min_d"], y2=d[ie]["lat_max_d"], color=colour, alpha=alpha, label="Egress Latitude Change")


ax1.tick_params(axis="x", labelbottom=False)
ax2.tick_params(axis="x", labelbottom=False)

ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")
# fig.tight_layout()

fig.savefig("altitude_resolution.png", dpi=300)