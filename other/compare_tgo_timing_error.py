# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:05:38 2021

@author: iant

COMPARE FILES BEFORE/AFTER TGO TIMING ERROR REPROCESSING
"""

import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator

from datetime import datetime

import platform
if platform.system() == "Linux":
    #UVIS timing error
    HDF5_DIRECTORY_BEFORE = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/hdf5/timing_error_210729/hdf5_level_0p2a/"
    HDF5_DIRECTORY_AFTER = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p2a/"
    

else:
    #SO timing error
    HDF5_DIRECTORY_BEFORE = r"D:\DATA\timing_error\before\hdf5_level_0p2a"
    HDF5_DIRECTORY_AFTER = r"D:\DATA\timing_error\after\hdf5_level_0p2a"

# OCCULTATION_CUTOFF_SECONDS = 60*60*24*10

#make filelists
hdf5_filepath_list_before = sorted(glob.glob(HDF5_DIRECTORY_BEFORE+r"\**\*.h5", recursive=True))
hdf5_filepath_list_after = sorted(glob.glob(HDF5_DIRECTORY_AFTER+r"\**\*.h5", recursive=True))

field_path = "Geometry/Point0/TangentAltAreoid"

hdf5_datetimes_i = []
hdf5_min_maxima_i = []
hdf5_datetimes_e = []
hdf5_min_maxima_e = []

for file_index, hdf5_filepath_before in enumerate(hdf5_filepath_list_before):
    
    hdf5_filepath_after = hdf5_filepath_before.replace("before", "after")
    hdf5_basename = os.path.basename(hdf5_filepath_before)
    
    if not os.path.exists(hdf5_filepath_after):
        print("Error: %s not found in new dataset" %hdf5_basename)
    else:
        
        
    
        with h5py.File(hdf5_filepath_before, "r") as f:
            tangent_alt_before = f[field_path][:,0]
            
        with h5py.File(hdf5_filepath_after, "r") as f:
            tangent_alt_after = f[field_path][:,0]
    
        
        tangent_alt_before[tangent_alt_before < -900] = np.nan
        tangent_alt_after[tangent_alt_after < -900] = np.nan
        
        if tangent_alt_before[0] > 200.0 and tangent_alt_before[-1] > 200.0:
            continue
        
        else:
            tangent_alt_before[tangent_alt_before > 200] = np.nan
            tangent_alt_after[tangent_alt_after > 200] = np.nan
        
            tangent_alt_diff = tangent_alt_before - tangent_alt_after
    
            # tangent_alt_diff[tangent_alt_diff > 900.] = np.nan
            # tangent_alt_diff[tangent_alt_diff < -900.] = np.nan
            # tangent_alt_diff[tangent_alt_diff == 0.] = np.nan
            
            if np.all(np.isnan(tangent_alt_diff)):
                tangent_alt_diff = [0.0,0.0]
            
            hdf5_datetime = datetime.strptime(hdf5_basename[0:15], "%Y%m%d_%H%M%S")
            
            #add nans when gap is large
            timedelta_i = 0.
            timedelta_e = 0.
            if len(hdf5_datetimes_i)>0:
                timedelta_i = (hdf5_datetime - hdf5_datetimes_i[-1]).total_seconds()
            if len(hdf5_datetimes_e)>0:
                timedelta_e = (hdf5_datetime - hdf5_datetimes_e[-1]).total_seconds()
                
            timedelta_obs = min([timedelta_i, timedelta_e])
            
            if hdf5_basename[26] == "I":
                hdf5_datetimes_i.append(hdf5_datetime)
                hdf5_min_maxima_i.append([np.nanmin(tangent_alt_diff), np.nanmax(tangent_alt_diff)])
                
                # if hdf5_min_maxima_i[-1][0] < -0.5 and hdf5_min_maxima_i[-1][1] > 0.5:
                #     stop()
            if hdf5_basename[26] == "E":
                hdf5_datetimes_e.append(hdf5_datetime)
                hdf5_min_maxima_e.append([np.nanmin(tangent_alt_diff), np.nanmax(tangent_alt_diff)])
        
hdf5_min_maxima_i = np.asfarray(hdf5_min_maxima_i)        
hdf5_min_maxima_e = np.asfarray(hdf5_min_maxima_e)        
        # stop()

fig, ax = plt.subplots(figsize=(12, 6))
ax.grid()
   
# ax.scatter(hdf5_datetimes_i, np.mean(hdf5_min_maxima_i, axis=1), color="tab:blue", alpha=0.7, label="Mean ingress tangent altitude difference")
ax.fill_between(hdf5_datetimes_i, y1=hdf5_min_maxima_i[:,0], y2=hdf5_min_maxima_i[:,1], color="tab:blue", alpha=1.0, label="Ingress max & min extent of altitude difference within each occultation")

# ax.scatter(hdf5_datetimes_e, np.mean(hdf5_min_maxima_e, axis=1), color="tab:red", alpha=0.7, label="Mean egress tangent altitude difference")
ax.fill_between(hdf5_datetimes_e, y1=hdf5_min_maxima_e[:,0], y2=hdf5_min_maxima_e[:,1], color="tab:red", alpha=1.0, label="Egress max & min extent of altitude difference within each occultation")

ax.set_title("NOMAD solar occultations: difference in areoid tangent altitude before - after TGO timing error correction")
ax.set_xlabel("Date")
ax.set_ylabel("Areoid Tangent Altitude Difference (km)")
ax.legend(loc="upper left")

ax.axvline(x=datetime(year=2018, month=4, day=21), color="k")
ax.text(datetime(year=2018, month=4, day=25), -4.5, "TGO mission start 21st April 2018")

ax.axvline(x=datetime(year=2019, month=5, day=18), color="k")
ax.text(datetime(year=2019, month=5, day=25), -4.5, "TGO clock reset 18th May 2019")

ax.axvline(x=datetime(year=2021, month=3, day=20), color="k")
ax.text(datetime(year=2021, month=3, day=25), -4.5, "Clock synchronisation\n20th March 2021")


ax.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=4, tz=None))    
fig.savefig("occultation_altitude_change_due_to_timing_error.png", dpi=300)