# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:21:00 2022

@author: iant

PLOT DIFFERENCE BETWEEN PLANNED TC20 EXECUTION TIMES AND REAL EXECUTION TIMES
"""

import os
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

from tools.sql.read_cache_db import get_filenames_from_cache
from tools.sql.read_itl_db import get_itl_dict



SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"


level = "hdf5_level_1p0a"


#read cache.db
cache = get_filenames_from_cache(os.path.join(SHARED_DIR_PATH, "db", level + ".db"))
cache_filenames = sorted([s.replace(".h5","") for s in cache[1]])


#read obs_type db
d_itl = get_itl_dict(os.path.join(SHARED_DIR_PATH, "db", "obs_type.db"))


#convert filenames to datetime
filename_dts = []
for filename in cache_filenames:
    dt = datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
    filename_dts.append(dt)


#define possible timedelta range
timedelta_ranges = [
    [datetime(2018, 3, 1), datetime(2018, 4, 1), -120, 0],
    [datetime(2018, 4, 1), datetime(2018, 4, 5), -80, 20],
    [datetime(2018, 4, 5), datetime(2018, 8, 25), -10, 20],
    [datetime(2018, 8, 25), datetime(2018, 11, 5), -5, 50],
    [datetime(2018, 11, 5), datetime(2021, 9, 1), -25, 25],
    [datetime(2021, 9, 1), datetime(2025, 1, 1), -35, 35],
]

#go through itl observations

#first, find first match between itl and tree
i = -1
j = -1
all_found = False

matches = {}


#loop through 
while i < len(filename_dts) - 1:
    
    i += 1
    found = False

    while not found:
        
        # i += 1
        j += 1
        
        
        itl_dt = d_itl["tc20_exec_start"][j]
        channels = d_itl["channels"][j].split(", ")
        
        filename_dt = filename_dts[i]
        filename = cache_filenames[i]
    
        tdelta = (itl_dt - filename_dt).total_seconds()
        
        #get timedelta range:
        for timedelta_range in timedelta_ranges:
            if timedelta_range[0] < filename_dt < timedelta_range[1]:
                start = timedelta_range[2]
                end = timedelta_range[3]
                
        
        if start < tdelta < end:
            for channel in channels:
                if channel in filename:
                    if np.mod(i, 1000) == 0:
                        print(tdelta, itl_dt, channel, filename)
                    matches[itl_dt] = [filename, tdelta]
                    
                    found = True
                    j -= 3




tdeltas = [v[1] for v in matches.values()]
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax.plot(matches.keys(), tdeltas)
ax.set_title("Planned execution time vs real execution time")
ax.set_xlabel("Observation time")
ax.set_ylabel("MITL execution time vs HDF5 filename (seconds)")
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=1))
ax.axhline(y=0, color="k", linestyle="--")
ax.tick_params(axis='x', labelrotation=90)
ax.grid()
fig.savefig("planned_vs_real_execution_time.png")

