# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:38:55 2022

@author: iant

LNO LIFETIMES
"""


import re
# import numpy as np
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist
# from tools.spectra.running_mean import running_mean_1d
from tools.general.progress_bar import progress


HDF5_DT_FORMAT = "%Y %b %d %H:%M:%S.%f"

# file_level = "hdf5_level_0p1a"
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_LNO_.*_CM")
# regex = re.compile(".*_SO_.*_CM")
# regex = re.compile("202201.._.*_LNO_.")
regex = re.compile("20......_.*_LNO_.")



def get_daily_usage(regex, file_level):
    h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False)
    
    
    h5_prefixes = []
    dates = {}
    
    for h5_ix, (h5_filepath, h5) in enumerate(progress(list(zip(h5_files, h5_filenames)))):
        
        h5_prefix = "%s" %(h5[0:15])
        
        if h5_prefix not in h5_prefixes:
            h5_prefixes.append(h5_prefix)
        
            h5_f = h5py.File(h5_filepath, "r")
            dt_str_start = h5_f["Geometry/ObservationDateTime"][0,0].decode()
            dt_str_end = h5_f["Geometry/ObservationDateTime"][-1, -1].decode()
            
            dt_start = datetime.strptime(dt_str_start, HDF5_DT_FORMAT)
            dt_end = datetime.strptime(dt_str_end, HDF5_DT_FORMAT)
        
            duration = (dt_end - dt_start).total_seconds() + 600.0
        
            year = h5[0:4]
            month = h5[4:6]
            day = h5[6:8]
            
            date = datetime(year=int(year), month=int(month), day=int(day))
            
            if date not in dates.keys():
                dates[date] = 0
            dates[date] += duration
    return dates

# load data only if not in memory
if "dates" not in globals():
    dates = get_daily_usage(regex, file_level)
        
    


days = [s for s in dates.keys()]
durations = [f/3600.0 for f in dates.values()]


#cumulative
usage = {}
cumul = 0
for date in dates.keys():
    cumul += dates[date]
    
    usage[date] = cumul

usage_days = [s for s in usage.keys()]
usage_cumul = [f/3600.0 for f in usage.values()]

fig1, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True, figsize=(13, 8), constrained_layout=True)
ax1a.scatter(days, durations)
ax1b.plot(usage_days, usage_cumul)
    
ax1a.set_title("LNO operating hours per day")
ax1a.set_ylabel("Operating hours per Earth day")
ax1a.grid()


ax1b.set_title("LNO cumulative operating hours")
ax1b.set_ylabel("Cumulative operating hours")
ax1b.grid()
ax1b.axhline(y=12000, color="k", linestyle="--")
ax1b.text(usage_days[0], 12500.0, "LNO detector mean time to failure")

fig1.savefig("LNO_operating_hours.png")


#extrapolate in time
reductions = [0.0, 0.25, 0.5]

for reduction in reductions:

    hours_per_day = 1100.0 / 365.0 * (1. - reduction)
    
    
    current_dt = usage_days[-1]
    current_usage = usage_cumul[-1]
    end_dt = datetime(year=2031, month=1, day=1)
    
    remaining_days = (end_dt - current_dt).days
    
    expected_usage = remaining_days * hours_per_day + current_usage
    ax1b.plot([current_dt, end_dt], [current_usage, expected_usage], label="%0.0f%% duty cycle reduction" %(reduction*100.0))
    
    days_to_mttf = (12000. - current_usage) / hours_per_day
    mttf_dt = current_dt + timedelta(days=days_to_mttf)
    
    ax1b.axvline(x=mttf_dt, color="k", linestyle="--")
    ax1b.text(mttf_dt, 12500.0, datetime.strftime(mttf_dt, "%Y-%m-%d"))
    
ax1b.legend()

fig1.savefig("LNO_operating_hours_extrapolated.png")
