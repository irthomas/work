# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:38:55 2022

@author: iant

SO AND LNO LIFETIMES
"""


import re
import numpy as np
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import platform


from tools.file.hdf5_functions import make_filelist
# from tools.spectra.running_mean import running_mean_1d
from tools.general.progress_bar import progress


channel = "LNO"
# channel = "SO"

#reload data even if already loaded?
# get_data = True
get_data = False

#save figures?
save_fig = True
# save_fig = False


#when calculating projected future operating time, how many days in the past to consider?
N_DAYS_PROJECT_BACK = 60
# N_DAYS_PROJECT_BACK = 30


HDF5_DT_FORMAT = "%Y %b %d %H:%M:%S.%f"

if channel == "SO":
    file_level = "hdf5_level_0p3a"
    # regex = re.compile("201812(?:0[0-9]|1[0-9]|2[0-8])_.*_SO_.*")
    regex = re.compile("20......_.*_SO_.*")

if channel == "LNO":
    #why 0.2A here?
    file_level = "hdf5_level_0p2a"
    regex = re.compile("20......_.*_LNO_.")

    

#overwrite local settings - can't use calibrated 1.0a files here
if platform.system() == "Windows":
    ROOT_DIR = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
else:
    ROOT_DIR = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"




channel_mttf = {"so":11000.0, "lno":12000.0}[channel.lower()]

def get_daily_usage(channel, regex, file_level):
    h5_filepaths, h5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False, path=ROOT_DIR)
    
    
    h5_prefixes = []
    dates = {}
    
    for h5_ix, (h5_filepath, h5) in enumerate(progress(list(zip(h5_filepaths, h5_filenames)))):
        
        h5_prefix = "%s" %(h5[0:15])
        
        #if h5_prefix already found, skip file
        if h5_prefix not in h5_prefixes:
            
            #all H, L, 1, 2 etc can be ignored as duration time taken from TC20
            # if "_SO_H_" in h5 or "_SO_L_" in h5:
            #     print("Skipping: ", h5)
            #     continue

            h5_prefixes.append(h5_prefix)
            # print(h5)
        
            h5_f = h5py.File(h5_filepath, "r")
            
            """assume 10 minutes"""
            # dt_str_start = h5_f["Geometry/ObservationDateTime"][0,0].decode()
            # dt_str_end = h5_f["Geometry/ObservationDateTime"][-1, -1].decode()
            # dt_start = datetime.strptime(dt_str_start, HDF5_DT_FORMAT)
            # dt_end = datetime.strptime(dt_str_end, HDF5_DT_FORMAT)
            # duration = (dt_end - dt_start).total_seconds() + 600.0
            
            """get actual duration"""
            duration = float(h5_f["Telecommand20/%sDurationTime" %channel.upper()][...]) #from switch on to switch off

        
            year = h5[0:4]
            month = h5[4:6]
            day = h5[6:8]
            
            date = datetime(year=int(year), month=int(month), day=int(day))
            
            if date not in dates.keys():
                dates[date] = 0
            dates[date] += duration
    return dates

# load data only if not in memory
if "dates" not in globals() or get_data:
    dates = get_daily_usage(channel, regex, file_level)
        
    
    
#fill in missing dates where no observations where made

sdate = sorted(list(dates.keys()))[0]
edate = sorted(list(dates.keys()))[-1]
dates_all = {}

for x in range((edate-sdate+timedelta(days=1)).days):
    dt = sdate+timedelta(days=x)
    if dt in dates.keys():
        dates_all[dt] = dates[dt]
    else:
        dates_all[dt] = 0.0
        



days = [s for s in dates_all.keys()]
durations = [f/3600.0 for f in dates_all.values()]


#cumulative
usage = {}
cumul = 0
for date in dates_all.keys():
    cumul += dates_all[date]
    
    usage[date] = cumul

usage_days = [s for s in usage.keys()]
usage_cumul = [f/3600.0 for f in usage.values()] #convert to hours

fig1, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True, figsize=(13, 8), constrained_layout=True)
ax1a.scatter(days, durations)
ax1b.plot(usage_days, usage_cumul)
    
ax1a.set_title("%s operating hours per day" %(channel.upper()))
ax1a.set_ylabel("Operating hours per Earth day")
ax1a.grid()


ax1b.set_title("%s cumulative operating hours" %(channel.upper()))
ax1b.set_ylabel("Cumulative operating hours")
ax1b.grid()
ax1b.axhline(y=channel_mttf, color="k", linestyle="--")
ax1b.text(usage_days[0], channel_mttf + 500, "%s detector mean time to failure" %(channel.upper()))

# fig1.savefig("%s_operating_hours.png" %(channel.upper()))


#extrapolate in time
reductions = [0.0, 0.25]#, 0.5]

for reduction in reductions:

    hours_per_day = np.mean(durations[-N_DAYS_PROJECT_BACK:]) * (1.0 - reduction)
    
    
    current_dt = usage_days[-1]
    current_usage = usage_cumul[-1]
    end_dt = datetime(year=2040, month=1, day=1)
    
    remaining_days = (end_dt - current_dt).days
    
    expected_usage = remaining_days * hours_per_day + current_usage
    
    if reduction == 0.0:
        #extend line backwards by duration
        previous_days = (days[-1] - days[-N_DAYS_PROJECT_BACK]).days
        previous_usage = current_usage - previous_days * hours_per_day
        previous_dt = current_dt - timedelta(days=previous_days)
        
        ax1b.plot([previous_dt, current_dt, end_dt], [previous_usage, current_usage, expected_usage], label="%0.0f%% duty cycle reduction" %(reduction*100.0))
        
        ax1a.axhline(y=hours_per_day, color="k", linestyle="--")
        ax1a.text(current_dt + timedelta(days=30), hours_per_day + 0.25, "Mean operating hours over past %i days = %0.2f hours" %(N_DAYS_PROJECT_BACK, hours_per_day))
        
    else:
        ax1b.plot([current_dt, end_dt], [current_usage, expected_usage], label="%0.0f%% duty cycle reduction" %(reduction*100.0))
    
    days_to_mttf = (channel_mttf - current_usage) / hours_per_day
    mttf_dt = current_dt + timedelta(days=days_to_mttf)
    
    ax1b.axvline(x=mttf_dt, color="k", linestyle="--")
    ax1b.text(mttf_dt, channel_mttf + 500, datetime.strftime(mttf_dt, "%Y-%m-%d"))
    

ax1b.scatter(usage_days[-1], usage_cumul[-1])

ax1b.legend(loc="center left")
ax1b.set_ylim((0, 13000))
ax1b.set_xlim(right=end_dt)

if save_fig:
    fig1.savefig("%s_operating_hours_extrapolated.png" %(channel.upper()))
