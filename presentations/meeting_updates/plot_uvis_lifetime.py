# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:17:15 2026

@author: iant
"""

import re
import numpy as np
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import platform


from tools.file.hdf5_functions import make_filelist
# from tools.spectra.running_mean import running_mean_1d
from tools.general.progress_bar import progress


channel = "UVIS"

# reload data even if already loaded?
# get_data = True
get_data = False

# save figures?
save_fig = True
# save_fig = False

# overwrite local settings - can't use calibrated 1.0a files here
if platform.system() == "Windows":
    ROOT_DIR = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
else:
    ROOT_DIR = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"


def get_daily_usage(channel, regex, file_level):

    h5_filepaths, h5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False, path=ROOT_DIR, skipbad=False)

    h5_prefixes = []
    dates = {}

    channel_mttf = {"so": 11000.0, "lno": 12000.0, "uvis": 26000.0}[channel.lower()]

    for h5_ix, (h5_filepath, h5) in enumerate(progress(list(zip(h5_filepaths, h5_filenames)))):

        if not os.path.exists(h5_filepath):
            print("Error: file %s doesn't not exist" % h5_filepath)
            continue

        h5_f = h5py.File(h5_filepath, "r")

        datatype = h5_f["Channel/ReverseFlagAndDataTypeFlagRegister"][...]

        datatype[datatype != 4] = 0
        diff = np.diff(datatype)

        stepper_movements = float(sum(diff != 0)) / 4.0  # one dark and back and occultation and back = 1 movement

        print(h5, stepper_movements)

        year = h5[0:4]
        month = h5[4:6]
        day = h5[6:8]

        date = datetime(year=int(year), month=int(month), day=int(day))

        if date not in dates.keys():
            dates[date] = 0
        dates[date] += stepper_movements
    return dates


if channel == "UVIS":
    file_level = "hdf5_level_0p1a"
    regex = re.compile("........_.*_UVIS")


# load data only if not in memory
if channel == "UVIS":
    if "dates_uvis" not in globals() or get_data:
        dates_uvis = get_daily_usage(channel, regex, file_level)


channel_mttf = {"so": 11000.0, "lno": 12000.0, "uvis": 26000.0}[channel.lower()]
if channel == "UVIS":
    dates = dates_uvis


# fill in missing dates where no observations where made
sdate = sorted(list(dates.keys()))[0]
edate = sorted(list(dates.keys()))[-1]
dates_all = {}

for x in range((edate - sdate + timedelta(days=1)).days):
    dt = sdate + timedelta(days=x)
    if dt in dates.keys():
        dates_all[dt] = dates[dt]
    else:
        dates_all[dt] = 0.0

days = [s for s in dates_all.keys()]
# TODO: why divide by 2 again?
stepper_movements = np.asarray(list(dates_all.values()), dtype=float)

# cumulative
usage = {}
cumul = 0
# TODO: why divide by 2 again?
for date in dates_all.keys():
    cumul += dates_all[date]

    usage[date] = cumul

usage_days = [s for s in usage.keys()]
usage_cumul = usage.values()


if channel.lower() == "uvis":

    fig1, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True, figsize=(13, 8), constrained_layout=True)
    ax1a.scatter(days, stepper_movements)
    ax1b.plot(usage_days, usage_cumul)

    ax1a.set_title("%s stepper motor movements" % (channel.upper()))
    ax1a.set_ylabel("Stepper motor movements per Earth day")
    ax1a.grid()

    ax1b.set_title("%s cumulative stepper motor movements" % (channel.upper()))
    ax1b.set_ylabel("Cumulative stepper motor movements")
    ax1b.grid()
    ax1b.axhline(y=channel_mttf, color="k", linestyle="--")
    ax1b.text(usage_days[0], channel_mttf + 500, "%s tested stepper motor movements" % (channel.upper()))
