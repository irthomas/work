# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:12:16 2021

@author: iant

GET LIST OF UVIS TYPE N HDF5 FILENAMES, THEN COMPARE TO SQL DUMP TO SEE IF LNO IS OPERATING

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


file_level = "hdf5_level_1p0a"
regex = re.compile("20......_.*_1p0a_UVIS_N")

SQL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

#first read in SQL dump
dump_csv = "nightside_nadirs.csv"
data_in = np.fromregex(dump_csv, "(....-..-..\s..:..:..),\d*,\w*\s\w*,(.*),......", dtype=str)

uvis_only_times = [str(t) for t,channel in data_in if channel == "UVIS"]
uvis_lno_times = [str(t) for t,channel in data_in if channel == '"LNO, UVIS"']

uvis_only_dts = [datetime.strptime(i, SQL_TIME_FORMAT) for i in uvis_only_times]
uvis_lno_dts = [datetime.strptime(i, SQL_TIME_FORMAT) for i in uvis_lno_times]


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level, open_files=False, silent=True)

lines = []

for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    # if np.mod(file_index, 100) == 0:
    #     print(file_index, hdf5_filename)
        
        
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]
    hour = hdf5_filename[9:11]
    minute = hdf5_filename[11:13]
    second = hdf5_filename[13:15]
    obs_datetime = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))

    uvis_only = nearest(uvis_only_dts, obs_datetime)
    uvis_lno = nearest(uvis_lno_dts, obs_datetime)
    
    uvis_only_delta = (obs_datetime - uvis_only).total_seconds()
    uvis_lno_delta = (obs_datetime - uvis_lno).total_seconds()
    
    # print(hdf5_filename)
    # print(uvis_only_delta)
    # print(uvis_lno_delta)
    
    if uvis_only_delta < 100:
        lines.append("%s,UVIS only" %hdf5_filename)
    elif uvis_lno_delta < 100:
        lines.append("%s,UVIS+LNO" %hdf5_filename)
    else:
        print("Error:", hdf5_filename)
        
with open("nomad_nightsides.csv", "w") as f:
    for line in lines:
        f.write("%s\n" %line)