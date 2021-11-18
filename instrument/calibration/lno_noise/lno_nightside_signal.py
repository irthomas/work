# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:43:57 2021

@author: iant

LNO NIGHTSIDE NADIR COUNTS

APPROX STD = 115 COUNTS AT -10C
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


d = {"dt":[], "t":[], "std":[]}


file_level = "hdf5_level_0p3a"

    
regex = re.compile("201804.._.*_0p3a_LNO_1_N_.*")
regex = re.compile("(2018.[12567890]..|2019....|2020....|2021....)_.*_0p3a_LNO_1_N_.*")


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level, open_files=False, silent=True)


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
        
        sza = f["Geometry/Point0/IncidenceAngle"][:, 0]
        
        indices = np.where(sza > 120)[0]

        y_all = f["Science/Y"][indices, :]
        t = f["Channel/MeasurementTemperature"][0]
        
    if len(y_all) == 0:
        continue
    
    y_all[:, 85] = np.nan
    
    
    y_means = np.nanmean(y_all, axis=1)
    y_stds = np.nanstd(y_all, axis=1)
    
    d["dt"].append(obs_datetime)
    d["t"].append(t)
    d["std"].append(np.mean(y_stds))
    

plt.figure()
plt.xlabel="Temperature"
plt.ylabel="Std"
plt.scatter(d["t"], d["std"])

# plt.figure()
# plt.xlabel="Time"
# plt.ylabel="Std"
# plt.scatter(d["dt"], d["std"])

# plt.figure()
# plt.xlabel="Time"
# plt.ylabel="Temperature"
# plt.scatter(d["dt"], d["t"])



"""check dayside counts order 168"""

file_level = "hdf5_level_0p3a"

    
# regex = re.compile("201804.._.*_0p3a_LNO_1_N_.*")
regex = re.compile("(2018.[12567890]..|2019....|2020....|2021....)_.*_0p3a_LNO_1_D_167")


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level, open_files=False, silent=True)


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
        
        sza = f["Geometry/Point0/IncidenceAngle"][:, 0]
        
        indices = np.where(sza < 20)[0]
                
        y_all = f["Science/Y"][indices, :]
        t = f["Channel/MeasurementTemperature"][0]
        
        n_sub = f.attrs["NSubdomains"]
        
    if len(y_all) == 0:
        continue
    
    # y_all[:, 85] = np.nan
    
    plt.figure()
    plt.plot(y_all.T)
    plt.title(n_sub)
    # stop()
    
    y_means = np.nanmean(y_all, axis=1)
    y_stds = np.nanstd(y_all, axis=1)
    
    d["dt"].append(obs_datetime)
    d["t"].append(t)
    d["std"].append(np.mean(y_stds))
    
    if file_index > 10:
        stop()
        

