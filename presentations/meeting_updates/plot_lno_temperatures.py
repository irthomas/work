# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:24:09 2020

@author: iant

GET MID-OBS TEMPERATURE FROM LNO OBSERVATIONS AND PLOT TEMPERATURE EVOLUTION
"""


import matplotlib.pyplot as plt
import numpy as np
# import os
import re
import h5py
from datetime import datetime

from tools.file.hdf5_functions import make_filelist
from tools.spectra.savitzky_golay import savitzky_golay

from tools.general.length import length
        


HDF5_DT_FORMAT = "%Y %b %d %H:%M:%S.%f"

# year = "2020[0-9][0-9][0-9][0-9]"
year = "20[0-9][0-9][0-9][0-9][0-9][0-9]"

regex = re.compile("%s_.*_LNO_1_D.*" %year)
file_level = "hdf5_level_1p0a"

datetime_strings = []
temperatures = []

hdf5_filepaths, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)
for file_index, (hdf5_filepath, hdf5_filename) in enumerate(zip(hdf5_filepaths, hdf5_filenames)):

    if np.mod(file_index, 100) == 0:
        print("%i/%i" %(file_index, len(hdf5_filenames)), hdf5_filename)

    
    with h5py.File(hdf5_filepath, "r") as f:
        temperature_in = f["Temperature/NominalLNO"][...]
        
        datetime_in = f["Temperature/TemperatureDateTime"][...]
        
        if length(temperature_in) > 20:
        
            temperatures.append(temperature_in[6])
            temperatures.append(temperature_in[-6])
            temperatures.append(np.nan)
        
            datetime_strings.append(datetime_in[6].decode())
            datetime_strings.append(datetime_in[-6].decode())
            datetime_strings.append(datetime_in[-6].decode())
        
datetimes = [datetime.strptime(x, HDF5_DT_FORMAT) for x in datetime_strings]

temperatures = np.asfarray(temperatures)
datetimes = np.asarray(datetimes)

datetimes_sg = datetimes[~np.isnan(temperatures)]
sg = savitzky_golay(np.asfarray(temperatures[~np.isnan(temperatures)]), 499, 1)

datetimes_sg =list(datetimes_sg)


regex = re.compile("%s.*_SO_A_I_134" %year)
file_level = "hdf5_level_1p0a"

so_i_datetime_strings = []
so_i_lats = []

hdf5_filepaths, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)
for file_index, (hdf5_filepath, hdf5_filename) in enumerate(zip(hdf5_filepaths, hdf5_filenames)):

    if np.mod(file_index, 100) == 0:
        print("%i/%i" %(file_index, len(hdf5_filenames)), hdf5_filename)
    
    with h5py.File(hdf5_filepath, "r") as f:
        latitudes_in = f["Geometry/Point0/Lat"][:, 0]
        
        datetime_in = f["Temperature/TemperatureDateTime"][...]
        
        mid_point = int(len(latitudes_in)/2)
        so_i_lats.append(latitudes_in[mid_point])

        mid_point = int(len(datetime_in)/2)
        so_i_datetime_strings.append(datetime_in[mid_point].decode())
        
so_i_datetimes = [datetime.strptime(x, HDF5_DT_FORMAT) for x in so_i_datetime_strings]




regex = re.compile("%s.*_SO_A_E_134" %year)
file_level = "hdf5_level_1p0a"

so_e_datetime_strings = []
so_e_lats = []

hdf5_filepaths, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)
for file_index, (hdf5_filepath, hdf5_filename) in enumerate(zip(hdf5_filepaths, hdf5_filenames)):

    if np.mod(file_index, 100) == 0:
        print("%i/%i" %(file_index, len(hdf5_filenames)), hdf5_filename)
    
    with h5py.File(hdf5_filepath, "r") as f:
        latitudes_in = f["Geometry/Point0/Lat"][:, 0]
        
        datetime_in = f["Temperature/TemperatureDateTime"][...]
        
        mid_point = int(len(latitudes_in)/2)
        so_e_lats.append(latitudes_in[mid_point])

        mid_point = int(len(datetime_in)/2)
        so_e_datetime_strings.append(datetime_in[mid_point].decode())
        
so_e_datetimes = [datetime.strptime(x, HDF5_DT_FORMAT) for x in so_e_datetime_strings]




#set up subplots
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 1)
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)

ax1a.scatter(so_i_datetimes, so_i_lats, label="Ingress")
ax1a.scatter(so_e_datetimes, so_e_lats, label="Egress")
ax1a.set_ylabel("SO Lats")
ax1a.legend()
ax1a.grid()

ax1a.set_title("Solar Occultation Latitudes")
ax1b.set_title("LNO Temperatures")

ax1b.plot(datetimes, temperatures, label="LNO Temperature")
ax1b.plot(datetimes_sg, sg, label="Rolling Mean")
ax1b.set_xlabel("Observation Date")
ax1b.set_ylabel("LNO Temperature")
ax1b.legend()
ax1b.grid()
       
plt.savefig("LNO_temperature_monitoring.png")
