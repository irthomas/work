# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:37:50 2020

@author: iant

PLOT HIGHEST ELLIPSOID ALTITUDE OF ALL INGRESS OCCULTATIONS
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

regex = re.compile("20[0-9][0-9][0-9][0-9][0-9][0-9]_.*_SO_A_I_134")
file_level = "hdf5_level_0p3k"

datetime_strings = []
max_alts = []
lats = []

hdf5_filepaths, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)
for hdf5_filepath, hdf5_filename in zip(hdf5_filepaths, hdf5_filenames):
    
    with h5py.File(hdf5_filepath, "r") as f:
        alts = f["Geometry/Point0/TangentAlt"][0, 0]
        latitudes_in = f["Geometry/Point0/Lat"][:, 0]
        
        datetime_in = f["Temperature/TemperatureDateTime"][...]
        
        if length(datetime_in) > 20:

            max_alts.append(alts)

            mid_point = int(len(datetime_in)/2)
            datetime_strings.append(datetime_in[mid_point].decode())

        
            mid_point = int(len(latitudes_in)/2)
            lats.append(latitudes_in[mid_point])

        
datetimes = [datetime.strptime(x, HDF5_DT_FORMAT) for x in datetime_strings]

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(18, 8), sharex=True)
ax1.scatter(datetimes, lats, label="Ingress")
ax1.set_ylabel("SO Midpoint Latitude")
ax1.legend()

ax2.plot(datetimes, max_alts, label="Ingress")
ax2.set_xlabel("Time")
ax2.set_ylabel("Starting Altitude")
ax2.legend()
ax2.grid()
       
plt.savefig("SO_starting_altitudes.png")
