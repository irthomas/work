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

regex = re.compile("2020[0-9][0-9][0-9][0-9]_.*_SO_A_I_134")
file_level = "hdf5_level_0p3k"

datetime_strings = []
max_alts = []

hdf5_filepaths, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)
for hdf5_filepath, hdf5_filename in zip(hdf5_filepaths, hdf5_filenames):
    
    with h5py.File(hdf5_filepath, "r") as f:
        alts = f["Geometry/Point0/TangentAlt"][:, 0]
        
        datetime_in = f["Temperature/TemperatureDateTime"][...]
        
        if length(datetime_in) > 20:

            max_alts.append(max(alts))

            mid_point = int(len(datetime_in)/2)
            datetime_strings.append(datetime_in[mid_point].decode())
        
datetimes = [datetime.strptime(x, HDF5_DT_FORMAT) for x in datetime_strings]

plt.figure()
plt.plot(datetimes, max_alts)