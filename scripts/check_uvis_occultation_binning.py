# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:28:41 2022

@author: iant
"""

import re
import numpy as np
import h5py

import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours

from tools.file.hdf5_filename_to_datetime import hdf5_filename_to_datetime

regex = re.compile(".*_1p0a_UVIS_[IE]")
file_level = "hdf5_level_1p0a"


h5s = []
h5_names = []

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, open_files=False)

dts = []
b = []
for i, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
    if np.mod(i, 500) == 0:
        print(i, hdf5_filename)

    if np.mod(i, 2) == 0:
        with h5py.File(hdf5_file, "r") as f:
        
            binning = f["Channel/HorizontalAndCombinedBinningSize"][...]
            
            b.append(np.mean(binning))
            
            dts.append(hdf5_filename_to_datetime(hdf5_filename))

plt.figure(figsize=(14, 4), constrained_layout=True)
plt.title("%s spectral binning" %regex.pattern)            
plt.scatter(dts, b)
plt.xlabel("Observation date")
plt.ylabel("Binning")
plt.grid()
plt.savefig("UVIS_spectral_binning.png")
