# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:23:09 2021

@author: iant

CHECK TEMPERATURES
"""
import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours


<<<<<<< HEAD
regex = re.compile("201807.._.*_SO_._[IE]_190")
file_level = "hdf5_level_1p0a"
chosen_bin = 3
=======
regex = re.compile(".*_SO_._C")
file_level = "hdf5_level_0p2a"
>>>>>>> 8dbf6066d746e2c2eb9ff7727853543853e43c4a


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
colours = get_colours(len(hdf5_filenames))

fig, ax = plt.subplots()

for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
    interpolated_ts = hdf5_file["Channel/InterpolatedTemperature"][...]
<<<<<<< HEAD
    bins = hdf5_file["Science/Bins"][:, 0]
    unique_bins = sorted(list(set(bins)))
    
    bin_indices = np.where(bins == unique_bins[chosen_bin])[0]

    ax.plot(interpolated_ts[bin_indices], label=hdf5_filename, color=colours[file_index])
=======
    ax.plot(interpolated_ts, label=hdf5_filename, color=colours[file_index])
>>>>>>> 8dbf6066d746e2c2eb9ff7727853543853e43c4a
    
ax.legend()


