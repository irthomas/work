# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:40:29 2022

@author: iant

READ LNO PHOBOS DATA AFTER FEB 2022 PATCH
PIPELINE UPDATED TO CONVERT 3D ARRAYS TO 2D
"""

import re
import matplotlib.pyplot as plt
import numpy as np


from tools.file.hdf5_functions import make_filelist

diffraction_order = 169

# regex = re.compile("20181029_203917_.*_UVIS_E")
# regex = re.compile("20180622_......_.*_UVIS_E")
regex = re.compile("20220304_......_.*_LNO_._P_%i" %diffraction_order)

file_level = "hdf5_level_0p3a"
# file_level = "hdf5_level_0p1d"



hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

h5 = hdf5_files[0]
h5_filename = hdf5_filenames[0]



y_all = np.asfarray(h5["Science/Y"][...])
bins = h5["Science/Bins"][:, 0]

h5.close()




unique_bins = sorted(list(set(bins)))

bin_d = {}
for unique_bin in unique_bins:
    idx = np.where(bins == unique_bin)[0]
    
    bin_d[unique_bin] = {}
    bin_d[unique_bin]["raw"] = y_all[idx, :]
    bin_d[unique_bin]["mean_px"] = np.mean(y_all[idx, :], axis=1)
    bin_d[unique_bin]["mean_all"] = np.mean(y_all[idx, :])


# fig = plt.figure(figsize=(16,10), constrained_layout=True)
# im = plt.imshow(y.T, aspect="auto")
# plt.colorbar(im)

plt.figure()
plt.plot([bin_d[i]["mean_all"] for i in bin_d.keys()])

grid_2d = np.asfarray([bin_d[i]["mean_px"] for i in bin_d.keys()])

plt.figure(figsize=(10,5), constrained_layout=True)
plt.title("LNO phobos observation after patching: %s" %h5_filename)
im = plt.imshow(grid_2d, extent=(0, grid_2d.shape[1], unique_bins[0], unique_bins[-1]+2), aspect="auto", origin="lower")
cbar = plt.colorbar(im)
cbar.set_label("Mean signal on each detector bin", rotation=270, labelpad=10)

plt.xlabel("Frame Number")
plt.ylabel("Detector row")
plt.savefig("%s_phobos_spectrum.png")

ill_bins = [146, 148, 150]

spectra = np.asfarray([np.mean(bin_d[i]["raw"], axis=0) for i in ill_bins])

plt.figure()
plt.plot(spectra.T)