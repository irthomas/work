# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:32:54 2025

@author: iant

PLOT AND CHECK NEW LNO FITTED PEAKS VS OTHER SPECTRA
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


ROOT_PATH = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a_peakfitting"


h5_filepaths = glob.glob(ROOT_PATH + os.sep + "**" + os.sep + "*_LNO_1_D*.h5", recursive=True)

h5s = [os.path.basename(h5_filepath) for h5_filepath in h5_filepaths]

h5_prefixes = {}

for h5, h5_filepath in zip(h5s, h5_filepaths):
    prefix = h5[0:15]
    if prefix not in h5_prefixes.keys():
        h5_prefixes[prefix] = []

    h5_prefixes[prefix].append([h5, h5_filepath])

# list(set(sorted([ for h5 in h5s])))

d = {}

for h5_prefix in h5_prefixes.keys():

    d[h5_prefix] = {}

    for h5, h5_filepath in h5_prefixes[h5_prefix]:

        with h5py.File(h5_filepath, "r") as h5f:
            y_fp = h5f["Science/YPeakReflectanceFactor"][...]
            dy_fp = h5f["Science/YPeakReflectanceFactorError"][...]

            y = h5f["Science/YReflectanceFactor"][...]
            y_mean = np.nanmean(y[:, 160:240], axis=1)

            inc = np.mean(h5f["Geometry/Point0/IncidenceAngle"][...], axis=1)
            lat = np.mean(h5f["Geometry/Point0/Lat"][...], axis=1)

        order = int(h5.split(".")[0].split("_")[-1])

        d[h5_prefix][order] = {"lat": lat, "inc": inc, "y_fp": y_fp, "dy_fp": dy_fp, "y_mean": y_mean}


for h5_prefix in list(d.keys())[0:5]:

    # x = "inc"
    x = "lat"

    plt.figure()
    plt.title(h5_prefix)
    for order in d[h5_prefix]:
        plt.errorbar(d[h5_prefix][order][x], d[h5_prefix][order]["y_fp"], yerr=d[h5_prefix][order]["dy_fp"], label=order)
        plt.plot(d[h5_prefix][order][x], d[h5_prefix][order]["y_mean"], label="%i mean" % order)

    plt.legend()
