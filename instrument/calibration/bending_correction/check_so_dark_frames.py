# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:54:49 2025

@author: iant

CHECK BG SPECTRA IN L1.0A FILES
CHECK CURVATURE VS D TR D Z
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist, open_hdf5_file

file_level = "hdf5_level_0p3k"
path = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5")
path = r"D:\DATA\hdf5"

regex = re.compile("201804.._......_...._SO_A_.*")

h5fs, h5s, _ = make_filelist(regex, file_level, open_files=True, path=path)


h5fs = h5fs[0:10]

# get background data

# if "d" not in globals():
if True:
    d = {}
    for h5f, h5 in zip(h5fs, h5s):

        prefix = h5[0:15]

        if prefix not in d.keys():
            sbsf = h5f["Channel/BackgroundSubtraction"][0]

            if sbsf == 0:

                bins = h5f["Science/Bins"][:, 0]
                y = h5f["Science/Y"][...]
                y_mean = np.mean(y[:, 160:240], axis=1)
                bg = h5f["Science/BackgroundY"][...]

                alts = h5f["Geometry/Point0/TangentAltAreoid"][:, 0]

                unique_bins = sorted(list(set(bins)))

                d[prefix] = {}
                for unique_bin in unique_bins:
                    ixs = np.where((bins == unique_bin) & (alts > -998.))[0]

                    d[prefix][unique_bin] = {"y": y[ixs, :], "y_mean": y_mean[ixs], "bg": bg[ixs, :], "alts": alts[ixs]}


for h5 in list(d.keys())[0:4]:

    plt.figure()
    for bin_ in list(d[h5].keys()):

        alts = d[h5][bin_]["alts"]
        bg_mean = np.mean(d[h5][bin_]["bg"][:, 160:240], axis=1)
        y_mean = d[h5][bin_]["y_mean"]
        # plt.plot(d[h5][bin_]["y_mean"], label=h5)

        # fig1, ax1 = plt.subplots()
        # ax2 = plt.twinx()
        # ax1.plot(alts, bg_mean, label=h5)
        # ax2.plot(alts, y_mean, label=h5)

        # plt.scatter(y_mean, bg_mean, label=h5)

        ixs_toa = np.where(alts > 200.0)[0]
        ixs_grnd = np.where(y_mean < 59000)[0]

        bin_mean_toa = np.mean(d[h5][bin_]["bg"][ixs_toa, :], axis=0)
        bin_mean_grnd = np.mean(d[h5][bin_]["bg"][ixs_grnd, :], axis=0)

        bin_mean_diff = bin_mean_toa - bin_mean_grnd

        plt.plot(bin_mean_diff, label="Det Top %i" % bin_)

        if bin_ == 132:
            plt.plot(bin_mean_grnd * (5000 / 58000))

    plt.title(h5)
    plt.legend()

    # stop()

# plt.legend()
