# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:07:13 2024

@author: iant


ANALYSE HDF5 FILE
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths


Y_FIELDS = {
    ("1p0a", "UVIS", "O"): "Y",
    # ("1p0a", "UVIS", "O"): "YOffsetCorr", #new nightside nadir/limb offset correction derived from 0.3c data
}


# h5 = "20240301_203034_1p0a_UVIS_O"
# h5 = "20240312_154849_1p0a_UVIS_O"
h5 = "20240313_135706_1p0a_UVIS_O"

h5_split = h5.split("_")
file_level = h5_split[2]
channel = h5_split[3]

if len(h5_split) == 7:
    obs_type = h5_split[-2]
    order = h5_split[-1]
else:
    obs_type = h5_split[-1]

h5_dirpath = os.path.join(paths["DATA_DIRECTORY"], "hdf5_level_%s" % file_level, h5[0:4], h5[4:6], h5[6:8])

h5f = open_hdf5_file(h5)
# h5f = open_hdf5_file(h5, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")


# find files in same day
h5_paths_same_dt = glob.glob(h5_dirpath + os.sep + h5[0:15] + "*.h5")

h5_names_same_dt = [os.path.splitext(os.path.basename(s))[0] for s in h5_paths_same_dt]

n_orders = len(h5_names_same_dt)

if file_level == "1p0a":

    geom_d = {}
    for geom_param in h5f["Geometry/Point0"].keys():
        geom_d[geom_param] = np.mean(h5f["Geometry/Point0/%s" % geom_param][...], axis=1)

    n_spectra = geom_d[geom_param].shape[0]

    inst_d = {}
    for inst_param in h5f["Channel"].keys():
        inst_d[inst_param] = h5f["Channel/%s" % inst_param][...]

    hsk_d = {}
    for hsk_param in h5f["Housekeeping"].keys():
        hsk_d[hsk_param] = h5f["Housekeeping/%s" % hsk_param][...]

    y_fieldname = Y_FIELDS[(file_level, channel, obs_type)]
    y = h5f["Science/%s" % y_fieldname][...]
    y_mean = np.mean(y[:, 160:240], axis=1)
    y_std = np.std(y[:, 160:240], axis=1)
    # sza = geom_d["SunSZA"]

if channel == "UVIS" and obs_type in ["O"]:

    fig1, ax1 = plt.subplots()
    ax1.set_title("%s Geometry" % h5)
    for key, value in geom_d.items():
        if np.min(value) < 1000.0:  # ignore surface alt etc.
            ax1.plot(value, label=key)

    ax1.legend()
    ax1.grid()

    fig2, ax2 = plt.subplots()
    ax2.set_title("%s HSK" % h5)

    for key, value in hsk_d.items():
        if key not in ["DateTime"]:
            ax2.plot(value[1:], label=key)

    ax2.legend()
    ax2.grid()

    for inst_param in ["VStart", "VEnd", "HStart", "HEnd", "HorizontalAndCombinedBinningSize"]:
        if np.all(inst_d[inst_param] == inst_d[inst_param][0]):
            print(inst_param, "=", inst_d[inst_param][0])

    # fig3, ax3 = plt.subplots()
    # ax3.set_title("%s Temperature" % h5)
    # ax3.plot(inst_d["InterpolatedTemperature"])
    # ax3.grid()

    fig4, ax4 = plt.subplots()
    ax4.set_title("%s Mean/std of signal" % h5)
    ax4.plot(y_mean, label="Mean signal")
    ax4.plot(y_std, label="Mean signal")
    ax4.grid()

    fig5, ax5 = plt.subplots()
    ax5.set_title("%s Spectra" % h5)
    ax5.plot(y.T)
    ax5.grid()
