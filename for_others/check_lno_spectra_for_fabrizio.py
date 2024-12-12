# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:46:58 2024

@author: iant

CHECK VARIATIONS IN LNO SPECTRA FOR FABRIZIO AND EMILIANO


"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths


DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"
orders = ["DF_186", "DF_193", "DF_198", "DF_199", "DP_187", "DP_194"]

h5_prefix = "20220130_045445"

d = {}
for order in orders:

    h5 = "%s_1p0a_LNO_1_%s" % (h5_prefix, order)
    h5_split = h5.split("_")
    file_level = h5_split[2]
    channel = h5_split[3]

    if len(h5_split) == 7:
        obs_type = h5_split[-2]
        order = h5_split[-1]
    else:
        obs_type = h5_split[-1]

    # h5_dirpath = os.path.join(DATA_PATH, "hdf5_level_%s" % file_level, h5[0:4], h5[4:6], h5[6:8])

    h5f = open_hdf5_file(h5, path=DATA_PATH)

    d[order] = {
        "raw": h5f["Science/YUnmodified"][...],
        "reff": h5f["Science/YReflectanceFactor"][...],
        "et": np.mean(h5f["Geometry/ObservationEphemerisTime"][...], axis=1),
        "inc": h5f["Geometry/MeanIncidenceAngle"][...],
    }

    k = "MOTOR_POWER_DAC_CODE_LNO"
    plt.figure()
    plt.title("LNO cryocooler output code")
    plt.xlabel("HSK packet index")
    plt.ylabel("DAC code value (higher = more power)")
    plt.plot(h5f["Housekeeping"][k][...])

    peak_px_ix = 180
    n_points = 70
    x = np.arange(-n_points, n_points)
    x2 = np.arange(n_points)

    fitted = []
    peak_ixs = []
    peak_vals = []
    left_vals = []
    right_vals = []

    y = d[order]["raw"]

    for ix in np.arange(y.shape[0]):

        polyfit_peak = np.polyfit(x, y[ix, peak_px_ix - n_points:peak_px_ix + n_points], 3)
        polyval_peak = np.polyval(polyfit_peak, x)
        peak_ix = np.where(polyval_peak == np.max(polyval_peak))[0][0]
        peak_val = polyval_peak[peak_ix]
        fitted.append(polyval_peak)
        peak_ixs.append(peak_ix + peak_px_ix - n_points)
        peak_vals.append(peak_val)

        polyfit_left = np.polyfit(x2, y[ix, 0:n_points], 2)
        polyval_left = np.polyval(polyfit_left, x2)
        left_val = polyval_left[int(n_points/2)]
        left_vals.append(left_val)

        polyfit_right = np.polyfit(x2, y[ix, -n_points:], 2)
        polyval_right = np.polyval(polyfit_right, x2)
        right_val = polyval_right[int(n_points/2)]
        right_vals.append(right_val)

    d[order]["fitted"] = np.asarray(fitted)
    d[order]["peak_px"] = np.asarray(peak_ixs)
    d[order]["peak_val"] = np.asarray(peak_vals)
    d[order]["peak_val"] = np.asarray(peak_vals)
    d[order]["left_val"] = np.asarray(left_vals)
    d[order]["right_val"] = np.asarray(right_vals)

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title(h5)
ax1.set_ylabel("Raw counts, fitted to the peak of the signal")
ax2.set_ylabel("Solar incidence angle (dotted lines)")
ax1.set_xlabel("Ephemeris time")
ax1.grid()

for order in list(d.keys()):
    ax1.plot(d[order]["et"], d[order]["peak_val"], "o-", label="Order %s" % order, alpha=0.7)
    ax2.plot(d[order]["et"], d[order]["inc"], ":")
    # plt.plot(np.arange(y.shape[0]), d[order]["left_val"], "o-")
    # plt.plot(np.arange(y.shape[0]), d[order]["right_val"], "o-")
ax1.legend()
