# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:46:58 2024

@author: iant

CHECK VARIATIONS IN LNO SPECTRA FOR FABRIZIO AND EMILIANO

SELECT 0.1A OR 1.0A FOR CHECKING


"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths


from instrument.nomad_lno_instrument_v02 import nu_mp
from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze


LEVEL = "0p1a"
# LEVEL = "1p0a"

DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

BAD_PX_ITERS = 5
BAD_PX_STD = 5.0

zp1a = LEVEL == "0p1a"

if zp1a:
    orders = [""]
else:
    orders = ["_DF_186", "_DP_187", "_DP_193", "_DF_194", "_DF_198", "_DF_199"]

h5_prefix = "20220130_045445"


def make_blaze(params, channel, order, ncols):

    [eff_n_px, t, scaler, offset] = params
    # print(params)

    px_nus = nu_mp(order, np.arange(ncols)*(eff_n_px/ncols), t)
    blaze = asymmetric_blaze(channel, order, px_nus)*scaler + offset
    return blaze


def min_blaze(params, args):
    # print(params)
    # print(args)

    [spectrum, channel, order] = args

    ncols = len(spectrum)
    blaze = make_blaze(params, channel, order, ncols)
    chisq_px = (spectrum - blaze)**2
    chisq = np.sum(chisq_px)
    return chisq


def fit_blaze_spectrum(spectrum, first_guess, channel, order):
    res = minimize(min_blaze, first_guess, args=[spectrum, channel, order])

    ncols = len(spectrum)
    # print(res.x)
    # plt.figure()
    # plt.plot(spectrum)
    blaze = make_blaze([*res.x[0:-1], 0.0], channel, order, ncols)
    # plt.plot(blaze)
    return blaze


d = {}
for order in orders:

    h5 = "%s_%s_LNO_1%s" % (h5_prefix, LEVEL, order)
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

    if zp1a:
        d[""] = {
            "raw": h5f["Science/Y"][...],
        }
    else:

        d[order] = {
            "raw": h5f["Science/YUnmodified"][...],
            "reff": h5f["Science/YReflectanceFactor"][...],
            "et": np.mean(h5f["Geometry/ObservationEphemerisTime"][...], axis=1),
            "inc": h5f["Geometry/MeanIncidenceAngle"][...],
        }

    # k = "MOTOR_POWER_DAC_CODE_LNO"
    # plt.figure()
    # plt.title("LNO cryocooler output code")
    # plt.xlabel("HSK packet index")
    # plt.ylabel("DAC code value (higher = more power)")
    # plt.plot(h5f["Housekeeping"][k][...])

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

    if zp1a:

        y_order = y[::6, :, :]  # every 6th is of same order

        fitted_spectra = []
        for i in range(76):
            spectrum = y_order[i, 0, :]

            x = np.arange(len(spectrum))

            good_ixs = np.ones_like(x, dtype=bool)

            for iteration in range(BAD_PX_ITERS):
                polyfit = np.polyfit(x[~np.isnan(spectrum)], spectrum[~np.isnan(spectrum)], 3)
                polyval = np.polyval(polyfit, x)
                # plt.plot(spectrum, label="%i" % iteration, color="C%i" % iteration)
                # plt.plot(polyval, "--", color="C%i" % iteration)

                diff = np.abs(spectrum - polyval)
                bad_ix = np.nanargmax(diff)
                diff_std = np.nanstd(diff)
                # print(bad_ix, diff[bad_ix]/diff_std, BAD_PX_STD)
                if diff[bad_ix]/diff_std > BAD_PX_STD:
                    spectrum[bad_ix] = np.nan
                else:
                    break

            # replace nans by polynomial value
            polyfit = np.polyfit(x[~np.isnan(spectrum)], spectrum[~np.isnan(spectrum)], 3)
            polyval = np.polyval(polyfit, x)
            bad_ixs = np.argwhere(np.isnan(spectrum))
            for bad_ix in bad_ixs:
                spectrum[bad_ix] = polyval[bad_ix]
            # plt.plot(spectrum, color="k")

            # plt.legend()

            # now fit the blaze function
            first_guess = [320., -5.0, np.max(spectrum)*0.95, np.mean(spectrum[0:50])]
            blaze = fit_blaze_spectrum(spectrum, first_guess, "lno", 186)

            fitted_spectra.append(blaze)

        plt.plot(np.asarray(fitted_spectra).T)
        stop()

        peak_mean = np.mean(y_order[:, :, 180:220], axis=2)
        plt.figure(figsize=(15, 6))
        plt.imshow(peak_mean.T, aspect="auto")

        left_mean = np.mean(y_order[:, :, 0:50], axis=2)
        plt.figure(figsize=(15, 6))
        plt.imshow(left_mean.T, aspect="auto")

        # plot each bin
        for i in range(y_order.shape[1]):
            plt.figure()
            plt.plot(y_order[:, i, :].T)

            mean = np.mean(y_order[:, i, :], axis=0)
            std = np.std(y_order[:, i, :], axis=0)

            plt.plot(mean, "k--")
            plt.plot(std, "k.-")

    else:
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

# fig1, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.set_title(h5)
# ax1.set_ylabel("Raw counts, fitted to the peak of the signal")
# ax2.set_ylabel("Solar incidence angle (dotted lines)")
# ax1.set_xlabel("Ephemeris time")
# ax1.grid()

# for order in list(d.keys()):
#     ax1.plot(d[order]["et"], d[order]["peak_val"], "o-", label="Order %s" % order, alpha=0.7)
#     ax2.plot(d[order]["et"], d[order]["inc"], ":")
#     # plt.plot(np.arange(y.shape[0]), d[order]["left_val"], "o-")
#     # plt.plot(np.arange(y.shape[0]), d[order]["right_val"], "o-")
# ax1.legend()
