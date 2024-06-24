# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:13:08 2023

@author: iant


STEP 2: ANALYSE THE DIAGONALLY-CORRECTED MINISCAN ARRAYS
DEFINE THE SOLAR LINES IN THE CORRECTED MINISCAN ARRAYS
MAKE A FILE CONTAINING THE FIT0 COEFFICIENT FOR EACH COLUMN I.E. THE BLAZE FUNCTION
SAVE IFIG FOR CHECKING LATER

CAN ALSO USE ITERATIVE BLAZEFIT FUNCTION

"""

import os
import h5py
from astropy.io import fits


import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal import savgol_filter
from lmfit import Model
# from scipy import integrate

# from instrument.nomad_so_instrument_v03 import aotf_peak_nu
# from instrument.nomad_lno_instrument_v02 import nu0_aotf

from tools.general.progress_bar import progress
from tools.plotting.save_load_figs import save_ifig

# from instrument.calibration.so_lno_2023.fit_absorption_miniscan_array import trap_absorption
from instrument.calibration.so_lno_2023.solar_line_dict import solar_line_dict

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

# channel = "so"
channel = "lno"

INPUT_FILE_TYPE = "fits"  # which file type should be loaded?
# INPUT_FILE_TYPE = "h5"


# save_ifigs = True
save_ifigs = False

# define what should be plotted
# fit coeffs is plotted separately
plot = ["uncorrected array", "residual array", "corrected array", "residual array 2", "corrected array 2", "residual spectra", "fit coeffs"]
# define the plotting axes, should be enough to include all above except fit coeffs
naxes = [2, 3]


# sinc_apriori = {"a":150000.0, "b":30000.0, "c":0.0, "d":0.5, "e":-8.0, "f":0.0}
sinc_apriori = {"a": 240000.0, "b": 50000.0, "c": 0.0175, "d": 7.0, "e": -30.0, "f": -7.0}


def sinefunction(x, a, b, c, d, e, f):
    """modified sine function for fitting to corrected miniscan columns"""
    return a + ((b + f * x) * np.sin(x*np.pi/180.0 + c*x + d)) + e*x


def index(list_, value):
    """get index of value in the list, or -1 if not in list"""
    try:
        ix = list_.index(value)
    except ValueError:
        return -1
    return ix


loop = 0
for h5_prefix, solar_line_data_all in solar_line_dict.items():  # loop through files
    channel = h5_prefix.split("-")[0].lower()

    # get data from miniscan file
    if INPUT_FILE_TYPE == "h5":
        with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" % h5_prefix), "r") as f:

            keys = list(f.keys())
            n_reps = len([i for i, key in enumerate(keys) if "array" in key])

            # aotfs = []
            # for i in range(n_reps):
            #     aotfs.append(f["aotf%02i" %i][...])

            arrs = []
            for i in range(n_reps):
                arrs.append(f["array%02i" % i][...])
    elif INPUT_FILE_TYPE == "fits":
        with fits.open(os.path.join(MINISCAN_PATH, channel, "%s.fits" % h5_prefix)) as hdul:
            keys = [i.name for i in hdul if i.name != "PRIMARY"]
            n_reps = len([i for i, key in enumerate(keys) if "ARRAY" in key])

            arrs = []
            for i in range(n_reps):
                arrs.append(hdul["ARRAY%02i" % i].data)

    aotf_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all]
    # blaze_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all if "blaze_rows" in solar_line_data.keys()][0]

    solar_line_data = aotf_solar_line_data[0]  # just take the first set of coeffs

    if len(plot) > 0:
        fig1, ax1 = plt.subplots(figsize=(14, 10), ncols=naxes[0], nrows=naxes[1], squeeze=0)  # , constrained_layout=True)
        if len(ax1) != 1:
            ax1 = ax1.flatten()

    # just do first array
    arr = arrs[0]
    # aotf = aotfs[0]

    print(loop, h5_prefix)
    loop += 1

    # HR_SCALER = int(arr.shape[1]/320)

    ix = index(plot, "uncorrected array")
    if ix > -1:
        im = ax1[ix].imshow(arr, aspect="auto")
        plt.colorbar(im, ax=ax1[ix])
        ax1[ix].set_title("Uncorrected miniscan array")

        # for d in aotf_solar_line_data:
        #     x = d["abs_region_cols"][0]
        #     y = d["abs_region_rows"][0]
        #     width = d["abs_region_cols"][1] - d["abs_region_cols"][0]
        #     if d["abs_region_rows"][1] == -1:
        #         height = arr.shape[0] - y
        #     rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        #     ax1[ix].add_patch(rect)

    fits = np.zeros((6, arr.shape[1]))
    residual = np.copy(arr)

    print("Performing fits 1")
    for i in progress(range(arr.shape[1])):
        y = arr[:, i]
        x = np.arange(len(y))

        smodel = Model(sinefunction)
        result = smodel.fit(y, x=x, a=sinc_apriori["a"], b=sinc_apriori["b"], c=sinc_apriori["c"],
                            d=sinc_apriori["d"], e=sinc_apriori["e"], f=sinc_apriori["f"])
        # print(result.best_values)

        yfit = result.best_fit
        # print(result.fit_report())
        fits[:, i] = np.array([v for v in result.best_values.values()])
        residual[:, i] /= yfit

        # plot centre column
        # if i == int(arr.shape[1]/2):
        #     plt.figure()
        #     plt.title("Fit iteration 1")
        #     plt.plot(x, y)
        #     plt.plot(x, yfit)
        #     stop()

    if "fit coeffs" in plot:
        fig2, axes2 = plt.subplots(figsize=(18, 5), ncols=5)
        for j in range(5):
            axes2[j].set_title("Fit coefficients %i" % j)
            axes2[j].plot(fits[j, :], label="1st iteration")

    ix = index(plot, "residual array")
    if ix > -1:
        vmax = np.min((np.nanmax(residual), 1.2))
        vmin = np.max((np.nanmin(residual), 0.8))
        im = ax1[ix].imshow(residual, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax1[ix])
        ax1[ix].set_title("Corrected 1 miniscan residuals")

    cutoff = solar_line_data["cutoffs"][0]

    arr2 = np.copy(arr)
    arr2[residual < cutoff] = np.nan

    ix = index(plot, "corrected array")
    if ix > -1:
        im = ax1[ix].imshow(arr2, aspect="auto")
        plt.colorbar(im, ax=ax1[ix])
        ax1[ix].set_title("Corrected 1 miniscan array cutoff=%0.3f" % cutoff)

    fits2 = np.zeros((6, arr.shape[1]))
    residual2 = np.copy(arr)

    print("Performing fits 2")
    for i in progress(range(arr2.shape[1])):
        y = arr2[:, i]
        x = np.arange(len(y))

        smodel = Model(sinefunction)
        result = smodel.fit(y, x=x, nan_policy="omit", a=sinc_apriori["a"], b=sinc_apriori["b"],
                            c=sinc_apriori["c"], d=sinc_apriori["d"], e=sinc_apriori["e"], f=sinc_apriori["f"])

        yfit2 = result.best_fit

        # print(result.fit_report())
        coeffs = np.array([v for v in result.best_values.values()])
        fits2[:, i] = coeffs

        yfit_tmp = np.zeros_like(y) + np.nan
        yfit_tmp[~np.isnan(y)] = yfit2

        fit_sim = sinefunction(x, *coeffs)

        residual2[:, i] /= fit_sim

        # plot centre column
        # if i == int(arr2.shape[1]/2):
        #     plt.figure()
        #     plt.title("Fit iteration 2")
        #     plt.plot(x, y)
        #     plt.plot(x, fit_sim)

    if "fit coeffs" in plot:
        for j in range(5):
            axes2[j].plot(fits2[j, :], label="2nd iteration")
        axes2[-1].legend()

    """temp code to plot first coeff and smooth"""
    plt.figure()
    plt.plot(fits2[0, :])
    smooth = savgol_filter(fits2[0, :], 199, 1)
    plt.plot(smooth)

    # print([fits2[0, i] for i in solar_line_data["abs_region_cols"]])
    # print([smooth[i] for i in solar_line_data["abs_region_cols"]])

    # save blaze to file
    np.savetxt(os.path.join(MINISCAN_PATH, channel, "%s_fit_coeff0.txt" % h5_prefix), smooth)

    # plt.figure()
    # plt.plot(fits2[4, :])
    # smooth = savgol_filter(fits2[4, :], 899, 1)
    # plt.plot(smooth)
    # print([fits2[4, i] for i in solar_line_data["abs_region_cols"]])
    # print([smooth[i] for i in solar_line_data["abs_region_cols"]])

    ix = index(plot, "residual array 2")
    if ix > -1:
        vmax = np.min((np.nanmax(residual), 1.2))
        vmin = np.max((np.nanmin(residual), 0.8))
        im = ax1[ix].imshow(residual, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax1[ix])
        ax1[ix].set_title("Corrected 2 miniscan residuals")

        # plot rectange of solar line
        for d in aotf_solar_line_data:
            x = d["abs_region_cols"][0]
            y = d["abs_region_rows"][0]
            width = d["abs_region_cols"][1] - d["abs_region_cols"][0]
            if d["abs_region_rows"][1] == -1:
                height = arr.shape[0] - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax1[ix].add_patch(rect)

    cutoff2 = solar_line_data["cutoffs"][1]

    arr3 = np.copy(arr)
    arr3[residual2 < cutoff2] = np.nan

    ix = index(plot, "corrected array 2")
    if ix > -1:
        im = ax1[ix].imshow(arr3, aspect="auto")
        plt.colorbar(im, ax=ax1[ix])
        ax1[ix].set_title("Corrected 2 miniscan array cutoff=%0.3f" % cutoff)

        for d in aotf_solar_line_data:
            x = d["abs_region_cols"][0]
            y = d["abs_region_rows"][0]
            width = d["abs_region_cols"][1] - d["abs_region_cols"][0]
            if d["abs_region_rows"][1] == -1:
                height = arr.shape[0] - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax1[ix].add_patch(rect)

    # make normalised rows
    # nanmax = np.nanmax(arr3, axis=1)
    # arr3_norm = arr3 / np.repeat(nanmax, arr3.shape[1]).reshape((-1, arr3.shape[1]))
    # arr3_norm_mean = np.nanmean(arr3_norm, axis=0)

    # arr4 = arr3 / arr3_norm_mean
    # plot horizontally normalised spectra i.e. blaze function removed
    # plt.figure()
    # plt.imshow(arr4)

    # find rows with nans and plot some of them - plot flattened (blaze removed) normalised spectra
    # nan_row_ixs = np.where(np.isnan(np.mean(arr4, axis=1)))[0]
    # plt.figure()
    # for i in np.linspace(0, len(nan_row_ixs)-1, num=10):
    #     plt.plot(arr4[nan_row_ixs[int(i)], :]/np.nanmean(arr4[nan_row_ixs[int(i)], :]), label=nan_row_ixs[int(i)])
    # plt.legend()
    # plt.ylim((0.8, 1.2))

    ix = index(plot, "residual spectra")
    if ix > -1:
        for i in range(0, residual2.shape[0], int(residual2.shape[0]/20)):
            ax1[ix].plot(residual2[i, :], alpha=0.5)
        ax1[ix].grid()
        # ax1[ix].legend()
        ax1[ix].set_title("Corrected 2 miniscan residuals")

        # plot solar line abs min and max
        for d in aotf_solar_line_data:
            for x in d["abs_region_cols"]:
                if x > 0:
                    ax1[ix].axvline(x=x, color="black", linestyle="--")

    # save ifig
    if save_ifigs:
        save_ifig(fig1, os.path.join(MINISCAN_PATH, channel, "%s_ifig.pkl" % h5_prefix))


# from tools.plotting.colours import get_colours


# arr_sec = arr[:, 650:800]
# max_row = np.max(arr_sec, axis=1)
# min_row = np.min(arr_sec, axis=1)

# max_row_rep = np.repeat(max_row, arr_sec.shape[1]).reshape((-1, arr_sec.shape[1]))
# min_row_rep = np.repeat(min_row, arr_sec.shape[1]).reshape((-1, arr_sec.shape[1]))

# arr_norm = (arr_sec - min_row_rep) / (max_row_rep - min_row_rep)
# arr_norm_mean = np.mean(arr_norm, axis=0)

# start_ixs = np.arange(0, 30)
# end_ixs = np.arange(120, 150)
# cont_ixs = np.concatenate((start_ixs, end_ixs))

# polyfit = np.polyfit(cont_ixs, arr_norm_mean[cont_ixs], 2)
# polyval = np.polyval(polyfit, np.arange(150))


# colours = get_colours(arr_sec.shape[0])
# for i, row in enumerate(arr_norm):
#     plt.plot(row, alpha=0.1, color=colours[i])
# plt.plot(polyval, "k")


# plt.scatter(arr_norm.T, alpha=0.1, color=np.asarray(colours))


# for arr_s in arr_sec:
#     plt.plot(arr_s / max(arr_s), alpha=0.1)
