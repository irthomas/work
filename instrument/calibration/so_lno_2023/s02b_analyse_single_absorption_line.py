# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:36:01 2024

@author: iant

ANALYSE A SINGLE ABSORPTION LINE
TEST FITTING ASYMMETRIC BLAZE

"""


import os
# import h5py
from astropy.io import fits
import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from scipy.ndimage import gaussian_filter
# from scipy.signal import argrelmin

# from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu_mp
# from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.baseline_als import baseline_als
from tools.general.progress_bar import progress
from tools.file.read_write_hdf5 import write_hdf5_from_dict_simple
from instrument.calibration.so_lno_2023.load_fits_miniscan import load_fits_miniscan
from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze
# from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs, make_ils
from tools.general.get_minima_maxima import get_local_maxima_or_equals

from instrument.calibration.so_lno_2023.fitting_functions import fit_blaze_spectrum


# channel = "so"
channel = "lno"

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [4]
# diffraction order of first spectrum in file
# starting_orders = list(range(163, 210))
starting_orders = [176]


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")


# check for data available in miniscan dir
filenames = os.listdir(os.path.join(MINISCAN_PATH, channel))
# list all fits files
h5_prefixes = [s.replace(".fits", "") for s in filenames if ".fits" in s and s]
# list those with chosen AOTF stepping (in KHz)
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-1]) in aotf_steppings]
# list those with chosen aotf diffraction orders
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-2]) in starting_orders]
print("%i files found matching the desired stepping and diffraction order start" % len(h5_prefixes))


# plot = ["asym_blaze", "std no abs"]
plot = ["asym_blaze"]

# just take 1 file - if more, ask the user
if len(h5_prefixes) == 1:
    h5_prefix = h5_prefixes[0]
else:
    for i, h5_prefix in enumerate(h5_prefixes):
        print(i, h5_prefix)
    inp_in = int(input("Please select one to plot (0 to %i): " % (len(h5_prefixes)-1)))
    if inp_in < len(h5_prefixes):
        h5_prefix = h5_prefixes[inp_in]
    else:
        print("Out of range, selecting the last one")
        h5_prefix = h5_prefixes[-1]

# get data from selected file
arrs, aotfs, ts = load_fits_miniscan(h5_prefix, MINISCAN_PATH)
nrows, ncols = arrs[0].shape

FIGSIZE = (8, 10)

# normalise arrays
array_norms = []
for arr_ix in range(len(arrs)):
    array_norm = np.zeros_like(arrs[arr_ix])
    for i, spectrum in enumerate(arrs[arr_ix]):
        array_norm[i, :] = spectrum / np.max(spectrum)
    array_norms.append(array_norm)


# find relative stds to find pixels without absorption lines
array_stds = [np.std(array_norm, axis=0)/np.mean(array_norm, axis=0) for array_norm in array_norms]

# now find the points with the smallest stds
baseline = baseline_als(1.0-array_stds[0], lam=500000, p=0.999)

# stds normalised
stds_norm = (1.0 - array_stds[0])/baseline

max_ixs = get_local_maxima_or_equals(stds_norm)

ixs = [i for i, f in enumerate(stds_norm) if i in max_ixs and f > 0.999]

# define manually based on stds
no_abs_ixs = np.asarray([18, 98, 123, 225, 284, 340, 663, 676, 694, 768, 790, 905, 955, 984, 1029, 1102, 1229, 1243, 1384, 1423, 1475, 1542])

if "std no abs" in plot:
    plt.figure()
    plt.plot(stds_norm)
    plt.scatter(no_abs_ixs, stds_norm[no_abs_ixs])


arr_ix = 0

# try fitting polynomial blaze to pixels without absorptions
baselines = np.zeros_like(arrs[arr_ix])
for i, spectrum in enumerate(arrs[arr_ix]):
    polyfit = np.polyfit(no_abs_ixs, spectrum[no_abs_ixs], 9)
    polyval = np.polyval(polyfit, np.arange(ncols))
    baselines[i, :] = polyval


# try comparing to asym blaze functions
if "asym_blaze" in plot:
    # order = int(h5_prefix.split("-")[-2]) + 2

    for order, row_ix in [[165, 200], [169, 770]]:

        # don't fit the blaze, just plot the expected shape
        px_nus = nu_mp(order, np.arange(ncols)*(290/ncols), -5.0)
        blaze = asymmetric_blaze(channel, order, px_nus)*1.01

        plt.figure()
        plt.plot(array_norms[0][row_ix, :])
        plt.plot(blaze)
        plt.scatter(no_abs_ixs, blaze[no_abs_ixs])

        # plt.figure()
        plt.plot(array_norms[0][row_ix, :]/blaze)


# fit blaze with minimise
fit_blaze_spectrum(spectrum, first_guess, channel, order)


if "polyfit" in plot:
    plt.figure()
    plt.plot(arrs[arr_ix][200, :])
    plt.plot(baselines[200, :])
    plt.plot(arrs[arr_ix][770, :])
    plt.plot(baselines[770, :])

array_corr = arrs[0]/baselines

# just take the first array (do the others in future)
rep_ix = 0

aotf_array = aotfs[rep_ix]
t_mean = np.mean(ts[rep_ix])

if "corrected array" in plot:
    plt.figure()
    plt.imshow(array_corr)

if "abs depth" in plot:
    plt.figure()
    plt.plot(array_corr[:, 1012])
