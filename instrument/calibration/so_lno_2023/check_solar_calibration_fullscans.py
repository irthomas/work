# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:19:17 2025

@author: iant

PLOT RAW SOLAR CALIBRATION FULLSCAN SPECTRA FOR DIFFERENT TEMPERATURES
COMPARE TO SIMULATED SPECTRA

"""

import os
import re
import numpy as np
# import h5py
# from astropy.io import fits
from scipy.optimize import minimize


import matplotlib.pyplot as plt
from tools.general.progress_bar import progress
from tools.plotting.colours import get_colours

from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze
from instrument.calibration.so_lno_2023.get_data import list_miniscan_data_1p0a, get_miniscan_data_1p0a
# from instrument.calibration.so_lno_2023.remove_oscillations import remove_oscillations
# from instrument.calibration.so_lno_2023.spectral_functions import find_peak_aotf_pixel  # , get_diagonal_blaze_indices
# from instrument.calibration.so_lno_2023.make_hr_array import make_hr_array
from tools.file.read_write_hdf5 import write_hdf5_from_dict_simple

channel = "lno"
# orders = [186, 187, 193, 194, 198, 199]
orders = range(165, 200)

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [156]
# aotf_steppings = [8]
# detector row binning
binnings = [0]
# diffraction order of first spectrum in file
starting_orders = [110]


if channel == "so":
    from instrument.nomad_so_instrument_v03 import nu_mp
elif channel == "lno":
    from instrument.nomad_lno_instrument_v02 import nu_mp


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
    blaze = make_blaze(res.x, channel, order, ncols)
    # plt.plot(blaze)
    return blaze


# in-flight
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel.upper())

regex = re.compile("20.*_%s_.*_CF" % channel.upper())  # search all files
# regex = re.compile("20180716.*_%s_.*_CM" % channel.upper())  # search specific file SO
# regex = re.compile("20181209_180348_1p0a_LNO_2_CM")  # search specific file LNO
# regex = re.compile("20210606_021551_1p0a_LNO_2_CM")  # search specific file LNO

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


# paths to files
H5_ROOT_PATH = r"C:\Users\iant\Documents\DATA\hdf5"


# list and colour code files matching the search parameters
# list_files = True
list_files = False

# force reloading of h5 data? If false and variable exists, use values in memory
# force_reload = True
force_reload = False


# for checking HR interpolation and oscillations in miniscan grid
# for checking the oscillation removal
# plot = ["check hr", "check fft", "raw", "corrected"]
plot = ["raw", "corrected"]
# plot = ["raw"]


# only reload from disk if not present
if "d_solar" not in globals():
    list_files = True


"""get data"""
# search through all minican files and list those matching the aotf step(s) and diffraction order(s) given above
# yellow = order matches but not the aotf Khz step
# blue = order and aotf step values match -> add this file to the list to analyse
if list_files:
    h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, file_level, channel, starting_orders,
                                                        aotf_steppings, binnings, path=H5_ROOT_PATH)

    # only reload from disk if not present
    if "d_solar" not in globals() or force_reload:
        d_solar = get_miniscan_data_1p0a(h5_filenames, channel, plot=plot, path=H5_ROOT_PATH)
    if h5_prefixes != list(d_solar.keys()):
        d_solar = get_miniscan_data_1p0a(h5_filenames, channel, plot=plot, path=H5_ROOT_PATH)


# get temperature range from all files
t_all = []
for h5_prefix in h5_prefixes:
    t_all.extend(d_solar[h5_prefix]["t"])

# get colour scale
t_min = min(t_all)
t_max = max(t_all)

colours = get_colours(256, cmap="gnuplot")


for order in orders:
    solar_spectra_d = {}
    # find matching orders
    for h5_prefix in progress(h5_prefixes):

        # n_reps = d_solar[h5_prefix]["t_rep"].shape[-1]
        aotfs = d_solar[h5_prefix]["a"]

        if channel == "so":
            from instrument.nomad_so_instrument_v03 import m_aotf
        elif channel == "lno":
            from instrument.nomad_lno_instrument_v02 import m_aotf

        orders_in_file = np.asarray([m_aotf(aotf) for aotf in aotfs])
        ts = d_solar[h5_prefix]["t"]
        ys = d_solar[h5_prefix]["y"]

        # indices where order is correct
        matching_ixs = np.where(orders_in_file == order)[0]

        # get indices of first and last spectrum of that order
        for ix in matching_ixs[[0, -1]]:
            t = ts[ix]

            spectrum = ys[ix, :]  # /np.max(ys[ix, :])
            solar_spectra_d[str(t)] = spectrum/np.max(spectrum)

    plt.figure()
    for t in list(sorted(solar_spectra_d.keys())):

        colour_ix = 256 * (float(t) - t_min)/(t_max - t_min)

        spectrum = solar_spectra_d[t]

        # fit the blaze function - doesn't work, AOTF shape also there
        # first_guess = [320., t, np.max(spectrum)*0.95, np.mean(spectrum[0:50])]
        # blaze = fit_blaze_spectrum(spectrum, first_guess, channel, order)
        # plt.plot(blaze, color=colours[int(colour_ix)], alpha=0.5)

        # plot normalised raw solar spectrum
        plt.plot(spectrum/np.max(spectrum), color=colours[int(colour_ix)], alpha=0.5, label="%0.2fK" % float(t))

    plt.grid()
    plt.legend()
    plt.xlabel("Pixel number")
    plt.ylabel("Normalised solar spectrum")
    plt.title("Solar calibrations for order %i" % order)

    write_hdf5_from_dict_simple("%s_%i_raw_solar_spectra_dict" % (channel, order), solar_spectra_d)
