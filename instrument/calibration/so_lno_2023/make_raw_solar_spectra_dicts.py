# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:01:20 2025

@author: iant

RAW SOLAR CALIBRATION FULLSCAN SPECTRA FOR DIFFERENT TEMPERATURES FOR LNO NADIR Y PEAK FITTING AUXILIARY FILES


"""

import re
import numpy as np
import matplotlib.pyplot as plt
from tools.general.progress_bar import progress
from tools.plotting.colours import get_colours
from instrument.calibration.so_lno_2023.get_data import list_miniscan_data_1p0a, get_miniscan_data_1p0a
from tools.file.read_write_hdf5 import write_hdf5_from_dict_simple

channel = "lno"
orders = range(118, 200)

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [156]
# aotf_steppings = [8]
# detector row binning
binnings = [0]
# diffraction order of first spectrum in file
starting_orders = [110]


# in-flight
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel.upper())

regex = re.compile("20.*_%s_.*_CF" % channel.upper())  # search all files

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
# plot = ["raw", "corrected"]
# plot = ["raw"]
plot = ["norm"]

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

    if "norm" in plot:
        plt.figure()
    for t in list(sorted(solar_spectra_d.keys())):

        colour_ix = 256 * (float(t) - t_min)/(t_max - t_min)

        spectrum = solar_spectra_d[t]

        # plot normalised raw solar spectrum
        if "norm" in plot:
            plt.plot(spectrum/np.max(spectrum), color=colours[int(colour_ix)], alpha=0.5, label="%0.2fK" % float(t))

    if "norm" in plot:
        plt.grid()
        plt.legend()
        plt.xlabel("Pixel number")
        plt.ylabel("Normalised solar spectrum")
        plt.title("Solar calibrations for order %i" % order)

    write_hdf5_from_dict_simple("%s_%i_raw_solar_spectra_dict" % (channel, order), solar_spectra_d)
