# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:52:02 2023

@author: iant

STEP 1: MINISCAN DIAGONAL CORRECTIONS:
    SEARCH THROUGH MINISCAN OBSERVATIONS
    APPLY FFT CORRECTIONS
    GENERATE H5 OR FITS FILES WITH CORRECTED DIAGONALS

    TODO: GET IT WORKING FOR 1KHZ SCANS

"""

import os
import re
import numpy as np
import h5py
from astropy.io import fits

import matplotlib.pyplot as plt
from tools.general.progress_bar import progress

from instrument.calibration.so_lno_2023.get_data import list_miniscan_data_1p0a, get_miniscan_data_1p0a
from instrument.calibration.so_lno_2023.remove_oscillations import remove_oscillations
from instrument.calibration.so_lno_2023.spectral_functions import find_peak_aotf_pixel, get_diagonal_blaze_indices
from instrument.calibration.so_lno_2023.make_hr_array import make_hr_array


# channel = "so"
channel = "lno"


# in-flight
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel.upper())

regex = re.compile("20.*_%s_.*_CM" % channel.upper())  # search all files
# regex = re.compile("20180716.*_%s_.*_CM" % channel.upper())  # search specific file SO
# regex = re.compile("20181209_180348_1p0a_LNO_2_CM")  # search specific file LNO

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


HR_SCALER = 10.  # make HR grid with N times more points than pixel columns

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

OUTPUT_FILE_TYPE = "fits"
# OUTPUT_FILE_TYPE = "h5"


list_files = True
# list_files = False

# for checking the oscillation removal
# plot_fft = True
plot_fft = False


if channel == "so":

    # search for miniscan files with the following characteristics
    # aotf step in kHz
    aotf_steppings = [4.0]
    # aotf_steppings = [2.0]

    # detector row binning
    binnings = [0]

    # diffraction order of first spectrum in file
    starting_orders = list(range(178, 210))
    # starting_orders = [188]


elif channel == "lno":
    # aotf_steppings = [8.0]
    aotf_steppings = [2.0, 4.0, 8.0]
    binnings = [0]

    # starting_orders = [194]
    starting_orders = list(range(180, 210))


# only reload from disk if not present
if "d" not in globals():
    list_files = True


"""get data"""
if list_files:
    h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, file_level, channel, starting_orders, aotf_steppings, binnings, path=None)

    # only reload from disk if not present
    if "d" not in globals():
        d = get_miniscan_data_1p0a(h5_filenames, channel)
    if h5_prefixes != list(d.keys()):
        d = get_miniscan_data_1p0a(h5_filenames, channel)


# remove oscillations, output spectra and other info to a dictionary d2
d2 = remove_oscillations(d, channel, plot=plot_fft)


"""make high resolution (hr) arrays"""
for h5_prefix in progress(h5_prefixes):

    n_reps = d2[h5_prefix]["nreps"]
    # HR array spectra for all repetitions
    aotfs = d2[h5_prefix]["aotf"]
    for rep in range(n_reps):

        # interpolate onto high res grid
        array = d2[h5_prefix]["array%i" % rep]
        array_hr, aotf_hr = make_hr_array(array, aotfs, HR_SCALER)
        d2[h5_prefix]["array%i_hr" % rep] = array_hr

    d2[h5_prefix]["aotf_hr"] = aotf_hr

    d2[h5_prefix]["t"] = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
    d2[h5_prefix]["t_range"] = [[np.min(d[h5_prefix]["t_rep"][:, rep]), np.max(d[h5_prefix]["t_rep"][:, rep])] for rep in range(n_reps)]

    for rep in range(d2[h5_prefix]["nreps"]):
        # calc blaze diagonals

        t = d2[h5_prefix]["t"][rep]
        px_ixs = np.arange(d2[h5_prefix]["array%i_hr" % rep].shape[1])

        px_peaks, aotf_nus = find_peak_aotf_pixel(t, d2[h5_prefix]["aotf_hr"], px_ixs, channel)
        px_peaks = np.asarray(px_peaks) * int(HR_SCALER)
        aotf_nus = np.asarray(aotf_nus)
        blaze_diagonal_ixs_all = get_diagonal_blaze_indices(px_peaks, px_ixs)

        # make diagonally corrected array
        diagonals = []
        diagonals_aotf = []

        for row in range(d2[h5_prefix]["array%i_hr" % rep].shape[0]-5):
            # find closest diagonal pixel number (in first column)
            closest_ix = np.argmin(np.abs(blaze_diagonal_ixs_all[:, 0] - row))
            row_offset = blaze_diagonal_ixs_all[closest_ix, 0] - row

            # apply offset to diagonal indices
            blaze_diagonal_ixs = (blaze_diagonal_ixs_all[closest_ix, :] - row_offset)

            if np.all(blaze_diagonal_ixs < d2[h5_prefix]["array%i_hr" % rep].shape[0]):
                diagonals.append(d2[h5_prefix]["array%i_hr" % rep][blaze_diagonal_ixs, px_ixs])
                diagonals_aotf.append(d2[h5_prefix]["aotf_hr"][blaze_diagonal_ixs])

        diagonals = np.asarray(diagonals)
        diagonals_aotf = np.asarray(diagonals_aotf)
        d2[h5_prefix]["array_diag%i_hr" % rep] = diagonals
        d2[h5_prefix]["aotf_diag%i_hr" % rep] = diagonals_aotf

        # print("Diagonal shape: ", diagonals.shape)
        # print("Diagonal AOTF shape: ", diagonals_aotf.shape)

    """Save figures and files"""
    # save diagonally-correct array and aot freqs to hdf5
    if OUTPUT_FILE_TYPE == "h5":
        with h5py.File(os.path.join(MINISCAN_PATH, channel.upper(), "%s.h5" % h5_prefix), "w") as f:
            for rep in range(d2[h5_prefix]["nreps"]):
                f.create_dataset("array%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["array_diag%i_hr" % rep],
                                 compression="gzip", shuffle=True)
                f.create_dataset("aotf%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["aotf_diag%i_hr" % rep],
                                 compression="gzip", shuffle=True)
                f.create_dataset("t%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["t_range"][rep],
                                 compression="gzip", shuffle=True)
    elif OUTPUT_FILE_TYPE == "fits":
        hdus = [fits.PrimaryHDU()]
        for rep in range(d2[h5_prefix]["nreps"]):
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["array_diag%i_hr" % rep], name="array%02i" % rep))
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["aotf_diag%i_hr" % rep], name="aotf%02i" % rep))
            hdus.append(fits.ImageHDU(data=d2[h5_prefix]["t_range"][rep], name="t%02i" % rep))
        hdul = fits.HDUList(hdus)
        hdul.writeto(os.path.join(MINISCAN_PATH, channel.upper(), "%s.fits" % h5_prefix), overwrite=True)

    # save miniscan png
    plt.figure(figsize=(8, 5), constrained_layout=True)
    plt.title(h5_prefix)
    plt.imshow(d2[h5_prefix]["array_diag%i_hr" % rep], aspect="auto")
    plt.savefig(os.path.join(MINISCAN_PATH, channel.upper(), "%s.png" % h5_prefix))
    plt.close()
