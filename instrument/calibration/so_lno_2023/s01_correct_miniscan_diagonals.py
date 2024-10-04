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

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [4]
# detector row binning
binnings = [0]
# diffraction order of first spectrum in file
# starting_orders = list(range(178, 210))
starting_orders = [164]


# in-flight
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel.upper())

regex = re.compile("20.*_%s_.*_CM" % channel.upper())  # search all files
# regex = re.compile("20180716.*_%s_.*_CM" % channel.upper())  # search specific file SO
# regex = re.compile("20181209_180348_1p0a_LNO_2_CM")  # search specific file LNO
# regex = re.compile("20210606_021551_1p0a_LNO_2_CM")  # search specific file LNO

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


HR_SCALER = 5.  # make HR grid with N times more points than pixel columns

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")
H5_ROOT_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

OUTPUT_FILE_TYPE = "fits"
# OUTPUT_FILE_TYPE = "h5"


# list and colour code files matching the search parameters
# list_files = True
list_files = False

# force reloading of h5 data? If false and variable exists, use values in memory
# force_reload = True
force_reload = False


# plot raw miniscan
# plot = ["raw", "diagonals"]
plot = ["diagonals"]
# plot_raw = False

# for checking the oscillation removal
# plot_fft = True
plot_fft = False

# for checking interpolation and oscillations in miniscan grid
# plot_hr_grid = True
plot_hr_grid = False


# only reload from disk if not present
if "d" not in globals():
    list_files = True


"""get data"""
# search through all minican files and list those matching the aotf step(s) and diffraction order(s) given above
# yellow = order matches but not the aotf Khz step
# blue = order and aotf step values match -> add this file to the list to analyse
if list_files:
    h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, file_level, channel, starting_orders,
                                                        aotf_steppings, binnings, path=H5_ROOT_PATH)

    # only reload from disk if not present
    if "d" not in globals() or force_reload:
        d = get_miniscan_data_1p0a(h5_filenames, channel, plot=plot, path=H5_ROOT_PATH)
    if h5_prefixes != list(d.keys()):
        d = get_miniscan_data_1p0a(h5_filenames, channel, plot=plot, path=H5_ROOT_PATH)


# remove oscillations, output spectra and other info to a dictionary d2
d2 = remove_oscillations(d, channel, plot=plot_fft)

# stop()

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

        if plot_hr_grid:
            plt.figure()
            for i in range(15, 320, 50):
                plt.plot(np.arange(array.shape[0]), array[:, i])
                plt.plot(np.arange(array_hr.shape[0])/HR_SCALER, array_hr[:, i*int(HR_SCALER)])

    d2[h5_prefix]["aotf_hr"] = aotf_hr
    d2[h5_prefix]["t_hr"] = [np.interp(np.arange(array_hr.shape[0])/HR_SCALER, np.arange(array.shape[0]),
                                       d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
    d2[h5_prefix]["t"] = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
    d2[h5_prefix]["t_range"] = [[np.min(d[h5_prefix]["t_rep"][:, rep]), np.max(d[h5_prefix]["t_rep"][:, rep])] for rep in range(n_reps)]

    for rep in range(d2[h5_prefix]["nreps"]):
        # calc blaze diagonals

        t = d2[h5_prefix]["t"][rep]

        aotf_hr = d2[h5_prefix]["aotf_hr"]
        array_hr = d2[h5_prefix]["array%i_hr" % rep]

        px_ixs_hr = np.arange(array_hr.shape[1])

        px_peaks, aotf_nus = find_peak_aotf_pixel(t, aotf_hr, px_ixs_hr/HR_SCALER, channel)
        # px_peaks is an array of size N spectra x max(orders) with the indices of HR pixel numbers where the AOTF is at max value for each order

        # find orders with non nan values and where there are sufficient points to interpolate between orders (i.e. not orders at start/end of miniscan)
        orders = np.where(np.sum(~np.isnan(px_peaks), axis=0) > 250)[0]

        order = orders[0]
        px_ixs = px_peaks[:, order]

        spectrum_ixs_not_nan = np.where(~np.isnan(px_ixs))[0]
        px_ixs_not_nan = np.int64(px_ixs[spectrum_ixs_not_nan])

        # now interpolate this onto the high res pixel number grid
        # spectrum = np.interp(px_ixs_hr, px_ixs_not_nan, array_hr[spectrum_ixs_not_nan, px_ixs_not_nan])
        px_ixs_interp = np.interp(px_ixs_hr, px_ixs_not_nan, spectrum_ixs_not_nan)

        # aotf_nus = np.asarray(aotf_nus)
        # px_peaks = np.asarray(px_peaks)
        # blaze_diagonal_ixs_all = get_diagonal_blaze_indices(px_peaks, px_ixs_hr)

        # make diagonally corrected array
        diagonals = []
        diagonals_aotf = []
        diagonals_t = []

        if "diagonals" in plot:

            array_hr_mask = array_hr.copy()

            for spectrum_ix in np.arange(array_hr_mask.shape[0]):
                # find orders without nans
                order_ixs = np.where(~np.isnan(px_peaks[spectrum_ix]))[0]
                for order_ix in order_ixs:
                    array_hr_mask[spectrum_ix, int(px_peaks[spectrum_ix, order_ix])] = np.nan

            plt.figure()
            plt.imshow(array_hr_mask)

        stop()

        for row in range(array_hr.shape[0]-5):
            # find closest diagonal pixel number (in first column)
            closest_ix = np.argmin(np.abs(blaze_diagonal_ixs_all[:, 0] - row))
            row_offset = blaze_diagonal_ixs_all[closest_ix, 0] - row

            # apply offset to diagonal indices
            blaze_diagonal_ixs = (blaze_diagonal_ixs_all[closest_ix, :] - row_offset)

            if np.all(blaze_diagonal_ixs < array_hr.shape[0]):
                diagonals.append(array_hr[blaze_diagonal_ixs, px_ixs_hr])
                diagonals_aotf.append(aotf_hr[blaze_diagonal_ixs])
                diagonals_t.append(d2[h5_prefix]["t_hr"][rep][blaze_diagonal_ixs])

        diagonals = np.asarray(diagonals)
        diagonals_aotf = np.asarray(diagonals_aotf)
        d2[h5_prefix]["array_diag%i_hr" % rep] = diagonals
        d2[h5_prefix]["aotf_diag%i_hr" % rep] = diagonals_aotf
        d2[h5_prefix]["t_diag%i_hr" % rep] = diagonals_t

        # print("Diagonal shape: ", diagonals.shape)
        # print("Diagonal AOTF shape: ", diagonals_aotf.shape)

    """Save figures and files"""
    # save diagonally-correct array and aot freqs to hdf5
    if OUTPUT_FILE_TYPE == "h5":
        print("Writing to file %s.h5" % h5_prefix)
        with h5py.File(os.path.join(MINISCAN_PATH, channel.upper(), "%s.h5" % h5_prefix), "w") as f:
            for rep in range(d2[h5_prefix]["nreps"]):
                f.create_dataset("array%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["array_diag%i_hr" % rep],
                                 compression="gzip", shuffle=True)
                f.create_dataset("aotf%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["aotf_diag%i_hr" % rep],
                                 compression="gzip", shuffle=True)
                f.create_dataset("trange%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["t_range"][rep],
                                 compression="gzip", shuffle=True)
                f.create_dataset("t%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["t_diag%i_hr" % rep][:, 0],
                                 compression="gzip", shuffle=True)  # just save 1st column for temperature
    elif OUTPUT_FILE_TYPE == "fits":
        hdus = [fits.PrimaryHDU()]
        print("Writing to file %s.fits" % h5_prefix)
        for rep in range(d2[h5_prefix]["nreps"]):
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["array_diag%i_hr" % rep], name="array%02i" % rep))
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["aotf_diag%i_hr" % rep], name="aotf%02i" % rep))
            hdus.append(fits.ImageHDU(data=d2[h5_prefix]["t_range"][rep], name="trange%02i" % rep))
            hdus.append(fits.ImageHDU(data=d2[h5_prefix]["t_diag%i_hr" % rep], name="t%02i" % rep))
        hdul = fits.HDUList(hdus)
        hdul.writeto(os.path.join(MINISCAN_PATH, channel.upper(), "%s.fits" % h5_prefix), overwrite=True)

    # save miniscan png
    plt.figure(figsize=(8, 5), constrained_layout=True)
    plt.title(h5_prefix)
    plt.imshow(d2[h5_prefix]["array_diag%i_hr" % rep], aspect="auto")
    plt.savefig(os.path.join(MINISCAN_PATH, channel.upper(), "%s.png" % h5_prefix))
    plt.close()
