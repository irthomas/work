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
from scipy.interpolate import RegularGridInterpolator


from instrument.calibration.so_lno_2023.get_data import list_miniscan_data_1p0a, get_miniscan_data_1p0a
from instrument.calibration.so_lno_2023.remove_oscillations import remove_oscillations
from instrument.calibration.so_lno_2023.spectral_functions import find_peak_aotf_pixel  # , get_diagonal_blaze_indices
from instrument.calibration.so_lno_2023.make_hr_array import make_hr_array


# channel = "so"
channel = "lno"

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [4]
# aotf_steppings = [8]
# detector row binning
binnings = [0]
# diffraction order of first spectrum in file
# starting_orders = list(range(178, 210))
# starting_orders = list(range(163, 210))
starting_orders = [176]


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

# paths to files
MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")
H5_ROOT_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

OUTPUT_FILE_TYPE = "fits"
# OUTPUT_FILE_TYPE = "h5"


# list and colour code files matching the search parameters
list_files = True
# list_files = False

# force reloading of h5 data? If false and variable exists, use values in memory
force_reload = True
# force_reload = False


# for checking HR interpolation and oscillations in miniscan grid
# for checking the oscillation removal
# plot = ["check hr", "check fft", "raw", "corrected"]
plot = ["corrected"]
# plot = ["raw"]


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
plot_fft = "check fft" in plot
d2 = remove_oscillations(d, channel, plot=plot_fft)


"""make high resolution (hr) arrays"""
for h5_prefix in progress(h5_prefixes):

    n_reps = d2[h5_prefix]["nreps"]
    # n_reps = 1
    # HR array spectra for all repetitions
    aotfs = d2[h5_prefix]["aotf"]
    for rep in range(n_reps):

        # interpolate onto high res grid
        array = d2[h5_prefix]["array%i" % rep]
        array_hr, aotf_hr = make_hr_array(array, aotfs, HR_SCALER)
        d2[h5_prefix]["array%i_hr" % rep] = array_hr

        if "check hr" in plot:
            plt.figure()
            for i in range(15, 320, 50):
                plt.plot(np.arange(array.shape[0]), array[:, i])
                plt.plot(np.arange(array_hr.shape[0])/HR_SCALER, array_hr[:, i*int(HR_SCALER)])

    # save to dictionary
    # aotf data, 1 entry per HR spectrum
    d2[h5_prefix]["aotf_hr"] = aotf_hr
    # instrument temperature 1 entry per HR spectrum
    d2[h5_prefix]["t_hr"] = [np.interp(np.arange(array_hr.shape[0])/HR_SCALER, np.arange(array.shape[0]),
                                       d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
    # mean temperature
    d2[h5_prefix]["t"] = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
    # min/max temperature
    d2[h5_prefix]["t_range"] = [[np.min(d[h5_prefix]["t_rep"][:, rep]), np.max(d[h5_prefix]["t_rep"][:, rep])] for rep in range(n_reps)]

    # loop through repeats if multiple miniscans in the file
    for rep in range(n_reps):

        t = d2[h5_prefix]["t"][rep]

        aotf_hr = d2[h5_prefix]["aotf_hr"]
        array_hr = d2[h5_prefix]["array%i_hr" % rep]

        px_ixs_hr = np.arange(array_hr.shape[1])

        # calc indices of the blaze diagonals for each order measured
        # px_peaks is an array of size N spectra x max(orders) with the indices of HR pixel numbers where the AOTF is at max value for each order
        px_peaks, aotf_nus = find_peak_aotf_pixel(t, aotf_hr, px_ixs_hr/HR_SCALER, channel)

        # copy the array to add nans where the aotf is at the peak (for plotting only)
        # array_hr_mask = array_hr.copy()
        array_hr_mask2 = array_hr.copy()

        # make dictionary of HR indices where AOTF is at the peak,
        peak_indices = {}

        for spectrum_ix in np.arange(array_hr.shape[0]):
            # find orders without nans
            order_ixs = np.where(~np.isnan(px_peaks[spectrum_ix]))[0]
            for order_ix in order_ixs:
                # array_hr_mask[spectrum_ix, int(px_peaks[spectrum_ix, order_ix])] = np.nan

                # make dictionary of peak indices in hr grid
                # order : [x, y]
                if order_ix not in peak_indices.keys():
                    peak_indices[order_ix] = []
                peak_indices[order_ix].append([spectrum_ix, int(px_peaks[spectrum_ix, order_ix])])

        # convert to arrays for interpolation between indices on the hr grid
        for order in peak_indices.keys():
            peak_indices[order] = np.asarray(peak_indices[order])

        spectra_per_order = [len(peak_indices[order]) for order in peak_indices.keys()]

        # interpolate the [x, y] peak indices onto the hr grid for each order
        # these are the diagonal indices where the aotf is at a peak on that pixel
        interp_ixs = []
        for order in peak_indices.keys():
            # check if a full order is present in the miniscan for each order
            if len(peak_indices[order]) > np.max(spectra_per_order) * 0.9:
                # interpolate diagonal pixel indices with polyfit
                polyfit = np.polyfit(peak_indices[order][:, 1], peak_indices[order][:, 0], 3)
                polyval = np.polyval(polyfit, px_ixs_hr)
                interp_ixs.append(polyval)

                polyval_ints = np.round(polyval).astype(int)

                array_hr_mask2[np.round(polyval).astype(int), px_ixs_hr] = np.nan

        interp_ixs = np.asarray(interp_ixs)

        if "raw" in plot and rep == 0:
            plt.figure()
            plt.imshow(array_hr_mask2)
            plt.savefig(os.path.join(MINISCAN_PATH, channel.upper(), "%s_raw_diagonals.png" % h5_prefix))

        # make diagonally corrected arrays
        # diagonals = []
        diagonals_aotf = []
        diagonals_t = []
        diagonals_ixs = []

        # now interpolate from one AOTF peak diagonal to the next AOTF peak diagonal i.e from one nan row to the next, repeat for each
        for peak_line_ix in np.arange(interp_ixs.shape[0] - 1):
            first_nans = interp_ixs[peak_line_ix, :]
            next_nans = interp_ixs[peak_line_ix + 1, :]
            first_nans_int = np.round(first_nans).astype(int)
            next_nans_int = np.round(next_nans).astype(int)

            # account for the fact that the line of nans is non-linear, there are more points between the two at the right hand side
            # diff_ixs = next_nans - first_nans
            # diff_ixs_col0 = diff_ixs - diff_ixs[0]
            # extra_per_col = diff_ixs_col0 / diff_ixs[0]

            aotfs_between_peaks = []
            ixs_between_peaks = []
            n_points = int(np.ceil(next_nans[0] - first_nans[0]))
            for column_ix in np.arange(array_hr.shape[1]):
                order_indices = np.linspace(first_nans[column_ix], next_nans[column_ix], num=n_points)
                # spectrum_columns = array_hr[np.round(order_indices).astype(int), column_ix]
                aotf_columns = aotf_hr[np.round(order_indices).astype(int)]

                if column_ix == 0:
                    diagonals_t.extend(d2[h5_prefix]["t_hr"][rep][np.round(order_indices).astype(int)])

                # spectra_between_peaks.append(spectrum_columns)
                aotfs_between_peaks.append(aotf_columns)
                ixs_between_peaks.append(order_indices)

            # spectra_between_peaks = np.asarray(spectra_between_peaks).T
            aotfs_between_peaks = np.asarray(aotfs_between_peaks).T
            ixs_between_peaks = np.asarray(ixs_between_peaks).T

            # diagonals.extend(spectra_between_peaks)
            diagonals_aotf.extend(aotfs_between_peaks)
            diagonals_ixs.extend(ixs_between_peaks)

        # diagonals = np.asarray(diagonals)
        diagonals_aotf = np.asarray(diagonals_aotf)
        diagonals_ixs = np.asarray(diagonals_ixs)
        diagonals_t = np.asarray(diagonals_t)

        # new method - 2d interpolation
        spectrum_all_ixs = np.arange(array_hr.shape[0])
        interp = RegularGridInterpolator((spectrum_all_ixs, px_ixs_hr), array_hr)

        diagonals_interp = []
        diagonals_aotf_out = []
        diagonals_t_out = []
        for spectrum_ix in np.arange(diagonals_ixs.shape[0]):
            spectrum_ixs = diagonals_ixs[spectrum_ix, :]

            if np.all(spectrum_ixs > 0.0):
                row_interp = interp((spectrum_ixs, px_ixs_hr))

                diagonals_interp.append(row_interp)
                diagonals_aotf_out.append(diagonals_aotf[spectrum_ix, :])
                diagonals_t_out.append(diagonals_t[spectrum_ix])

        diagonals_interp = np.asarray(diagonals_interp)
        diagonals_aotf_out = np.asarray(diagonals_aotf_out)
        diagonals_t_out = np.asarray(diagonals_t_out)

        # plt.figure()
        # plt.imshow(diagonals, aspect="auto")
        # plt.figure()
        # plt.imshow(diagonals_interp, aspect="auto")

        d2[h5_prefix]["array_diag%i_hr" % rep] = diagonals_interp
        d2[h5_prefix]["aotf_diag%i_hr" % rep] = diagonals_aotf_out
        d2[h5_prefix]["t_diag%i_hr" % rep] = diagonals_t_out

        # print("Diagonal shape: ", diagonals_interp.shape)
        # print("Diagonal AOTF shape: ", diagonals_aotf.shape)

    """Save figures and files"""
    # save diagonally-correct array and aot freqs to hdf5
    if OUTPUT_FILE_TYPE == "h5":
        print("Writing to file %s.h5" % h5_prefix)
        with h5py.File(os.path.join(MINISCAN_PATH, channel.upper(), "%s.h5" % h5_prefix), "w") as f:
            for rep in range(n_reps):
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
        for rep in range(n_reps):
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["array_diag%i_hr" % rep], name="array%02i" % rep))
            hdus.append(fits.CompImageHDU(data=d2[h5_prefix]["aotf_diag%i_hr" % rep], name="aotf%02i" % rep))
            hdus.append(fits.ImageHDU(data=d2[h5_prefix]["t_range"][rep], name="trange%02i" % rep))
            hdus.append(fits.ImageHDU(data=d2[h5_prefix]["t_diag%i_hr" % rep], name="t%02i" % rep))
        hdul = fits.HDUList(hdus)
        hdul.writeto(os.path.join(MINISCAN_PATH, channel.upper(), "%s.fits" % h5_prefix), overwrite=True)

    # save miniscan png
    if "corrected" in plot:
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.title(h5_prefix)
        plt.imshow(d2[h5_prefix]["array_diag%i_hr" % rep], aspect="auto")
        plt.savefig(os.path.join(MINISCAN_PATH, channel.upper(), "%s_corrected.png" % h5_prefix))
        plt.close()
