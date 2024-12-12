# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:43:14 2024

@author: iant

STEP2:
    NEW VERSION OF STEP02: READ IN DIAGONALLY CORRECT MINISCANS, FIT BLAZE AND AUTOMATICALLY FIND SOLAR LINE POSITIONS
    DERIVE AOTF FUNCTIONS AND SAVE TO PDF AND TEXT FILES


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
# from instrument.nomad_lno_instrument_v02 import nu_mp, nu0_aotf, F_aotf_sinc
# from tools.spectra.solar_spectrum import get_solar_hr
# from tools.spectra.baseline_als import baseline_als
from tools.general.progress_bar import progress
from tools.file.read_write_hdf5 import write_hdf5_from_dict_simple
from instrument.calibration.so_lno_2023.fit_blaze_shape import fit_blaze
# from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs, make_ils

# channel = "so"
channel = "lno"

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [4, 8]
# diffraction order of first spectrum in file
starting_orders = list(range(163, 210))
# starting_orders = [164]


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

FIGSIZE = (8, 10)

# save_ifigs = True
# save_ifigs = False

# plot_solar = True
plot_solar = False

force_reload = True
# force_reload = False

# save the fitted blaze function to a hdf5 file for further analysis?
save_blaze_fits = True
# save_blaze_fits = False


plot = ["blaze_fits", "blaze_norm"]

# new version finds solar line positions automatically rather than using the solar line dict
# check for data available in miniscan dir
filenames = os.listdir(os.path.join(MINISCAN_PATH, channel))
# list all fits files
h5_prefixes = [s.replace(".fits", "") for s in filenames if ".fits" in s and s]
# list those with chosen stepping
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-1]) in aotf_steppings]
# list those with chosen aotf diffraction orders
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-2]) in starting_orders]
print("%i files found matching the desired stepping and diffraction order start" % len(h5_prefixes))


if save_blaze_fits:
    blaze_d = {}

with PdfPages("aotfs.pdf") as pdf:
    # for h5_prefix, solar_line_data_all in solar_line_dict.items():  # loop through files
    for file_ix, h5_prefix in enumerate(h5_prefixes):  # loop through files
        print("%i/%i: %s" % (file_ix+1, len(h5_prefixes), h5_prefix))
        channel = h5_prefix.split("-")[0].lower()

        # get data from miniscan file
        with fits.open(os.path.join(MINISCAN_PATH, channel, "%s.fits" % h5_prefix)) as hdul:
            keys = [i.name for i in hdul if i.name != "PRIMARY"]
            n_reps = len([i for i, key in enumerate(keys) if "ARRAY" in key])

            arrs = []
            aotfs = []
            ts = []
            for i in range(n_reps):
                arrs.append(hdul["ARRAY%02i" % i].data)
                aotfs.append(hdul["AOTF%02i" % i].data)
                ts.append(hdul["T%02i" % i].data)
                print(arrs[-1].shape, aotfs[-1].shape, ts[-1].shape)

        # just take the first array (do the others in future)
        rep_ix = 0

        aotf_array = aotfs[rep_ix]
        scan_array = arrs[rep_ix]
        t_mean = np.mean(ts[rep_ix])
        nrows, ncols = scan_array.shape

        # fit blaze shape to every spectrum in the whole array
        # if "scan_array_norm" not in globals() or force_reload:

        if "blaze_fits" in plot:
            fig1, ax1 = plt.subplots(figsize=FIGSIZE)
            ax1.set_title(h5_prefix)

        scan_blaze_fits = np.zeros_like(scan_array)
        for i, scan_line in enumerate(progress(scan_array)):
            scan_blaze_fits[i, :] = fit_blaze(scan_line, max_rms=0.002)

            if "blaze_fits" in plot and i % 100 == 0:
                p = ax1.plot(scan_line)
                ax1.plot(scan_blaze_fits[i, :], color=p[-1].get_color(), linestyle="--")
        if "blaze_fits" in plot:
            pdf.savefig()
            plt.close()
        # normalise to flat spectrum by removing blaze
        scan_array_norm = scan_array / scan_blaze_fits
        if "blaze_norm" in plot:
            fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=FIGSIZE)
            ax1a.set_title("%s: blaze fitting" % h5_prefix)
            ax1b.set_title("Blaze removed")
            ax1a.imshow(scan_blaze_fits)

            for i in np.arange(0, scan_array_norm.shape[0], 100):
                ax1b.plot(scan_array_norm[i, :], label="%i" % i)
            ax1b.legend()

            # ax1b.imshow(scan_array_norm)
            pdf.savefig()
            plt.close()

        if save_blaze_fits:
            blaze_d[h5_prefix] = {"ts": ts[rep_ix], "blaze": scan_blaze_fits, "aotf_col0": aotf_array[:, 0]}

        # find local minima
        # find smallest values in array, then block off x points around
        min_array = np.zeros_like(scan_array_norm)

        # get row and col indices of minima in order, with deepest absorption first
        sorted_ixs = np.array(np.unravel_index(np.argsort(scan_array_norm, axis=None), scan_array_norm.shape))

        n_found = 0
        # loop through absorption rows and col indices
        for i in np.arange(sorted_ixs.shape[1]):
            ix = sorted_ixs[:, i]

            # if index too close to top/bottom of detector, skip
            if ix[1] < 100 or ix[1] > ncols-50:
                continue

            # if index too close to left/right edge of detector, skip
            if ix[0] < 100 or ix[0] > nrows-100:
                continue

            # if chosen point is not already blocked off
            if min_array[ix[0], ix[1]] == 0:
                n_found += 1
                # print(ix)
                # set absorption centre = 2
                min_array[ix[0], ix[1]] = 2

                # set points arround it = 1 to block them off, so they aren't used in future iterations
                x_around = [i for i in np.arange(ix[0]-400, ix[0]+400, 1) if i >= 0 and i < nrows]
                y_around = [i for i in np.arange(ix[1]-40, ix[1]+40, 1) if i >= 0 and i < ncols]
                for x in x_around:
                    for y in y_around:
                        if min_array[x, y] == 0:
                            min_array[x, y] = 1

            # once N absorptions have been found, stop
            if n_found == 5:
                break

        fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=FIGSIZE)
        ax1a.set_title("%s: blaze removed" % h5_prefix)
        ax1b.set_title("Absorption search")
        ax1a.imshow(scan_array_norm)
        ax1b.imshow(min_array)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=FIGSIZE)
        for col_ix, row_ix in zip(np.where(min_array == 2)[1], np.where(min_array == 2)[0]):
            column = scan_array_norm[:, col_ix]
            depth = scan_array_norm[row_ix, col_ix]
            aotf_khzs = aotf_array[:, col_ix]
            aotf_khz_centre = aotf_array[row_ix, col_ix]
            aotf_func = 1.0 - (column / depth)
            plt.plot(aotf_khzs - aotf_khz_centre, aotf_func, label=col_ix)
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.close()


if save_blaze_fits:
    write_hdf5_from_dict_simple(os.path.join(MINISCAN_PATH, "blaze_functions"), blaze_d)
