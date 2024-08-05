# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:52:06 2024

@author: iant

STEP2:
    NEW VERSION OF STEP02: READ IN DIAGONALLY CORRECT MINISCANS, FIT BLAZE AND AUTOMATICALLY FIND SOLAR LINE POSITIONS
    DERIVE AOTF FUNCTIONS AND SAVE TO PDF AND TEXT FILES

TODO:
    SIMULATE MINISCAN DATA WITH SOLAR LINES TO CHECK AOTF AND BLAZE ARE CORRECT

CONVERT SOLAR LINE INTO AOTF/PIXEL RANGE

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


from instrument.calibration.so_lno_2023.fit_blaze_shape import fit_blaze

# from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs, make_ils

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

# channel = "so"
channel = "lno"


aotf_steppings = [4]

# save_ifigs = True
# save_ifigs = False

# plot_solar = True
plot_solar = False

force_reload = True
# force_reload = False


plot = ["blaze_fits", "blaze_norm"]

# new version finds solar line positions automatically rather than using the solar line dict
# check for data available in miniscan dir
filenames = os.listdir(os.path.join(MINISCAN_PATH, channel))
# list all fits files
h5_prefixes = [s.replace(".fits", "") for s in filenames if ".fits" in s and s]
# list those with chosen stepping
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-1]) in aotf_steppings]


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

        # just take the first array (do the others in future)
        aotf_array = aotfs[0]
        scan_array = arrs[0]
        t_mean = np.mean(ts[0])
        nrows, ncols = scan_array.shape

        # fit blaze shape to every spectrum in the whole array
        # if "scan_array_norm" not in globals() or force_reload:

        if "blaze_fits" in plot:
            fig1, ax1 = plt.subplots()

        scan_blaze_fits = np.zeros_like(scan_array)
        for i, scan_line in enumerate(progress(scan_array)):
            scan_blaze_fits[i, :] = fit_blaze(scan_line, max_rms=0.02)

            if "blaze_fits" in plot and i % 100 == 0:
                p = ax1.plot(scan_line)
                ax1.plot(scan_blaze_fits[i, :], color=p[-1].get_color(), linestyle="--")
        if "blaze_fits" in plot:
            pdf.savefig()
            plt.close()
        # normalise to flat spectrum by removing blaze
        scan_array_norm = scan_array / scan_blaze_fits
        if "blaze_norm" in plot:
            fig1, (ax1a, ax1b) = plt.subplots(nrows=2)
            ax1a.set_title("Blaze fitting")
            ax1b.set_title("Blaze removed")
            ax1a.imshow(scan_blaze_fits)
            ax1b.imshow(scan_array_norm)
            pdf.savefig()
            plt.close()

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

        fig1, (ax1a, ax1b) = plt.subplots(nrows=2)
        ax1a.set_title("Blaze removed")
        ax1b.set_title("Absorption search")
        ax1a.imshow(scan_array_norm)
        ax1b.imshow(min_array)
        pdf.savefig()
        plt.close()

        plt.figure()
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

# SIMULATE MINISCANS WITH SOLAR LINE
# # find upper/lower nus for high res solar spectrum
# min_aotf_khz = np.min(aotf_array)
# max_aotf_khz = np.max(aotf_array)

# aotf_search_range_khz = np.arange(18000, 32000)
# aotf_search_range_nu = nu0_aotf(aotf_search_range_khz)

# min_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, min_aotf_khz)]
# max_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, max_aotf_khz)]

# # get solar spectrum
# nu_hr = np.arange(min_nu - 50.0, max_nu + 50.0, 0.001)
# solar_hr_raw = get_solar_hr(nu_hr)
# # normalise + correct solar with baseline ALS
# solar_cont = baseline_als(solar_hr_raw, lam=1e11, p=0.9999)
# solar_hr = solar_hr_raw / solar_cont

# if plot_solar:
#     plt.figure()
#     plt.plot(nu_hr, solar_hr_raw)
#     plt.plot(nu_hr, solar_cont)
#     plt.figure()
#     plt.plot(nu_hr, solar_hr)


# aotf_line = aotf_array[0, :]
# scan_line = scan_array[0, :]
# # plt.plot(aotf_line, scan_line)


# # from aotf frequency get aotf function
# # first get px grid for each order
# order_search_range = np.arange(110, 210)
# order_search_range_nu = nu_mp(order_search_range, 0.0, 0.0)

# # get list of orders
# min_order = order_search_range[np.searchsorted(order_search_range_nu, min_nu)]
# max_order = order_search_range[np.searchsorted(order_search_range_nu, max_nu)]
# orders = np.arange(min_order, max_order+1)

# px_ixs = np.arange(320.0)
# px_ixs_hr = np.arange(ncols) / ncols * 320.0


# solar_spec_d = {}
# for order in orders:
#     px_nus = nu_mp(order, px_ixs, t_mean)
#     solar_spec_d[order] = {}

#     solar_spec = np.zeros_like(px_nus)
#     for i, px_nu in enumerate(px_nus):
#         solar_ixs = np.searchsorted(nu_hr, [px_nu-0.2, px_nu+0.2])
#         solar_spec[i] = np.mean(solar_hr[solar_ixs])

#     solar_spec_d[order]["solar_spec"] = solar_spec


# # ILS convolution
# # sum of ILS for each pixel for T=1
# ils_sum = np.zeros(len(px_nus))
# ils_sums = np.zeros((len(orders), len(px_nus)))
# # sum of ILS for each pixel including absorptions
# ils_sums_spectrum = np.zeros((len(orders), len(px_nus)))

# for order_ix, order in enumerate(orders):
#     # spectral calibration of each pixel in order
#     px_nus = nu_mp(order, px_ixs, t_mean)
#     # plt.plot(px_nus, solar_spec_d[order]["solar_spec"], label="Order %i" % order)

#     # convolve with ILS to make low resolution plot
#     # approx value for order
#     aotf_nu_centre = np.mean(aotf_line)
#     ils_d = get_ils_coeffs(channel, aotf_nu_centre)

#     for px, px_nu in enumerate(px_nus):

#         # get bounding indices of solar grid
#         ix_start = np.searchsorted(nu_hr, px_nu - 0.7)
#         ix_end = np.searchsorted(nu_hr, px_nu + 0.7)

#         width = ils_d["ils_width"][px]  # only ILS width known for LNO
#         # displacement = ils_d["ils_displacement"][px]
#         # amplitude = ils_d["ils_amplitude"][px]
#         displacement = 0.0
#         amplitude = 0.0

#         nu_grid = nu_hr[ix_start:ix_end] - px_nu
#         solar_grid = solar_hr[ix_start:ix_end]

#         ils = make_ils(nu_grid, width/6., displacement, amplitude)
#         # summed ils without absorption lines - different for different pixels but v. similar for orders
#         ils_sum[px] = np.sum(ils)
#         ils_sums[order_ix, px] = ils_sum[px]
#         # summed ils with absorption lines - changes with pixel and order
#         ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_grid)

# spectrum = ils_sums_spectrum / ils_sums


# plt.figure()
# for order_ix, order in enumerate(orders):
#     # px_nus = nu_mp(order, px_ixs, t_mean)
#     px_nus_hr = nu_mp(order, px_ixs_hr, t_mean)

#     spectrum_norm = spectrum[order_ix, :] / np.max(spectrum[order_ix, :])

#     spectrum_norm_hr = np.interp(px_ixs_hr, px_ixs, spectrum_norm)

#     plt.plot(px_nus_hr, spectrum_norm_hr, label="Order %i" % order)

#     if order == 194:
#         plt.plot(px_nus_hr[10:], scan_array_norm[75, 10:], linestyle="--")
#     if order == 195:
#         plt.plot(px_nus_hr[10:], scan_array_norm[120, 10:], linestyle="--")


# plt.legend()
# plt.grid()

# CHECK DATA
# norm_scan_array = np.zeros_like(scan_array)
# for i in range(aotf_array.shape[0]):
#     norm_scan_array[i, :] = scan_array[i, :]/np.max(scan_array[i, :])


# plt.plot(norm_scan_array[:, 10])

# for i in range(0, aotf_array.shape[0], 49):
#     plt.plot(scan_array[i, :]/np.max(scan_array[i, :]), c=[i/aotf_array.shape[0], i/aotf_array.shape[0], i/aotf_array.shape[0]])


# #check if horizontal strips line up correctly
# for i in range(0, aotf_array.shape[1], 49):
#     plt.plot(scan_array[:, i]/np.max(scan_array[:, i]), c=[i/aotf_array.shape[1], i/aotf_array.shape[1], i/aotf_array.shape[1]])


# convert aotf line to nus in all orders


# import numpy as np

# arr4 = np.zeros_like(arr)

# for i in range(arr.shape[0]):
#     norm=(arr[i,:]/np.max(arr[i,:]))/(smooth/np.max(smooth))
#     # plt.plot(norm)

#     arr4[i, :] = norm

# plt.figure()
# plt.imshow(arr4[:, 50:])

# plt.figure()
# plt.plot(arr4[:, 2130+50])

# plt.figure()
# plt.plot(arr4[:, 50+50])


# 10000.0
