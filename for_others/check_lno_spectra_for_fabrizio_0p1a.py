# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:46:58 2024

@author: iant

CHECK VARIATIONS IN LNO SPECTRA FOR FABRIZIO AND EMILIANO
NEW VERSION:
    APPLY CORRECTIONS BEFORE BINNING ALL DETECTOR ROWS TOGETHER
    POLYFIT TO REMOVE BAD PIXELS
    FIT THE BLAZE WITH SEVERAL PARAMETERS INCLUDING OFFSET
    APPLY OFFSET CORRECTION BASED ON SOLAR CALS
    
    
THIS WORKS ON 0.1A DATA ONLY, BEFORE BINNING

NEXT STEPS:
    use minimise to fit shape to solar cal rather than blaze
    get std of variation before/after correction
    


"""

from tools.file.read_write_hdf5 import read_hdf5_to_dict
from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze
from instrument.nomad_lno_instrument_v02 import nu_mp, m_aotf
from tools.file.paths import paths
from tools.file.hdf5_functions import open_hdf5_file
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

BAD_PX_ITERS = 5
BAD_PX_STD = 5.0

h5_prefix = "20220130_045445"
h5_t = -12.377  # from 1a0 file

orders = [186, 187, 193, 194, 198, 199]


def make_blaze2(params, channel, order, ncols, t):

    # [t, scaler, offset] = params
    [scaler, offset] = params
    eff_n_px = 320.0

    px_nus = nu_mp(order, np.arange(ncols)*(eff_n_px/ncols), t)
    blaze = asymmetric_blaze(channel, order, px_nus)*scaler + offset
    return blaze


def make_blaze3(params, channel, order, ncols):

    # [eff_n_px, t, scaler, offset] = params
    [t, scaler, offset] = params
    # [scaler, offset] = params
    eff_n_px = 320.0
    # t = 13.4

    px_nus = nu_mp(order, np.arange(ncols)*(eff_n_px/ncols), t)
    blaze = asymmetric_blaze(channel, order, px_nus)*scaler + offset
    return blaze


def min_blaze2(params, args):

    [spectrum, channel, order, t] = args

    ncols = len(spectrum)
    blaze = make_blaze2(params, channel, order, ncols, t)
    chisq_px = (spectrum - blaze)**2
    chisq = np.sum(chisq_px)
    return chisq


def min_blaze3(params, args):

    [spectrum, channel, order] = args

    ncols = len(spectrum)
    blaze = make_blaze3(params, channel, order, ncols)
    chisq_px = (spectrum - blaze)**2
    chisq = np.sum(chisq_px)
    return chisq


def fit_blaze_spectrum2(spectrum, first_guess, channel, order, t):
    res = minimize(min_blaze2, first_guess, args=[spectrum, channel, order, t])

    ncols = len(spectrum)
    # print(res.x)
    # plt.figure()
    # plt.plot(spectrum)
    blaze = make_blaze2(res.x, channel, order, ncols, t)
    blaze_corr = make_blaze2([*res.x[0:-1], 0.0], channel, order, ncols, t)
    # plt.plot(blaze)
    return blaze, blaze_corr, res.x


def fit_blaze_spectrum3(spectrum, first_guess, channel, order):
    res = minimize(min_blaze3, first_guess, args=[spectrum, channel, order])

    ncols = len(spectrum)
    # print(res.x)
    # plt.figure()
    # plt.plot(spectrum)
    blaze = make_blaze3(res.x, channel, order, ncols)
    blaze_corr = make_blaze3([*res.x[0:-1], 0.0], channel, order, ncols)
    # plt.plot(blaze)
    return blaze, blaze_corr, res.x


def get_data(h5_prefix):

    print("Getting data from file")
    h5 = "%s_0p1a_LNO_1" % (h5_prefix)
    h5f = open_hdf5_file(h5, path=DATA_PATH)

    aotfs = h5f["Channel/AOTFFrequency"][...]
    orders = np.asarray([m_aotf(aotf) for aotf in aotfs])
    t = np.median(h5f["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][0:10])  # approx temperature, before obs starts

    # get data, sort by order
    unique_orders = sorted(list(set(orders)))

    d = {}

    for order in unique_orders:
        order_ixs = np.where(orders == order)[0]
        d[order] = {
            "raw": h5f["Science/Y"][order_ixs, :, :],
            "t": t,
        }

    return d


def bad_px_correction(d):

    unique_orders = d.keys()

    for order in unique_orders:
        y = d[order]["raw"]

        nspectra = y.shape[0]
        nbins = y.shape[1]
        npixels = y.shape[2]

        badpx_removed = np.zeros((nspectra, nbins, npixels))

        print("Bad pixel correction")

        for i in range(nspectra):

            if i % 20 == 0:
                print(order, i)

            for j in range(nbins):
                spectrum = y[i, j, :]

                x = np.arange(len(spectrum))

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

                badpx_removed[i, j, :] = spectrum

        d[order]["goodpx"] = badpx_removed

    return d


def scale_spectrum(params, spectrum):
    [scaler, offset] = params

    return spectrum * scaler + offset


def min_solar(params, args):

    [spectrum, solar_spectrum] = args

    chisq_px = (scale_spectrum(params, solar_spectrum) - spectrum)**2
    chisq = np.sum(chisq_px)
    return chisq


def fit_solar_spectrum(spectrum, first_guess, solar_spectrum):
    res = minimize(min_solar, first_guess, args=[spectrum, solar_spectrum])

    scaled_spectrum = scale_spectrum(res.x, solar_spectrum)
    corr_spectrum = scale_spectrum([res.x[0], 0], solar_spectrum)
    return scaled_spectrum, corr_spectrum, res.x


def fit_spectra(d, channel):

    unique_orders = d.keys()

    for order in unique_orders:

        # t_median = np.median(d[order]["ts"])

        y = d[order]["raw"]
        solar_spectrum = d[order]["solar_spectrum"]

        nspectra = y.shape[0]
        nbins = y.shape[1]
        npixels = y.shape[2]

        fitted_spectra = np.zeros((nspectra, nbins, npixels))
        corr_spectra = np.zeros((nspectra, nbins, npixels))
        fitted_params = np.zeros((nspectra, nbins, 2))

        print("Fitting blaze functions to spectra")

        for i in range(nspectra):

            if i % 20 == 0:
                print(order, i)

            for j in range(nbins):

                spectrum = y[i, j, :]

                # plt.plot(spectrum/np.max(spectrum))
                # plt.plot(solar_spectrum)

                # now fit the solar spectrum function
                first_guess = [np.max(spectrum), np.mean(spectrum[0:50])]
                # params are: scaler, offset
                scaled_spectrum, corr_spectrum, params = fit_solar_spectrum(spectrum, first_guess, solar_spectrum)

                fitted_spectra[i, j, :] = scaled_spectrum
                corr_spectra[i, j, :] = corr_spectrum
                fitted_params[i, j, :] = params

        d[order]["fitted_spectra"] = fitted_spectra
        d[order]["corr_spectra"] = corr_spectra
        d[order]["params"] = fitted_params
        d[order]["nspectra"] = nspectra
        d[order]["nbins"] = nbins
        d[order]["npixels"] = npixels

    return d


# h5 = "%s_0p1a_LNO_1" % (h5_prefix)
# h5_split = h5.split("_")
# channel = h5_split[3]

# d = get_data(h5_prefix)
# d = bad_px_correction(d)

# for order in orders:

#     # get solar spectrum of nearest temperature to nadir obs
#     solar_spectra_name = "lno_%i_raw_solar_spectra_dict" % order
#     solar_spectra_d = read_hdf5_to_dict(solar_spectra_name)[0]
#     solar_spectra_ts = np.asfarray(list(solar_spectra_d.keys()))

#     ix = np.argmin(np.abs(solar_spectra_ts - h5_t))
#     solar_spectrum = solar_spectra_d[str(solar_spectra_ts[ix])]

#     d[order]["solar_spectrum"] = solar_spectrum

# d = fit_spectra(d, channel)

# plt.figure()
for order in orders:
    # for i in [100]:
    #     plt.figure()
    #     plt.title("%s: Order %i, Spectrum %i" % (h5_prefix, order, i))
    #     plt.xlabel("Pixel number")
    #     plt.ylabel("Uncorrected signal")
    #     plt.grid()
    #     for j in range(d[order]["nbins"]):
    #         plt.plot(d[order]["raw"][i, j, :], color="C%i" % j, alpha=0.5, label="Bin %i" % j)
    #         # plt.plot(d[order]["raw"][i, j, :], color="C%i" % j, linestyle="None", marker="o", alpha=0.3)
    #         plt.plot(d[order]["fitted_spectra"][i, j, :], color="C%i" % j)
    #     plt.legend()

    #     plt.figure()
    #     plt.title("%s: Order %i, Spectrum %i" % (h5_prefix, order, i))
    #     plt.xlabel("Pixel number")
    #     plt.ylabel("Corrected signal")
    #     plt.grid()
    #     for j in range(d[order]["nbins"]):
    #         plt.plot(d[order]["corr_spectra"][i, j, :].T, color="C%i" % j, label="Bin %i" % j)
    #     plt.legend()

    # # plot peak of fitted spectra before linear correction
    # plt.figure()
    # for j in range(d[order]["nbins"]):
    #     fitted = d[order]["fitted_spectra"][:, j, :]
    #     plt.plot(np.max(fitted, axis=1), color="C%i" % j)

    # find linear fit between the peak value of each bin of the fitted spectra
    bin1 = np.max(d[order]["fitted_spectra"][:, 1, :], axis=1)
    bin01_polyfit = np.polyfit(np.max(d[order]["fitted_spectra"][:, 0, :], axis=1), bin1, 1)
    bin21_polyfit = np.polyfit(np.max(d[order]["fitted_spectra"][:, 2, :], axis=1), bin1, 1)
    bin31_polyfit = np.polyfit(np.max(d[order]["fitted_spectra"][:, 3, :], axis=1), bin1, 1)
    # bin01_polyfit = [1.0, 0.0]
    # bin21_polyfit = [1.0, 0.0]
    # bin31_polyfit = [1.0, 0.0]
    print(bin01_polyfit, bin21_polyfit, bin31_polyfit)

    # normalise to bin 1, correct for small offset/gradients between bins
    bin0_polyval = np.polyval(bin01_polyfit, np.max(d[order]["fitted_spectra"][:, 0, :], axis=1))
    bin2_polyval = np.polyval(bin21_polyfit, np.max(d[order]["fitted_spectra"][:, 2, :], axis=1))
    bin3_polyval = np.polyval(bin31_polyfit, np.max(d[order]["fitted_spectra"][:, 3, :], axis=1))

    # # plot scatter plots to check if linear correction is good
    # plt.figure()
    # plt.scatter(bin0_polyval, np.max(d[order]["fitted_spectra"][:, 1, :], axis=1))
    # plt.scatter(bin2_polyval, np.max(d[order]["fitted_spectra"][:, 1, :], axis=1))
    # plt.scatter(bin3_polyval, np.max(d[order]["fitted_spectra"][:, 1, :], axis=1))
    # plt.plot([0, 400], [0, 400])

    # plot the peaks of the 4 bins separately for all acquisitions
    # plt.figure()
    # plt.plot(bin1, color="r")
    # plt.plot(bin0_polyval, color="g")
    # plt.plot(bin2_polyval, color="b")
    # plt.plot(bin3_polyval, color="y")

    # plot mean and std for the bins
    bins_peak = np.asarray([bin0_polyval, bin1, bin2_polyval, bin3_polyval]).T
    # bins_peak = np.asarray([bin1, bin2_polyval]).T
    bin_peak_sum = np.sum(bins_peak, axis=1)
    bin_peak_std = np.std(bins_peak, axis=1) * float(bins_peak.shape[1])  # scale std by number of bins
    d[order]["spectra_peak_sum"] = bin_peak_sum
    d[order]["spectra_peak_std"] = bin_peak_std

    # plt.figure()
    # plt.errorbar(np.arange(len(bin_peak_mean)), bin_peak_sum, bin_peak_std)

    # normalise mean and std to compare each order
    mean_bin_peak_sum = np.mean(bin_peak_sum) * 5  # scaler to be defined
    d[order]["scaler"] = mean_bin_peak_sum
    plt.errorbar(np.arange(len(bin_peak_sum)), bin_peak_sum/mean_bin_peak_sum, bin_peak_std/mean_bin_peak_sum, label="Order %i" % order)
    plt.xlabel("Spectrum index")
plt.grid()
plt.legend()


# finally plot all orders for some measurements
# for i in [18, 19, 43, 42]:
#     plt.figure(figsize=(5, 4))
#     x = np.asarray([10000000/nu_mp(order, 160, -10) for order in orders])
#     y = np.asarray([d[order]["spectra_peak_sum"][i] for order in orders])
#     y_scaler = np.mean(y) / 0.15
#     # y_scaler = np.asarray([d[order]["scaler"] for order in orders])
#     yerr = np.asarray([d[order]["spectra_peak_std"][i] for order in orders])

#     plt.errorbar(x, y/y_scaler, yerr/y_scaler, linestyle="None", marker="x", color="r")
#     plt.ylim([0, 0.25])
#     plt.xlim([375, 2550])
#     frame1 = plt.gca()
#     frame1.axes.xaxis.set_ticklabels([])
#     frame1.axes.yaxis.set_ticklabels([])
#     # plt.savefig("%i.png" % i)
# plt.show()


# plt.plot(np.mean(d[186]["raw"][100, :, :], axis=0), "k--", alpha=0.6)
# plt.plot(np.mean(d[186]["raw"][99:102, :, :], axis=(0, 1)), "k:")
# plt.plot(d[186]["blaze"][100, :, :].T)
# print(d[186]["params"][100, :, :])
# plt.plot(np.asarray(fitted_spectra).T)
# stop()

# peak_mean = np.mean(y_order[:, :, 180:220], axis=2)
# plt.figure(figsize=(15, 6))
# plt.imshow(peak_mean.T, aspect="auto")

# left_mean = np.mean(y_order[:, :, 0:50], axis=2)
# plt.figure(figsize=(15, 6))
# plt.imshow(left_mean.T, aspect="auto")

# plot each bin
# for i in range(y_order.shape[1]):
#     plt.figure()
#     plt.plot(y_order[:, i, :].T)

#     mean = np.mean(y_order[:, i, :], axis=0)
#     std = np.std(y_order[:, i, :], axis=0)

#     plt.plot(mean, "k--")
#     plt.plot(std, "k.-")


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
