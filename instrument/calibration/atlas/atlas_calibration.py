# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:12:13 2025

@author: iant

ANALYSE ATLAS COMET OBS USING NEW LNO OFFSET CORRECTION


"""

# import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from numpy.polynomial import Polynomial
from scipy import interpolate

from instrument.nomad_lno_instrument_v01 import nu_mp
# from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2

# from tools.general.normalise_values_to_range import normalise_values_to_range
# from tools.datasets.get_phobos_crism_data import get_phobos_crism_data

from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order
from instrument.calibration.lno_phobos.lno_offset_correction_phobos import fit_spectra
from instrument.calibration.lno_phobos.lno_bad_pixel_correction_phobos import bad_pixel_correction

from instrument.calibration.lno_phobos.phobos_save_mean_shape_test import save_phobos_ideal_spectrum


# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3a"

px_range = range(120, 280)


"""decide what to plot:
    0 runs through the full calibration plotting everything
    to just plot the final result, select option 4"""

# plot_level = -2  # check signal noise
# plot_level = -1  # check bad pixels

# plot_level = 0
# plot_level = 2
plot_level = 3
# plot_level = 4

# PLOT_TYPES = ["bad_pixel_fits", "solar_fits"]
PLOT_TYPES = []

SAVE_FIGURES = False
# SAVE_FIGURES = True

CLOSE_FIGURES = False
# CLOSE_FIGURES = True

offset_correction = "solar"  # first_bin, last_bin

bad_pixel_type = "iteration"  # dict


"""observation name:{"h5":hdf5 filename, "orders_crism":orders to fit to CRISM spectra}"""

obs_types = {

    "LNO order 169": {"h5": re.compile("20251003_044802_0p3a_LNO_1_X_169")},

}


json_d = {}
cal_d = {}

for obs_type in obs_types.keys():

    h5_pattern = obs_types[obs_type]["h5"]

    if isinstance(h5_pattern, re.Pattern):
        # search for matching files and open them
        h5fs, h5s, _ = make_filelist2(h5_pattern, file_level, path=data_path)
    else:
        # if single filename given, open it
        h5s = [h5_pattern]
        h5fs = [open_hdf5_file(h5_pattern, path=data_path)]

    counts_d = {}
    for h5_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):

        order = 169

        """get solar calibration info from an LNO solar cal fullscan if the data for this order is not yet processed.
        This gives the sensitivity of the instrument in each order and solar radiance"""
        # get solar calibration data
        cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
        cal_d = rad_cal_order(cal_h5, [order], centre_indices=px_range, path=data_path, silent=True)
        cal_d[order]["solar_scalar"] = cal_d[order]["y_centre_mean"] / 2.0e6
        cal_d[order]["solar_spectrum"] = cal_d[order]["y_spectrum"] / np.max(cal_d[order]["y_spectrum"])

        # get detector row binning
        top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
        unique_bins = sorted(list(set(top_bins)))
        n_bins = len(unique_bins)
        binning = unique_bins[1] - unique_bins[0]

        # get y, reshape to 3d
        y_all = h5f["Science/Y"][...]
        y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

        """correct for different total integration times"""
        integration_time = h5f["Channel/IntegrationTime"][0]  # detector row of top of each bin
        n_accs = h5f["Channel/NumberOfAccumulations"][0]  # detector row of top of each bin
        total_integration_time = integration_time * n_accs / 1000  # seconds

        """correct bad pixels"""
        if bad_pixel_type == "iteration":
            print("Correcting bad pixels")
            if "bad_pixel_fits" in PLOT_TYPES:
                y_3d = bad_pixel_correction(y_3d, plot=[[0, 2], [1, 4], [3, 2]])
            else:
                y_3d = bad_pixel_correction(y_3d)

        if plot_level < 1:
            # plot raw y data spectrally binned, after bad pixel but before offset correction
            plt.figure(figsize=(12, 4))
            plt.title("%s: raw Y data, spectrally binned before offset correction" % h5)
            plt.xlabel("Frame index")
            plt.ylabel("Raw signal (counts)")
            for bin_ix in range(n_bins):
                plt.plot(np.mean(y_3d[:, bin_ix, :], axis=1), label="Detector Bin %i" % bin_ix)
            plt.legend()
            plt.grid()

            plt.figure(figsize=(12, 3))
            plt.title("%s: raw Y data, spectrally binned" % h5)
            plt.xlabel("Frame index")
            plt.ylabel("Bin number")
            im = plt.imshow(np.mean(y_3d[:, :, :], axis=2).T, aspect="auto")
            cbar = plt.colorbar(im)
            cbar.set_label("Raw signal (counts)", rotation=270, labelpad=10)
            plt.subplots_adjust(bottom=0.15)
            # plt.savefig("lno_phobos_raw_imshow.png")

        """fit to solar spectrum to correct offset"""
        if offset_correction == "solar":
            print("Applying solar offset correction")
            solar_spectrum = cal_d[order]["solar_spectrum"]
            if "solar_fits" in PLOT_TYPES:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, solar_spectrum, plot=[[1, 3], [1, 4], [1, 5]])
            else:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, solar_spectrum)

        # TODO : check why binning is multiplied
        y_spectral_mean = np.max(corr_spectra, axis=2)  # / total_integration_time * binning
        # print(total_integration_time, binning)

        if plot_level < -1:
            # plot raw data after solar correction for every bin to check for noisy regions
            plt.figure(figsize=(14, 8))
            plt.plot(y_spectral_mean, label=np.arange(y_spectral_mean.shape[1]))
            plt.title(h5)
            plt.legend()
            sys.exit()  # stop execution

        if plot_level < 0:
            # plot all good spectra for a specific bin to check bad pixel removal
            for bin_ in range(y_3d.shape[1]):
                plt.figure(figsize=(14, 8))
                plt.title("Raw spectra for bin %i (detector row %i; binning %i)" % (bin_, unique_bins[bin_], binning))
                plt.xlabel("Pixel number")
                plt.ylabel("Signal (counts)")
                for i, y_spectrum in enumerate(y_3d[:, bin_, :]):
                    plt.plot(y_spectrum, alpha=0.1)
            for bin_ in range(y_3d.shape[1]):
                plt.figure(figsize=(14, 8))
                plt.title("Corrected spectra for bin %i (detector row %i; binning %i)" % (bin_, unique_bins[bin_], binning))
                plt.xlabel("Pixel number")
                plt.ylabel("Signal (counts)")
                for i, y_spectrum in enumerate(corr_spectra[:, bin_, :]):
                    plt.plot(y_spectrum, alpha=0.5)

            sys.exit()  # stop execution

        if plot_level < 1:
            # plot raw y data spectrally binned, after bad pixel and offset correction
            plt.figure(figsize=(12, 3))
            plt.title("%s: raw Y data, spectrally binned" % h5)
            plt.xlabel("Frame index")
            plt.ylabel("Bin number")
            im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
            cbar = plt.colorbar(im)
            cbar.set_label("Raw signal (counts)", rotation=270, labelpad=10)
            plt.subplots_adjust(bottom=0.15)
            # plt.savefig("lno_phobos_raw_imshow.png")

            plt.figure(figsize=(12, 4))
            plt.title("%s: offset corrected Y data, spectrally binned" % h5)
            plt.xlabel("Frame index")
            plt.ylabel("Raw signal (counts)")
            for bin_ix in range(n_bins):
                plt.plot(y_spectral_mean[:, bin_ix], label="Detector Bin %i" % bin_ix)
            plt.legend()
            plt.grid()

        # scale raw counts by solar spectrum
        # y_spectral_solar_scaled = y_spectral_mean / cal_d[phobos_unique_order]["solar_scalar"]

        # add entry to dictionary
        if order not in counts_d.keys():
            counts_d[order] = {}

        # add info to dictionary
        # raw spectra after bad pixel + offset correction
        counts_d[order]["y_3d"] = y_3d
        counts_d[order]["fitted_params"] = fitted_params  # scalar, offset
        # spectral detector pixels for a single order are averaged together
        counts_d[order]["y_spectral_mean"] = y_spectral_mean
        # spectral detector pixels, scaled to solar spectrum
        # counts_d[phobos_unique_order]["y_spectral_solar_scaled"] = y_spectral_solar_scaled

        # average of spectral detector pixels in all frames
        counts_d[order]["y_spectral_frame_mean"] = np.mean(y_spectral_mean, axis=0)
        # stdev of spectral detector pixels in all frames
        counts_d[order]["y_spectral_frame_std"] = np.std(y_spectral_mean, axis=0)
        # average of spectral detector pixels in all frames, scaled to solar spectrum
        # counts_d[phobos_unique_order]["y_spectral_frame_solar_scaled_mean"] = np.mean(y_spectral_solar_scaled, axis=0)
        # stdev of spectral detector pixels in all frames, scaled to solar spectrum
        # counts_d[phobos_unique_order]["y_spectral_frame_solar_scaled_std"] = np.std(y_spectral_solar_scaled, axis=0)

        # check signal on each row
        bin_ratios = counts_d[order]["y_spectral_frame_mean"]
        print("%s: %s" % (h5, str([float(f"{x:.2f}") for x in bin_ratios])))

    mean_shapes_d = save_phobos_ideal_spectrum(counts_d, [0, 1, 2, 3, 4, 5, 6, 7], plot=False, title="%s" % h5, min_spectra=20)

    x = nu_mp(order, np.arange(320), -10)

    plt.figure()
    for bin_ix in mean_shapes_d[order].keys():
        spectrum = mean_shapes_d[order][bin_ix]
        if bin_ix in [3, 4]:
            colour = "k"
            alpha = 1.0
        else:
            colour = "b"
            alpha = 0.5
        plt.plot(x, spectrum, color=colour, alpha=alpha, label="Bin %i" % bin_ix)
    plt.legend()
    plt.title("ATLAS mean spectra: %s" % h5)
    plt.grid()
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Raw signal after bad pixel and offset correction")

    # all data collected, now plot it
    # if plot_level < 2:
    #     # plot solar calibration spectra
    #     plt.figure(figsize=(10, 5))
    #     plt.title("Solar calibration spectra")
    #     plt.plot(cal_d[order]["y_spectrum"], label="Diffraction order %i" % order)
    #     plt.plot([0, len(cal_d[order]["y_spectrum"])-1], [cal_d[order]["y_centre_mean"], cal_d[order]["y_centre_mean"]],
    #              linestyle=":", label="Order %i mean signal" % order)
    #     plt.legend()
    #     plt.xlabel("Detector pixel number")
    #     plt.ylabel("Raw solar signal (counts)")
    #     plt.grid()
    # plt.savefig("lno_phobos_solar_cal_spectra.png")

    if plot_level < 2.5:

        """plot spectrally averaged signal for each bin individually"""
        fig, axes = plt.subplots(nrows=len(bin_ratios), ncols=1, figsize=(10, len(bin_ratios)*5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle("Spectrally binned signal for each order")
        # plt.subplots_adjust(bottom=0.15)

        for axes_ix, bin_ix in enumerate(range(len(bin_ratios))):

            linestyle = "-"

            axes[axes_ix].set_title("Detector bin %i" % bin_ix)
            x_plt = np.arange(counts_d[order]["y_spectral_mean"].shape[0])
            axes[axes_ix].scatter(x_plt, counts_d[order]["y_spectral_mean"][:, bin_ix], label="Order %i" % order)

            axes[axes_ix].set_xlabel("Frame index")
            axes[axes_ix].set_ylabel("Raw signal (counts)")
            if axes_ix == 0:
                axes[axes_ix].legend(loc="upper left")
            axes[axes_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s_lno_atlas_signal.png" % h5)
        if CLOSE_FIGURES:
            plt.close()

    # for order in phobos_unique_orders:

    #     """corect for mean phase angle"""
    #     # TODO: move all this stuff to section above before plotting
    #     # 1 value per binned detector row
    #     counts_d[order]["y_spectral_frame_corr_mean"] = np.zeros(y_spectral_mean.shape[1])
    #     counts_d[order]["y_spectral_frame_corr_std"] = np.zeros(y_spectral_mean.shape[1])
    #     counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"] = np.zeros(y_spectral_mean.shape[1])
    #     counts_d[order]["y_spectral_frame_solar_scaled_corr_std"] = np.zeros(y_spectral_mean.shape[1])

    #     # loop through binned rows where signal is >75% of the best binned detector row
    #     for axes_ix, bin_ix in enumerate(good_row_indices):

    #         linestyle = "-"

    #         if plot_level < 3.0:
    #             axes[axes_ix].set_title("Detector bin %i" % bin_ix)
    #         x_plt = counts_d[order]["phase_angle"][:, bin_ix]
    #         x_mean = np.mean(x_plt)
    #         y = counts_d[order]["y_spectral_mean"][:, bin_ix]

    #         # correct for phase angle by applying pre-calculated gradient
    #         y_plt = y - (x_plt - np.mean(x_plt)) * counts_d[order]["phase_correction"]

    #         # check if a gradient remains in the data after correction
    #         poly = Polynomial.fit(x_plt, y_plt, 1)
    #         yfit = poly(x_plt)

    #         counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] = np.mean(y_plt)
    #         counts_d[order]["y_spectral_frame_corr_std"][bin_ix] = np.std(y_plt)

    #         solar_scalar = cal_d[order]["solar_scalar"]

    #         counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] = counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] / solar_scalar
    #         counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] = counts_d[order]["y_spectral_frame_corr_std"][bin_ix] / solar_scalar

    #         # snr_before = counts_d[order]["y_spectral_frame_mean"][bin_ix] / counts_d[order]["y_spectral_frame_std"][bin_ix]
    #         # snr_after = counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] / counts_d[order]["y_spectral_frame_corr_std"][bin_ix]
    #         # print(order, bin_ix, "SNR:", snr_before, "->", snr_after)

    #         if plot_level < 3.0:
    #             axes[axes_ix].scatter(x_plt, y_plt, color=colour_d[order], label="Order %i" % order)
    #             axes[axes_ix].plot([np.min(x_plt), np.max(x_plt)],
    #                                [np.mean(y_plt), np.mean(y_plt)],
    #                                color=colour_d[order], linestyle=linestyle)

    #             if order == phobos_unique_orders[-1]:
    #                 axes[axes_ix].set_xlabel("Phase angle (degrees)")
    #                 axes[axes_ix].set_ylabel("Signal with phase angle correction (counts)")
    #                 if axes_ix == 0:
    #                     axes[axes_ix].legend(loc="upper left")
    #                 axes[axes_ix].grid()
    #     if plot_level < 3.0:
    #         fig.subplots_adjust(bottom=0.15)
    #         if SAVE_FIGURES:
    #             plt.savefig("%s_lno_phobos_signal_phase_corr.png" % h5_prefix)
    #         if CLOSE_FIGURES:
    #             plt.close()

    # if plot_level < 4:

    #     """plot error bars without and with accounting for phase angle correction"""
    #     fig1, axes1 = plt.subplots(nrows=3, figsize=(10, 9))
    #     # fig1.suptitle("%s: Instrument sensitivity correction" % h5)
    #     fig1.suptitle("LNO Phobos Observation: Instrument Sensitivity and Solar Continuum Correction")
    #     axes1[0].plot(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders], label="Solar calibration scalar")
    #     axes1[0].scatter(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders])
    #     axes1[0].legend(loc="upper left")
    #     axes1[0].grid()
    #     axes1[0].set_ylabel("Signal (counts)")

    #     unique_bins_rem = unique_bins[:]

    #     for bin_ix in range(len(unique_bins_rem)):
    #         axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_mean"][bin_ix] for order in phobos_unique_orders],
    #                           yerr=[counts_d[order]["y_spectral_frame_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
    #                           label="Y counts spectral and frame mean bin %i" % bin_ix, color="C%i" % bin_ix)
    #         axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] for order in phobos_unique_orders],
    #                           yerr=[counts_d[order]["y_spectral_frame_corr_std"][bin_ix] for order in phobos_unique_orders], capsize=2, color="C%i" % bin_ix)
    #     axes1[1].legend(loc="upper left")
    #     axes1[1].grid()
    #     axes1[1].set_ylabel("Signal (counts)")

    #     # plot counts, scaled by instrument sensitivity, for each bin separately with error
    #     for bin_ix in range(len(unique_bins_rem)):
    #         axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_mean"][bin_ix] for order in phobos_unique_orders],
    #                           yerr=[counts_d[order]["y_spectral_frame_solar_scaled_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
    #                           label="Y counts mean scaled to solar bin %i" % bin_ix, color="C%i" % bin_ix)
    #         axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for order in phobos_unique_orders],
    #                           yerr=[counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] for order in phobos_unique_orders],
    #                           capsize=2, color="C%i" % bin_ix)

    #     axes1[2].legend(loc="upper left")
    #     axes1[2].grid()
    #     axes1[2].set_ylabel("Counts scaled to instrument sensitivity")

    #     axes1[2].set_xlabel("Diffraction order")
    #     if SAVE_FIGURES:
    #         plt.savefig("%s_lno_phobos_solar_corr.png" % h5_prefix)
    #     if CLOSE_FIGURES:
    #         plt.close()

    # # now average together good bins
    # counts_d[order]["y_processed_mean"] = counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix]
