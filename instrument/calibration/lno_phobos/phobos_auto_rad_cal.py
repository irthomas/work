# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:53:21 2025

@author: iant

ANALYSE ALL PHOBOS 0.3A DATA USING THE LATEST CORRECTIONS

IDEAS:
    CHECK WHY PHASE ANGLE CORRECTION AROUND 35 DEGREES (OFF-POINTING?)
    ADD ABILITY TO SELECT PHASE ANGLE, LONGITUDINAL RANGES, ETC.
"""

# import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from numpy.polynomial import Polynomial
from scipy import interpolate

# from instrument.nomad_lno_instrument_v01 import m_aotf
# from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2

# from tools.general.normalise_values_to_range import normalise_values_to_range
from tools.datasets.get_phobos_crism_data import get_phobos_crism_data

from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order
from instrument.calibration.lno_phobos.lno_offset_correction_phobos import fit_spectra
from instrument.calibration.lno_phobos.lno_bad_pixel_correction_phobos import bad_pixel_correction


# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3a"

px_range = range(120, 280)


"""decide what to plot:
    0 runs through the full calibration plotting everything
    to just plot the final result, select option 4"""

# plot_level = -2  # check signal noise
# plot_level = -1  # check bad pixels

# plot_level = 0  # raw line plots per bin and raw images after bad pixel, before and after offset correction
# plot_level = 2  # line plots each bin before and after phase correction
# plot_level = 2.5  # line plots after phase angle correction
# plot_level = 3  # solar scaling
plot_level = 4  # crism fit all bins seperately
# plot_level = 4.5 # crism mean of best bins fit only
# plot_level = 5  # nothing

# PLOT_TYPES = ["bad_pixel_fits", "solar_fits"]
PLOT_TYPES = []

# SAVE_FIGURES = False
SAVE_FIGURES = True

# CLOSE_FIGURES = False
CLOSE_FIGURES = True

offset_correction = "solar"  # first_bin, last_bin

bad_pixel_type = "iteration"  # dict

ignore_edge_bins = True  # don't include bins at edge of FOV in analysis (counts unrelibale if FOV slightly off the moon)
# ignore_edge_bins = False

# derived from phase_phase_angle_correction.py
phase_correction_coeff = -0.08374207277211694  # corrected negatives, taking solar spectra mean of pixels, ignoring edge bins and bad file 20250628_083013

# remove data above this phase angle
max_phase_angle = 34.0  # cuts off a couple of high phase angle observations
# max_phase_angle = 180.0

# choose only trailing or leading hemisphere. not yet implemented
max_longitude = -999.0

"""observation name:{"h5":hdf5 filename, "orders_crism":orders to fit to CRISM spectra, "bins":(optional:) manually select the bins}"""

obs_types = {
    # "Hydration band 2 order stepping #1 tracking 60s": {"h5": re.compile("20250628_083013_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},  # investigate
    "Hydration band 2 order stepping #2 tracking 60s": {"h5": re.compile("20250622_070511_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #3 tracking 60s": {"h5": re.compile("20250619_101826_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #4 tracking 60s": {"h5": re.compile("20250619_022647_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},

    "Hydration band 3 order stepping #1 tracking 60s": {"h5": re.compile("20250411_203101_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #2 tracking 60s": {"h5": re.compile("20250408_080101_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #3 tracking 60s": {"h5": re.compile("20250405_190547_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #4 tracking 60s": {"h5": re.compile("20250402_221907_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #5 tracking 60s": {"h5": re.compile("20250402_142717_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},

    "Hydration band 2 order stepping #5 inertial 60s": {"h5": re.compile("20241229_103444_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #6 inertial 60s": {"h5": re.compile("20241213_052910_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #7 inertial 60s": {"h5": re.compile("20241210_084211_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #8 inertial 60s": {"h5": re.compile("20241207_040411_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},

    # carbonates and phyllosilicates don't have a full analysis of the phase angle correction for these orders, to be done later
    # "Carbonates #1 inertial 60s": {"h5": re.compile("20241011_012228_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192], "bins": [1]},
    # "Carbonates #2 inertial 60s": {"h5": re.compile("20241004_235732_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192], "bins": [1]},
    # "Carbonates #3 inertial 60s": {"h5": re.compile("20241001_191914_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192], "bins": [1]},
    # "Carbonates #4 inertial 60s": {"h5": re.compile("20240922_210734_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192], "bins": [1]},
    # "Carbonates #5 inertial 60s": {"h5": re.compile("20240911_224710_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192], "bins": [1]},

    "Hydration band 4 orders #1 inertial 60s": {"h5": re.compile("20240831_034846_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #2 inertial 60s": {"h5": re.compile("20240825_022347_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #3 inertial 60s": {"h5": re.compile("20240819_005847_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #4 inertial 60s": {"h5": re.compile("20240805_163944_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #5 inertial 60s": {"h5": re.compile("20240702_042211_0p3a_LNO_1_.*"), "orders_crism": [169]},
    # "Hydration band 4 orders #6 inertial 60s": {"h5": re.compile("20240627_023139_0p3a_LNO_1_.*"), "orders_crism": [169]},  # investigate
    "Hydration band 4 orders #7 inertial 60s": {"h5": re.compile("20240620_092319_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #8 inertial 60s": {"h5": re.compile("20240614_075827_0p3a_LNO_1_.*"), "orders_crism": [169]},

    # "Phyllosilicates #1 inertial 60s": {"h5": re.compile("20240606_012950_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #2 inertial 60s": {"h5": re.compile("20240531_075626_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #3 inertial 60s": {"h5": re.compile("20240508_064538_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #4 inertial 60s": {"h5": re.compile("20240503_124624_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #5 inertial 60s": {"h5": re.compile("20240502_131156_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #6 inertial 60s": {"h5": re.compile("20240427_112108_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    # "Phyllosilicates #7 inertial 60s": {"h5": re.compile("20240412_193543_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},


}


crism_d = get_phobos_crism_data()


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

    # get diffraction orders from filenames
    phobos_orders = [int(s.split("_")[-1]) for s in h5s]
    phobos_unique_orders = sorted(list(set(phobos_orders)))

    # get solar calibration data
    cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
    cal_d = rad_cal_order(cal_h5, phobos_unique_orders, centre_indices=px_range, path=data_path)
    for order in phobos_unique_orders:
        cal_d[order]["solar_scalar"] = cal_d[order]["y_centre_mean"] / 2.0e6
        cal_d[order]["solar_spectrum"] = cal_d[order]["y_spectrum"] / np.max(cal_d[order]["y_spectrum"])

    h5_prefix = h5s[0][:15]

    # dictionary of colours for plotting orders in same colours
    colour_d = {order: "C%i" % i for i, order in enumerate(phobos_unique_orders)}

    counts_d = {}
    for h5_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):

        phobos_unique_order = int(h5f["Channel/DiffractionOrder"][0])

        """get solar calibration info from an LNO solar cal fullscan if the data for this order is not yet processed.
        This gives the sensitivity of the instrument in each order and solar radiance"""
        if phobos_unique_order not in cal_d.keys():
            cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
            cal_d[phobos_unique_order] = rad_cal_order(cal_h5, phobos_unique_order, centre_indices=px_range, path=data_path)
            cal_d[phobos_unique_order]["solar_scalar"] = cal_d[phobos_unique_order]["y_centre_mean"] / 2.0e6
            cal_d[phobos_unique_order]["solar_spectrum"] = cal_d[phobos_unique_order]["y_spectrum"] / np.max(cal_d[phobos_unique_order]["y_spectrum"])

        # get detector row binning
        top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
        unique_bins = sorted(list(set(top_bins)))
        n_bins = len(unique_bins)
        binning = unique_bins[1] - unique_bins[0]

        # get y, reshape to 3d
        y_all = h5f["Science/Y"][...]
        y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

        # get phase angle
        phase_angle = np.mean(h5f["Geometry/Point0/PhaseAngle"][:, :], axis=1)
        phase_angle[phase_angle == -999] = np.nan
        phase_angle_2d = np.reshape(phase_angle, [-1, n_bins])

        # correct nans in phase angle (needs more investigation why geometry seems wrong)
        nan_frames, nan_rows = np.where(np.isnan(phase_angle_2d))
        for frame, row in zip(nan_frames, nan_rows):
            good_rows = np.where(~np.isnan(phase_angle_2d[frame, :]))[0]
            phase_angle_2d[frame, row] = np.mean(phase_angle_2d[frame, good_rows])

        # additional check for nans in all rows of a frame
        if np.any(np.isnan(phase_angle_2d)):
            for row in range(phase_angle_2d.shape[1]):
                x = np.arange(phase_angle_2d.shape[0])
                y = phase_angle_2d[:, row]
                nan_ixs = np.where(np.isnan(y))[0]
                nnan_ixs = np.where(~np.isnan(y))[0]
                y_new = interpolate.interp1d(x[nnan_ixs], y[nnan_ixs], fill_value="extrapolate")(x)
                phase_angle_2d[:, row] = y_new

        # remove spectra outside of phase angle / longitude range
        # get indices only for first h5 file
        if h5_ix == 0:
            # compare mean phase angle for all bins
            good_phase_ixs = np.where(np.mean(phase_angle_2d, axis=1) < max_phase_angle)[0]

        # resize everything from all files to remove spectra and data
        phase_angle_2d = phase_angle_2d[good_phase_ixs, :]
        y_3d = y_3d[good_phase_ixs, :, :]

        if y_3d.shape[0] == 0:
            # skip if all data removed
            continue

        """correct for different total integration times"""
        integration_time = h5f["Channel/IntegrationTime"][0]  # detector row of top of each bin
        n_accs = h5f["Channel/NumberOfAccumulations"][0]  # detector row of top of each bin
        total_integration_time = integration_time * n_accs / 1000  # seconds

        """correct bad pixels"""
        if bad_pixel_type == "iteration":
            if "bad_pixel_fits" in PLOT_TYPES:
                y_3d = bad_pixel_correction(y_3d, plot=[[0, 2], [1, 4], [3, 2]])
            else:
                y_3d = bad_pixel_correction(y_3d)

        if plot_level < 1:
            # plot raw y data spectrally binned, after bad pixel but before offset correction
            plt.figure(figsize=(12, 4))
            plt.title("%s: raw Y data, spectrally binned" % h5)
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
            solar_spectrum = cal_d[phobos_unique_order]["solar_spectrum"]
            if "solar_fits" in PLOT_TYPES:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, solar_spectrum, plot=[[1, 3], [1, 4], [1, 5]])
            else:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, solar_spectrum)

        # set negatives to nan
        nan_ixs = np.where(corr_spectra == 0.0)
        corr_spectra[nan_ixs] = np.nan

        # TODO : check why binning is multiplied (only matters for 6 bin 4 order obs)
        # y_spectral_mean = np.nanmax(corr_spectra, axis=2) / total_integration_time * binning
        y_spectral_mean = np.mean(corr_spectra[:, :, px_range], axis=2) / total_integration_time * binning
        # print(total_integration_time, binning)

        if plot_level < -1:
            # plot raw data after solar correction for every bin to check for noisy regions
            plt.figure(figsize=(14, 8))
            plt.plot(y_spectral_mean, label=np.arange(y_spectral_mean.shape[1]))
            plt.title(h5)
            plt.legend()
            sys.exit()

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
        y_spectral_solar_scaled = y_spectral_mean / cal_d[phobos_unique_order]["solar_scalar"]

        # add entry to dictionary
        if phobos_unique_order not in counts_d.keys():
            counts_d[phobos_unique_order] = {}

        # add info to dictionary
        # spectral detector pixels for a single order are averaged together
        counts_d[phobos_unique_order]["y_spectral_mean"] = y_spectral_mean
        # spectral detector pixels, scaled to solar spectrum
        counts_d[phobos_unique_order]["y_spectral_solar_scaled"] = y_spectral_solar_scaled

        # average of spectral detector pixels in all frames before phase correction
        counts_d[phobos_unique_order]["y_spectral_frame_mean"] = np.nanmean(y_spectral_mean, axis=0)
        # stdev of spectral detector pixels in all frames
        counts_d[phobos_unique_order]["y_spectral_frame_std"] = np.nanstd(y_spectral_mean, axis=0)
        # average of spectral detector pixels in all frames, scaled to solar spectrum
        counts_d[phobos_unique_order]["y_spectral_frame_solar_scaled_mean"] = np.nanmean(y_spectral_solar_scaled, axis=0)
        # stdev of spectral detector pixels in all frames, scaled to solar spectrum
        counts_d[phobos_unique_order]["y_spectral_frame_solar_scaled_std"] = np.nanstd(y_spectral_solar_scaled, axis=0)

        counts_d[phobos_unique_order]["phase_angle"] = phase_angle_2d
        # correction for phase angle gradient (calculated elsewhere), scaled to solar spectrum
        counts_d[phobos_unique_order]["phase_correction"] = phase_correction_coeff * cal_d[phobos_unique_order]["solar_scalar"]

        # define which are good rows
        # the selection is made on the first file and applied to the other orders
        if h5_ix == 0:

            if "bins" in obs_types[obs_type].keys():
                good_row_indices = obs_types[obs_type]["bins"]
                best_bin_ratio = counts_d[phobos_unique_order]["y_spectral_frame_mean"] / np.nanmax(counts_d[phobos_unique_order]["y_spectral_frame_mean"])

            elif not ignore_edge_bins:
                best_bin_ratio = counts_d[phobos_unique_order]["y_spectral_frame_mean"] / np.nanmax(counts_d[phobos_unique_order]["y_spectral_frame_mean"])
                # if signal is >75% of the best binned detector row, also include it in the analysis
                good_row_indices = np.where(best_bin_ratio > 0.75)[0]
            else:
                bins_to_consider = np.arange(1, len(counts_d[phobos_unique_order]["y_spectral_frame_mean"])-1)
                best_bin_ratio = counts_d[phobos_unique_order]["y_spectral_frame_mean"][bins_to_consider] / \
                    np.nanmax(counts_d[phobos_unique_order]["y_spectral_frame_mean"][bins_to_consider])
                # if signal is >75% of the best binned detector row, also include it in the analysis
                good_row_indices = np.where(best_bin_ratio > 0.75)[0] + 1

            print("%s: rows %s have been selected - ratios:" % (h5_prefix, str(good_row_indices)), [float(f"{x:.2f}") for x in best_bin_ratio])

    # print("Signals on each bin for each order:")
    # for phobos_unique_order in counts_d.keys():
    #     print(phobos_unique_order, ":", counts_d[phobos_unique_order]["y_spectral_frame_mean"])

    # skip whole observation if no spectra (due to phase angle / longitude criteria)
    if len(counts_d) == 0:
        print("Skipping", obs_type)
        continue

    # all data collected, now plot it
    if plot_level < 2:
        # plot solar calibration spectra
        plt.figure(figsize=(10, 5))
        plt.title("Solar calibration spectra")
        for order in phobos_unique_orders:
            plt.plot(cal_d[order]["y_spectrum"], color=colour_d[order], label="Diffraction order %i" % order)
            plt.plot([0, len(cal_d[order]["y_spectrum"])-1], [cal_d[order]["y_centre_mean"], cal_d[order]["y_centre_mean"]],
                     color=colour_d[order], linestyle=":", label="Order %i mean signal" % order)
        plt.legend()
        plt.xlabel("Detector pixel number")
        plt.ylabel("Raw solar signal (counts)")
        plt.grid()
        # plt.savefig("lno_phobos_solar_cal_spectra.png")

    if plot_level < 2.5:

        """plot spectrally averaged signal for each bin individually"""
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(10, len(good_row_indices)*5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle("Spectrally binned signal for each order")
        # plt.subplots_adjust(bottom=0.15)

        for axes_ix, bin_ix in enumerate(good_row_indices):
            for order in phobos_unique_orders:

                linestyle = "-"

                axes[axes_ix].set_title("Detector bin %i" % bin_ix)
                x_plt = np.arange(counts_d[order]["y_spectral_mean"].shape[0])
                axes[axes_ix].scatter(x_plt, counts_d[order]["y_spectral_mean"][:, bin_ix], color=colour_d[order], label="Order %i" % order)
                # axes[axes_ix].plot([np.min(x_plt), np.max(x_plt)],
                #                    [np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix]), np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix])],
                #                    color=colour_d[order], linestyle=linestyle)

            axes[axes_ix].set_xlabel("Frame index")
            axes[axes_ix].set_ylabel("Raw signal (counts)")
            if axes_ix == 0:
                axes[axes_ix].legend(loc="upper left")
            axes[axes_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s_lno_phobos_signal.png" % h5_prefix)
        if CLOSE_FIGURES:
            plt.close()

        """plot vs phase angle"""
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(10, len(good_row_indices)*5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle("Spectrally binned signal for each order and detector row bin before phase correction")
        # plt.subplots_adjust(bottom=0.15)

        for axes_ix, bin_ix in enumerate(good_row_indices):
            for order in phobos_unique_orders:

                linestyle = "-"

                axes[axes_ix].set_title("Detector bin %i" % bin_ix)
                x_plt = counts_d[order]["phase_angle"][:, bin_ix]
                y_plt = counts_d[order]["y_spectral_mean"][:, bin_ix]

                good_ix = np.where(~np.isnan(x_plt) & ~np.isnan(y_plt))

                # linear fit to the data
                poly = Polynomial.fit(x_plt[good_ix], y_plt[good_ix], 1)
                # straight line for plotting
                yfit = poly(x_plt[good_ix])

                axes[axes_ix].scatter(x_plt, y_plt, color=colour_d[order], label="Order %i" % order)
                axes[axes_ix].plot(x_plt[good_ix], yfit, color=colour_d[order], linestyle=linestyle)

            axes[axes_ix].set_xlabel("Phase angle (degrees)")
            axes[axes_ix].set_ylabel("Raw signal (counts)")
            if axes_ix == 0:
                axes[axes_ix].legend(loc="upper left")
            axes[axes_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s_lno_phobos_signal_phase.png" % h5_prefix)
        if CLOSE_FIGURES:
            plt.close()

    if plot_level < 3.0:
        """plot vs phase angle after counts correction"""
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(10, len(good_row_indices)*5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle("Spectrally binned signal for each order and detector row bin after phase correction")

    for order in phobos_unique_orders:

        """corect for mean phase angle"""
        # TODO: move all this stuff to section above before plotting
        # 1 value per binned detector row
        counts_d[order]["y_spectral_frame_corr_mean"] = np.zeros(y_spectral_mean.shape[1])
        counts_d[order]["y_spectral_frame_corr_std"] = np.zeros(y_spectral_mean.shape[1])
        counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"] = np.zeros(y_spectral_mean.shape[1])
        counts_d[order]["y_spectral_frame_solar_scaled_corr_std"] = np.zeros(y_spectral_mean.shape[1])

        # loop through binned rows where signal is >75% of the best binned detector row
        for axes_ix, bin_ix in enumerate(good_row_indices):

            linestyle = "-"

            if plot_level < 3.0:
                axes[axes_ix].set_title("Detector bin %i" % bin_ix)
            x_plt = counts_d[order]["phase_angle"][:, bin_ix]
            x_mean = np.mean(x_plt)
            y = counts_d[order]["y_spectral_mean"][:, bin_ix]

            good_ix = np.where(~np.isnan(x_plt) & ~np.isnan(y))

            # correct for phase angle by applying pre-calculated gradient
            # doesn't change the y values, only reduces the standard deviation
            y_plt = y[good_ix] - (x_plt[good_ix] - np.mean(x_plt)) * counts_d[order]["phase_correction"]

            # check if a gradient remains in the data after correction
            poly = Polynomial.fit(x_plt[good_ix], y_plt, 1)
            yfit = poly(x_plt[good_ix])

            counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] = np.mean(y[good_ix])
            counts_d[order]["y_spectral_frame_corr_std"][bin_ix] = np.std(y_plt)

            solar_scalar = cal_d[order]["solar_scalar"]

            counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] = counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] / solar_scalar
            counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] = counts_d[order]["y_spectral_frame_corr_std"][bin_ix] / solar_scalar

            # snr_before = counts_d[order]["y_spectral_frame_mean"][bin_ix] / counts_d[order]["y_spectral_frame_std"][bin_ix]
            # snr_after = counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] / counts_d[order]["y_spectral_frame_corr_std"][bin_ix]
            # print(order, bin_ix, "SNR:", snr_before, "->", snr_after)

            if plot_level < 3.0:
                axes[axes_ix].scatter(x_plt[good_ix], y_plt, color=colour_d[order], label="Order %i" % order)
                # plot flat line
                axes[axes_ix].plot([np.min(x_plt[good_ix]), np.max(x_plt[good_ix])],
                                   [np.mean(y_plt), np.mean(y_plt)],
                                   color=colour_d[order], linestyle=linestyle)
                # plot linear fit
                axes[axes_ix].plot(x_plt[good_ix], yfit, color=colour_d[order], linestyle=linestyle)

                if order == phobos_unique_orders[-1]:
                    axes[axes_ix].set_xlabel("Phase angle (degrees)")
                    axes[axes_ix].set_ylabel("Signal with phase angle correction (counts)")
                    if axes_ix == 0:
                        axes[axes_ix].legend(loc="upper left")
                    axes[axes_ix].grid()
            # stop()
        if plot_level < 3.0:
            fig.subplots_adjust(bottom=0.15)
            if SAVE_FIGURES:
                plt.savefig("%s_lno_phobos_signal_phase_corr.png" % h5_prefix)
            if CLOSE_FIGURES:
                plt.close()

    # print("Phase corrected signals on each bin for each order:")
    # for order in phobos_unique_orders:
    #     print(order, [counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for bin_ix in good_row_indices])

    if plot_level < 4:

        """plot error bars without and with accounting for phase angle correction"""
        fig1, axes1 = plt.subplots(nrows=3, figsize=(10, 9))
        # fig1.suptitle("%s: Instrument sensitivity correction" % h5)
        fig1.suptitle("LNO Phobos Observation: Instrument Sensitivity and Solar Continuum Correction")
        axes1[0].plot(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders], label="Solar calibration scalar")
        axes1[0].scatter(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders])
        axes1[0].legend(loc="upper left")
        axes1[0].grid()
        axes1[0].set_ylabel("Signal (counts)")

        unique_bins_rem = unique_bins[:]

        for bin_ix in range(len(unique_bins_rem)):
            axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_mean"][bin_ix] for order in phobos_unique_orders],
                              yerr=[counts_d[order]["y_spectral_frame_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
                              label="Y counts spectral and frame mean bin %i" % bin_ix, color="C%i" % bin_ix)
            axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] for order in phobos_unique_orders],
                              yerr=[counts_d[order]["y_spectral_frame_corr_std"][bin_ix] for order in phobos_unique_orders], capsize=2, color="C%i" % bin_ix)
        axes1[1].legend(loc="upper left")
        axes1[1].grid()
        axes1[1].set_ylabel("Signal (counts)")

        # plot counts, scaled by instrument sensitivity, for each bin separately with error
        for bin_ix in range(len(unique_bins_rem)):
            axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_mean"][bin_ix] for order in phobos_unique_orders],
                              yerr=[counts_d[order]["y_spectral_frame_solar_scaled_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
                              label="Y counts mean scaled to solar bin %i" % bin_ix, color="C%i" % bin_ix)
            axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for order in phobos_unique_orders],
                              yerr=[counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] for order in phobos_unique_orders],
                              capsize=2, color="C%i" % bin_ix)

        axes1[2].legend(loc="upper left")
        axes1[2].grid()
        axes1[2].set_ylabel("Counts scaled to instrument sensitivity")

        axes1[2].set_xlabel("Diffraction order")
        if SAVE_FIGURES:
            plt.savefig("%s_lno_phobos_solar_corr.png" % h5_prefix)
        if CLOSE_FIGURES:
            plt.close()

    # now average together good bins
    counts_d[order]["y_processed_mean"] = counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix]

    """scale solar-corrected counts to CRISM"""

    # plot CRISM red and blue first
    if plot_level < 5:

        fig2, ax2a = plt.subplots(figsize=(10, 5))
        # fig2.suptitle("Phobos Radiance Calibration: %s (%s)" % (obs_type, h5))
        fig2.suptitle("Phobos I/F Calibration: Observation %s scaled to CRISM" % h5_prefix)
        ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="tab:red", marker="x", alpha=0.7, label="CRISM Phobos red")
        ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="tab:blue", marker="x", alpha=0.7, label="CRISM Phobos blue")

    # for each good row, get the mean and standard deviation
    good_bin_means = []
    good_bin_stds = []
    for bin_ix in good_row_indices:
        good_bin_means.append([counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for order in phobos_unique_orders])
        good_bin_stds.append([counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] for order in phobos_unique_orders])

    good_bin_means = np.array(good_bin_means)
    good_bin_stds = np.array(good_bin_stds)
    good_bin_snrs = good_bin_means / good_bin_stds
    good_bin_snr = np.sqrt(np.sum(good_bin_snrs**2, axis=0))

    print("SNRs per order:", good_bin_snr)

    """scale to crism"""
    # first interpolate crism to cover spectral gaps
    orders_to_scale = np.array(obs_types[obs_type]["orders_crism"])
    order_ums = [10000./cal_d[order]["x_mean"] for order in orders_to_scale]
    crism_red_f = interpolate.interp1d(crism_d["x"], crism_d["phobos_red"], kind="quadratic")
    crism_blue_f = interpolate.interp1d(crism_d["x"], crism_d["phobos_blue"], kind="quadratic")

    crism_red_interp = crism_red_f(order_ums)
    crism_blue_interp = crism_blue_f(order_ums)

    """plot each bin scaled to crism"""
    if plot_level < 4.5:
        for bin_ix in good_row_indices:
            y_column_mean_norm = {order: counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for order in phobos_unique_orders}
            y_column_std_norm = {order: counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] for order in phobos_unique_orders}

            # divide values for each order by crism interpolated value to that wavelength
            # calculate scaling factor to calibrate to crism red and blue
            # scale each bin individually
            red_scalar = np.mean(crism_red_interp / np.asarray([y_column_mean_norm[order] for order in orders_to_scale]))
            blue_scalar = np.mean(crism_blue_interp / np.asarray([y_column_mean_norm[order] for order in orders_to_scale]))

            y_column_mean_norm_red = {order: y_column_mean_norm[order]*red_scalar for order in y_column_mean_norm.keys()}
            y_column_mean_norm_blue = {order: y_column_mean_norm[order]*blue_scalar for order in y_column_mean_norm.keys()}
            y_column_std_norm_red = {order: y_column_std_norm[order]*red_scalar for order in y_column_std_norm.keys()}
            y_column_std_norm_blue = {order: y_column_std_norm[order]*blue_scalar for order in y_column_std_norm.keys()}

            x_plt = [10000./cal_d[order]["x_mean"] for order in y_column_mean_norm.keys()]

            y_plt1 = np.array([y_column_mean_norm_red[order] for order in y_column_mean_norm.keys()])
            # y_err = np.array([y_column_std_norm_red[order] for order in y_column_std_norm.keys()])
            y_err1 = y_plt1 / good_bin_snr

            # ax2a.plot(x_plt, y_pl1t, color="darkred", label="LNO scaled to Phobos red")
            ax2a.plot(x_plt, y_plt1, color="darkred", alpha=0.3)
            ax2a.scatter(x_plt, y_plt1, color="darkred", alpha=0.3)

            y_plt2 = np.array([y_column_mean_norm_blue[order] for order in y_column_mean_norm.keys()])
            # y_err = np.array([y_column_std_norm_blue[order] for order in y_column_std_norm.keys()])
            y_err2 = y_plt2 / good_bin_snr

            # ax2a.plot(x_plt, y_plt2, color="darkblue", label="LNO scaled to Phobos blue")
            ax2a.plot(x_plt, y_plt2, color="darkblue", alpha=0.3)
            ax2a.scatter(x_plt, y_plt2, color="darkblue", alpha=0.3)

    """plot mean of all good bins"""
    # get column-scaled data for each order
    y_column_mean_norm = {order: np.mean([counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix]
                                         for bin_ix in good_row_indices]) for order in phobos_unique_orders}
    y_column_std_norm = {order: np.mean([counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix]
                                        for bin_ix in good_row_indices]) for order in phobos_unique_orders}

    # divide values for each order by crism interpolated value to that wavelength
    # calculate scaling factor to calibrate to crism red and blue
    # scale mean of bins
    red_scalar = np.mean(crism_red_interp / np.asarray([y_column_mean_norm[order] for order in orders_to_scale]))
    blue_scalar = np.mean(crism_blue_interp / np.asarray([y_column_mean_norm[order] for order in orders_to_scale]))

    y_column_mean_norm_red = {order: y_column_mean_norm[order]*red_scalar for order in y_column_mean_norm.keys()}
    y_column_mean_norm_blue = {order: y_column_mean_norm[order]*blue_scalar for order in y_column_mean_norm.keys()}
    y_column_std_norm_red = {order: y_column_std_norm[order]*red_scalar for order in y_column_std_norm.keys()}
    y_column_std_norm_blue = {order: y_column_std_norm[order]*blue_scalar for order in y_column_std_norm.keys()}

    x_plt = [10000./cal_d[order]["x_mean"] for order in y_column_mean_norm.keys()]

    y_plt1 = np.array([y_column_mean_norm_red[order] for order in y_column_mean_norm.keys()])
    # y_err = np.array([y_column_std_norm_red[order] for order in y_column_std_norm.keys()])
    y_err1 = y_plt1 / good_bin_snr

    if plot_level < 5:

        # ax2a.plot(x_plt, y_pl1t, color="darkred", label="LNO scaled to Phobos red")
        ax2a.errorbar(x_plt, y=y_plt1, yerr=y_err1, color="darkred", capsize=2, label="LNO scaled to Phobos red")
        ax2a.scatter(x_plt, y_plt1, color="darkred")

    y_plt2 = np.array([y_column_mean_norm_blue[order] for order in y_column_mean_norm.keys()])
    # y_err = np.array([y_column_std_norm_blue[order] for order in y_column_std_norm.keys()])
    y_err2 = y_plt2 / good_bin_snr

    if plot_level < 5:

        # ax2a.plot(x_plt, y_plt2, color="darkblue", label="LNO scaled to Phobos blue")
        ax2a.errorbar(x_plt, y=y_plt2, yerr=y_err2, color="darkblue", capsize=2, label="LNO scaled to Phobos blue")
        ax2a.scatter(x_plt, y_plt2, color="darkblue")

    min_phase_angle = np.min([counts_d[order]["phase_angle"] for order in counts_d.keys()])
    mean_phase_angle = np.mean([counts_d[order]["phase_angle"] for order in counts_d.keys()])

    if plot_level < 5:

        ax2a.text(0.5, 0.06, "Phase angle min=%0.2f, mean=%0.2f" % (min_phase_angle, mean_phase_angle))

        ax2a.grid()
        ax2a.legend()
        ax2a.set_ylim((0.0, 0.1))
        ax2a.set_xlabel("Wavelength (microns)")
        ax2a.set_ylabel("CRISM Phobos I/F (Fraeman 2014)")
        fig2.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s_phobos_radcal.png" % h5_prefix)
        if CLOSE_FIGURES:
            plt.close()

    # lines = []
    json_d[h5_prefix] = {
        "orders": [int(i) for i in y_column_mean_norm.keys()],
        "um": x_plt,
        "norm_vals": [float(f) for f in y_column_mean_norm.values()],
        "norm_stds": [float(f) for f in y_column_std_norm.values()],
        "red_scaled": [float(f) for f in y_plt1],
        "blue_scaled": [float(f) for f in y_plt2],
    }

with open("lno_phobos_output.json", "w", encoding="utf-8") as f:
    json.dump(json_d, f, ensure_ascii=False, indent=4)
