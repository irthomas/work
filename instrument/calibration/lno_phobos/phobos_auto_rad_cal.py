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
from instrument.calibration.lno_phobos.phobos_save_mean_shape_test import save_phobos_ideal_spectrum


# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3a"

# pixel range to consider for mean of high res spectra
px_range = range(190, 210)
# px_range = range(150, 250)  # full peak centred
# px_range = range(120, 280)
# px_range = [-1]  # take maximum of raw spectrum
# px_range = range(320)

"""decide what to plot:
    0 runs through the full calibration plotting everything
    to just plot the final result, select option 4 or 4.5"""

# plot_level = -2  # check signal noise
# plot_level = -1  # check bad pixels

# plot_level = 0  # raw line plots per bin and raw images after bad pixel, before and after offset correction
# plot_level = 2  # line plots each bin before and after phase correction
# plot_level = 2.5  # line plots after phase angle correction
# plot_level = 3  # solar scaling
# plot_level = 4  # crism fit all bins seperately
plot_level = 4.5  # crism mean of best bins fit only
# plot_level = 5  # nothing

# PLOT_TYPES = ["bad_pixel_fits", "offset_fits"]
# PLOT_TYPES = ["offset_fits"]
PLOT_TYPES = []

# SAVE_FIGURES = False
SAVE_FIGURES = True

# CLOSE_FIGURES = False
CLOSE_FIGURES = True

# offset_correction = "solar"  # first_bin, last_bin not yet implemented in this version
offset_correction = "mean_shape"  # get mean phobos spectral shape from a json

bad_pixel_type = "iteration"

ignore_edge_bins = True  # don't include bins at edge of FOV in analysis (counts unrelibale if FOV slightly off the moon)
# ignore_edge_bins = False

# derived from phase_phase_angle_correction.py
# phase_correction_coeff = -0.08374207277211694  # corrected negatives, taking solar spectra mean of pixels, ignoring edge bins and bad file 20250628_083013
# phase_correction_coeff = -0.06617203722695901  # different phase correction for orders 174+?
phase_correction_coeff = -0.07026119494783746  # 190-210 pixel range and phobos mean spectrum phase shift

# remove data above this phase angle
max_phase_angle = 34.0  # cuts off a couple of high phase angle observations
# max_phase_angle = 180.0

# choose only trailing or leading hemisphere. not yet implemented
max_longitude = -999.0

"""observation name:{"h5":hdf5 filename, "orders_crism":orders to fit to CRISM spectra, "bins":(optional:) manually select the bins}"""


obs_types = {
    # # 160 162 164 166 168 170
    # # order 168 very spiky
    "Hydration band 2 order stepping #1 tracking 60s": {"h5": re.compile("20250628_083013_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170], "skip": True},
    # # # # this one for the cal paper
    "Hydration band 2 order stepping #2 tracking 60s": {"h5": re.compile("20250622_070511_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #3 tracking 60s": {"h5": re.compile("20250619_101826_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #4 tracking 60s": {"h5": re.compile("20250619_022647_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},

    # # 157 160 163 166 169 172
    "Hydration band 3 order stepping #1 tracking 60s": {"h5": re.compile("20250411_203101_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #2 tracking 60s": {"h5": re.compile("20250408_080101_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #3 tracking 60s": {"h5": re.compile("20250405_190547_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},
    "Hydration band 3 order stepping #4 tracking 60s": {"h5": re.compile("20250402_221907_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172], "skip": True},
    "Hydration band 3 order stepping #5 tracking 60s": {"h5": re.compile("20250402_142717_0p3a_LNO_1_.*"), "orders_crism": [166, 169, 172]},

    # # 160 162 164 166 168 170
    # # all very low signal
    "Hydration band 2 order stepping #5 inertial 60s": {"h5": re.compile("20241229_103444_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},
    "Hydration band 2 order stepping #6 inertial 60s": {"h5": re.compile("20241213_052910_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170], "skip": True},
    "Hydration band 2 order stepping #7 inertial 60s": {"h5": re.compile("20241210_084211_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170], "skip": True},
    "Hydration band 2 order stepping #8 inertial 60s": {"h5": re.compile("20241207_040411_0p3a_LNO_1_.*"), "orders_crism": [166, 168, 170]},

    # # carbonates and phyllosilicates don't have a full analysis of the phase angle correction for these orders, to be done later
    "Carbonates #1 inertial 60s": {"h5": re.compile("20241011_012228_0p3a_LNO_1_P_17."), "orders_crism": [174, 175, 176]},
    "Carbonates #2 inertial 60s": {"h5": re.compile("20241004_235732_0p3a_LNO_1_P_17."), "orders_crism": [174, 175, 176]},
    "Carbonates #3 inertial 60s": {"h5": re.compile("20241001_191914_0p3a_LNO_1_P_17."), "orders_crism": [174, 175, 176]},
    "Carbonates #4 inertial 60s": {"h5": re.compile("20240922_210734_0p3a_LNO_1_P_17."), "orders_crism": [174, 175, 176]},
    "Carbonates #5 inertial 60s": {"h5": re.compile("20240911_224710_0p3a_LNO_1_P_17."), "orders_crism": [174, 175, 176]},

    "Carbonates #1b inertial 60s": {"h5": re.compile("20241011_012228_0p3a_LNO_1_P_19."), "orders_crism": [190, 191, 192], "suffix": "b"},
    "Carbonates #2b inertial 60s": {"h5": re.compile("20241004_235732_0p3a_LNO_1_P_19."), "orders_crism": [190, 191, 192], "suffix": "b"},
    "Carbonates #3b inertial 60s": {"h5": re.compile("20241001_191914_0p3a_LNO_1_P_19."), "orders_crism": [190, 191, 192], "suffix": "b"},
    "Carbonates #4b inertial 60s": {"h5": re.compile("20240922_210734_0p3a_LNO_1_P_19."), "orders_crism": [190, 191, 192], "suffix": "b"},
    "Carbonates #5b inertial 60s": {"h5": re.compile("20240911_224710_0p3a_LNO_1_P_19."), "orders_crism": [190, 191, 192], "suffix": "b"},

    # "Carbonates #1 inertial 60s": {"h5": re.compile("20241011_012228_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192]},
    # "Carbonates #2 inertial 60s": {"h5": re.compile("20241004_235732_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192]},
    # "Carbonates #3 inertial 60s": {"h5": re.compile("20241001_191914_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192]},
    # "Carbonates #4 inertial 60s": {"h5": re.compile("20240922_210734_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192]},
    # "Carbonates #5 inertial 60s": {"h5": re.compile("20240911_224710_0p3a_LNO_1_.*"), "orders_crism": [174, 175, 176, 190, 191, 192]},

    # 163 165 167 169
    # bin 2 mean shape spectra are much better
    "Hydration band 4 orders #1 inertial 60s": {"h5": re.compile("20240831_034846_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #2 inertial 60s": {"h5": re.compile("20240825_022347_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #3 inertial 60s": {"h5": re.compile("20240819_005847_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #4 inertial 60s": {"h5": re.compile("20240805_163944_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #5 inertial 60s": {"h5": re.compile("20240702_042211_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #6 inertial 60s": {"h5": re.compile("20240627_023139_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #7 inertial 60s": {"h5": re.compile("20240620_092319_0p3a_LNO_1_.*"), "orders_crism": [169]},
    "Hydration band 4 orders #8 inertial 60s": {"h5": re.compile("20240614_075827_0p3a_LNO_1_.*"), "orders_crism": [169]},

    # # 189, 190, 191, 192, 193, 201
    "Phyllosilicates #1 inertial 60s": {"h5": re.compile("20240606_012950_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    "Phyllosilicates #2 inertial 60s": {"h5": re.compile("20240531_075626_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    "Phyllosilicates #3 inertial 60s": {"h5": re.compile("20240508_064538_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    "Phyllosilicates #4 inertial 60s": {"h5": re.compile("20240503_124624_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    "Phyllosilicates #5 inertial 60s": {"h5": re.compile("20240502_131156_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201]},
    "Phyllosilicates #6 inertial 60s": {"h5": re.compile("20240427_112108_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201], "skip": True},
    "Phyllosilicates #7 inertial 60s": {"h5": re.compile("20240412_193543_0p3a_LNO_1_.*"), "orders_crism": [189, 190, 191, 192, 193, 201], "skip": True},

    # "Early test": {"h5": re.compile("20211123_173753_0p3a_LNO_1_P.*")},
}


crism_d = get_phobos_crism_data()


json_d = {}
cal_d = {}

mean_shapes_d = {}

for obs_type in obs_types.keys():

    if "skip" in obs_types[obs_type].keys():
        skip_obs = obs_types[obs_type]["skip"]
    else:
        skip_obs = False
    if skip_obs:
        print("Skipping %s" % obs_type)
        continue

    if "suffix" in obs_types[obs_type].keys():
        suffix = obs_types[obs_type]["suffix"]
    else:
        suffix = ""

    h5_pattern = obs_types[obs_type]["h5"]

    if isinstance(h5_pattern, re.Pattern):
        # search for matching files and open them
        h5fs, h5s, _ = make_filelist2(h5_pattern, file_level, path=data_path, silent=True)
    else:
        # if single filename given, open it
        h5s = [h5_pattern]
        h5fs = [open_hdf5_file(h5_pattern, path=data_path, silent=True)]

    # get diffraction orders from filenames
    phobos_orders = [int(s.split("_")[-1]) for s in h5s]
    phobos_unique_orders = sorted(list(set(phobos_orders)))

    # get solar calibration data
    cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
    cal_d = rad_cal_order(cal_h5, phobos_unique_orders, centre_indices=px_range, path=data_path, silent=True)
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
        # if phobos_unique_order not in cal_d.keys():
        #     cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
        #     cal_d[phobos_unique_order] = rad_cal_order(cal_h5, phobos_unique_order, centre_indices=px_range, path=data_path, silent=True)
        #     cal_d[phobos_unique_order]["solar_scalar"] = cal_d[phobos_unique_order]["y_centre_mean"] / 2.0e6
        #     cal_d[phobos_unique_order]["solar_spectrum"] = cal_d[phobos_unique_order]["y_spectrum"] / np.max(cal_d[phobos_unique_order]["y_spectrum"])

        # get detector row binning
        top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
        unique_bins = sorted(list(set(top_bins)))
        n_bins = len(unique_bins)
        binning = unique_bins[1] - unique_bins[0]

        # get y, reshape to 3d
        y_all = h5f["Science/Y"][...]
        y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

        if h5_ix == 0:
            ts = h5f["Channel/InterpolatedTemperature"][...]
            print("%s%s temperature range: %0.1f %0.1f" % (h5_prefix, suffix, np.min(ts), np.max(ts)))

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
            plt.title("%s: raw Y data, spectrally binned" % h5_prefix)
            plt.xlabel("Detector frame index (temporal direction")
            plt.ylabel("Raw signal (counts)")
            for bin_ix in range(n_bins):
                plt.plot(np.mean(y_3d[:, bin_ix, :], axis=1), label="Detector Bin %i" % bin_ix)
            plt.legend()
            plt.grid()

            plt.figure(figsize=(12, 6))
            plt.title("NOMAD-LNO observation %s: raw Y data, spectrally binned, before offset correction" % h5_prefix)
            plt.xlabel("Detector frame index (temporal direction)")
            plt.ylabel("Binned detector row index (spatial direction)")
            im = plt.imshow(np.mean(y_3d[:, :, :], axis=2).T, aspect="auto")
            cbar = plt.colorbar(im)
            cbar.set_label("Spectrally binned signal on each detector row bin (counts)", rotation=270, labelpad=10)
            plt.subplots_adjust(bottom=0.15)
            # plt.savefig("lno_phobos_raw_imshow.png")

        """fit to solar spectrum to correct offset"""
        if offset_correction == "solar":
            ideal_spectrum = cal_d[phobos_unique_order]["solar_spectrum"]

        elif offset_correction == "mean_shape":
            # load json already prepopulated with mean shapes
            with open("lno_phobos_mean_shapes.json", "r") as f:
                mean_shape_d = json.load(f)
            mean_shape_d = {int(k): np.asarray(v) for k, v in mean_shape_d[h5_prefix].items()}
            ideal_spectrum = mean_shape_d[phobos_unique_order]

        if offset_correction in ["solar", "mean_shape"]:
            if "offset_fits" in PLOT_TYPES:
                title = "NOMAD-LNO observation %s:" % h5_prefix
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, ideal_spectrum, plot=[[1, 4], [1, 5]], title=title)
            else:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, ideal_spectrum)

        # set negatives to nan
        nan_ixs = np.where(corr_spectra == 0.0)
        corr_spectra[nan_ixs] = np.nan

        # TODO : check why binning is multiplied (only matters for 6 bin 4 order obs)
        if px_range[0] == -1:
            y_spectral_mean = np.max(corr_spectra[:, :, :], axis=2) / total_integration_time * binning
        else:
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
            plt.figure(figsize=(12, 6))
            plt.title("NOMAD-LNO observation %s: raw Y data, spectrally binned, after offset correction" % h5_prefix)
            plt.xlabel("Detector frame index (temporal direction)")
            plt.ylabel("Binned detector row index (spatial direction)")
            im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
            cbar = plt.colorbar(im)
            cbar.set_label("Spectrally binned signal on each detector row bin (counts)", rotation=270, labelpad=10)
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

            # plot the mean before and after offset correction
            if px_range[0] == -1:
                mean_before = np.max(y_3d[:, :, :], axis=2) / total_integration_time * binning
            else:
                mean_before = np.mean(y_3d[:, :, px_range], axis=2) / total_integration_time * binning
            mean_rows_before = np.mean(mean_before, axis=0)
            std_rows_before = np.std(np.concatenate((mean_rows_before[:10], mean_rows_before[13:])))

            mean_rows_after = np.mean(y_spectral_mean, axis=0)
            std_rows_after = np.std(np.concatenate((mean_rows_after[:10], mean_rows_after[13:])))

            plt.figure(figsize=(11, 4))
            plt.title("NOMAD-LNO observation %s: mean counts per bin, averaging all detector frames" % h5_prefix)
            plt.plot(mean_rows_before, label="Uncorrected signal", color="C0", marker="o")
            plt.plot(mean_rows_after, label="Offset-corrected signal", color="C1", marker="o")
            plt.axhline(std_rows_before, label="Stdev of uncorrected signal", color="C0", linestyle="--")
            plt.axhline(-std_rows_before, color="C0", linestyle="--")
            plt.axhline(std_rows_after, label="Stdev of uncorrected signal", color="C1", linestyle="--")
            plt.axhline(-std_rows_after, color="C1", linestyle="--")
            plt.legend()
            plt.grid()
            plt.xlabel("Binned detector row index (spatial direction)")
            plt.ylabel("Mean signal on each binned detector row")

        # scale raw counts by solar spectrum
        y_spectral_solar_scaled = y_spectral_mean / cal_d[phobos_unique_order]["solar_scalar"]

        # add entry to dictionary
        if phobos_unique_order not in counts_d.keys():
            counts_d[phobos_unique_order] = {}

        # add info to dictionary
        # raw spectra after bad pixel + offset correction
        counts_d[phobos_unique_order]["y_3d"] = y_3d
        counts_d[phobos_unique_order]["fitted_params"] = fitted_params  # scalar, offset
        # spectral detector pixels for a single order are averaged together
        counts_d[phobos_unique_order]["y_spectral_mean"] = y_spectral_mean  # same as the
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
                bins_to_consider = np.arange(len(counts_d[phobos_unique_order]["y_spectral_frame_mean"]))

            elif not ignore_edge_bins:
                best_bin_ratio = counts_d[phobos_unique_order]["y_spectral_frame_mean"] / np.nanmax(counts_d[phobos_unique_order]["y_spectral_frame_mean"])
                # if signal is >75% of the best binned detector row, also include it in the analysis
                good_row_indices = np.where(best_bin_ratio > 0.75)[0]
                bins_to_consider = np.arange(len(counts_d[phobos_unique_order]["y_spectral_frame_mean"]))
            else:
                bins_to_consider = np.arange(1, len(counts_d[phobos_unique_order]["y_spectral_frame_mean"])-1)
                best_bin_ratio = counts_d[phobos_unique_order]["y_spectral_frame_mean"][bins_to_consider] / \
                    np.nanmax(counts_d[phobos_unique_order]["y_spectral_frame_mean"][bins_to_consider])
                # if signal is >75% of the best binned detector row, also include it in the analysis
                good_row_indices = np.where(best_bin_ratio > 0.75)[0] + 1

            print("%s%s: rows %s have been selected - ratios:" % (h5_prefix, suffix, str(good_row_indices)), [float(f"{x:.2f}") for x in best_bin_ratio])

    # print("Signals on each bin for each order:")
    # for phobos_unique_order in counts_d.keys():
    #     print(phobos_unique_order, ":", counts_d[phobos_unique_order]["y_spectral_frame_mean"])

    # skip whole observation if no spectra (due to phase angle / longitude criteria)
    if len(counts_d) == 0:
        print("Skipping", obs_type)
        continue

    # check shape of high-res Phobos spectra, saving mean spectral shape for the chosen bins
    # mean_shape_d = save_phobos_ideal_spectrum(counts_d, plot=True, title="%s" % h5_prefix)
    if "solar" in offset_correction:
        mean_shapes_d["%s%s" % (h5_prefix, suffix)] = save_phobos_ideal_spectrum(counts_d, good_row_indices, plot=False, title="%s" % h5_prefix)

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
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(11, len(good_row_indices)*4), squeeze=False)
        axes = axes.flatten()
        # fig.suptitle("Spectrally binned signal for each order")
        # plt.subplots_adjust(bottom=0.15)

        for axes_ix, bin_ix in enumerate(good_row_indices):
            for order in phobos_unique_orders:

                linestyle = "-"

                axes[axes_ix].set_title("NOMAD-LNO observation %s: Counts per order after offset correction" % h5_prefix)
                x_plt = np.arange(counts_d[order]["y_spectral_mean"].shape[0])
                axes[axes_ix].scatter(x_plt, counts_d[order]["y_spectral_mean"][:, bin_ix], color=colour_d[order], label="Order %i" % order)
                # axes[axes_ix].plot([np.min(x_plt), np.max(x_plt)],
                #                    [np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix]), np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix])],
                #                    color=colour_d[order], linestyle=linestyle)

            axes[axes_ix].set_xlabel("Detector frame index")
            axes[axes_ix].set_ylabel("Offset-corrected signal (counts)")
            if axes_ix == 0:
                axes[axes_ix].legend(loc="upper left")
            axes[axes_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s%s_lno_phobos_signal.png" % (h5_prefix, suffix))
        if CLOSE_FIGURES:
            plt.close()

        """plot vs phase angle"""
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(11, len(good_row_indices)*4), squeeze=False)
        axes = axes.flatten()
        # fig.suptitle("Spectrally binned signal for each order and detector row bin before phase correction")
        # plt.subplots_adjust(bottom=0.15)

        for axes_ix, bin_ix in enumerate(good_row_indices):
            for order in phobos_unique_orders:

                linestyle = "-"

                axes[axes_ix].set_title("NOMAD-LNO observation %s: Counts per order after offset correction" % h5_prefix)
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
            axes[axes_ix].set_ylabel("Offset-corrected signal (counts)")
            if axes_ix == 0:
                axes[axes_ix].legend(loc="upper left")
            axes[axes_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s%s_lno_phobos_signal_phase.png" % (h5_prefix, suffix))
        if CLOSE_FIGURES:
            plt.close()

    if plot_level < 3.0:
        """plot vs phase angle after counts correction"""
        fig, axes = plt.subplots(nrows=len(good_row_indices), ncols=1, figsize=(11, len(good_row_indices)*4), squeeze=False)
        axes = axes.flatten()
        # fig.suptitle("Spectrally binned signal for each order and detector row bin after phase correction")

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
                axes[axes_ix].set_title("NOMAD-LNO observation %s: Counts per order after phase angle correction" % h5_prefix)
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
                # axes[axes_ix].plot([np.min(x_plt[good_ix]), np.max(x_plt[good_ix])],
                #                    [np.mean(y_plt), np.mean(y_plt)],
                #                    color=colour_d[order], linestyle=linestyle)
                # plot linear fit
                axes[axes_ix].plot(x_plt[good_ix], yfit, color=colour_d[order], linestyle=linestyle)

                if order == phobos_unique_orders[-1]:
                    axes[axes_ix].set_xlabel("Phase angle (degrees)")
                    axes[axes_ix].set_ylabel("Signal after phase angle correction (counts)")
                    if axes_ix == 0:
                        axes[axes_ix].legend(loc="upper left")
                    axes[axes_ix].grid()
            # stop()
        if plot_level < 3.0:
            fig.subplots_adjust(bottom=0.15)
            if SAVE_FIGURES:
                plt.savefig("%s%s_lno_phobos_signal_phase_corr.png" % (h5_prefix, suffix))
            if CLOSE_FIGURES:
                plt.close()

    # print("Phase corrected signals on each bin for each order:")
    # for order in phobos_unique_orders:
    #     print(order, [counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for bin_ix in good_row_indices])

    if plot_level < 4:

        """plot error bars without and with accounting for phase angle correction"""
        fig1, axes1 = plt.subplots(nrows=3, figsize=(11, 8), constrained_layout=True)
        # fig1.suptitle("%s: Instrument sensitivity correction" % h5)
        fig1.suptitle("NOMAD-LNO observation %s: instrument sensitivity and solar spectrum correction" % h5_prefix)
        axes1[0].plot(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders], label="Solar calibration scalar")
        axes1[0].scatter(phobos_unique_orders, [cal_d[order]["y_centre_mean"] for order in phobos_unique_orders])
        axes1[0].legend(loc="upper left")
        axes1[0].grid()
        axes1[0].set_ylabel("Solar signal (counts)")

        unique_bins_rem = unique_bins[:]

        for bin_ix in range(len(unique_bins_rem)):
            if bin_ix in bins_to_consider:
                if bin_ix not in good_row_indices:
                    axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_mean"][bin_ix] for order in phobos_unique_orders],
                                      yerr=[counts_d[order]["y_spectral_frame_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
                                      label="Y counts spectral and frame mean, bin %i" % bin_ix, color="C%i" % bin_ix)
                else:
                    axes1[1].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_corr_mean"][bin_ix] for order in phobos_unique_orders],
                                      yerr=[counts_d[order]["y_spectral_frame_corr_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
                                      label="Y counts spectral and frame mean, bin %i" % bin_ix, color="C%i" % bin_ix)
        axes1[1].legend(loc="upper left")
        axes1[1].grid()
        axes1[1].set_ylabel("Phobos signal (counts)")

        # plot counts, scaled by instrument sensitivity, for each bin separately with error
        for bin_ix in range(len(unique_bins_rem)):
            if bin_ix in bins_to_consider:
                if bin_ix not in good_row_indices:
                    axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_mean"][bin_ix] for order in phobos_unique_orders],
                                      yerr=[counts_d[order]["y_spectral_frame_solar_scaled_std"][bin_ix] for order in phobos_unique_orders], capsize=2,
                                      label="Y counts mean scaled to solar, bin %i" % bin_ix, color="C%i" % bin_ix)
                else:
                    axes1[2].errorbar(phobos_unique_orders, [counts_d[order]["y_spectral_frame_solar_scaled_corr_mean"][bin_ix] for order in phobos_unique_orders],
                                      yerr=[counts_d[order]["y_spectral_frame_solar_scaled_corr_std"][bin_ix] for order in phobos_unique_orders],
                                      capsize=2, label="Y counts mean scaled to solar, bin %i" % bin_ix, color="C%i" % bin_ix)

        axes1[2].legend(loc="upper left")
        axes1[2].grid()
        axes1[2].set_ylabel("Phobos signal scaled by\ninstrument sensitivity")

        axes1[2].set_xlabel("Diffraction order")
        if SAVE_FIGURES:
            plt.savefig("%s%s_lno_phobos_solar_corr.png" % (h5_prefix, suffix))
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

        # ax2a.text(0.5, 0.06, "Phase angle min=%0.2f, mean=%0.2f" % (min_phase_angle, mean_phase_angle))

        ax2a.grid()
        ax2a.legend()
        ax2a.set_ylim((0.0, 0.1))
        ax2a.set_xlabel("Wavelength (microns)")
        ax2a.set_ylabel("CRISM Phobos I/F (Fraeman 2014)")
        fig2.subplots_adjust(bottom=0.15)
        if SAVE_FIGURES:
            plt.savefig("%s%s_phobos_radcal.png" % (h5_prefix, suffix))
        if CLOSE_FIGURES:
            plt.close()

    # lines = []
    json_d["%s%s" % (h5_prefix, suffix)] = {
        "orders": [int(i) for i in y_column_mean_norm.keys()],
        "um": x_plt,
        "norm_vals": [float(f) for f in y_column_mean_norm.values()],
        "norm_stds": [float(f) for f in y_column_std_norm.values()],
        "red_scaled": [float(f) for f in y_plt1],
        "blue_scaled": [float(f) for f in y_plt2],
    }

with open("lno_phobos_output.json", "w", encoding="utf-8") as f:
    json.dump(json_d, f, ensure_ascii=False, indent=4)

if "solar" in offset_correction:
    with open("lno_phobos_shape_output.json", "w", encoding="utf-8") as f:
        json.dump(mean_shapes_d, f, ensure_ascii=False, indent=4)
