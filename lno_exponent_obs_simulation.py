# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:15:22 2025

@author: iant

LNO EXPONENT SIMULATION

CHECK EXPONENTS VS SIGNAL AND TEMPERATURE
AIM IS TO BE ABLE TO PREDICT EXPONENT - SHOULD BE DOMINATED BY INST TEMPERATURE BUT ALSO BAD PIXELS AND SIGNAL
BAD PIXELS ARE PARTICULARLY DIFFICULT AS THEY CHANGE RANDOMLY

ONE EXPONENT PER SUBDOMAIN (FRAME)
BAD PIXELS GOING ABOVE THE 2**11*N (2048*N) CUTOFF WILL INCREASE THE EXPONENT BY 1

NACCS, INTT, BINNING, PX_INT, TEMP
14	200	11	30800	-9 ALL EXPS=5
14	200	11	30800	-16 EXPS ARE HALF 4 & HALF 5
9	205	17	31365	-3.5 EXPS 5->6
9	205	17	31365	-2 EXPS ARE HALF 5 & HALF 6


BINNING EXPONENT TEMPERATURE
12      5        -14.4 +- 1.3
18      6        -0.8 +- 0.3
24      6        -8.6 +- 0.9


BIAS IS 1146 +- 3 COUNTS PER DETECTOR ROW I.E. MULTIPLY BY NUMBER OF ROWS TO GET CORRECT VALUE




"""
import re
import numpy as np
from datetime import datetime
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2
# # from tools.file.download_h5 import download_h5
from tools.general.progress_bar import progress

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tools.sql.heaters_temp import get_temperature_range
from numpy.polynomial import Polynomial

import math

NOISE_AMP = 2.0  # for "20190102_092151_0p1d_LNO_1_D_169" mid_ix = 125
# NOISE_AMP = 1.9  # for "20190102_092151_0p1d_LNO_1_D_169" mid_ix = 125
CORR_FACT = 1.03  # for "20190102_092151_0p1d_LNO_1_D_169" mid_ix = 125

# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"
# # download_h5(h5, path=r"C:\Users\iant\Documents\DATA\hdf5")


""" load a random LNO nadir spectrum"""
# mean counts 500 around px 180-220 for binning=12, 14 accs and 200ms IT with 33600 ms px (int * naccs/2 * binning+1)
# std around 36 counts across all pixels, exp=5

# get a mean spectrum and standard dev
ref_h5 = "20190102_092151_0p1d_LNO_1_D_169"
ref_h5f = open_hdf5_file(ref_h5, path=data_path)

# get detector row binning
top_bins = ref_h5f["Science/Bins"][:, 0]  # detector row of top of each bin
unique_bins = sorted(list(set(top_bins)))
n_bins = len(unique_bins)

binning = int(ref_h5f["Channel/Binning"][0] + 1)
intt = ref_h5f["Channel/IntegrationTime"][0]
nacc = ref_h5f["Channel/NumberOfAccumulations"][0]
intt_acq = intt * int(nacc / 2)

# real detector bins
sim_d = {"intt": intt, "wtop": unique_bins[0], "wbottom": unique_bins[-1] + binning, "nacc": nacc, "binning": binning, "nreps": 1}
sim_d["wheight"] = sim_d["wbottom"] - sim_d["wtop"]
sim_d["nsubd"] = 24 / (sim_d["wheight"] / sim_d["binning"])

# get y, reshape to 3d
y_all = ref_h5f["Science/Y"][...]
y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

mid_ix = int(y_3d.shape[0] / 2)
meas_spectrum = np.sum(y_3d[mid_ix, :, :], axis=0)  # mean of bins in frame

exponent = ref_h5f["Channel/Exponent"][mid_ix]

smoothed = savgol_filter(meas_spectrum, 11, 2)  # smoothed but follows absorptions
smoothed2 = savgol_filter(meas_spectrum, 51, 2)  # general shape only
# get error vs heavily smoothed spectrum
y_errors = meas_spectrum - smoothed2

# get the general smoothed shape of an LNO spectrum after bg subtraction
LNO_SHAPE = smoothed / sim_d["wheight"] / intt_acq  # counts per row per millisecond per acc


def execution_time(win_height, integ_time, n_accum):
    return (n_accum + 1) * (integ_time + 0.071 + (win_height) * 0.320 + 1) + 0.337


# cut detector bins
# sim_d = {"intt": 220, "wtop": 80, "wbottom": 212, "nacc": 8, "binning": 11, "nreps": 3}

print(execution_time(sim_d["wheight"], sim_d["intt"], sim_d["nacc"]) * sim_d["nsubd"], "ms for 6 subds")
print(sim_d)


"""get full frame int time stepping plots from MCO"""
# don't reload each time
if "full_frame" not in globals():
    file_level = "hdf5_level_0p2a"
    regex = re.compile("20161125_(0[6-9]|1[0-3]).*")

    frame_ix = -1  # auto selection
    # frame_ix = 0  # for bias
    # frame_ix = 1  # for bias
    # frame_ix = 250  # for higher int time
    # frame_ix = 28  # 201 ms

    h5fs, h5s, _ = make_filelist2(regex, file_level, path=data_path, silent=False, pre_psp=True)

    means = []
    full_frame = np.zeros((256, 320)) + np.nan
    exp_frame = np.zeros(256)
    for h5_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):
        # for h5_ix, (h5, h5f) in enumerate(zip(h5s[0:1], h5fs[0:1])):

        nacc = h5f["Channel/NumberOfAccumulations"][0]
        binning = int(h5f["Channel/Binning"][0] + 1)
        intt = h5f["Channel/IntegrationTime"][...]

        # choose closest
        if frame_ix == -1:
            frame_ix = np.argmin(np.abs(intt - sim_d["intt"]))

        int_px = intt * binning * int(nacc / 2)
        # npixel = binning *

        # get detector row binning
        # 2016 cal data, bins is 3D!
        unique_bins = h5f["Science/Bins"][0, :, 0]
        n_bins = len(unique_bins)

        # 2016 cal data, Y is 3D!
        y_all = h5f["Science/Y"][...]

        exps = h5f["Channel/Exponent"][...]

        # get bias
        frame = y_all[frame_ix, :, :]
        # print(np.mean(frame))
        means.append(np.mean(frame))

        full_frame[unique_bins, :] = frame
        exp_frame[unique_bins] = exps[frame_ix]

        # plt.plot(y_all[0, :, :].T, alpha=0.2)

    # get raw dark frame
    FULL_FRAME = full_frame[sim_d["wtop"]:sim_d["wbottom"], :]

    plt.figure()
    plt.title("LNO Raw Dark Frame")
    plt.xlabel("Pixel number (spectral direction)")
    plt.ylabel("Detector row (spatial direction)")
    plt.imshow(FULL_FRAME, vmin=1900, vmax=4000, extent=(0, 320, sim_d["wbottom"], sim_d["wtop"]))
    plt.colorbar()

# plt.figure()
# plot_exps = []
# for frame_ix, frame in enumerate(full_frame):
#     exp = exp_frame[frame_ix]
#     if exp not in plot_exps:
#         label = "Exp = %i" % exp
#         plot_exps.append(exp)
#     else:
#         label = ""
#     plt.plot(frame, color="C%i" % exp, label=label, alpha=0.3)
# plt.title("Int time: %0.1f, Pxs int time: %0.1f" % (intt[frame_ix], int_px[frame_ix]))
# plt.legend()

# bad_ixs = np.where(full_frame > np.mean(means)*1.5)
# for bad_ixx, bad_ixy in zip(bad_ixs[1], bad_ixs[0]):
#     value = full_frame[bad_ixy, bad_ixx]
#     plt.text(bad_ixx, value, "%i,%i" % (bad_ixx, bad_ixy))

# print("Mean", np.mean(means), "+-", np.std(means))
# bias at intt = 0 is 1146.0603 +- 3.3355103 counts

# find pixels where counts are always too high
# not yet done


def sim_one_frame(win_height, integ_time, sig_flag):
    """simulate one unbinned frame, taking background level from dark measurement and adding noise
    then add signal from lno shape and if sig_flag is True"""
    bkg_lev = integ_time
    noise = np.random.normal(bkg_lev, NOISE_AMP, (win_height, 320))
    win = np.copy(FULL_FRAME) + noise
    # print("Bg:", win[100, 100])
    # plt.figure()
    # plt.plot(win.T)
    if sig_flag:
        win += np.copy(LNO_SHAPE * integ_time) * CORR_FACT
        # print("Bg + signal:", win[100, 100])
    return win


def apply_binning(win, n_of_bin):
    return np.stack(tuple(map(lambda b: np.sum(b, 0), np.split(win, n_of_bin))))


def apply_exp(win, exp):
    return np.left_shift(np.right_shift(win, exp), exp)


def sim_accum(win_height, integ_time, n_accum, binning_fact, apply_exp_f=True, exponent=-1):
    """simulate multiple accumulations, optionally applying exponent bug and given exponent"""
    n_of_bin = win_height // binning_fact
    win = np.zeros((n_of_bin, 320), dtype=int)
    vmax = 0
    # loop through accumulations
    for _ in range(n_accum // 2):
        frame = sim_one_frame(win_height, integ_time, True)
        # print("Light:", frame[100, 100])
        win += apply_binning(np.int16(frame), n_of_bin)
        vmax = max(vmax, np.max(win))
        # print(vmax)
        frame_dark = sim_one_frame(win_height, integ_time, False)
        # print("Dark:", frame_dark[100, 100])
        win -= apply_binning(np.int16(frame_dark), n_of_bin)
        vmax = max(vmax, np.max(win))
        # print(vmax)
    # e.g. 33000 counts = 2**11 + 2**5 (i.e. 2**11 roundup down to 10)
    exp = max(0, int(math.log2(vmax)) - 10)
    if apply_exp_f:
        if exponent != -1:
            exp = exponent
        win = apply_exp(win, exp)
    return win, exp


def sim_accum_spectrum(win_height, integ_time, n_accum, binning_fact, apply_exp_f=True, exponent=-1):
    """simulate bins, sum bins and return summed spectrum and exponent"""
    win, exp = sim_accum(win_height, integ_time, n_accum, binning_fact, apply_exp_f, exponent=exponent)
    win_binned = np.sum(win, axis=0)
    return win_binned, exp


def sim_only(win_height, integ_time, n_accum, binning_fact, apply_exp_f=True):
    """simulate bins, sum bins and get std dev comparing to smoothed spectrum"""
    win, exp = sim_accum(win_height, integ_time, n_accum, binning_fact, apply_exp_f)
    win_binned = np.sum(win, axis=0)
    smoothed = savgol_filter(win_binned, 51, 2)
    y_errors = win_binned - smoothed
    return np.max(smoothed), np.std(y_errors)


def sim_and_plot(win_height, integ_time, n_accum, binning_fact, apply_exp_f=True):
    win, exp = sim_accum(win_height, integ_time, n_accum, binning_fact, apply_exp_f)
    params = (win_height, integ_time, n_accum, binning_fact)
    print("Window height=%d  Integ_time=%d  N_accum=%d  Binning_fact=%d" % params)
    print("Exec time %0.1fms" % (execution_time(win_height, integ_time, n_accum)))
    if apply_exp_f:
        print("Exponent = ", exp)
    else:
        print("Values in accum memory")
    plt.figure()
    win_binned = np.sum(win, axis=0)
    # plt.plot(win[0])
    plt.plot(win_binned)

    smoothed = savgol_filter(win_binned, 51, 2)
    plt.plot(smoothed)

    y_errors = win_binned - smoothed
    plt.plot(y_errors)
    print("Sim peak:", np.max(smoothed), "Stdev:", np.std(y_errors))
    # return win


# sim_and_plot(144, 590, 4, 24)
# sim_and_plot(144, 590, 4, 24, False)
# sim_and_plot(chosen_window, chosen_int_time, chosen_nacc, chosen_binning)
# sim_and_plot(144, 1190, 2, 24)
# sim_and_plot(144, 190, 14, 24)

fig1, ax1 = plt.subplots()

# Find the choice of NOISE_AMP and CORR_FACT that match the spectrum
# MC analysis
sim_spectra = []
diff_sim_spectra = []
mean_sim_diff = []
exps = []

peaks = []
stdevs = []
snrs = []
# TODO: simulated diff is too high compared to measurement. Changing exponent doesn't change simulated diff, need another measure
# TODO: check if changing wbottom changes measured SNR or simulated SNR
chosen_exp = -1
print("Simulating an exponent of", chosen_exp)
for i in progress(range(100)):
    sim_spectrum, exp = sim_accum_spectrum(sim_d["wheight"], sim_d["intt"], sim_d["nacc"], sim_d["binning"], apply_exp_f=True, exponent=chosen_exp)
    sim_spectra.append(sim_spectrum)
    exps.append(exp)
    diff_sim_spectrum = np.diff(sim_spectrum)
    diff_sim_spectra.append(diff_sim_spectrum)
    mean_sim_diff.append(np.mean(np.abs(diff_sim_spectrum)))

    if i == 0:
        ax1.plot(sim_spectrum, label="Sum of bins (simulated)", alpha=0.1)
    else:
        ax1.plot(sim_spectrum, alpha=0.3)

    sim_smoothed2 = savgol_filter(sim_spectrum, 51, 2)  # general shape only
    # get error vs heavily smoothed spectrum
    sim_error = sim_spectrum - sim_smoothed2

    peaks.append(np.max(sim_smoothed2))
    stdevs.append(np.std(sim_error))
    snrs.append(np.max(sim_smoothed2) / np.std(sim_error))

print("Measured exponent:", exponent)
print("Simulated exponent range:", np.min(exps), "to", np.max(exps))
print("Measured peak:", np.max(smoothed2), "Stdev:", np.std(y_errors), "SNR:", np.max(smoothed2) / np.std(y_errors))
print("Simulated peak:", np.mean(peaks), "+-", np.std(peaks), "Stdev:", np.mean(stdevs), "+-", np.std(stdevs), "SNR:", np.mean(snrs), "+-", np.std(snrs))


ax1.plot(meas_spectrum, "k--", label="Sum of bins (measured)")
# ax1.plot(smoothed, label="Light smoothing")
# ax1.plot(smoothed2, label="Heavy smoothing")
ax1.legend()
ax1.grid()
ax1.set_xlabel("Pixel number (spectral dimension)")
ax1.set_ylabel("Counts")
# sim_spectra = np.asarray(sim_spectra[0])
# exps = np.asarray(exps)

# plt.plot(sim_spectra.T, "b--")
# mean_sim_spectrum = np.mean(spectra, axis=0)
# plt.plot(mean_sim_spectrum, "b--", label="Mean simulation")

# Simulated SNR


plt.legend()

# remove bad pixel manually
meas_spectrum_no_bp = np.copy(meas_spectrum)
meas_spectrum_no_bp[40] = np.mean((meas_spectrum_no_bp[39], meas_spectrum_no_bp[41]))

diff_sim_spectra = np.diff(sim_spectra)
diff_meas_spectrum = np.diff(meas_spectrum_no_bp)

# plt.figure()
# plt.title("Diffs")
# plt.plot(diff_sim_spectra.T, label="Simulated spectrum")
# plt.plot(diff_meas_spectrum, label="Measured spectrum")
# plt.legend()

print("Simulated diff:", np.mean(mean_sim_diff), "vs measured diff:", np.mean(np.abs(diff_meas_spectrum)))

# print("Sim peak:", np.mean(peaks), "+-", np.std(peaks), "Stdev:", np.mean(stdevs), "+-", np.std(stdevs), "SNR:", np.mean(peaks) / np.mean(stdevs))
