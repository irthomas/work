# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:32:17 2026

@author: iant
"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
from datetime import datetime


from tools.file.hdf5_functions import open_hdf5_file

from instrument.calibration.lno_phobos.lno_offset_correction_phobos import fit_spectra
from instrument.calibration.lno_phobos.lno_bad_pixel_correction_phobos import bad_pixel_correction
from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order
from tools.spectra.non_uniform_savgol import non_uniform_savgol

DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"
px_range = range(150, 250)  # full peak centred


orders = [121, 134, 149, 168, 189, 196]
h5s = ["20260623_060729_1p0a_LNO_1_L_%i" % i for i in orders]
frame_ixs_range = [25, 99]
frame_ixs_range_u = [103, 232]
good_bin_ixs = [0, 1, 2]

if "limb_d" not in globals():
    cal_d = {}
    limb_d = {}
    for order, h5 in zip(orders, h5s):

        cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
        if order not in cal_d.keys():
            cal_d = rad_cal_order(cal_h5, [order], centre_indices=px_range, path=DATA_PATH, silent=True)
            cal_d[order]["solar_spectrum"] = cal_d[order]["y_spectrum"] / np.max(cal_d[order]["y_spectrum"])
        ideal_spectrum = cal_d[order]["solar_spectrum"]

        h5f = open_hdf5_file(h5, path=DATA_PATH)

        top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
        unique_bins = sorted(list(set(top_bins)))
        n_bins = len(unique_bins)
        binning = unique_bins[1] - unique_bins[0]

        frame_indices = np.arange(frame_ixs_range[0] * n_bins, frame_ixs_range[1] * n_bins)

        y_all = h5f["Science/Y"][frame_indices, :]
        y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

        x_centre = np.mean(h5f["Science/X"][0, px_range])

        # y_mean = np.mean(y[:, 100:300], axis=1)

        alt = h5f["Geometry/Point0/TangentAltAreoid"][frame_indices, 0]
        alt_2d = np.reshape(alt, [-1, n_bins])

        obs_dt_str_2d = h5f["Geometry/ObservationDateTime"][frame_indices, :]
        obs_dt_str_3d = np.reshape(obs_dt_str_2d, [-1, n_bins, obs_dt_str_2d.shape[1]])

        obs_dt_str = obs_dt_str_3d[:, 0, :]

        dt_s = [datetime.strptime(s[0].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]
        dt_e = [datetime.strptime(s[1].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]

        dt_means = [((e - s) / 2) + s for s, e in zip(dt_s, dt_e)]

        # et_s = np.asfarray([sp.utc2et(s[0]) for s in obs_dt_str])
        # et_e = np.asfarray([sp.utc2et(s[1]) for s in obs_dt_str])

        # ets = np.mean([et_s, et_s], axis=0)

        y_3d = bad_pixel_correction(y_3d)
        fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, ideal_spectrum)

        y_spectral_mean = np.mean(corr_spectra[:, :, px_range], axis=2)
        y_solar_corr = np.mean(corr_spectra[:, :, px_range] / cal_d[order]["y_spectrum"][px_range], axis=2)

        limb_d[order] = {}

        limb_d[order] = {"ys": y_spectral_mean, "yscorr": y_solar_corr, "alts": alt_2d, "dts": dt_means, "x": x_centre}

fig1, axes1 = plt.subplots(nrows=7, figsize=(13, 9), constrained_layout=True, sharex=True)

max_val = np.max([np.max(limb_d[order]["yscorr"]) for order in limb_d.keys()])
min_val = np.min([np.min(limb_d[order]["yscorr"]) for order in limb_d.keys()])

for order_ix, order in enumerate(limb_d.keys()):
    plt.suptitle("LNO AMEC Limb 20260623_060729_1p0a_LNO_1_L")
    # axes1[order_ix]
    # plt.title("%s: raw Y data, spectrally binned" % h5)
    # plt.xlabel("Frame index (all orders)")
    # plt.ylabel("Bin number")
    im = axes1[order_ix].imshow(limb_d[order]["yscorr"].T, aspect="auto", vmin=min_val, vmax=max_val)
    axes1[order_ix].text(10, 2, "Order %i" % order)
    # cbar = axes1[order_ix].colorbar(im)
    # cbar.set_label("Signal (counts)", rotation=270, labelpad=10)
    # plt.subplots_adjust(bottom=0.15)

    axes1[order_ix].set_ylabel("Detector bin")

    for bin_ix in range(n_bins):
        axes1[-1].plot(limb_d[order]["alts"][:, bin_ix], color="C%i" % order_ix, linestyle=["-", ":", "--", "-."][bin_ix])
axes1[-1].set_ylim(bottom=-10)
axes1[-1].set_xlim(left=0)
axes1[-1].set_xlabel("Frame index (time)")
axes1[-1].text(10, 200, "Tangent Altitude (km)")
# plt.grid()
# plt.legend()

plt.savefig("LNO_AMEC_Limb_20260623_060729_solar_scaled_counts.png")

h5u = "20260623_060729_1p0a_UVIS_L"

h5fu = open_hdf5_file(h5u, path=DATA_PATH)

y_allu = h5fu["Science/Y"][np.arange(frame_ixs_range_u[0], frame_ixs_range_u[1]), :]
x = h5fu["Science/X"][0, :]
altu = h5fu["Geometry/Point0/TangentAltAreoid"][np.arange(frame_ixs_range_u[0], frame_ixs_range_u[1]), 0]

y_spectral_mean_u1 = np.mean(y_allu[:, 0:400], axis=1)
y_spectral_mean_u2 = np.mean(y_allu[:, 400:800], axis=1)
y_spectral_mean_u3 = np.mean(y_allu[:, 800:1024], axis=1)
x1 = "UVIS spectral bin 1 %0.3f-%0.3f um" % (x[0] / 1000, x[399] / 1000)
x2 = "UVIS spectral bin 2 %0.3f-%0.3f um" % (x[400] / 1000, x[799] / 1000)
x3 = "UVIS spectral bin 3 %0.3f-%0.3f um" % (x[800] / 1000, x[1023] / 1000)

fig2, axes2 = plt.subplots(nrows=2, figsize=(13, 4), constrained_layout=True, sharex=True)
plt.suptitle("UVIS AMEC Limb 20260623_060729_1p0a_UVIS_L")
axes2[0].plot(y_spectral_mean_u1, label=x1)
axes2[0].plot(y_spectral_mean_u2, label=x2)
axes2[0].plot(y_spectral_mean_u3, label=x3)
axes2[1].plot(altu)

axes2[0].text(10, 0.01, "UVIS spectrally binned radiance")
axes2[0].legend()
axes2[0].set_ylabel("Radiance")
axes2[1].set_ylim(bottom=-10)
axes2[1].set_xlim(left=0)
axes2[1].set_xlabel("Frame index (time)")
axes2[1].text(10, 200, "Tangent Altitude (km)")
plt.savefig("UVIS_AMEC_Limb_20260623_060729_radiance.png")

# %%
fig3, axes3 = plt.subplots(figsize=(13, 4))
plot_ix_range = np.arange(45)

for order_ix, order in enumerate(limb_d.keys()):
    for bin_ix in good_bin_ixs:

        axes3.plot(limb_d[order]["alts"][plot_ix_range, :][:, bin_ix], limb_d[order]["yscorr"][plot_ix_range, :][:, bin_ix],
                   color="C%i" % order_ix, alpha=0.2)
    sg = non_uniform_savgol(np.mean(limb_d[order]["alts"][plot_ix_range, :][:, good_bin_ixs], axis=1),
                            np.mean(limb_d[order]["yscorr"][plot_ix_range, :][:, good_bin_ixs], axis=1), 15, 1)
    axes3.plot(np.mean(limb_d[order]["alts"][plot_ix_range, :][:, good_bin_ixs], axis=1), sg,
               label="Order %i" % (order), color="C%i" % order_ix)

plt.legend()

# %%
fig4, axes4 = plt.subplots(figsize=(13, 4), constrained_layout=True)
plot_ix_range = np.arange(10, 42)
plot_ix_range_u = np.arange(17, 72)

for order_ix, order in enumerate(limb_d.keys()):
    um = 10000 / limb_d[order]["x"]
    # axes4.plot(np.mean(limb_d[order]["alts"][plot_ix_range, :][:, good_bin_ixs], axis=1),
    #            np.mean(limb_d[order]["yscorr"][plot_ix_range, :][:, good_bin_ixs], axis=1),
    #            label="Order %i" % (order), color="C%i" % order_ix, alpha=0.3)

    sg = non_uniform_savgol(np.mean(limb_d[order]["alts"][plot_ix_range, :][:, good_bin_ixs], axis=1),
                            np.mean(limb_d[order]["yscorr"][plot_ix_range, :][:, good_bin_ixs], axis=1), 15, 1)
    axes4.plot(np.mean(limb_d[order]["alts"][plot_ix_range, :][:, good_bin_ixs], axis=1), sg,
               label="Order %i (%0.3f um) smoothed" % (order, um), color="C%i" % order_ix)
axes4.plot(altu[plot_ix_range_u], y_spectral_mean_u1[plot_ix_range_u] / 1000, label="UVIS %s radiance/1000" % x1, color="C%i" % (order_ix + 1))
axes4.plot(altu[plot_ix_range_u], y_spectral_mean_u2[plot_ix_range_u] / 1000, label="UVIS %s radiance/1000" % x2, color="C%i" % (order_ix + 2))
axes4.plot(altu[plot_ix_range_u], y_spectral_mean_u3[plot_ix_range_u] / 1000, label="UVIS %s radiance/1000" % x3, color="C%i" % (order_ix + 3))
plt.grid()
plt.legend()
plt.axhline(y=0, color="k", linestyle="--")
plt.xlabel("Tangent altitude of detector bin centre above Areoid (km)")
plt.ylabel("Raw signal / solar signal (LNO); radiance/1000 (UVIS)")
plt.savefig("AMEC_Limb_20260623_060729_counts_vs_altitude.png")
