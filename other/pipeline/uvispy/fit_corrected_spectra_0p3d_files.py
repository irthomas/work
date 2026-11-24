# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:22:42 2026

@author: iant

CALCULATE OPTIMUM STRAYLIGHT FROM YANNICK FIT
"""

import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tools.general.progress_bar import progress

from other.pipeline.uvispy.fit_uvis_transmittance import fit_to_spectra

filepaths = [r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d\2024\05\01\20240501_021829_0p3d_UVIS_E.h5"]
filepaths = glob.glob(r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d\2024\05\**\*UVIS_I.h5", recursive=True)

good_fit_d = {}

for src in progress(filepaths[0:30]):
    # dst = os.path.basename(src).replace("0p3d", "0p3e")
    #     if os.path.exists(dst):
    #         os.remove(dst)

    h5_basename = os.path.basename(src).split(".")[0]
    h5_l1 = h5_basename.replace("0p3d", "1p0a")
    print(h5_basename)

    with h5py.File(src, 'r') as f:
        # Keep only type-4 science frames and transpose detector data to
        # (spectral column, detector row, frame) for the calculations below.
        # acq_mode = f['Channel/AcquisitionMode'][0]  # 0 = Full CCD, 1 = Vertical Binning, 2 =Horizontal / Combined binning

        # print(f["Science"].keys())

        vstart = int(f['Channel/VStart'][0]) + 1
        # vend = int(f['Channel/VEnd'][0]) + 1
        # nbr_ccd_lines = vend - vstart + 1
        integration_time = float(f['/Channel/IntegrationTime'][0])
        # Instrument = f.attrs['InstName']
        Y = f["Science/Y"][...]
        YWellIllumMask = f["Science/YIllumMask"][...]
        YNonIllumMask = f["Science/YNonIllumMask"][...]

        x = f['Science/X'][0, :]
        xpix_1b = f['Science/Xpix_1b'][...]  # array of pixel centre indices for real columns in matlab indices e.g. [12.5, 20.5, 28.5,...]
        # xnb_bin = f['Science/XNbBin'][...]  # array of binning factors for real columns e.g. [8,8,8,...]
        # x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

        # YNb = f['Science/YNb']
        alts_in = np.mean(f["Geometry/Point0/TangentAltAreoid"][...], axis=1)

    nbr_obs = Y.shape[0]
    nbr_lambda_pixels = Y.shape[2]  # n real columns e.g. 128 or 1024
    nbr_rows = Y.shape[1]  # n real columns e.g. 128

    # divide Y by binning factor to match binned and unbinned
    Y /= (1024 / nbr_lambda_pixels)

    ymask_illum = YWellIllumMask

    # multiply raw data by masks to mask each detector zone
    Y_illum = Y * ymask_illum

    # first get list of non-illuminated frames with sufficient signal
    middle_col_ix = int(nbr_lambda_pixels / 2)
    middle_col_sums = np.nansum(Y_illum[:, :, middle_col_ix], axis=1)
    middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

    uv_col_ix = int(nbr_lambda_pixels / 10)

    toa_ix = np.where(middle_col_trans > 0.99)[0][-1]  # example index above atmosphere
    atmo_ix = np.argmin(np.abs(middle_col_trans - 0.5))
    atmo_ixs = np.arange(atmo_ix - 2, atmo_ix + 3, 1)

    # plot binned well-illuminated transmittance with hill straylight added in various amounts corrected for transmittance
    # no hill correction
    # FIXME: check Y_illum rather than Y
    y_toa_unbinned = Y[toa_ix, :, :]
    y_toa_binned = np.sum(y_toa_unbinned, axis=0)
    y_trans_unbinned = []
    y_trans = []
    y_trans_error = []
    alts = []
    for frame_ix in atmo_ixs:
        y_trans_unbinned.append(Y[frame_ix, :, :])
        y_atmos_binned = np.sum(Y[frame_ix, :, :], axis=0)
        y_trans.append(y_atmos_binned / y_toa_binned)
        y_trans_error.append((y_atmos_binned / y_toa_binned) / 100)
        alts.append(alts_in[frame_ix])

    # fit to uncorrected
    # frame_ixs is the internal list of frames that have slant columns
    # px_ixs is the spectral indices on which the fitting was done
    y_trans_fitted, frame_ixs, px_ixs = fit_to_spectra(h5_l1, x, y_trans, y_trans_error, alts, do_no_ozone=True, which_xs="Zachary",
                                                       reference_aerosol_wavelength=510.0, particle_sizes=None,
                                                       dust_fraction_step=0.025, fit_nms=[316, 640], plot_results=False, with_error_bars=False)

    if len(frame_ixs) != len(y_trans):
        print("%s: not all frames have slant columns" % h5_basename)

    # check fits. If too bad don't use for further analysis
    for frame_ix in frame_ixs:
        chisq = 1024 * np.sum((y_trans[frame_ix][px_ixs] - y_trans_fitted[frame_ix, :][px_ixs])**2) / len(px_ixs)
        print(h5_basename, frame_ix, "chisq=", chisq)
        if chisq < 0.005:
            if h5_l1 not in good_fit_d.keys():
                good_fit_d[h5_l1] = {}
                good_fit_d[h5_l1]["x"] = x
                good_fit_d[h5_l1]["toa_frame"] = Y[toa_ix, :, :]
                good_fit_d[h5_l1]["atmos_frames"] = []
                good_fit_d[h5_l1]["alts"] = []

                good_fit_d[h5_l1]["y_trans"] = []
                good_fit_d[h5_l1]["y_trans_error"] = []
                good_fit_d[h5_l1]["y_fit"] = []

            good_fit_d[h5_l1]["atmos_frames"].append(y_trans_unbinned[frame_ix])
            good_fit_d[h5_l1]["alts"].append(alts[frame_ix])

            good_fit_d[h5_l1]["y_trans"].append(y_trans[frame_ix])
            good_fit_d[h5_l1]["y_trans_error"].append(y_trans_error[frame_ix])
            good_fit_d[h5_l1]["y_fit"].append(y_trans_fitted[frame_ix])
        else:
            print("%s: %i not fitted sufficiently" % (h5_basename, frame_ix))

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.title(h5_basename)
    for frame_ix in frame_ixs:
        plt.plot(x, y_trans[frame_ix], color=f"C{frame_ix}")
        plt.plot(x, y_trans_fitted[frame_ix, :], color=f"C{frame_ix}", linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.grid()
    plt.savefig("%s_y_fitting.png" % h5_basename)

# %%

# h5_l1 = "20240512_084850_1p0a_UVIS_I"
h5_l1 = "20240512_144234_1p0a_UVIS_I"
# h5_l1 = "20240512_223426_1p0a_UVIS_I"
# h5_l1 = "20240516_130419_1p0a_UVIS_I"

data = good_fit_d[h5_l1]
toa_frame = data["toa_frame"]
atmos_frames = np.asarray(data["atmos_frames"])
y_fits = np.asarray(data["y_fit"])
x = data["x"]

frame_ixs = np.arange(len(data["alts"]))

# check fit in UV only this time
fit_nms = [215, 295]
# convert nanometres into pixel indices
px_ixs = np.arange(np.argmin(np.abs(x - fit_nms[0])), np.argmin(np.abs(x - fit_nms[1])), 1)

# fit the best correction to minimise the chisq
y_toa_binned = np.sum(toa_frame, axis=0)
for corr in [0, 50, 100, 150]:
    # correct for hill by constant amount
    y_toa_binned1 = np.sum(np.maximum(toa_frame - corr, 0.0), axis=0)

    plt.figure(figsize=(15, 6))
    plt.title("UVIS transmittance: straylight removal testing")
    for frame_ix in frame_ixs:
        # uncorrected
        y_atmos_binned = np.sum(atmos_frames[frame_ix, :, :], axis=0)
        y_trans_binned = y_atmos_binned / y_toa_binned

        # do first run with constant correction
        y_atmos_binned1 = np.sum(np.maximum(atmos_frames[frame_ix, :, :] - corr, 0.0), axis=0)
        y_trans_binned1 = y_atmos_binned1 / y_toa_binned1

        # correct for hill by variable amount based on max fitted transmittance
        corr2 = corr * (np.max(y_fits[frame_ix]))
        y_toa_binned2 = np.sum(np.maximum(toa_frame - corr2, 0.0), axis=0)
        y_atmos_binned2 = np.sum(np.maximum(atmos_frames[frame_ix, :, :] - corr2, 0.0), axis=0)
        y_trans_binned2 = y_atmos_binned2 / y_toa_binned2

        chisq = 1024 * np.sum((y_trans_binned2[px_ixs] - y_trans_binned[px_ixs])**2) / len(px_ixs)

        print(corr, frame_ix, chisq, np.mean(y_trans_binned1), np.mean(corr2))

        # plt.plot(x, y_trans_binned, label="Spectrum %i uncorrected" % frame_ix, color="C%0i" % frame_ix, linestyle=":")
        plt.plot(x, y_fits[frame_ix], label="Spectrum %i fit" % frame_ix, color="C%0i" % frame_ix)
        plt.plot(x, y_trans_binned1, label="Spectrum %i +%i counts" % (frame_ix, corr), color="C%0i" % frame_ix, linestyle=":")
        plt.plot(x, y_trans_binned2, label="Spectrum %i variable correction" % frame_ix, color="C%0i" % frame_ix, linestyle="--")
        # plt.plot(x, y_trans_binned1/y_fits[frame_ix], label="Spectrum %i +%i counts" % (frame_ix, corr), color="C%0i" % frame_ix, linestyle=":")
        # plt.plot(x, y_trans_binned2/y_fits[frame_ix], label="Spectrum %i variable correction" % frame_ix, color="C%0i" % frame_ix, linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.grid()
    plt.legend(loc="lower right")
    # plt.ylim([-0.1, 1.0])
    # plt.savefig("%s_correction_types.png")
