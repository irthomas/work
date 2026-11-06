# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:20:29 2026

@author: iant

TEST UVIS STRAYLIGHT REMOVAL

# something is still going wrong with the straylight linear fit for pixels < 150 - straylight not linear in UV?

"""
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
import scipy
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# from datetime import datetime

FRAMES_TO_CALIBRATE = [121, 122, 129, 130, 131, 132, 133, 134]
FRAMES_TOA = [0, 1]
FRAMES_PLOT = [2, 3, 4, 5]
PXS_TO_CALIBRATE = []
# PXS_TO_CALIBRATE = np.arange(0, 160)
# PXS_TO_CALIBRATE = np.arange(170, 200)

BP_N_ITERS_LIN = 10
BP_N_ITERS_HILL = 2


def plot_frame_col_norm(frame, vmin=None, vmax=None):
    """function to plot a column-normalised 2d detector frame"""

    y_col_max = np.max(frame, axis=1)
    frame_col_max = frame / y_col_max[:, None]
    plt.figure()
    plt.imshow(frame_col_max.T, vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')


def define_top_bottom_well_illuminated_lines(X):
    """outputs same as matlab with indices subtracted by 1. Only for pFM occultation at present"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    x_top = np.asarray([[1, 172], [100, 172], [200, 170], [300, 169], [400, 167], [500, 166], [
                       600, 162], [700, 160], [800, 158], [900, 156], [1032, 152]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    x_bottom = np.asarray([[1, 181], [100, 181], [200, 181], [300, 181], [400, 181], [500, 181], [
                          600, 181], [700, 181], [800, 182], [900, 183], [1032, 185]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    return iLineTopStart, iLineBottomStart


def define_top_bottom_non_illuminated_lines(X):
    """outputs the top and bottom row indices of the non-illuminated zone"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    # new values
    # x_top = np.asarray([[9, 150], [80, 150], [158, 154], [180, 154], [404, 152], [433, 152], [752, 154], [875, 147], [1032, 145]])
    x_top = np.asarray([[9, 150], [59, 153], [109, 155], [159, 156], [209, 157], [259, 155], [309, 155], [359, 155], [409, 154], [
        459, 154], [509, 154], [609, 154], [709, 152], [809, 152], [909, 147], [1009, 144]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    # x_bottom = np.asarray([[9, 191], [116, 191], [200, 191], [300, 193], [420, 194], [626, 195], [799, 197], [1032, 199]])
    x_bottom = np.asarray([[9, 189], [59, 189], [109, 189], [159, 189], [209, 191], [259, 191], [309, 191], [359, 191], [409, 192], [
        459, 191], [509, 191], [609, 193], [709, 195], [809, 195], [909, 197], [1009, 198]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    iLineTopStart -= 2  # offset to avoid overlap with sl hill
    iLineBottomStart += 1  # offset to avoid overlap with sl hill

    return iLineTopStart, iLineBottomStart


def define_top_bottom_hill_lines(X):
    """outputs the top and bottom row indices of the hill zone"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    # straylight hill
    x_top_start = np.asarray([[9, 150], [59, 153], [109, 155], [159, 156], [209, 157], [259, 155], [309, 155], [359, 155], [409, 154], [
                             459, 154], [509, 154], [609, 154], [709, 152], [809, 152], [909, 147], [1009, 144]])
    x_top_end = np.asarray([[9, 170], [59, 169], [109, 169], [159, 168], [209, 168], [259, 166], [309, 165], [359, 165], [409, 164], [
                           459, 163], [509, 162], [609, 159], [709, 158], [809, 154], [909, 151], [1009, 147]])
    x_bottom_start = np.asarray([[9, 185], [59, 186], [109, 186], [159, 186], [209, 186], [259, 187], [309, 187], [359, 187], [409, 188], [
                                459, 188], [509, 188], [609, 190], [709, 190], [809, 192], [909, 193], [1009, 195]])
    x_bottom_end = np.asarray([[9, 189], [59, 189], [109, 189], [159, 189], [209, 191], [259, 191], [309, 191], [359, 191], [409, 192], [
                              459, 191], [509, 191], [609, 193], [709, 195], [809, 195], [909, 197], [1009, 198]])

    iLineTopStart = np.interp(x, x_top_start[:, 0], x_top_start[:, 1], left=None, right=None)
    iLineTopEnd = np.interp(x, x_top_end[:, 0], x_top_end[:, 1], left=None, right=None)
    iLineBottomStart = np.interp(x, x_bottom_start[:, 0], x_bottom_start[:, 1], left=None, right=None)
    iLineBottomEnd = np.interp(x, x_bottom_end[:, 0], x_bottom_end[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineTopEnd = np.floor(iLineTopEnd).astype(int)
    iLineBottomStart = np.ceil(iLineBottomStart).astype(int)
    iLineBottomEnd = np.floor(iLineBottomEnd).astype(int)

    return iLineTopStart, iLineTopEnd, iLineBottomStart, iLineBottomEnd


def define_top_bottom_partially_illuminated_lines(X):
    """outputs same as matlab with indices subtracted by 1. Only for pFM occultation at present"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    # x_top = np.asarray([[1, 172], [100, 172], [200, 170], [300, 169], [400, 167], [500, 166], [
    #                    600, 162], [700, 160], [800, 158], [900, 156], [1032, 152]])
    x_top = np.asarray([[9, 150], [59, 153], [109, 155], [159, 156], [209, 157], [259, 155], [309, 155], [359, 155], [409, 154], [
        459, 154], [509, 154], [609, 154], [709, 152], [809, 152], [909, 147], [1009, 144]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    # x_bottom = np.asarray([[1, 181], [100, 181], [200, 181], [300, 181], [400, 181], [500, 181], [
    #                       600, 181], [700, 181], [800, 182], [900, 183], [1032, 185]])
    x_bottom = np.asarray([[9, 189], [59, 189], [109, 189], [159, 189], [209, 191], [259, 191], [309, 191], [359, 191], [409, 192], [
        459, 191], [509, 191], [609, 193], [709, 195], [809, 195], [909, 197], [1009, 198]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    return iLineTopStart, iLineBottomStart


src = "20240516_130419_0p3b_UVIS_I.h5"  # unbinned, solar line peak seen

with h5py.File(src, 'r') as f:
    acq_mode = f['Channel/AcquisitionMode'][0]  # 0 = Full CCD, 1 = Vertical Binning, 2 =Horizontal / Combined binning

    itypedata = f['Channel/ReverseFlagAndDataTypeFlagRegister'][:]
    frame_ixs_type4 = np.where(itypedata == 4)[0]

    vstart = int(f['Channel/VStart'][0]) + 1
    vend = int(f['Channel/VEnd'][0]) + 1
    nbr_ccd_lines = vend - vstart + 1
    nbr_obs = len(frame_ixs_type4)
    # integration_time = float(f['/Channel/IntegrationTime'][0])
    Instrument = f.attrs['InstName']
    Y = np.swapaxes(f["Science/Y"][...], 0, 2)[:, :, frame_ixs_type4]
    YError = np.swapaxes(f["Science/YError"][...], 0, 2)[:, :, frame_ixs_type4]
    YMask = np.swapaxes(f["Science/YMask"][...], 0, 2)[:, :, frame_ixs_type4]
    YErrorSysNL = np.swapaxes(f["Science/YErrorSysNL"][...], 0, 2)[:, :, frame_ixs_type4]

    x = np.swapaxes(f['Science/X'][...], 0, 1)[:, frame_ixs_type4]
    xpix_1b = f['Science/Xpix_1b'][...]  # array of pixel centre indices for real columns in matlab indices e.g. [12.5, 20.5, 28.5,...]
    xnb_bin = f['Science/XNbBin'][...]  # array of binning factors for real columns e.g. [8,8,8,...]
    x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

    CircuitNoise = np.swapaxes(f['Science/CircuitNoise'], 0, 1)
    YNb = f['Science/YNb'][frame_ixs_type4]
    # stop()

if x_calib_ref == 'N/A':
    print('lambda_IASB_v1: manual')
    x_calib_ref = 'lambda_IASB_v1'

if len(FRAMES_TO_CALIBRATE) > 0:
    Y = Y[:, :, FRAMES_TO_CALIBRATE]
    YError = YError[:, :, FRAMES_TO_CALIBRATE]
    YMask = YMask[:, :, FRAMES_TO_CALIBRATE]
    YErrorSysNL = YErrorSysNL[:, :, FRAMES_TO_CALIBRATE]
    nbr_obs = len(FRAMES_TO_CALIBRATE)

if len(PXS_TO_CALIBRATE) > 0:
    x = x[PXS_TO_CALIBRATE, :]
    xpix_1b = xpix_1b[PXS_TO_CALIBRATE]
    xnb_bin = xnb_bin[PXS_TO_CALIBRATE]

    Y = Y[PXS_TO_CALIBRATE, :, :]
    YError = YError[PXS_TO_CALIBRATE, :, :]
    YMask = YMask[PXS_TO_CALIBRATE, :, :]
    YErrorSysNL = YErrorSysNL[PXS_TO_CALIBRATE, :, :]

nbr_lambda_pixels = xnb_bin.shape[0]  # n real columns e.g. 128


# skip illumination shift for now

# get top and bottom rows of each zone and subtract vstart
ix_non_illum_top, ix_non_illum_bottom = define_top_bottom_non_illuminated_lines(xpix_1b)
ix_non_illum_top -= vstart
ix_non_illum_bottom -= vstart

ix_illum_top, ix_illum_bottom = define_top_bottom_well_illuminated_lines(xpix_1b)
ix_illum_top -= vstart
ix_illum_bottom -= vstart

ix_sl_top_start, ix_sl_top_end, ix_sl_bottom_start, ix_sl_bottom_end = define_top_bottom_hill_lines(xpix_1b)
ix_sl_top_start -= vstart
ix_sl_top_end -= vstart
ix_sl_bottom_start -= vstart
ix_sl_bottom_end -= vstart

ix_par_illum_top, ix_par_illum_bottom = define_top_bottom_partially_illuminated_lines(xpix_1b)
ix_par_illum_top -= vstart
ix_par_illum_bottom -= vstart

# skip partially illuminated pixels for now
# make masks of each zone
ymask_non_illum = np.zeros_like(Y[:, :, 0])
for col_ix in range(nbr_lambda_pixels):
    ymask_non_illum[col_ix, :ix_non_illum_top[col_ix]] = 1
    ymask_non_illum[col_ix, ix_non_illum_bottom[col_ix]:] = 1

ymask_illum = np.zeros_like(Y[:, :, 0])
for col_ix in range(nbr_lambda_pixels):
    ymask_illum[col_ix, ix_illum_top[col_ix]:ix_illum_bottom[col_ix]] = 1

ymask_sl = np.zeros_like(Y[:, :, 0])
for col_ix in range(nbr_lambda_pixels):
    # vertical indices to fit
    sl_ixs = np.concatenate((
        np.arange(ix_sl_top_start[col_ix], ix_sl_top_end[col_ix] + 1),
        np.arange(ix_sl_bottom_start[col_ix], ix_sl_bottom_end[col_ix] + 1)
    ))
    ymask_sl[col_ix, sl_ixs] = 1

ymask_par_illum = np.zeros_like(Y[:, :, 0])
for col_ix in range(nbr_lambda_pixels):
    ymask_par_illum[col_ix, ix_par_illum_top[col_ix]:ix_par_illum_bottom[col_ix]] = 1


# compare masks on one image
masks = np.zeros_like(ymask_non_illum)
masks += ymask_non_illum
masks += ymask_illum * 2
masks += ymask_sl * 4
plt.figure()
plt.imshow(masks.T, aspect="auto", interpolation="none")
plt.xlabel("Pixel column (spectral direction)")
plt.ylabel("Pixel row (illumination direction)")
plt.title("Illumination masks")

# multiply raw data by masks to mask each detector zone
Y_non_illum = Y * ymask_non_illum[:, :, None]
Y_illum = Y * ymask_illum[:, :, None]
Y_par_illum = Y * ymask_par_illum[:, :, None]


# optional plotting
# plot_frame_col_norm(Y[:, :, 0])

plot_frame_col_norm(Y_non_illum[:, :, 0])
plt.xlabel("Pixel column (spectral direction)")
plt.ylabel("Pixel row (illumination direction)")
plt.title("Non-illuminated detector signal")

# plot_frame_col_norm(Y_illum[:, :, 0])

# first get list of non-illuminated frames with sufficient signal
middle_col_ix = int(nbr_lambda_pixels / 2)
middle_col_sums = np.nansum(Y_non_illum[middle_col_ix, :, :], axis=0)
middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

lin_straylight_frames = np.zeros_like(Y)

print("Fittinglinear straylight")
plt.figure()
for frame_ix in range(nbr_obs):

    # for frame_ix in range(1):
    frame = np.copy(Y_non_illum[:, :, frame_ix])
    frame_non_illum_ixs = np.where((frame != 0) & ~np.isnan(frame))

    # only fit linear, and report error if signal sufficient
    if middle_col_trans[frame_ix] > 0.1:
        poly_degree = 2
        chisq_cutoff = 11
    elif middle_col_trans[frame_ix] > 0.03:  # if low signal, fit linear
        poly_degree = 2
        chisq_cutoff = 20
    else:  # if very low signal, fit linear, ignore error
        poly_degree = 2
        chisq_cutoff = 99e99

    # straylight_frame = np.full_like(frame, np.nan)
    # obs_method_performed_on_lambda = np.ones(nbr_lambda_pixels)
    straylight_spectrum = np.zeros(nbr_lambda_pixels)

    for col_ix in range(nbr_lambda_pixels):
        # iterate to remove uncorrected bad and noisy pixels
        non_illum_ixs = frame_non_illum_ixs[1][frame_non_illum_ixs[0] == col_ix]
        pxs_removed = []
        for loop in range(BP_N_ITERS_LIN):

            poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
            straylight = poly(np.arange(nbr_ccd_lines))
            chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / (poly(non_illum_ixs) + 500)  # 500 offset to get away from zero
            if col_ix == 300 and frame_ix == 0:
                plt.scatter(non_illum_ixs, frame[col_ix, non_illum_ixs] - straylight[non_illum_ixs], label=loop, alpha=0.5)
                chisq = np.sum(np.abs(chisq_px)) / len(non_illum_ixs)
                print(chisq)
                print(frame[col_ix, non_illum_ixs])

            # TODO check this - the std successively removes the first pixels
            # chisq_px[0:5] /= 10.0
            bad_ix = np.where(chisq_px == np.max(chisq_px))[0][0]
            pxs_removed.append(bad_ix)
            frame[col_ix, non_illum_ixs[bad_ix]] = straylight[non_illum_ixs[bad_ix]]
        poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
        straylight = poly(np.arange(nbr_ccd_lines))
        chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / (poly(non_illum_ixs) + 500)
        chisq = np.sum(np.abs(chisq_px)) / len(non_illum_ixs)

        lin_straylight_frames[col_ix, :, frame_ix] = straylight

# subtract linear straylight from raw data
Y_no_lin_straylight = Y - lin_straylight_frames  # 3d raw data - straylight frames

# %%
# optional plotting

# plot_frame_col_norm(Y_no_lin_straylight[:, :, 0])
# plt.xlabel("Pixel column (spectral direction)")
# plt.ylabel("Pixel row (illumination direction)")
# plt.title(src)


# plt.figure()
# for frame_ix in range(Y_no_lin_straylight.shape[2]):
#     for i, col_ix in enumerate(range(50, 101, 109)):
#         y = Y_no_lin_straylight[col_ix, :, frame_ix]
#         plt.plot(y, color="C%i" % (i * 2), alpha=0.5)
#         y = Y[col_ix, :, frame_ix]
#         plt.plot(y, color="C%i" % (i * 2 + 1), alpha=0.5)
# plt.grid()
# plt.xlabel("Detector column")
# plt.ylabel("Detector signal")
# plt.title("Signal on detector column 50")

# plt.figure()
# for frame_ix in range(Y_no_lin_straylight.shape[2]):
#     for i, col_ix in enumerate(range(300, 301, 109)):
#         y = Y_no_lin_straylight[col_ix, :, frame_ix]
#         plt.plot(y / max(y), color="C%i" % (i), alpha=0.5)
# plt.grid()
# plt.xlabel("Detector column")
# plt.ylabel("Detector signal")
# plt.title("Signal on detector column %i" % col_ix)


# for rolling_mid in np.arange(50, 1000, 100):
#     plt.figure()
#     # rolling_mid = 100
#     rolling_ixs_all = np.lib.stride_tricks.sliding_window_view(np.arange(rolling_mid - 10, rolling_mid + 10), 10)
#     for rolling_ixs in rolling_ixs_all:
#         for frame_ix in range(Y_no_lin_straylight.shape[2]):
#             mean = np.mean(Y_no_lin_straylight[rolling_ixs, :, frame_ix], axis=0)
#             std = np.std(Y_no_lin_straylight[rolling_ixs, :, frame_ix], axis=0)

#             if (rolling_ixs == rolling_ixs_all[0]).all() and frame_ix == Y_no_lin_straylight.shape[2] - 1:
#                 plt.plot(mean / np.max(mean), color="r", alpha=0.1, label="Mean signal of nearby columns")
#                 plt.plot(std / np.max(std), color="b", alpha=0.1, label="Standard deviation of nearby columns")
#             else:
#                 plt.plot(mean / np.max(mean), color="r", alpha=0.1)
#                 plt.plot(std / np.max(std), color="b", alpha=0.1)
#     plt.legend()
#     plt.grid()
#     plt.xlabel("Detector column")
#     plt.ylabel("Detector signal")
#     plt.title("Comparing signal levels of nearby detector columns")


# %%


print("Fitting hill straylight")
Y_sl = Y_no_lin_straylight * ymask_sl[:, :, None]
hill_straylight_frames = np.zeros_like(Y)

plt.figure()
for frame_ix in range(nbr_obs):

    # for frame_ix in range(1):
    frame = np.copy(Y_sl[:, :, frame_ix])
    frame_sl_ixs = np.where((frame != 0) & ~np.isnan(frame))

    # only fit linear, and report error if signal sufficient
    if middle_col_trans[frame_ix] > 0.1:
        poly_degree = 2
        chisq_cutoff = 11
    elif middle_col_trans[frame_ix] > 0.03:  # if low signal, fit linear
        poly_degree = 2
        chisq_cutoff = 20
    else:  # if very low signal, fit linear, ignore error
        poly_degree = 2
        chisq_cutoff = 99e99

    # straylight_frame = np.full_like(frame, np.nan)
    # obs_method_performed_on_lambda = np.ones(nbr_lambda_pixels)
    straylight_spectrum = np.zeros(nbr_lambda_pixels)

    for col_ix in range(nbr_lambda_pixels):
        # iterate to remove uncorrected bad and noisy pixels
        sl_ixs = frame_sl_ixs[1][frame_sl_ixs[0] == col_ix]
        sl_applied_ixs = np.arange(min(sl_ixs), max(sl_ixs))
        pxs_removed = []
        for loop in range(BP_N_ITERS_HILL):

            poly = Polynomial.fit(sl_ixs, frame[col_ix, sl_ixs], poly_degree)
            straylight = poly(np.arange(nbr_ccd_lines))
            chisq_px = (frame[col_ix, sl_ixs] - poly(sl_ixs))**2 / (poly(sl_ixs) + 500)  # 500 offset to get away from zero
            if col_ix == 0 and frame_ix == 0:
                plt.scatter(sl_ixs, frame[col_ix, sl_ixs] - straylight[sl_ixs], label=loop, alpha=0.5)
                # chisq = np.sum(np.abs(chisq_px)) / len(sl_ixs)
                # print(chisq)
                # print(frame[col_ix, sl_ixs])

            # TODO check this - the std successively removes the first pixels
            # chisq_px[0:5] /= 10.0
            bad_ix = np.where(chisq_px == np.max(chisq_px))[0][0]
            pxs_removed.append(bad_ix)
            frame[col_ix, sl_ixs[bad_ix]] = straylight[sl_ixs[bad_ix]]
        poly = Polynomial.fit(sl_ixs, frame[col_ix, sl_ixs], poly_degree)
        straylight = poly(np.arange(nbr_ccd_lines))
        chisq_px = (frame[col_ix, sl_ixs] - poly(sl_ixs))**2 / (poly(sl_ixs) + 500)
        chisq = np.sum(np.abs(chisq_px)) / len(sl_ixs)

        hill_straylight_frames[col_ix, sl_applied_ixs, frame_ix] = straylight[sl_applied_ixs]

# remove hill straylight from raw data (already corrected for linear straylight)
Y_no_hill_straylight = Y_no_lin_straylight - hill_straylight_frames  # 3d raw data - straylight frames

# smooth hill straylight
y_hill = np.nansum(hill_straylight_frames, axis=1)
hill_straylight_frames_smooth = np.asarray([savgol_filter(y_hill[:, i], 19, 1) for i in range(y_hill.shape[1])]).T
plt.figure()
plt.plot(y_hill[:, 0])
plt.plot(hill_straylight_frames_smooth[:, 0])


plt.figure()
plt.imshow(hill_straylight_frames[:, :, 0].T, aspect="auto", interpolation="none", vmax=400)
# plt.plot(Y_no_lin_straylight[:, :, 0].T)


# check hill sl fits
plot_frame_col_norm(Y_no_hill_straylight[:, :, 0])
plot_frame_col_norm(hill_straylight_frames[:, :, 0])

# %%

# lots of plotting - convert to transmittance and plot spectra


y_plot = np.nansum(Y, axis=1).T
toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
y_cal = y_plot / toa[None, :]
plt.figure(figsize=(10, 10))
plt.plot(y_cal[FRAMES_PLOT, :].T, "r", alpha=0.5)
for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
    plt.text(100, y_cal[i, 100] + 0.05, i)
plt.title("No linear or hill straylight removal performed, summing of all detector columns")
plt.xlabel("Pixel number")
plt.ylabel("Transmittance")
plt.grid()

y_plot = np.nansum(Y_no_hill_straylight * ymask_illum[:, :, None], axis=1).T
toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
y_cal = y_plot / toa[None, :]
plt.figure(figsize=(10, 10))
plt.plot(y_cal[FRAMES_PLOT, :].T, "g", alpha=0.5)
for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
    plt.text(100, y_cal[i, 100] + 0.05, i)
plt.title("Linear and hill straylight removed, summing of well-illuminated columns only")
plt.xlabel("Pixel number")
plt.ylabel("Transmittance")
plt.grid()

# attempt to combine hill straylight removal in UV with just linear removal in the visible
y_plot = np.nansum(Y_no_hill_straylight * ymask_illum[:, :, None], axis=1).T
y_plot_vis = np.nansum(Y_no_lin_straylight * ymask_illum[:, :, None], axis=1).T
y_plot[:, 400:] = y_plot_vis[:, 400:]
toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
y_cal = y_plot / toa[None, :]
plt.figure(figsize=(10, 10))
plt.plot(y_cal[FRAMES_PLOT, :].T, "lightgreen", alpha=0.5)
for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
    plt.text(100, y_cal[i, 100] + 0.05, i)
plt.title("Combine hill straylight removal in UV with just linear removal in the visible")
plt.xlabel("Pixel number")
plt.ylabel("Transmittance")
plt.grid()

# attempt to smooth hill straylight
y_plot = np.nansum(Y_no_lin_straylight * ymask_par_illum[:, :, None], axis=1).T - hill_straylight_frames_smooth.T
toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
y_cal = y_plot / toa[None, :]
plt.figure(figsize=(10, 10))
plt.plot(y_cal[FRAMES_PLOT, :].T, "darkgreen", alpha=0.5)
for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
    plt.text(100, y_cal[i, 100] + 0.05, i)
plt.title("Hill and linear straylight removal - smoothed")
plt.xlabel("Pixel number")
plt.ylabel("Transmittance")
plt.grid()

# y_plot = np.nansum(Y_no_lin_straylight * ymask_par_illum[:, :, None], axis=1).T
# toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
# y_cal = y_plot / toa[None, :]
# plt.figure(figsize=(10, 10))
# plt.plot(y_cal[FRAMES_PLOT, :].T, "b", alpha=0.5)
# for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
#     plt.text(100, y_cal[i, 100] + 0.05, i)
# plt.title("Y no lin sl par illum mask")
# plt.xlabel("Pixel number")
# plt.ylabel("Transmittance")
# plt.grid()

# y_plot = np.nansum(Y_no_lin_straylight * ymask_illum[:, :, None], axis=1).T
# toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
# y_cal = y_plot / toa[None, :]
# plt.figure(figsize=(10, 10))
# plt.plot(y_cal[FRAMES_PLOT, :].T, "lightblue", alpha=0.5)


# y_plot = np.nansum(Y_no_lin_straylight, axis=1).T
# toa = np.mean(y_plot[FRAMES_TOA, :], axis=0)
# y_cal = y_plot / toa[None, :]
# plt.figure(figsize=(10, 10))
# plt.plot(y_cal[FRAMES_PLOT, :].T, "k", alpha=0.5)
# for i in np.arange(y_cal.shape[0])[FRAMES_PLOT]:
#     plt.text(100, y_cal[i, 100] + 0.05, i)
# plt.title("y no lin sum")
# plt.xlabel("Pixel number")
# plt.ylabel("Transmittance")
# plt.grid()


# %%

# ongoing work, can ignore
means = []
stds = []
rolling_mid = 200
rolling_ixs_all = np.lib.stride_tricks.sliding_window_view(np.arange(rolling_mid - 10, rolling_mid + 10), 10)
for rolling_ixs in rolling_ixs_all:
    mean = np.mean(Y_no_hill_straylight[rolling_ixs, :, 0], axis=0)
    std = np.std(Y_no_hill_straylight[rolling_ixs, :, 0], axis=0)
    means.append(mean)
    stds.append(std)
means = np.asarray(means)
stds = np.asarray(stds)

mean_mean = np.mean(means, axis=0)
mean_mean_max = np.max(mean_mean)
mean_mean /= mean_mean_max
std_std = np.std(stds, axis=0)
std_std /= np.max(std_std)

plt.figure()
plt.plot(np.arange(len(mean_mean)) + vstart, mean_mean)
plt.plot(np.arange(len(mean_mean)) + vstart, std_std)

x_top_end = np.where(std_std > 0.02)[0][0] - 2
x_bottom_start = np.where(std_std > 0.02)[0][-1] + 2

x_top_start = np.where(mean_mean > 0.0015)[0][0]
x_bottom_end = np.where(mean_mean > 0.0015)[0][-1]

plt.axvline(x_top_start + vstart)
plt.axvline(x_top_end + vstart)
plt.axvline(x_bottom_start + vstart)
plt.axvline(x_bottom_end + vstart)

line = ", [%i, %i], [%i, %i], [%i, %i], [%i, %i]" % (
    rolling_mid + xpix_1b[0],
    x_top_start + vstart,
    rolling_mid + xpix_1b[0],
    x_top_end + vstart,
    rolling_mid + xpix_1b[0],
    x_bottom_start + vstart,
    rolling_mid + xpix_1b[0],
    x_bottom_end + vstart,
)
print(line)

# plt.figure()
# for mean, std in zip(means, stds):
#     plt.plot(mean, alpha=0.1)
#     plt.plot(std, alpha=0.1)

col = Y_no_hill_straylight[rolling_mid, :, 0]
ixs = np.concatenate((np.arange(x_top_start, x_top_end + 1), np.arange(x_bottom_start, x_bottom_end)))
ixs_all = np.arange(x_top_start, x_bottom_end)

poly = Polynomial.fit(ixs, col[ixs], 2)
poly_vals = poly(ixs_all)
plt.scatter(ixs + vstart, col[ixs] / np.max(poly_vals))
plt.scatter(ixs_all + vstart, poly_vals / np.max(poly_vals))

Y_no_hill_straylight2 = np.copy(Y_no_hill_straylight)

Y_no_hill_straylight2[rolling_mid, ixs_all, 0] -= poly_vals

plt.figure()
plt.plot(Y_no_hill_straylight[rolling_mid, :, 0])
plt.plot(Y_no_hill_straylight2[rolling_mid, :, 0])
