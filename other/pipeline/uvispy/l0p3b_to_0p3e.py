# -*- coding: utf-8 -*-
"""
Convert 0p3b to 0p3c

Also combines the post_matlab_cleanup script, removing non-science frames

ONLY FOR PFM OCCULTATIONS

"""


import glob
import matplotlib.pyplot as plt
import logging
import h5py
import numpy as np
import os
from scipy.interpolate import interp1d
from datetime import datetime
from numpy.polynomial import Polynomial
# import shutil
# import nomad_ops.core.hdf5.generic_functions as generics
# from nomad_ops.core.tools.progress_bar import progress


__project__ = "NOMAD"
__author__ = "Ian Thomas, adapted from code by Yannick Willame"
__contact__ = "ian.thomas@aeronomie.be"


# TODO: change back
# logger = logging.getLogger(__name__)
import sys
logger = logging.getLogger()
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


VERSION = 90
OUTPUT_VERSION = "0.3E"

# PLOT_TYPES = ["linear_fit", "rises_falls", "illumination"]
# PLOT_TYPES = ["rises_falls"]
# PLOT_TYPES = ["illumination"]
# PLOT_TYPES = ["illumination", "non_illumination"]
PLOT_TYPES = []


ATTRIBUTES_TO_BE_REMOVED = ["NSpec"]
DATASETS_TO_BE_REMOVED = [
    "Science/Y",
    "Science/YMask",
    "Science/YValidFlag",
    "Science/YError",
    "Science/YErrorRandom",
    "Science/YErrorSystematic",
    "Science/StraylightInflight",
    "Science/X",
    "Science/CircuitNoise",
    "Science/YNb",
    "Science/XNbBin",
    "Science/Xpix_1b",
    "Science/YMaskROI",
    "Channel/VStart_YMaskROI_0b",

    "Science/YErrorSysNL",
]

DATASETS_NOT_TO_BE_RESHAPED = [
    'BadPixelMap',
    'PointXY',
    #        'Housekeeping',
    'Telecommand20/',
    'QualityFlag/',
    'CircuitNoise',  # for UVIS
    'TM11/',
    'TM29/',
    'XNbBin',
    'Xpix_1b',
    'FirstDC_Corr',
    'FirstDC_LevInit',
    'VStart_YMaskROI_0b',
    'Temperature/',
    'InstrF_allROIPixels',
    'nbrBinV_allROIPixels',
    'YMaskROI_allROIPixels',
    'nbrBinV_ExtendROI',
    'Percent_AnomalyDark',
    'Percent_HotPixels'
]

DATASETS_NOT_TO_BE_RESHAPED_COMPRESSED = [
    'Telecommand20',
    'QualityFlag'
]


MASK_VALUES = {
    # "NonLinCorr": 1,

    # "HotPix": 10,
    "Saturation": 20,
    "Anomaly": 30,

    "L02A_NaN": 90,
    "L02C_NaN": 80,

    # For SO measurements only
    "NotUsedPixSO": 50,
    "NotUsedLineSO": 60,
    "NotUsedLambdaSO": 70,

    "NotInBinROI": 100,
}

CRIT_MASK = {
    "GoodPixelsKept": 1.5,
    "HotPixelsKept": 11.5,
}


"""from 0p2b to 0p3b: modified"""


def define_roi(vstart, xpix_1b):
    """Define the top and bottom illuminated lines. Smearing and checks on data size have been removed"""
    # get well illuminated zone start end rows on full frame grid
    # offset compared to matlab
    i_line_top_start, i_line_bottom_start = define_top_bottom_well_illuminated_lines(xpix_1b)

    # convert to illuminated rows
    # offset compared to matlab
    i_start_line_roi = np.asarray(i_line_top_start - vstart + 1, dtype=int)
    i_end_line_roi = np.asarray(i_line_bottom_start - vstart + 1, dtype=int)

    # same values as matlab
    n_lines_roi = (i_end_line_roi - i_start_line_roi) + 1

    # Create the matrix for the binned ROI
    mat_bin_roi = np.zeros((xpix_1b.shape[0], int(np.max(n_lines_roi))), dtype=bool)
    # mat_bin_roi = np.zeros((xpix_1b.shape[0], int(np.max(n_lines_roi))), dtype=float)

    # TODO: replace with floats to solve sawtooth
    for ipix in range(i_line_bottom_start.shape[0]):
        mat_bin_roi[ipix, int(i_start_line_roi[ipix] - np.min(i_start_line_roi)):int(i_end_line_roi[ipix] - np.min(i_start_line_roi)) + 1] = 1

    return i_start_line_roi, i_end_line_roi, n_lines_roi, mat_bin_roi


def define_bounded_roi(vstart, xpix_1b):
    """Define the top and bottom illuminated lines. Smearing and checks on data size have been removed"""
    # get well illuminated zone start end rows on full frame grid
    # offset compared to matlab
    i_line_top_start, i_line_bottom_start = define_top_bottom_bounding_illuminated_lines(xpix_1b)

    # convert to illuminated rows
    # offset compared to matlab
    i_start_line_roi = np.asarray(i_line_top_start - vstart + 1, dtype=int)
    i_end_line_roi = np.asarray(i_line_bottom_start - vstart + 1, dtype=int)

    # same values as matlab
    n_lines_roi = (i_end_line_roi - i_start_line_roi) + 1

    # Create the matrix for the binned ROI
    mat_bin_roi = np.zeros((xpix_1b.shape[0], int(np.max(n_lines_roi))), dtype=bool)
    # mat_bin_roi = np.zeros((xpix_1b.shape[0], int(np.max(n_lines_roi))), dtype=float)

    # TODO: replace with floats to solve sawtooth
    for ipix in range(i_line_bottom_start.shape[0]):
        mat_bin_roi[ipix, int(i_start_line_roi[ipix] - np.min(i_start_line_roi)):int(i_end_line_roi[ipix] - np.min(i_start_line_roi)) + 1] = 1

    return i_start_line_roi, i_end_line_roi, n_lines_roi, mat_bin_roi


def define_top_bottom_well_illuminated_lines(X):
    """outputs same as matlab with indices subtracted by 1. Only for pFM occultation at present"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    x_top = np.asarray([[1, 172], [100, 172], [200, 170], [300, 169], [400, 167], [500, 165], [
                       600, 162], [700, 160], [800, 158], [900, 156], [1032, 152]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    x_bottom = np.asarray([[1, 178], [100, 178], [200, 178], [300, 178], [400, 179], [500, 179], [
                          600, 180], [700, 181], [800, 182], [900, 183], [1032, 185]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    return iLineTopStart, iLineBottomStart


def define_top_bottom_bounding_illuminated_lines(X):
    # TODO: finish this
    """outputs same as matlab with indices subtracted by 1. Only for pFM occultation at present"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    x_top = np.asarray([[1, 172 - 5], [100, 172 - 5], [200, 170 - 5], [300, 169 - 5], [400, 167 - 5], [500, 165 - 5], [
                       600, 162 - 5], [700, 160 - 5], [800, 158 - 5], [900, 156 - 5], [1032, 152 - 5]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    x_bottom = np.asarray([[1, 178 + 5], [100, 178 + 5], [200, 178 + 5], [300, 178 + 5], [400, 179 + 5], [500, 179 + 5], [
                          600, 180 + 5], [700, 181 + 5], [800, 182 + 5], [900, 183 + 5], [1032, 185 + 5]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    return iLineTopStart, iLineBottomStart


def define_top_bottom_non_illuminated_lines(X):
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    # for non-illuminated region
    # x_ref_bottom = np.array([9, 102, 195, 288, 381, 474, 567, 660, 753, 846, 939, 1032])
    # x_ref_top = np.array([9, 102, 195, 288, 381, 474, 567, 660, 753, 846, 939, 1032])
    # iLineBottomStart = np.interp(x, x_ref_bottom, np.array(
    #     [167, 164, 163, 161, 158, 157, 155, 153, 151, 149, 146, 144]), left=None, right=None)
    # iLineTopStart = np.interp(x, x_ref_top, np.array(
    #     [183, 183, 184, 185, 185, 186, 187, 188, 189, 190, 192, 193]), left=None, right=None)

    # new values
    x_top = np.asarray([[9, 169], [80, 169], [158, 161], [180, 154], [404, 152], [433, 150], [752, 154], [875, 149], [1032, 146]])
    iLineTopStart = np.interp(x, x_top[:, 0], x_top[:, 1], left=None, right=None)

    x_bottom = np.asarray([[9, 185], [116, 185], [200, 191], [300, 193], [420, 194], [626, 195], [799, 197], [1032, 199]])
    iLineBottomStart = np.interp(x, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    iLineTopStart = np.ceil(iLineTopStart).astype(int)
    iLineBottomStart = np.floor(iLineBottomStart).astype(int)

    return iLineTopStart, iLineBottomStart


"""end of 02pb functions"""
# prepare_integration_interval_door_bps(lambda_values) has been removed
# do_binning_v(Y, U2Y, YMask, nbr_bin_lines) has been removed


def test_enough_signal_present(YFrame, YMaskFrame, iStartLineROI, nbrROILines, xpix_1b):
    # Calculate the signal, masking out unwanted areas
    signal_sum = np.sum(YFrame[:, iStartLineROI:iStartLineROI + nbrROILines] * ~YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines], axis=1)
    mask_sum = np.sum(~YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines], axis=1)

    test_signal = signal_sum / mask_sum

    # Normalize by number of binned pixels
    nbr_pix_binned = xpix_1b[1] - xpix_1b[0]
    test_signal /= nbr_pix_binned

    # Filter out NaN values
    vect_no_nan = ~np.isnan(test_signal)

    # Count number of saturated pixels (where mask equals 2)
    nbr_saturation = np.sum(YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines] == MASK_VALUES["Saturation"])

    # Determine if there's enough signal
    if (np.mean(test_signal[vect_no_nan]) > 1000) or nbr_saturation >= 10:
        return 1
    else:
        # print("Not enough signal")
        return 0


def straylight_fitpoly2(i_start_line_ccd, nbr_ccd_lines, i_start_line_bin_roi, nbr_bin_roi_lines, y_frame, ymask_frame, xpix_1b, xnb_bin):

    # i_start_line_bin_roi is offset by 1
    nbr_lambda_pixels = y_frame.shape[0]

    # if nan_issue_top_cdd_missing_last_packets == -1: #for calib and occultations: all pixels
    first_pix_to_treat = 0
    last_pix_to_treat = nbr_lambda_pixels
    # else:
    #     first_pix_to_treat = np.min(np.where(xpix_1b >= 728))
    #     last_pix_to_treat = nbr_lambda_pixels

    xpix_1b = xpix_1b.flatten()
    xnb_bin = np.array(xnb_bin, dtype=np.float32).flatten()
    x = np.floor(xpix_1b + (xnb_bin / 2)).astype(int)

    i_line_top_start, i_line_bottom_start = define_top_bottom_non_illuminated_lines(x)

    # i_line_bottom_start and top start are the same as matlab except the last value!
    i_line_bottom_start = (i_line_bottom_start + 1 - i_start_line_ccd)
    i_line_top_start = (i_line_top_start + 1 - i_start_line_ccd)
    i_line_bottom_start[-1] -= 1
    i_line_top_start[-1] += 1

    num_line = np.arange(y_frame.shape[1]) + 1  # same as matlab

    yes_enough_signal = test_enough_signal_present(y_frame, ymask_frame, np.max(i_start_line_bin_roi), np.min(nbr_bin_roi_lines), xpix_1b)
    # print(yes_enough_signal)

    straylight_frame = np.full_like(y_frame, np.nan)
    obs_method_performed_on_lambda = np.zeros(nbr_lambda_pixels)

    straylight_spectrum = np.zeros(len(range(first_pix_to_treat, last_pix_to_treat)))

    # i_lambda is one less than matlab
    for i_lambda in range(first_pix_to_treat, last_pix_to_treat):
        y_line = y_frame[i_lambda, :]

        if not yes_enough_signal:
            temp_x = np.arange(1, i_line_bottom_start[i_lambda] + 1).astype(int)
            temp_x_ok = temp_x[~ymask_frame[i_lambda, temp_x - 1].astype(bool)]

            if len(temp_x_ok) > 1:
                p_bottom = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
                y_line[temp_x - 1] = np.polyval(p_bottom, temp_x)

            temp_x = np.arange(i_line_top_start[i_lambda], len(y_line) + 1).astype(int)
            temp_x_ok = temp_x[~ymask_frame[i_lambda, temp_x - 1].astype(bool)]

            if len(temp_x_ok) > 1:
                p_top = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
                y_line[temp_x - 1] = np.polyval(p_top, temp_x)

        obs_method_performed_on_lambda[i_lambda] = 1

        nbr_pts_adj = 8  # for occultation
        weight = np.zeros_like(y_line)

        # weight true indices are the same as matlab
        if (i_line_bottom_start[i_lambda] >= 3) and (i_line_top_start[i_lambda] <= (nbr_ccd_lines - 2)):
            weight[int(max(0, i_line_bottom_start[i_lambda] - (nbr_pts_adj + 2))):int(i_line_bottom_start[i_lambda]) - 1] = 1
            weight[int(i_line_top_start[i_lambda]):int(min(len(y_line), i_line_top_start[i_lambda] + (nbr_pts_adj + 1)))] = 1
        else:
            obs_method_performed_on_lambda[i_lambda] = 0

        weight2 = weight.astype(bool) & ~ymask_frame[i_lambda, :].astype(bool) & ~np.isnan(y_line)

        if np.sum(weight2) > 0:
            p = np.polyfit(num_line[weight2], y_line[weight2], 2)
            # this gives the same values as matlab
            straylight_frame[i_lambda, :] = np.polyval(p, num_line)

            # if i_lambda == 0:
            # print([(i,v) for i,v in enumerate(weight2)])
            # print(num_line[weight2], y_line[weight2], straylight_frame[i_lambda, :])
            # print(straylight_frame[i_lambda, :])

        else:
            straylight_frame[i_lambda, :] = np.nan

        # index values offset by 1
        num_line_illum = np.arange(i_start_line_bin_roi[i_lambda], i_start_line_bin_roi[i_lambda] + nbr_bin_roi_lines[i_lambda])
        # if i_lambda == 0:
        # print(num_line_illum)

        straylight_spectrum[i_lambda] = np.nansum(
            straylight_frame[i_lambda, num_line_illum] * (ymask_frame[i_lambda, num_line_illum] < CRIT_MASK["HotPixelsKept"])
        ) / np.nansum(ymask_frame[i_lambda, num_line_illum] < CRIT_MASK["HotPixelsKept"])
        # if i_lambda == 0:
        # print(i_lambda, straylight_frame[i_lambda, num_line_illum], ymask_frame[i_lambda, num_line_illum])
        # print(straylight_spectrum[i_lambda])

    straylight_spectrum[:int(first_pix_to_treat)] = 0

    vect_no_nan = ~np.isnan(straylight_spectrum)
    if np.sum(vect_no_nan) > 1:
        if np.sum(vect_no_nan) < len(vect_no_nan):
            interp_func = interp1d(x[vect_no_nan], straylight_spectrum[vect_no_nan], kind='linear', bounds_error=False, fill_value='extrapolate')
            straylight_spectrum = interp_func(x)
        else:
            straylight_spectrum = straylight_spectrum
    else:
        straylight_spectrum = np.full(nbr_lambda_pixels, np.nan)

    # else:
    #     straylight_spectrum = np.full(nbr_lambda_pixels, np.nan)
    #     obs_method_performed_on_lambda = np.zeros(nbr_lambda_pixels)

    return straylight_spectrum, obs_method_performed_on_lambda


# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/20240101_090648_0p3b_UVIS_I.h5"
# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/01/01/20240101_090648_0p3b_UVIS_I.h5"
# dst= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3e/2024/01/01/20240101_090648_0p3e_UVIS_I.h5"

src = "20230126_081530_0p3b_UVIS_I.h5"  # binned
# src = "20180522_051504_0p3b_UVIS_I.h5"  # missing last rows
# src = "20180522_051504_0p3b_UVIS_I.h5"  # good?
# src = "20240819_202356_0p3b_UVIS_I.h5"  # binned high ozone?

# src = "20240818_165255_0p3b_UVIS_I.h5"


# filepaths = glob.glob(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p3b\2024\10\20\*UVIS_I.h5")

# for src in filepaths:
dst = os.path.basename(src).replace("0p3b", "0p3e")
#     if os.path.exists(dst):
#         os.remove(dst)


# def remove_straylight(src, dst):
if True:
    # aux_path = os.path.join(PFM_AUXILIARY_FILES, "matlab", "v_07")

    hdf5_basename = os.path.basename(src).split(".")[0]

    d_out = {}

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
        nbr_lambda_pixels = xnb_bin.shape[0]  # n real columns e.g. 128
        x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

        CircuitNoise = np.swapaxes(f['Science/CircuitNoise'], 0, 1)
        YNb = f['Science/YNb'][frame_ixs_type4]

    logger.info("%s: acq_mode=%i, nbr_lambda_pixels=%i, vstart=%i, vend=%i, nbr_ccd_lines=%i, Yshape=%s",
                hdf5_basename, acq_mode, nbr_lambda_pixels, vstart, vend, nbr_ccd_lines, Y.shape)

    if x_calib_ref == 'N/A':
        print('lambda_IASB_v1: manual')
        x_calib_ref = 'lambda_IASB_v1'

    # mat_bin_roi is the boolean matrix defining the illuminated pixel indices
    start_line_bin_roi, end_line_bin_roi, nbr_bin_roi_lines, mat_bin_roi = define_roi(vstart, xpix_1b)

    iStartLineROI = min(start_line_bin_roi)
    iEndLineROI = max(end_line_bin_roi)
    nbrROILines = iEndLineROI - iStartLineROI + 1
    VStart_YMaskROI_0b = (iStartLineROI + vstart - 1) - 1

    logger.info("%s: iStartLineROI=%i, iEndLineROI=%i, nbrROILines=%i, VStart_YMaskROI_0b=%i, Yshape=%s",
                hdf5_basename, iStartLineROI, iEndLineROI, nbrROILines, vend, Y.shape)

    """calculate illumination shift"""
    # on which row ranges (approx) do the illumination patterns rise and fall for the given wavelengths?
    nms = [350, 375, 400, 425, 450, 500, 550, 600]
    illum_d = {s: {nm: {"px": [], "fit": []} for nm in nms} for s in ["rises", "falls"]}

    # which indices to check (need enough signal)
    frame_ixs = np.where((np.mean(Y, axis=(0, 1)) > np.max(np.mean(Y, axis=(0, 1))) * 0.2))[0]
    frame_ixs_all = np.arange(nbr_obs)

    if "linear_fit" in PLOT_TYPES:
        fig0, axes0 = plt.subplots(ncols=len(nms), figsize=(16, 10))
        cmap = plt.get_cmap("brg")
        colours1 = [cmap(i) for i in np.arange(len(frame_ixs)) / len(frame_ixs)]
        colours2 = [cmap(i) for i in np.arange(len(nms)) / len(nms)]

    if "rises_falls" in PLOT_TYPES:
        fig1, (ax1a, ax1b) = plt.subplots(ncols=2, figsize=(16, 10))

    for i, nm in enumerate(nms):
        nm_ix = np.argmin(np.abs(x[:, 0] - nm))

        illum_d[nm] = {"px": nm_ix}

        line_range_rise = [start_line_bin_roi[nm_ix] - 3, start_line_bin_roi[nm_ix] + 4]
        line_range_fall = [end_line_bin_roi[nm_ix] - 3 + 4, end_line_bin_roi[nm_ix] + 4 + 4]

        # check number of rows at bottom of detector, trim last range if needed
        if line_range_fall[1] > nbr_ccd_lines:
            line_range_fall[1] = nbr_ccd_lines

        for j, frame_ix in enumerate(frame_ixs):

            col = Y[nm_ix, :, frame_ix] / np.max(Y[nm_ix, :, frame_ix])

            if "linear_fit" in PLOT_TYPES:
                line = np.arange(col.shape[0], dtype=float) + vstart
                # need to flatten col to get real shape
                col_start_end = np.concatenate((col[:5], col[-5:]))
                ixs_start_end = np.concatenate((np.arange(5), np.arange(col.shape[0] - 5, col.shape[0])))
                col_baseline = np.interp(np.arange(col.shape[0]), ixs_start_end, col_start_end)
                col -= col_baseline
                col /= np.max(col)

                axes0[i].plot(line, col, color=colours1[j])

            x_rise = np.interp(0.5, col[line_range_rise[0]:line_range_rise[1]], np.arange(
                line_range_rise[0], line_range_rise[1]) + vstart)
            x_fall = np.interp(0.5, col[line_range_fall[0]:line_range_fall[1]][::-1],
                               np.arange(line_range_fall[0], line_range_fall[1])[::-1] + vstart)

            illum_d["rises"][nm]["px"].append(x_rise)
            illum_d["falls"][nm]["px"].append(x_fall)

            if "linear_fit" in PLOT_TYPES:
                axes0[i].scatter([x_rise, x_fall], [0.5, 0.5], color=colours1[j])

        illum_d["rises"][nm]["px"] = np.asarray(illum_d["rises"][nm]["px"])
        illum_d["falls"][nm]["px"] = np.asarray(illum_d["falls"][nm]["px"])
        p_rises = Polynomial.fit(frame_ixs, illum_d["rises"][nm]["px"], 4)
        p_falls = Polynomial.fit(frame_ixs, illum_d["falls"][nm]["px"], 4)
        illum_d["rises"][nm]["fit"] = p_rises(frame_ixs_all)
        illum_d["falls"][nm]["fit"] = p_falls(frame_ixs_all)

        illum_d["rises"][nm]["mean"] = np.mean(illum_d["rises"][nm]["px"])
        illum_d["falls"][nm]["mean"] = np.mean(illum_d["falls"][nm]["px"])

        if "rises_falls" in PLOT_TYPES:
            ax1a.plot(illum_d["rises"][nm]["px"] - illum_d["rises"][nm]["mean"], color=colours2[i], label="%i nm" % nm)
            ax1b.plot(illum_d["falls"][nm]["px"] - illum_d["falls"][nm]["mean"], color=colours2[i], label="%i nm" % nm)
            ax1a.plot(illum_d["rises"][nm]["fit"] - illum_d["rises"][nm]["mean"], color=colours2[i], label="%i nm" % nm)
            ax1b.plot(illum_d["falls"][nm]["fit"] - illum_d["falls"][nm]["mean"], color=colours2[i], label="%i nm" % nm)

    # relative shift from the mean, averaged for all calculated wavelength columns
    illum_d["rises"]["mean_fit"] = np.mean(np.asarray([illum_d["rises"][nm]["fit"] - illum_d["rises"][nm]["mean"] for nm in nms]), axis=0)
    illum_d["falls"]["mean_fit"] = np.mean(np.asarray([illum_d["falls"][nm]["fit"] - illum_d["falls"][nm]["mean"] for nm in nms]), axis=0)

    start_rows_roi = np.asarray([illum_d["rises"][nm]["mean"] + illum_d["rises"]["mean_fit"] for nm in nms]) - vstart
    end_rows_roi = np.asarray([illum_d["falls"][nm]["mean"] + illum_d["falls"]["mean_fit"] for nm in nms]) - vstart
    col_ixs = np.asarray([illum_d[nm]["px"] for nm in nms])

    if "rises_falls" in PLOT_TYPES:
        ax1a.plot(illum_d["rises"]["mean_fit"], color="k")
        ax1b.plot(illum_d["falls"]["mean_fit"], color="k")
        ax1a.legend()
        ax1b.legend()

    # xpix_1b = xpix_1b.flatten()
    # xnb_bin = np.array(xnb_bin, dtype=np.float32).flatten()
    x_new = np.floor(xpix_1b + (xnb_bin / 2)).astype(int)

    """Calculate for all columns. Only for pFM occultation at present"""
    px_ixs = np.arange(nbr_lambda_pixels)

    ix_non_illum_top, ix_non_illum_bottom = define_top_bottom_non_illuminated_lines(x_new)

    illum_matrix = np.zeros_like(Y)  # between 0 and 1
    non_illum_matrix = np.zeros_like(Y)  # only 0 or 1, no values in between
    for obs_ix in np.arange(nbr_obs):
        poly = Polynomial.fit(col_ixs, start_rows_roi[:, obs_ix], 2)
        fit_line_top_start = poly(px_ixs)
        poly = Polynomial.fit(col_ixs, end_rows_roi[:, obs_ix], 2)
        fit_line_bottom_start = poly(px_ixs)

        for col_ix in range(nbr_lambda_pixels):
            top_rounded = int(np.round(fit_line_top_start[col_ix]))
            bottom_rounded = int(np.round(fit_line_bottom_start[col_ix]))

            illum_matrix[col_ix, top_rounded + 1:bottom_rounded, obs_ix] = 1
            partial_illum = top_rounded - fit_line_top_start[col_ix] + 0.5
            illum_matrix[col_ix, top_rounded, obs_ix] = partial_illum
            partial_illum = bottom_rounded - fit_line_bottom_start[col_ix] + 0.5
            illum_matrix[col_ix, bottom_rounded, obs_ix] = 1 - partial_illum

            # define non-illuninated regions
            non_illum_matrix[col_ix, 0:(ix_non_illum_top[col_ix] - vstart), obs_ix] = 1
            non_illum_matrix[col_ix, (ix_non_illum_bottom[col_ix] - vstart):, obs_ix] = 1

        if obs_ix in [10] and "illumination" in PLOT_TYPES:
            plt.figure()
            plt.plot(px_ixs, fit_line_top_start, "r", linewidth=2)
            plt.plot(px_ixs, fit_line_bottom_start, "r", linewidth=2)
            plt.imshow(illum_matrix[:, :, obs_ix].T, cmap="Greys", aspect="auto")
        if obs_ix in [10] and "non_illumination" in PLOT_TYPES:
            plt.figure()
            plt.plot(px_ixs, fit_line_top_start)
            plt.plot(px_ixs, fit_line_bottom_start)
            plt.imshow(non_illum_matrix[:, :, obs_ix].T)

    # check if a pixel/bin is saturating in SO, then this pixel will not be binned (for all observations)
    # YMask is the noise on each pixel/bin in each frame, calculated in previous level
    # find pixels/bins where noise is too high in all frames
    ymaskframe_NotBinForOccult = np.sum(YMask > CRIT_MASK["HotPixelsKept"], axis=2, dtype=float)
    ymaskframe_NotBinForOccult[ymaskframe_NotBinForOccult > 0] = np.nan

    # ymaskframe_NotBinForOccult[]

    ymaskframes = np.repeat(ymaskframe_NotBinForOccult[:, :, None], nbr_obs, axis=2)

    logger.info("%s: %i bins not considered" % (hdf5_basename, np.sum(np.isnan(ymaskframe_NotBinForOccult))))

    # new: make region of interest ymask for each frame 0 = non illuminated, 1=illuminated, nan = bad
    ymaskframesBinROI = np.repeat(mat_bin_roi[:, :, None], nbr_obs, axis=2).astype(float)
    ymaskROI = ymaskframes + illum_matrix
    # ymaskROI = ymaskframes + ymaskframesBinROI

    """vertically bin data"""
    # new: do the binning here
    Y_bins = np.nansum(Y * ymaskROI, axis=1)
    U2_bins = np.nansum(YError * ymaskROI, axis=1)
    U2SysNL_bins = np.nansum(YErrorSysNL * ymaskROI, axis=1)
    nbr_bin_mat = np.nansum(ymaskROI, axis=1)  # number of good illuminated rows per pixel per frame
    nbr_satur_mat = np.sum(np.isnan(ymaskROI), axis=1)  # number of saturated and hot pixels now per pixel per frame

    # Create the hot pixel/saturation masks
    ymask = (nbr_bin_mat <= (nbr_bin_roi_lines[:, None] * 0.5)).astype(np.uint8)
    vect_saturated = (nbr_satur_mat >= (nbr_bin_roi_lines[:, None] * 0.5))

    # Mask the data
    Y_bins = np.where(~ymask, Y_bins, np.nan)
    U2_bins = np.where(~ymask, U2_bins, np.nan)

    # Replace infinities with NaN
    Y_bins[np.isinf(Y_bins)] = np.nan
    U2_bins[np.isinf(U2_bins)] = np.nan

    data_SmeaRem = Y_bins
    u2_data_SmeaRem = U2_bins
    u2_data_SysNL = U2SysNL_bins
    # nbrBinMat = nbr_bin_mat
    VectSaturated = vect_saturated

    u2_data_Rdm = (u2_data_SmeaRem / (nbr_bin_mat**2))  # error obtained up to here were random error -> to be saved in h5 file
    u2_data_SysNL = (u2_data_SysNL / (nbr_bin_mat**2))

    # data_OM is the same as matlab
    data_OM = (data_SmeaRem / nbr_bin_mat)
    data_OM2 = (data_SmeaRem / nbr_bin_mat)
    u2_data_OM = u2_data_Rdm + u2_data_SysNL  # add the systematic error due to Non-Linearity correction (if applied)
    # Straylight_OM = np.full_like(data_SmeaRem, np.nan)
    Straylight_OM2 = np.full_like(data_SmeaRem, np.nan)

    yvalidflag = np.zeros(nbr_obs, dtype=np.uint8)

    """straylight"""
    BP_N_ITERS = 3
    straylight_frames = np.zeros_like(Y)

    # use calculated values, each variable is 3d here
    ynon_maskROI = ymaskframes + non_illum_matrix
    Ynon = Y * ynon_maskROI

    # first get list of frames where sufficient signal
    middle_col_ix = int(nbr_lambda_pixels / 2)
    middle_col_sums = np.nansum(Ynon[middle_col_ix, :, :], axis=0)
    middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

    for frame_ix in range(nbr_obs):
        # for frame_ix in range(1):
        frame = Ynon[:, :, frame_ix]
        frame_non_illum_ixs = np.where((frame != 0) & ~np.isnan(frame))

        # only fit quadratic and report error if signal sufficient
        if middle_col_trans[frame_ix] > 0.1:
            poly_degree = 2
            chisq_cutoff = 10
        elif middle_col_trans[frame_ix] > 0.03:  # if low signal, fit linear
            poly_degree = 1
            chisq_cutoff = 20
        else:  # if very low signal, fit linear, ignore error
            poly_degree = 1
            chisq_cutoff = 99e99

        straylight_frame = np.full_like(frame, np.nan)
        obs_method_performed_on_lambda = np.ones(nbr_lambda_pixels)
        straylight_spectrum = np.zeros(nbr_lambda_pixels)

        for col_ix in range(nbr_lambda_pixels):
            # iterate to remove uncorrected bad and noisy pixels
            for loop in range(BP_N_ITERS):
                non_illum_ixs = frame_non_illum_ixs[1][frame_non_illum_ixs[0] == col_ix]

                poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
                straylight = poly(np.arange(nbr_ccd_lines))
                chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / poly(non_illum_ixs)
                chisq = np.sum(chisq_px) / len(non_illum_ixs)
                bad_ix = np.where(chisq_px == np.max(chisq_px))[0][0]
                frame[col_ix, non_illum_ixs[bad_ix]] = straylight[non_illum_ixs[bad_ix]]
            poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
            straylight = poly(np.arange(nbr_ccd_lines))
            chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / poly(non_illum_ixs)
            chisq = np.sum(np.abs(chisq_px)) / len(non_illum_ixs)

            if chisq > chisq_cutoff:
                # remove extra noisy pixels if chisq still too high
                print(frame_ix, col_ix, middle_col_trans[frame_ix], chisq)
                for loop in range(BP_N_ITERS):
                    non_illum_ixs = frame_non_illum_ixs[1][frame_non_illum_ixs[0] == col_ix]

                    poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
                    straylight = poly(np.arange(nbr_ccd_lines))
                    chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / poly(non_illum_ixs)
                    chisq = np.sum(np.abs(chisq_px)) / len(non_illum_ixs)
                    bad_ix = np.where(chisq_px == np.max(chisq_px))[0][0]
                    frame[col_ix, non_illum_ixs[bad_ix]] = straylight[non_illum_ixs[bad_ix]]
                poly = Polynomial.fit(non_illum_ixs, frame[col_ix, non_illum_ixs], poly_degree)
                straylight = poly(np.arange(nbr_ccd_lines))
                chisq_px = (frame[col_ix, non_illum_ixs] - poly(non_illum_ixs))**2 / poly(non_illum_ixs)
                chisq = np.sum(np.abs(chisq_px)) / len(non_illum_ixs)

                print(frame_ix, col_ix, middle_col_trans[frame_ix], chisq)
                if chisq > chisq_cutoff:
                    plt.figure()
                    plt.title("Non-illuminated zone and removed pixels")
                    plt.xlabel("Spectral dimension")
                    plt.ylabel("Detector row")
                    plot_frame = Ynon[:, :, frame_ix]
                    frame_col_max = np.nanmax(plot_frame, axis=1)
                    im = plt.imshow((plot_frame / frame_col_max[:, None]).T, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    # im = plt.imshow(Ynon[:, :, frame_ix].T, vmax=20000, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    plt.colorbar(im)

                    plt.figure()
                    plt.title("Column-normalised detector frame")
                    plt.xlabel("Spectral dimension")
                    plt.ylabel("Detector row")
                    plot_frame = Y[:, :, frame_ix]
                    frame_col_max = np.nanmax(plot_frame, axis=1)
                    im = plt.imshow((plot_frame / frame_col_max[:, None]).T, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    # im = plt.imshow(Ynon[:, :, frame_ix].T, vmax=20000, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    plt.colorbar(im)

                    plt.figure()
                    plt.title("Column-normalised illuminated zone")
                    plt.xlabel("Spectral dimension")
                    plt.ylabel("Detector row")
                    plot_frame = Y[:, :, frame_ix] * ymaskROI[:, :, frame_ix]
                    frame_col_max = np.nanmax(plot_frame, axis=1)
                    im = plt.imshow((plot_frame / frame_col_max[:, None]).T, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    # im = plt.imshow(Ynon[:, :, frame_ix].T, vmax=20000, extent=[xpix_1b[0], xpix_1b[-1], vstart + nbr_ccd_lines, vstart], aspect="auto")
                    plt.colorbar(im)

                    plt.figure()
                    plt.title("Straylight fit to non-illuminated rows in detector column %i" % col_ix)
                    plt.xlabel("Detector row")
                    plt.ylabel("Signal in non-illuminated zone")
                    plt.plot(frame[col_ix, :])
                    plt.scatter(non_illum_ixs, frame[col_ix, non_illum_ixs])
                    plt.plot(straylight)
                    stop()

            # total illuminated lines minus total nans
            n_not_nans = np.nansum((illum_matrix[col_ix, :, frame_ix]) < CRIT_MASK["HotPixelsKept"])
            straylight_spectrum[col_ix] = np.nansum(straylight * illum_matrix[col_ix, :, frame_ix]) / n_not_nans

            straylight_frames[col_ix, :, frame_ix] = straylight

            # if col_ix == 21 and middle_col_trans[frame_ix] < 0.97:
            #     plt.plot(frame[col_ix, :])
            #     plt.scatter(non_illum_ixs, frame[col_ix, non_illum_ixs])
            #     plt.plot(straylight)

        Straylight_OM2[:, frame_ix] = straylight_spectrum
        data_OM[:, frame_ix] -= straylight_spectrum
        yvalidflag[frame_ix] = 1

        print(frame_ix, middle_col_trans[frame_ix], chisq)

    # check non-illuminated lines
    # for i in range(0, 121, 10):
    #     plt.figure()
    #     plt.title(i)
    #     # plt.ylim(-100, 6000)
    #     # plt.plot(Y[i, :, 400:410])
    #     # plt.plot(Ynon[i, :, 400:410])
    #     plt.ylim(-100, 12000)
    #     plt.plot(Y[i, :, 162:172])
    #     plt.plot(Ynon[i, :, 162:172])

    Y_no_straylight = Y - straylight_frames

    data_OM = np.nansum(Y_no_straylight, axis=1)  # raw sum including top and bottom straylight

    plt.figure()
    plt.plot(data_OM[:, 400:410] / np.nanmax(data_OM[:, 400:410], axis=0)[None, :])

    # check illuminated lines
    Yillum = Y_no_straylight * ymaskROI
    data_OM = np.nansum(Yillum, axis=1)  # selected illuminated rows after straylight correction

    for i in [0, 1, 2, 3, 4, 13, 21, 23]:  # range(0, 121, 10):
        plt.figure()
        plt.title(i)
        if src == "20230126_081530_0p3b_UVIS_I.h5":
            # plt.ylim(-100, 6000)
            # plt.plot(Y_no_straylight[i, :, 400:410])
            plt.plot(Yillum[i, :, 400:410] / np.nanmax(Yillum[i, :, 400:410], axis=0)[None, :])
        # plt.ylim(-100, 12000)
        # plt.plot(Y[i, :, 162:172])
        # plt.plot(Ynon[i, :, 162:172])

    # plt.figure()
    # for i in range(5, nbr_obs, 10):
    #     for j in range(5, nbr_lambda_pixels, 130):
    #         plt.plot(Ynon[j, :, i])

    # first_pix_to_treat = 0
    # last_pix_to_treat = nbr_lambda_pixels

    # i_line_top_start, i_line_bottom_start = define_top_bottom_non_illuminated_lines(x_new)

    # # i_line_bottom_start and top start are the same as matlab except the last value!
    # i_line_bottom_start = (i_line_bottom_start + 1 - vstart)
    # i_line_top_start = (i_line_top_start + 1 - vstart)
    # i_line_bottom_start[-1] -= 1
    # i_line_top_start[-1] += 1

    # for i_obs in range(nbr_obs):
    #     # if itypedata[i_obs] == 4:  # science measurement
    #     # Straylight calculation

    #     num_line = np.arange(Y[:, :, i_obs].shape[1]) + 1  # same as matlab
    #     yes_enough_signal = test_enough_signal_present(Y[:, :, i_obs], YMask[:, :, i_obs], np.max(start_line_bin_roi), np.min(nbr_bin_roi_lines), xpix_1b)
    #     # print(yes_enough_signal)

    #     straylight_frame = np.full_like(Y[:, :, i_obs], np.nan)
    #     obs_method_performed_on_lambda = np.zeros(nbr_lambda_pixels)

    #     straylight_spectrum = np.zeros(len(range(first_pix_to_treat, last_pix_to_treat)))

    #     # i_lambda is one less than matlab
    #     for i_lambda in range(first_pix_to_treat, last_pix_to_treat):
    #         y_line = Y[i_lambda, :, i_obs]

    #         if not yes_enough_signal:
    #             temp_x = np.arange(1, i_line_bottom_start[i_lambda] + 1).astype(int)
    #             temp_x_ok = temp_x[~YMask[i_lambda, temp_x - 1, i_obs].astype(bool)]

    #             if len(temp_x_ok) > 1:
    #                 p_bottom = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
    #                 y_line[temp_x - 1] = np.polyval(p_bottom, temp_x)

    #             temp_x = np.arange(i_line_top_start[i_lambda], len(y_line) + 1).astype(int)
    #             temp_x_ok = temp_x[~YMask[i_lambda, temp_x - 1, i_obs].astype(bool)]

    #             if len(temp_x_ok) > 1:
    #                 p_top = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
    #                 y_line[temp_x - 1] = np.polyval(p_top, temp_x)

    #         obs_method_performed_on_lambda[i_lambda] = 1

    #         nbr_pts_adj = 8  # for occultation
    #         weight = np.zeros_like(y_line)

    #         # weight true indices are the same as matlab
    #         if (i_line_bottom_start[i_lambda] >= 3) and (i_line_top_start[i_lambda] <= (nbr_ccd_lines - 2)):
    #             weight[int(max(0, i_line_bottom_start[i_lambda] - (nbr_pts_adj + 2))):int(i_line_bottom_start[i_lambda]) - 1] = 1
    #             weight[int(i_line_top_start[i_lambda]):int(min(len(y_line), i_line_top_start[i_lambda] + (nbr_pts_adj + 1)))] = 1
    #         else:
    #             obs_method_performed_on_lambda[i_lambda] = 0

    #         weight2 = weight.astype(bool) & ~YMask[i_lambda, :, i_obs].astype(bool) & ~np.isnan(y_line)

    #         if np.sum(weight2) > 0:
    #             p = np.polyfit(num_line[weight2], y_line[weight2], 2)
    #             # this gives the same values as matlab
    #             straylight_frame[i_lambda, :] = np.polyval(p, num_line)

    #             # if i_lambda == 0:
    #             # print([(i,v) for i,v in enumerate(weight2)])
    #             # print(num_line[weight2], y_line[weight2], straylight_frame[i_lambda, :])
    #             # print(straylight_frame[i_lambda, :])

    #         else:
    #             straylight_frame[i_lambda, :] = np.nan

    #         # index values offset by 1
    #         num_line_illum = np.arange(start_line_bin_roi[i_lambda], start_line_bin_roi[i_lambda] + nbr_bin_roi_lines[i_lambda])
    #         # if i_lambda == 0:
    #         # print(num_line_illum)

    #         straylight_spectrum[i_lambda] = np.nansum(
    #             straylight_frame[i_lambda, num_line_illum] *
    #             (YMask[i_lambda, num_line_illum, i_obs] < CRIT_MASK["HotPixelsKept"])
    #         ) / np.nansum(YMask[i_lambda, num_line_illum, i_obs] < CRIT_MASK["HotPixelsKept"])

    #     straylight_spectrum[:int(first_pix_to_treat)] = 0

    #     vect_no_nan = ~np.isnan(straylight_spectrum)
    #     if np.sum(vect_no_nan) > 1:
    #         if np.sum(vect_no_nan) < len(vect_no_nan):
    #             interp_func = interp1d(x_new[vect_no_nan], straylight_spectrum[vect_no_nan], kind='linear', bounds_error=False, fill_value='extrapolate')
    #             straylight_spectrum = interp_func(x_new)
    #         else:
    #             straylight_spectrum = straylight_spectrum
    #     else:
    #         straylight_spectrum = np.full(nbr_lambda_pixels, np.nan)

    #     data_OM2[:, i_obs] -= straylight_spectrum
    #     Straylight_OM[:, i_obs] = straylight_spectrum
    #     yvalidflag[i_obs] = 1 if np.sum(~np.isnan(data_OM[:, i_obs])) > 0.5 else 0

        # stop()
        # else:
        #     data_OM[:, i_obs] = np.nan
        #     yvalidflag[i_obs] = 0  # set to non valid measurement

    # Handling NaNs and transposing
    u2_data_Rdm[np.isinf(u2_data_Rdm)] = np.nan
    # u2_data_SysNL[np.isinf(u2_data_SysNL)] = np.nan
    data_OM[np.isinf(data_OM)] = np.nan
    u2_data_OM[np.isinf(u2_data_OM)] = np.nan
    # Straylight_OM[np.isinf(Straylight_OM)] = np.nan
    Straylight_OM2[np.isinf(Straylight_OM2)] = np.nan
    u2_data_Sys_OM = u2_data_OM - u2_data_Rdm
    # data_NoCorr[np.isinf(data_NoCorr)] = np.nan
    # u2_data_NoCorr[np.isinf(u2_data_NoCorr)] = np.nan

    # Convert XNbBin to uint8
    xnb_bin = xnb_bin.astype(np.uint8)

    # # Resizing due to too many NaNs in SO measurements
    # if nbr_lambda_pixels > np.sum(obs_method_performed_on_lambda):
    #     nbr_lambda_pixels = np.sum(obs_method_performed_on_lambda)

    #     mask = obs_method_performed_on_lambda.astype(bool)

    #     data_OM = data_OM[mask, :]
    #     u2_data_OM = u2_data_OM[mask, :]
    #     u2_data_Sys_OM = u2_data_Sys_OM[mask, :]
    #     u2_data_Rdm = u2_data_Rdm[mask, :]
    #     Straylight_OM = Straylight_OM[mask, :]
    #     ymask = ymask[mask, :]
    #     ymaskROI = ymaskROI[mask, :, :]
    #     CircuitNoise = CircuitNoise[mask, :]
    #     x = x[mask, :]

    #     YNb[:] = nbr_lambda_pixels
    #     xnb_bin = xnb_bin[mask].astype(np.uint8)
    #     xpix_1b = xpix_1b[mask]

    d_out["Science/Y"] = data_OM.astype(np.float32)
    d_out["Science/YMask"] = ymask
    d_out["Science/YValidFlag"] = yvalidflag

    d_out["Science/YError"] = u2_data_OM
    d_out["Science/YErrorRandom"] = u2_data_Rdm
    d_out["Science/YErrorSystematic"] = u2_data_Sys_OM
    # d_out["Science/StraylightInflight"] = Straylight_OM
    d_out["Science/StraylightInflight"] = Straylight_OM2

    d_out["Science/X"] = x
    d_out["Science/CircuitNoise"] = CircuitNoise
    d_out["Science/YNb"] = YNb
    d_out["Science/XNbBin"] = xnb_bin
    d_out["Science/Xpix_1b"] = xpix_1b
    d_out["Science/YMaskROI"] = ymaskROI
    d_out["Channel/VStart_YMaskROI_0b"] = np.uint8(VStart_YMaskROI_0b)

    # TODO: remove plotting
    y_plot = d_out["Science/Y"].T

    if src == '20230126_081530_0p3b_UVIS_I.h5':
        toa_frames = [298, 348]
        plot_frames = [378, 418]
    elif src == '20180522_051504_0p3b_UVIS_I.h5':
        toa_frames = [50, 75]
        plot_frames = [150, 165]
    else:
        toa_frames = [50, 75]
        plot_frames = [100, 200]

    toa = np.mean(y_plot[toa_frames[0]:toa_frames[1], :], axis=0)
    y_cal = y_plot / toa[None, :]
    plt.figure(figsize=(10, 10))
    # plt.plot(y_cal[380:420, :].T, "k")
    plt.plot(y_cal[plot_frames[0]:plot_frames[1], :].T, "k")
    # check for changes
    plt.savefig("uvis_trans_%s.png" % (str(datetime.now())[:19].replace("-", "_").replace(" ", "_").replace(":", "_")), dpi=200)

    # indrec = np.argwhere(itypedata == 4)[:, 0]
    indrec = np.arange(nbr_obs)
    logger.info("NSpec old=%i, NSpec new=%i", len(itypedata), len(indrec))

    with h5py.File(src, 'r') as hdf5FileIn:
        with h5py.File(dst, 'w') as hdf5FileOut:

            # generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)

            hdf5FileOut.attrs["NSpec"] = len(indrec)

            # # don't copy all datasets to new file
            # for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            #     # print(dset_path)
            #     if dset_path in DATASETS_TO_BE_REMOVED:  # don't copy
            #         # print("Not copying %s" %dset_path)
            #         continue

            #     dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])

            #     if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
            #         hdf5FileIn.copy(dset_path, dest)
            #         # print("Copying but not reshaping %s" %dset_path)
            #         continue

            #     dsetcopy = np.array(dset)
            #     # print("%s dataset is being reshaped (shape=%s)" %(dset_path, dsetcopy.shape))
            #     if dsetcopy.ndim == 0:
            #         hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy)
            #     elif dsetcopy.ndim == 1:  # if 1D, use compression
            #         # print("Length = %i" %dsetcopy.shape[0])
            #         hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec], compression="gzip", shuffle=True)
            #     elif len(dsetcopy.shape) == 2:  # if 2D, use compression
            #         # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
            #         hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :], compression="gzip", shuffle=True)
            #     elif len(dsetcopy.shape) == 3:  # if 3D, use compression
            #         # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
            #         hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :, :], compression="gzip", shuffle=True)

            for key, value in d_out.items():
                # print("%s dataset (shape=%i)" %(key, len(value.shape)))
                if key == "Science/YMask":
                    dtype = "uint8"
                else:
                    dtype = "float32"

                if value.ndim == 1:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value, dtype=dtype, compression="gzip", shuffle=True)
                    else:
                        dset = value[indrec]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, compression="gzip", shuffle=True)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                elif value.ndim == 2:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value.T, dtype=dtype, chunks=(
                            value.shape[1], min(24, value.shape[0])), compression="gzip", compression_opts=4)
                    else:
                        dset = value.T[indrec, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(
                            len(indrec), min(24, value.shape[0])), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                elif value.ndim == 3:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=np.swapaxes(value, 0, 2), dtype=dtype, chunks=(
                            1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                    else:
                        dset = np.swapaxes(value, 0, 2)[indrec, :, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                else:
                    hdf5FileOut.create_dataset(key, data=value, dtype=dtype)


# def convert(hdf5file_path):
#     """this function should perform the same function as nomad_ops/matlab/src/RemoveStraylight.m"""
#     logger.info("Convert: %s", hdf5file_path)
#     tmp_file = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
#     # shutil.copyfile(hdf5file_path, tmp_file)
#     remove_straylight(hdf5file_path, tmp_file)
#     return [tmp_file]


# with h5py.File(dst, "r") as f:
#     y_new = f["Science/Y"][...]

# dst_old = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3e/2018/05/22/20180522_051504_0p3c_UVIS_I.h5"
# with h5py.File(dst_old, "r") as f:
#     y_old = f["Science/Y"][...]

# plt.figure()
# plt.imshow(y_old)
# plt.title("Y Old")
# plt.show()

# plt.figure()
# plt.imshow(y_new)
# plt.title("Y New")
# plt.show()


# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/01/01/20240101_090648_0p3b_UVIS_I.h5"
# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/05/02/20240502_193435_0p3b_UVIS_E.h5"
# convert(src)
