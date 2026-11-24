# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:40:21 2026

@author: iant

CHECK ALL 0.3D COLUMN SLICES
"""

import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tools.general.progress_bar import progress

filepaths = glob.glob(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p3d\2024\05\**\*UVIS_I.h5", recursive=True)
filepaths = glob.glob(r"X:\projects\work\NOMAD\test\iant\hdf5\hdf5_level_0p3d\2024\05\**\*UVIS_I.h5", recursive=True)
# filepaths = glob.glob(r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d/2024/05/**/*UVIS_I.h5", recursive=True)

# filepaths = glob.glob(r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d/2024/05/12/20240512_0*UVIS_I.h5", recursive=True)
# filepaths = ["20240501_021829_0p3d_UVIS_E.h5"]
# filepaths = [r"X:\projects\work\NOMAD\test\iant\hdf5\hdf5_level_0p3d\2024\05\01\20240501_021829_0p3d_UVIS_E.h5"]
# filepaths = [r"X:\projects\work\NOMAD\test\iant\hdf5\hdf5_level_0p3d\2024\05\12\20240512_092439_0p3d_UVIS_E.h5"]

filepaths = [r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d\2024\05\01\20240501_021829_0p3d_UVIS_E.h5"]
# filepaths = [r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d\2024\05\12\20240512_092439_0p3d_UVIS_E.h5"]


def plot_frame_col_norm(frame, vmin=None, vmax=None, extent=None, figsize=None):
    """function to plot a column-normalised 2d detector frame"""

    y_col_max = np.max(frame, axis=1)
    frame_col_max = frame / y_col_max[:, None]
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    if extent:
        plt.imshow(frame_col_max.T, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", interpolation='none')
    else:
        plt.imshow(frame_col_max.T, vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')


def define_top_bottom_hill_lines(X):
    """outputs the top and bottom row indices of the hill zone"""
    delta_x = X[1] - X[0]
    # when binned, take the value of the rightmost binned pix
    x = np.floor(X + delta_x / 2)

    # straylight hill
    x_top_start = np.asarray([[9, 164], [59, 162], [109, 160], [159, 160], [209, 157], [259, 155], [309, 155], [359, 155], [409, 154], [
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


for src in progress(filepaths[0:1]):
    # dst = os.path.basename(src).replace("0p3d", "0p3e")
    #     if os.path.exists(dst):
    #         os.remove(dst)

    h5_basename = os.path.basename(src).split(".")[0]
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
        # YError = np.swapaxes(f["Science/YError"][...], 0, 2)
        # YMask = np.swapaxes(f["Science/YMask"][...], 0, 1)  # not correct
        # YErrorSysNL = np.swapaxes(f["Science/YErrorSysNL"][...], 0, 2)
        YWellIllumMask = f["Science/YIllumMask"][...]
        YNonIllumMask = f["Science/YNonIllumMask"][...]

        x = f['Science/X'][0, :]
        xpix_1b = f['Science/Xpix_1b'][...]  # array of pixel centre indices for real columns in matlab indices e.g. [12.5, 20.5, 28.5,...]
        # xnb_bin = f['Science/XNbBin'][...]  # array of binning factors for real columns e.g. [8,8,8,...]
        # x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

        # YNb = f['Science/YNb']
        alts = np.mean(f["Geometry/Point0/TangentAltAreoid"][...], axis=1)

    nbr_obs = Y.shape[0]
    nbr_lambda_pixels = Y.shape[2]  # n real columns e.g. 128 or 1024
    nbr_rows = Y.shape[1]  # n real columns e.g. 128

    # divide Y by binning factor to match binned and unbinned
    Y /= (1024 / nbr_lambda_pixels)

    ix_sl_top_start, ix_sl_top_end, ix_sl_bottom_start, ix_sl_bottom_end = define_top_bottom_hill_lines(xpix_1b)
    ix_sl_top_start -= vstart
    ix_sl_top_end -= vstart
    ix_sl_bottom_start -= vstart
    ix_sl_bottom_end -= vstart

    ymask_illum = YWellIllumMask
    ymask_non_illum = YNonIllumMask

    ymask_sl = np.zeros_like(Y[0, :, :])
    for col_ix in range(nbr_lambda_pixels):
        # vertical indices to fit
        sl_ixs = np.concatenate((
            np.arange(ix_sl_top_start[col_ix], ix_sl_top_end[col_ix] + 1),
            np.arange(ix_sl_bottom_start[col_ix], ix_sl_bottom_end[col_ix] + 1)
        ))
        ymask_sl[sl_ixs, col_ix] = 1

    ymask_sl_3d = np.repeat(ymask_sl[None, :, :], nbr_obs, axis=0)

    # multiply raw data by masks to mask each detector zone
    # Y_non_illum = Y * ymask_non_illum
    Y_illum = Y * ymask_illum
    Y_hill_illum = Y * ymask_sl_3d

    # first get list of non-illuminated frames with sufficient signal
    middle_col_ix = int(nbr_lambda_pixels / 2)
    middle_col_sums = np.nansum(Y_illum[:, :, middle_col_ix], axis=1)
    middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

    uv_col_ix = int(nbr_lambda_pixels / 10)

    toa_ix = np.where(middle_col_trans > 0.99)[0][-1]  # example index above atmosphere
    atmo_ix = np.argmin(np.abs(middle_col_trans - 0.5))


# %%
    # compare masks on one image
    masks = np.zeros_like(ymask_non_illum[0, :, :])
    masks += ymask_non_illum[0, :, :]
    masks += ymask_illum[0, :, :] * 2
    masks += ymask_sl_3d[0, :, :] * 4
    plt.figure()
    plt.imshow(masks, aspect="auto", interpolation="none", extent=(200, 650, vstart + 71, vstart))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")
    plt.title("Linear straylight mask")

# %%
    # optional plotting
    # plot frame
    plt.figure(figsize=(9, 5))
    plt.title("Raw UVIS Occultation Frame")
    plt.imshow(Y[toa_ix, :, :], extent=(200, 650, vstart + 71, vstart), aspect="auto")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")

# %%
    plt.figure(figsize=(9, 5))
    plt.title("UVIS Raw Occultation Spectrum")
    plt.plot(x, np.sum(Y[toa_ix, :, :], axis=0), label={128: "Spectral binning", 1024: "Unbinned"}[nbr_lambda_pixels], alpha=0.7)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Summed Counts")
    plt.legend()
    plt.grid()
    plt.axhline(y=0)

# %%
    # simple transmittance without hill SL removal
    trans = np.sum(Y[atmo_ix, :, :], axis=0) / np.sum(Y[toa_ix, :, :], axis=0)
    plt.title("UVIS Transmittance Spectrum")
    plt.plot(x, trans, label={128: "Spectral binning", 1024: "Unbinned"}[nbr_lambda_pixels], alpha=0.7)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Summed Counts")
    plt.legend()
    plt.grid()

# %%
    # plot illum pattern shift
    # plot frame shift difference
    plt.figure(figsize=(9, 5))
    plt.title("UVIS Occultation Frame Index %i - Index %i" % (0, toa_ix))
    plt.imshow(Y[0, :, :] - Y[toa_ix, :, :], aspect="auto")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")

# %%
    # plot ymask_illum
    plt.figure(figsize=(9, 5))
    plt.title("UVIS Well-Illuminated Pixels Mask Frame %i" % (0))
    plt.imshow(ymask_illum[-1, :, :], aspect="auto", extent=(200, 650, vstart + 71, vstart))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")

# %%
    plot_frame_col_norm(Y[toa_ix, :, :].T, extent=(200, 650, vstart + 71, vstart), figsize=(9, 5))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")
    plt.title("Column-Normalised UVIS Occultation Frame")

# %%
    # plot raw slices
    plt.figure(figsize=(9, 5))
    plt.xlabel("Pixel column (spectral direction)")
    plt.ylabel("Pixel row (illumination direction)")
    plt.title("Non-illuminated detector signal")

    plt.plot(Y[toa_ix, :, :])
    # plt.plot(Y[toa_ix, :, uv_col_ix])
    # plt.plot(Y[toa_ix, :, middle_col_ix - 10])
    # plt.plot(Y[toa_ix, :, middle_col_ix])
    # plt.plot(Y[toa_ix, :, middle_col_ix + 10])

# %%
    # # plot raw slices
    plt.figure(figsize=(9, 5))
    loop = 0
    for i in np.arange(20, int(nbr_lambda_pixels / 1.5), 10):
        plt.plot(Y[toa_ix, :, i] / np.max(Y[toa_ix, :, i]), label="%0.0f nm" % (x[i]), color="C%0i" % loop)
        plt.plot(Y_hill_illum[toa_ix, :, i] / np.max(Y[toa_ix, :, i]), label="%0.0f nm" % (x[i]), linestyle="--", color="C%0i" % loop)
        loop += 1
    plt.legend()
    plt.grid()
    plt.title("Normalised detector vertical slices")
    plt.xlabel("Detector row")
    plt.ylabel("Normalised pixel counts")
    plt.axhline(y=0, color="k", linestyle="--")

    plt.figure(figsize=(9, 5))
    for i in np.arange(20, int(nbr_lambda_pixels / 1.5), 10):
        plt.plot(Y[toa_ix, :, i], label="%0.0f nm" % (x[i]))
    plt.legend()
    plt.grid()
    plt.title("Detector vertical slices")
    plt.xlabel("Detector row")
    plt.ylabel("Pixel counts")
    plt.axhline(y=0, color="k", linestyle="--")

# %%
    # plot binned transmittance with hill straylight present
    y_toa_binned = np.sum(Y[toa_ix, :, :], axis=0)

    plt.figure(figsize=(9, 5))
    plt.title("UVIS transmittances with only linear straylight removed")
    for ix in [atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]:
        y_atmos_binned = np.sum(Y[ix, :, :], axis=0)
        y_trans_binned = y_atmos_binned / y_toa_binned

        plt.plot(x, y_trans_binned, label="Spectrum %i" % ix)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.grid()
    plt.legend()

# %%
    plot_frame_col_norm(Y_hill_illum[toa_ix, :, :].T, extent=(200, 650, vstart + 71, vstart), figsize=(9, 5))
    # plt.imshow(Y_hill_illum[toa_ix, :, :], extent=(200, 650, vstart + 71, vstart))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pixel row (illumination direction)")
    plt.title("Column-Normalised UVIS Occultation Frame")

# %%
    # plot binned well-illuminated transmittance with hill straylight added in various amounts
    for corr in [0, 50, 100, 150]:
        Y_illum_corr = np.maximum(Y_illum - corr, 0.0)
        y_toa_binned = np.sum(Y_illum_corr[toa_ix, :, :], axis=0)

        plt.figure(figsize=(9, 5))
        plt.title("UVIS transmittance: straylight removal testing")
        # plt.title(corr)
        for ix in [atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]:
            y_atmos_binned = np.sum(Y_illum_corr[ix, :, :], axis=0)
            y_trans_binned = y_atmos_binned / y_toa_binned

            plt.plot(x, y_trans_binned, label="Spectrum %i" % ix)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")
        plt.grid()
        plt.legend()
        plt.ylim([-0.1, 0.8])
        # see _animations for animated version

# %%
    # plot binned well-illuminated transmittance with hill straylight added in various amounts corrected for transmittance
    # no hill correction
    y_toa_binned = np.sum(Y[toa_ix, :, :], axis=0)
    for corr in [0, 50, 100, 150]:
        # correct for hill by constant amount
        y_toa_binned1 = np.sum(np.maximum(Y_illum[toa_ix, :, :] - corr, 0.0), axis=0)

        plt.figure(figsize=(9, 5))
        plt.title("UVIS transmittance: straylight removal testing")
        # plt.title(corr)
        for loop_ix, ix in enumerate([atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]):
            y_atmos_binned = np.sum(Y[ix, :, :], axis=0)
            y_trans_binned = y_atmos_binned / y_toa_binned

            # do first run with 50% correction
            y_atmos_binned1 = np.sum(np.maximum(Y_illum[ix, :, :] - corr, 0.0), axis=0)
            y_trans_binned1 = y_atmos_binned1 / y_toa_binned1

            # correct for hill by variable amount based on mean transmittance
            corr2 = corr * np.mean(y_trans_binned1)
            y_toa_binned2 = np.sum(np.maximum(Y_illum[toa_ix, :, :] - corr2, 0.0), axis=0)
            y_atmos_binned2 = np.sum(np.maximum(Y_illum[ix, :, :] - corr2, 0.0), axis=0)
            y_trans_binned2 = y_atmos_binned2 / y_toa_binned2
            print(corr, ix, np.mean(y_trans_binned1), np.mean(corr2))

            plt.plot(x, y_trans_binned, label="Spectrum %i uncorrected" % ix, color="C%0i" % loop_ix, linestyle=":")
            plt.plot(x, y_trans_binned1, label="Spectrum %i +%i counts" % (ix, corr), color="C%0i" % loop_ix, linestyle="--")
            plt.plot(x, y_trans_binned2, label="Spectrum %i variable correction" % ix, color="C%0i" % loop_ix)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")
        plt.grid()
        plt.legend(loc="lower right")
        plt.ylim([-0.1, 1.0])
