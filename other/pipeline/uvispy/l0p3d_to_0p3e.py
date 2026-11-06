# -*- coding: utf-8 -*-
"""
Convert 0p3b to 0p3e remove occultation straylight

Also combines the post_matlab_cleanup script, removing non-science frames

"""


import logging
import h5py
import numpy as np
import os
from scipy.interpolate import interp1d
import shutil

import importlib
if importlib.util.find_spec("nomad_ops"):
    LINUX = True
else:
    LINUX = False

if LINUX:
    import nomad_ops.core.hdf5.generic_functions as generics

    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
    from nomad_ops.core.hdf5.l0p2b_to_0p3b.l0p2b_to_0p3b import Define_ROI, Define_Top_Bottom_NonIlluminated_Lines, \
        define_mask_values
    from nomad_ops.core.tools.progress_bar import progress
else:
    from tools.general.progress_bar import progress
    import matplotlib.pyplot as plt

    import sys
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


__project__ = "NOMAD"
__author__ = "Ian Thomas"
__contact__ = "ian.thomas@aeronomie.be"

logger = logging.getLogger(__name__)

VERSION = 90
OUTPUT_VERSION = "0.3E"

PLOT_TYPES = []

ATTRIBUTES_TO_BE_REMOVED = []
DATASETS_TO_BE_REMOVED = []
DATASETS_NOT_TO_BE_RESHAPED = [
    'BadPixelMap',
    'PointXY',
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
    'Percent_HotPixels',
    "Science/IllumStartRows",
    "Science/IllumEndRows",
    "Science/IllumColIxs",
    "Science/IllumColNms",
    "LinSLCoeffs",
]
"""remove"""


def plot_frame_col_norm(frame, vmin=None, vmax=None):
    """function to plot a column-normalised 2d detector frame"""

    y_col_max = np.max(frame, axis=1)
    frame_col_max = frame / y_col_max[:, None]
    plt.figure()
    plt.imshow(frame_col_max.T, vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')


"""from generics"""


def createIntermediateGroups(root, groups):
    """Create each group in a nested HDF5 path and return the deepest group."""
    for g in groups:
        root = root.require_group(g)
    return root


def iter_datasets(group, root=()):
    """Yield every dataset in an HDF5 group as (slash-separated path, dataset)."""
    for name, obj in group.items():
        path = root + (name,)
        if isinstance(obj, h5py.Dataset):
            yield "/".join(path), obj
        elif isinstance(obj, h5py.Group):
            yield from iter_datasets(obj, path)


"""from 0p2b to 0p3b: modified"""


def define_well_illuminated_lines(xpix_1b):
    """Interpolate pFM detector-row bounds of the illuminated region by column."""

    x_top = np.asarray([[1, 172], [100, 172], [200, 170], [300, 169], [400, 167], [500, 165], [
                        600, 162], [700, 160], [800, 158], [900, 156], [1032, 152]])
    i_line_top_start = np.interp(xpix_1b, x_top[:, 0], x_top[:, 1], left=None, right=None)

    x_bottom = np.asarray([[1, 178], [100, 178], [200, 178], [300, 178], [400, 179], [500, 179], [
                          600, 180], [700, 181], [800, 182], [900, 183], [1032, 185]])
    i_line_bottom_start = np.interp(xpix_1b, x_bottom[:, 0], x_bottom[:, 1], left=None, right=None)

    i_line_top_start = np.ceil(i_line_top_start).astype(int)
    i_line_bottom_start = np.floor(i_line_bottom_start).astype(int)

    return i_line_top_start, i_line_bottom_start


"""new"""


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


if not LINUX:
    DIR_PATH = r"X:\projects\work\NOMAD\test\iant\hdf5"
    h5 = "20240501_021829_0p3d_UVIS_E"  # 128 cols
    from tools.file.hdf5_functions import get_filepath
    src = get_filepath(h5, path=DIR_PATH)

# src = "20240818_165255_0p3D_UVIS_I.h5"  # binned high ozone steep T loss
# src = "20240516_130419_0p3b_UVIS_I.h5"  # unbinned, solar line peak seen

# src = "20240818_165255_0p3D_UVIS_I.h5"  # binned high ozone steep T loss

# filepaths = glob.glob(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p3b\2024\10\20\*UVIS_I.h5")

# src = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p3b\2024\05\16\20240516_130419_0p3b_UVIS_I.h5"

# for src in filepaths:
dst = os.path.basename(src).replace("0p3d", "0p3e")
#     if os.path.exists(dst):
#         os.remove(dst)


# def remove_nonlinear_straylight(src, dst):
if True:
    hdf5_basename = os.path.basename(src).split(".")[0]

    d_out = {}

    with h5py.File(src, 'r') as f:
        # Keep only type-4 science frames and transpose detector data to
        # (spectral column, detector row, frame) for the calculations below.
        acq_mode = f['Channel/AcquisitionMode'][0]  # 0 = Full CCD, 1 = Vertical Binning, 2 =Horizontal / Combined binning

        vstart = int(f['Channel/VStart'][0]) + 1
        vend = int(f['Channel/VEnd'][0]) + 1
        # nbr_ccd_lines = vend - vstart + 1
        # integration_time = float(f['/Channel/IntegrationTime'][0])
        # Instrument = f.attrs['InstName']
        Y = f["Science/Y"][...]
        # YError = np.swapaxes(f["Science/YError"][...], 0, 2)
        # YMask = np.swapaxes(f["Science/YMask"][...], 0, 1)  # not correct
        # YErrorSysNL = np.swapaxes(f["Science/YErrorSysNL"][...], 0, 2)
        YWellIllumMask = f["Science/YIllumMask"][...]
        YNonIllumMask = f["Science/YNonIllumMask"][...]

        # x = f['Science/X'][...]
        xpix_1b = f['Science/Xpix_1b'][...]  # array of pixel centre indices for real columns in matlab indices e.g. [12.5, 20.5, 28.5,...]
        # xnb_bin = f['Science/XNbBin'][...]  # array of binning factors for real columns e.g. [8,8,8,...]
        # x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

        # YNb = f['Science/YNb']
        # stop()

    nbr_obs = Y.shape[0]
    nbr_lambda_pixels = Y.shape[2]  # n real columns e.g. 128
    nbr_rows = Y.shape[1]  # n real columns e.g. 128

    logger.info("%s: acq_mode=%i, nbr_lambda_pixels=%i, vstart=%i, vend=%i, nbr_rows=%i, Yshape=%s",
                hdf5_basename, acq_mode, nbr_lambda_pixels, vstart, vend, nbr_rows, Y.shape)

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

    # compare masks on one image
    masks = np.zeros_like(ymask_non_illum[0, :, :])
    masks += ymask_non_illum[0, :, :]
    masks += ymask_illum[0, :, :] * 2
    masks += ymask_sl_3d[0, :, :] * 4
    # plt.figure()
    # plt.imshow(masks, aspect="auto", interpolation="none")
    # plt.xlabel("Pixel column (spectral direction)")
    # plt.ylabel("Pixel row (illumination direction)")
    # plt.title("Illumination masks")

    # multiply raw data by masks to mask each detector zone
    Y_non_illum = Y * ymask_non_illum
    Y_illum = Y * ymask_illum
    Y_hill_illum = Y * ymask_sl_3d

    # first get list of non-illuminated frames with sufficient signal
    middle_col_ix = int(nbr_lambda_pixels / 2)
    middle_col_sums = np.nansum(Y_illum[:, :, middle_col_ix], axis=1)
    middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

    toa_ix = np.where(middle_col_trans > 0.99)[0][0]  # example index above atmosphere

    # optional plotting
    # plot_frame_col_norm(Y[toa_ix, :, :].T)

    # plot_frame_col_norm(Y_non_illum[toa_ix, :, :].T)
    # plt.xlabel("Pixel column (spectral direction)")
    # plt.ylabel("Pixel row (illumination direction)")
    # plt.title("Non-illuminated detector signal")

    # plot slices
    plt.plot(Y[toa_ix, :, 10])
    plt.plot(Y[toa_ix, :, middle_col_ix - 10])
    plt.plot(Y[toa_ix, :, middle_col_ix])
    plt.plot(Y[toa_ix, :, middle_col_ix + 10])
    plt.axhline(y=0)

    stop()

    """Write corrected arrays and retained input datasets to the output HDF5."""
    indrec = np.arange(nbr_obs)  # all obs
    with h5py.File(src, 'r') as hdf5FileIn:
        with h5py.File(dst, 'w') as hdf5FileOut:

            if LINUX:
                generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)

            # Copy metadata/datasets that are not recalculated, dropping the
            # listed obsolete products and selecting only retained science frames.
            # FIXME: put this back
            # for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            for dset_path, dset in iter_datasets(hdf5FileIn):
                # print(dset_path)
                if dset_path in DATASETS_TO_BE_REMOVED:  # don't copy
                    # print("Not copying %s" %dset_path)
                    continue

                # FIXME: put this back
                # dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                dest = createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])

                if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
                    # Preserve housekeeping and other frame-independent data
                    # in their original layout.
                    hdf5FileIn.copy(dset_path, dest)
                    # print("Copying but not reshaping %s" % dset_path)
                    continue

                dsetcopy = np.array(dset)
                # print("%s dataset is being reshaped (shape=%s)" %(dset_path, dsetcopy.shape))
                if dsetcopy.ndim == 0:
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy)
                # Other copied datasets use the file's frame-first convention;
                # select type-4 frames along the leading observation axis.
                elif dsetcopy.ndim == 1:  # if 1D, use compression
                    # print(dset_path, "length = %i" %dsetcopy.shape[0])
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec], compression="gzip", shuffle=True)
                elif len(dsetcopy.shape) == 2:  # if 2D, use compression
                    # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :], compression="gzip", shuffle=True)
                elif len(dsetcopy.shape) == 3:  # if 3D, use compression
                    # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :, :], compression="gzip", shuffle=True)

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
                    # Calculated arrays are kept internally column-first; write
                    # frame-first output by transposing before selecting frames.
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value.T, dtype=dtype, chunks=(
                            value.shape[1], min(24, value.shape[0])), compression="gzip", compression_opts=4)
                    else:
                        dset = value.T[indrec, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(
                            len(indrec), min(24, value.shape[0])), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                elif value.ndim == 3:
                    # Restore the HDF5 convention (frame, row, column) and chunk
                    # along the frame dimension for efficient access.
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value, dtype=dtype, chunks=(
                            1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                    else:
                        dset = value[indrec, :, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                else:
                    hdf5FileOut.create_dataset(key, data=value, dtype=dtype)


def convert(hdf5file_path):
    """this function should perform the second part of the straylight / sawtooth removal, subtracting the 'hill' straylight"""
    logger.info("Convert: %s", hdf5file_path)
    tmp_file = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
    # shutil.copyfile(hdf5file_path, tmp_file)
    remove_nonlinear_straylight(hdf5file_path, tmp_file)
    return [tmp_file]
