# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:54:33 2026

@author: iant

PLOT 0P3D FRAMES FOR CHECKING LINEAR STRAYLIGHT AND SAWTOOTH CORRECTIONS

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from tools.file.hdf5_functions import get_filepath

cut_off = 0.01

DIR_PATH = r"X:\projects\work\NOMAD\test\iant\hdf5"


# h5 = "20230126_081530_0p3d_UVIS_I"
h5 = "20240501_033224_0p3d_UVIS_I"

h5_filepath = get_filepath(h5, path=DIR_PATH)


"""code to plot transmittance of existing file"""
with h5py.File(h5_filepath, "r") as h5f:
    x = h5f["Science/X"][0, :]
    y = h5f["Science/Y"][...]
    illum_mask = h5f["Science/YIllumMask"][...]
    non_illum_mask = h5f["Science/YNonIllumMask"][...]

    y_illum = y * illum_mask
    y_non_illum = y * non_illum_mask

    # lin_sl_coeffs = h5["Science/LinSLCoeffs"][...]


ingress = "UVIS_I" in h5

# find indices at TOA and atmosphere
y_mean = np.mean(y_illum, axis=(1, 2))

if ingress:
    y_diff = np.abs(np.diff(y_mean)) / np.mean(y_mean[0:10])
    toa_ix = np.min(np.where(y_diff > cut_off)) - 5
    atmos_ix = np.min(np.where(y_mean < 0.1))
    # plt.plot(y_diff)
    # plt.axhline(cut_off)
    # plt.axvline(toa_ix)

    toa_ixs = np.arange(toa_ix - 10, toa_ix)
    # atmos_ixs = np.arange(toa_ix, atmos_ix, (atmos_ix - toa_ix) // 20)
    atmos_ixs = np.arange(toa_ix, atmos_ix)

y_2d = np.sum(y_illum, axis=1)
y_toa = np.mean(y_2d[toa_ixs, :], axis=0)

trans = y_2d[atmos_ixs] / y_toa

plt.plot(trans.T)

# plot single frames
frame_ixs = [85, 86, 87, 88]
col_ixs = [20, 50, 100]
row_ixs = [5, 10, -10, -5]

# plt.figure()
# plt.grid()

# row_ix = -1
# col_ix = -1
# frame_ix = -1
# for frame_ix in frame_ixs:
#     # for col_ix in col_ixs:
#     #     plot = y[frame_ix, :, col_ix]
#     for row_ix in row_ixs:
#         plot = y[frame_ix, row_ix, :]

#         plt.plot(plot / np.max(plot), label="%i;%i;%i" % (frame_ix, col_ix, row_ix))
#         plt.legend()

# for frame_ix in frame_ixs:
#     plt.figure()
#     plt.imshow(y[frame_ix, :, :] - y[frame_ixs[0], :, :], vmin=-100, vmax=1000)

# plt.figure()
# # plt.plot(x, y[40:70, :].T, "k", alpha=0.5)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Transmittance")
# plt.grid()
