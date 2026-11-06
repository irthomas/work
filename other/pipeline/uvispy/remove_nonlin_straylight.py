# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:29:57 2026

@author: iant

"""

# import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tools.file.hdf5_functions import get_filepath


def plot_frame_col_norm(frame, vmin=None, vmax=None):
    """function to plot a column-normalised 2d detector frame"""

    y_col_max = np.max(frame, axis=1)
    frame_col_max = frame / y_col_max[:, None]
    plt.figure()
    plt.imshow(frame_col_max.T, vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')


cut_off = 0.01

DIR_PATH = r"X:\projects\work\NOMAD\test\iant\hdf5"
# DIR_PATH = r"C:\Users\iant\Documents\DATA\hdf5"
# DIR_PATH = r"C:\Users\iant\Dropbox\NOMAD\Python"

# h5 = "20240516_130419_0p3d_UVIS_I"
# h5 = "20230126_081530_0p3d_UVIS_I"
# h5 = "20240501_033224_0p3d_UVIS_I"
# h5 = "20240502_030733_0p3d_UVIS_I"

# h5 = "20180501_013906_0p3d_UVIS_I"  # only 61 lines -> not working
# h5 = "20180501_021814_0p3d_UVIS_E"
# h5 = "20180501_132643_0p3d_UVIS_I"
# h5 = "20180501_140514_0p3d_UVIS_E"  # only 61 lines -> not working
# h5 = "20180501_231624_0p3d_UVIS_I"  # only 61 lines -> not working
# h5 = "20180501_235421_0p3d_UVIS_E"  # only 65 lines -> not working

# h5 = "20240501_021829_0p3d_UVIS_E"  # 128 cols
# h5 = "20240501_033224_0p3d_UVIS_I"
# h5 = "20240502_193435_0p3d_UVIS_E"

h5_filepath = get_filepath(h5, path=DIR_PATH)
# h5_filepath = h5 + ".h5"

"""code to plot transmittance of existing file"""
with h5py.File(h5_filepath, "r") as h5f:
    x = h5f["Science/X"][0, :]
    y = h5f["Science/Y"][...]
    illum_mask = h5f["Science/YIllumMask"][...]
    non_illum_mask = h5f["Science/YNonIllumMask"][...]

    y_illum = y * illum_mask
    y_non_illum = y * non_illum_mask

    xpix_1b = h5f["Science/Xpix_1b"][...]
    vstart = h5f["Channel/VStart"][0]

    start_rows_roi = h5f["Science/IllumStartRows"][...]
    end_rows_roi = h5f["Science/IllumEndRows"][...]
    col_nms = h5f["Science/IllumColNms"][...]


# print 400nm illum row position
ix_400 = [ix for ix, i in enumerate(col_nms) if i == 400][0]

start_rows_400 = start_rows_roi[:, ix_400]
end_rows_400 = end_rows_roi[:, ix_400]

print(h5, "vstart=", vstart, "400nm start=", np.mean(start_rows_400) + vstart, "400nm end=", np.mean(end_rows_400) + vstart)
# lin_sl_coeffs = h5["Science/LinSLCoeffs"][...]


ingress = "UVIS_I" in h5
n_frames = y_illum.shape[0]
frame_ix = 10


fig, ax1 = plt.subplots()

im = ax1.imshow(y_non_illum[0, :, :], animated=True, vmin=-100, vmax=100, aspect="auto", extent=[0, 1024, vstart + 71, vstart])
cbar = fig.colorbar(im, ax=ax1, shrink=0.8)

plt.tight_layout()


def update_frame(frame_ix, frames):

    # y_col_max = np.max(frames[frame_ix, :, :], axis=0)
    # frame_col_max = frames[frame_ix, :, :] / y_col_max[None, :]

    # im.set_array(frame_col_max)
    im.set_array(y_non_illum[frame_ix, :, :])
    # print("%i/%i" % (frame_ix, y_non_illum.shape[0]))
    return im,


anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=n_frames,
    fargs=(y_non_illum,),
    interval=10,
    blit=True
)

plt.show()

# writer = animation.FFMpegWriter(fps=30, codec='h264', extra_args=['-pix_fmt', 'yuv420p'])
# anim.save("%s.mp4" % h5, writer=writer, dpi=300)


# fig = plt.figure()
# plt.imshow(np.mean(y_non_illum[10:100, :, :], axis=0))
# # plot_frame_col_norm(y_non_illum[frame_ix, :, :].T)

# # plt.plot(y_non_illum[10, :, 91])


# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print("[%0.0f, %0.0f], " % (xpix_1b[int(np.round(ix))], iy + vstart), end="")


# cid = fig.canvas.mpl_connect('button_press_event', onclick)


# plt.show()


def define_non_illuminated_lines(xpix_1b):
    """outputs same as matlab with indices subtracted by 1. Only for pFM occultation at present"""
    x_top_start = np.asarray([[12, 150], [100, 147], [212, 144], [252, 140], [356, 139], [484, 139], [604, 139], [652, 136], [1028, 136]])
    x_top_end = np.asarray([[12, 165], [60, 161], [100, 156], [148, 151], [204, 149], [276, 147], [348, 145], [468, 145],
                           [572, 146], [644, 145], [716, 143], [812, 143], [884, 142], [972, 142], [1028, 142]])
    x_bottom_start = np.asarray([[12, 185], [108, 185], [148, 190], [188, 194], [244, 197], [316, 199], [388, 200], [460, 199],
                                [532, 199], [604, 199], [692, 199], [804, 199], [900, 200], [964, 200], [1028, 200]])
    x_bottom_end = np.asarray([[12, 192], [68, 192], [100, 193], [148, 196], [196, 199], [228, 201],
                              [236, 207], [460, 207], [684, 207], [700, 207], [1028, 207]])

    i_line_top_start = np.interp(xpix_1b, x_top_start[:, 0], x_top_start[:, 1], left=None, right=None)
    i_line_top_end = np.interp(xpix_1b, x_top_end[:, 0], x_top_end[:, 1], left=None, right=None)
    i_line_bottom_start = np.interp(xpix_1b, x_bottom_start[:, 0], x_bottom_start[:, 1], left=None, right=None)
    i_line_bottom_end = np.interp(xpix_1b, x_bottom_end[:, 0], x_bottom_end[:, 1], left=None, right=None)

    i_line_top_start = np.ceil(i_line_top_start).astype(int)
    i_line_top_end = np.floor(i_line_top_end).astype(int)
    i_line_bottom_start = np.ceil(i_line_bottom_start).astype(int)
    i_line_bottom_end = np.floor(i_line_bottom_end).astype(int)

    return i_line_top_start, i_line_top_end, i_line_bottom_start, i_line_bottom_end


def define_non_illiminated_matrix(vstart, xpix_1b, n_rows):
    """Define the top and bottom non illuminated lines. Smearing and checks on data size have been removed"""
    # get well illuminated zone start end rows on full frame grid
    # offset compared to matlab
    i_line_top_start, i_line_top_end, i_line_bottom_start, i_line_bottom_end = define_non_illuminated_lines(xpix_1b)

    # convert to illuminated rows
    # offset compared to matlab
    i_line_top_start -= vstart + 1
    i_line_top_end -= vstart + 1
    i_line_bottom_start -= vstart + 1
    i_line_bottom_end -= vstart + 1

    # i_start_line_roi = np.asarray(i_line_top_start - vstart + 1, dtype=int)
    # i_end_line_roi = np.asarray(i_line_bottom_start - vstart + 1, dtype=int)

    # same values as matlab
    # n_lines_roi = (i_end_line_roi - i_start_line_roi) + 1

    # Create the matrix for the binned ROI
    non_illiminated_matrix = np.zeros((xpix_1b.shape[0], n_rows), dtype=bool)
    # mat_bin_roi = np.zeros((xpix_1b.shape[0], int(np.max(n_lines_roi))), dtype=float)

    for ipix in range(xpix_1b.shape[0]):
        non_illiminated_matrix[ipix, int(i_line_top_start[ipix]):int(i_line_top_end[ipix])] = 1
        non_illiminated_matrix[ipix, int(i_line_bottom_start[ipix]):int(i_line_bottom_end[ipix])] = 1

    return non_illiminated_matrix


# n_rows = y.shape[1]
# non_illiminated_matrix = define_non_illiminated_matrix(vstart, xpix_1b, n_rows)

# fig = plt.figure()
# plt.imshow(non_illiminated_matrix.T)
# plt.show()
# plt.imshow(mat_bin_roi.T * np.mean(y_non_illum[10:100, :, :], axis=0))
