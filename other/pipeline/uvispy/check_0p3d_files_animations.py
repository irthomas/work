# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:29:57 2026

@author: iant

"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

filepaths = glob.glob(r"X:\projects\work\NOMAD\test\iant\hdf5\hdf5_level_0p3d\2024\05\**\*UVIS_I.h5", recursive=True)


h5_filepath = filepaths[0]

h5_basename = os.path.basename(h5_filepath).split(".")[0]
print(h5_basename)

"""code to plot frames of existing file"""
with h5py.File(h5_filepath, "r") as h5f:
    x = h5f["Science/X"][0, :]
    Y = h5f["Science/Y"][...]
    alts = h5f["Geometry/Point0/TangentAltAreoid"][...]
    YWellIllumMask = h5f["Science/YIllumMask"][...]
    YNonIllumMask = h5f["Science/YNonIllumMask"][...]

    # xpix_1b = h5f["Science/Xpix_1b"][...]
    vstart = h5f["Channel/VStart"][0]

    # start_rows_roi = h5f["Science/IllumStartRows"][...]
    # end_rows_roi = h5f["Science/IllumEndRows"][...]
    # col_nms = h5f["Science/IllumColNms"][...]

nbr_obs = Y.shape[0]
nbr_lambda_pixels = Y.shape[2]  # n real columns e.g. 128 or 1024
nbr_rows = Y.shape[1]  # n real columns e.g. 128

# divide Y by binning factor to match binned and unbinned
Y /= (1024 / nbr_lambda_pixels)

ymask_illum = YWellIllumMask
ymask_non_illum = YNonIllumMask

Y_illum = Y * ymask_illum
Y_non_illum = Y * ymask_non_illum

# first get list of non-illuminated frames with sufficient signal
middle_col_ix = int(nbr_lambda_pixels / 2)
middle_col_sums = np.nansum(Y_illum[:, :, middle_col_ix], axis=1)
middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)

uv_col_ix = int(nbr_lambda_pixels / 10)

toa_ix = np.where(middle_col_trans > 0.99)[0][-1]  # example index above atmosphere
atmo_ix = np.argmin(np.abs(middle_col_trans - 0.5))

# # print 400nm illum row position
# ix_400 = [ix for ix, i in enumerate(col_nms) if i == 400][0]

# start_rows_400 = start_rows_roi[:, ix_400]
# end_rows_400 = end_rows_roi[:, ix_400]

# print(h5, "vstart=", vstart, "400nm start=", np.mean(start_rows_400) + vstart, "400nm end=", np.mean(end_rows_400) + vstart)
# # lin_sl_coeffs = h5["Science/LinSLCoeffs"][...]


# ingress = "UVIS_I" in h5
n_frames = Y_illum.shape[0]
# frame_ix = 10

# %%
fig, ax1 = plt.subplots()

im = ax1.imshow(Y_non_illum[0, :, :], animated=True, vmin=-100, vmax=100, aspect="auto", extent=(200, 650, vstart + 71, vstart))
cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Pixel row (illumination direction)")


def update_frame(frame_ix, frames):

    # y_col_max = np.max(frames[frame_ix, :, :], axis=0)
    # frame_col_max = frames[frame_ix, :, :] / y_col_max[None, :]

    # im.set_array(frame_col_max)
    im.set_array(Y_non_illum[frame_ix, :, :])
    # print("%i/%i" % (frame_ix, y_non_illum.shape[0]))
    return im,


anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=n_frames,
    fargs=(Y_non_illum,),
    interval=10,
    blit=True
)

plt.show()

# writer = animation.FFMpegWriter(fps=30, codec='h264', extra_args=['-pix_fmt', 'yuv420p'])
# anim.save("%s.mp4" % h5, writer=writer, dpi=300)

# %%

# plot binned well-illuminated transmittance with varying amounts of hill straylight removed
fig, ax1 = plt.subplots()
corrs = np.arange(0, 160.0, 2)
Y_illum_corr = np.maximum(Y_illum - corrs[0], 0.0)
y_toa_binned = np.sum(Y_illum_corr[toa_ix, :, :], axis=0)
lines1 = []
for ix in [atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]:
    y_atmos_binned = np.sum(Y_illum_corr[ix, :, :], axis=0)
    y_trans_binned = y_atmos_binned / y_toa_binned

    line1, = plt.plot(x, y_trans_binned, label="Spectrum %i" % ix)
    lines1.append(line1)

text1 = ax1.text(0.05, 0.95, "Straylight +%0i counts per pixel" % 0, transform=ax1.transAxes)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.grid()
plt.legend()


def update_frame(frame_ix, corrs):

    corr = corrs[frame_ix]
    Y_illum_corr = np.maximum(Y_illum - corr, 0.0)
    y_toa_binned = np.sum(Y_illum_corr[toa_ix, :, :], axis=0)

    for line_ix, ix in enumerate([atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]):
        y_atmos_binned = np.sum(Y_illum_corr[ix, :, :], axis=0)
        y_trans_binned = y_atmos_binned / y_toa_binned

        lines1[line_ix].set_ydata(y_trans_binned)

        text1.set_text("Straylight +%0i counts per pixel" % corr)

    return tuple(lines1) + (text1,)


anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=len(corrs),
    fargs=[corrs],
    interval=40,
    blit=True
)

plt.show()

# writer = animation.FFMpegWriter(fps=30, codec='h264', extra_args=['-pix_fmt', 'yuv420p'])
# anim.save("straylight_addition.mp4", writer=writer, dpi=300)

# %%

# plot binned well-illuminated transmittance with varying amounts of hill straylight removed
fig, ax1 = plt.subplots()
corrs = np.arange(0, 160.0, 2)
y_toa_binned1 = np.sum(np.maximum(Y_illum[toa_ix, :, :] - corrs[0], 0.0), axis=0)
lines1 = []
for ix in [atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]:
    y_atmos_binned = np.sum(Y_illum_corr[ix, :, :], axis=0)
    y_trans_binned = y_atmos_binned / y_toa_binned

    line1, = plt.plot(x, y_trans_binned1, label="Spectrum %i" % ix)
    line2, = plt.plot(x, y_trans_binned2, label="Spectrum %i" % ix)
    lines1.append(line1)

text1 = ax1.text(0.05, 0.95, "Straylight +%0i counts per pixel" % 0, transform=ax1.transAxes)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.grid()
plt.legend()


def update_frame(frame_ix, corrs):

    corr = corrs[frame_ix]
    Y_illum_corr = np.maximum(Y_illum - corr, 0.0)
    y_toa_binned = np.sum(Y_illum_corr[toa_ix, :, :], axis=0)

    for line_ix, ix in enumerate([atmo_ix - 2, atmo_ix - 1, atmo_ix, atmo_ix + 1, atmo_ix + 2]):
        y_atmos_binned = np.sum(Y_illum_corr[ix, :, :], axis=0)
        y_trans_binned = y_atmos_binned / y_toa_binned

        lines1[line_ix].set_ydata(y_trans_binned)

        text1.set_text("Straylight +%0i counts per pixel" % corr)

    return tuple(lines1) + (text1,)


anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=len(corrs),
    fargs=[corrs],
    interval=40,
    blit=True
)

plt.show()

# writer = animation.FFMpegWriter(fps=30, codec='h264', extra_args=['-pix_fmt', 'yuv420p'])
# anim.save("straylight_addition.mp4", writer=writer, dpi=300)
