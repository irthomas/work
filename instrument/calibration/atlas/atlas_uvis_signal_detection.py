# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 17:06:51 2025

@author: iant

ATLAS UVIS DATA CHECK:

    GET UVIS DEIMOS OBSERVATION
    MAKE MASK OF ILLUMINATED PIXELS
    APPLY MASK TO ATLAS TO BIN ONLY ILLUMINATED PIXELS
    APPLY RUNNING MEAN TO CHECK IF SIGNAL VISIBLE IN LAST 1/3 OF OBSERVATION
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file

# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3a"


deimos_h5s = [
    "20250830_183857_0p3b_UVIS_Q"
    # "20251003_044802_0p3b_UVIS_X"
]

h5 = "20251003_044802_0p3b_UVIS_X"

for deimos_h5 in deimos_h5s[0:1]:

    deimos_h5f = open_hdf5_file(deimos_h5, path=data_path)

    v_start_deimos = int(deimos_h5f["Channel/VStart"][0])
    v_end_deimos = int(deimos_h5f["Channel/VEnd"][0])
    print(v_start_deimos, v_end_deimos)

    print(deimos_h5f["Science/Y"][...].shape)
    y_deimos_all = deimos_h5f["Science/Y"][:, :, :]

# combine all frames
y_frame_mean_deimos_all = np.mean(y_deimos_all, axis=0)

# make full frame nan array and populate with mean frame
y_full_frame_deimos = np.zeros((256, 1024)) + np.nan
y_full_frame_deimos[v_start_deimos:v_end_deimos+1, :] = y_frame_mean_deimos_all

# remove bad points
y_full_frame_deimos[y_full_frame_deimos < -1000.0] = np.nan
y_full_frame_deimos[y_full_frame_deimos > 2000.0] = np.nan

# remean
y_full_frame_deimos -= np.nanmean(y_full_frame_deimos)

plt.figure(figsize=(8, 6))
plt.title("UVIS Deimos Mean Frames")
plt.xlabel("Detector column (spectral direction)")
plt.ylabel("Detector row")
plt.imshow(y_full_frame_deimos, aspect="auto", vmin=0, vmax=600)

# cut down the frame to illuminated zone
y_frame_mean_deimos = y_full_frame_deimos[115:145, 200:900]
# remean
y_frame_mean_deimos -= np.nanmean(y_frame_mean_deimos)


mask1_ixs = np.where(y_frame_mean_deimos > 200.0)
mask1 = np.zeros_like(y_frame_mean_deimos)
mask1[mask1_ixs] = 1

# plt.figure()
# plt.imshow(mask1, aspect="auto")

# find how many high signal neighbours each pixel has
neighbours = scipy.signal.convolve2d(mask1, np.ones((3, 3)), mode='same')

# plt.figure()
# plt.imshow(neighbours, aspect="auto")

mask_ixs = np.where(neighbours > 6)
mask = np.zeros_like(neighbours)
mask[mask_ixs] = 1

dark_mask_ixs = np.where(neighbours < 2)
dark_mask = np.zeros_like(neighbours)
dark_mask[dark_mask_ixs] = 1

# plt.figure()
# plt.title("UVIS Deimos Illumination Mask")
# plt.xlabel("Detector column (spectral direction)")
# plt.ylabel("Detector row")
# plt.imshow(mask, aspect="auto")

# plt.figure()
# plt.title("UVIS Deimos Dark Mask")
# plt.xlabel("Detector column (spectral direction)")
# plt.ylabel("Detector row")
# plt.imshow(dark_mask, aspect="auto")

# apply to Atlas

h5f = open_hdf5_file(h5, path=data_path)

v_start = h5f["Channel/VStart"][0]
v_end = h5f["Channel/VEnd"][0]
print(v_start, v_end)

print(h5f["Science/Y"][...].shape)
y_all = h5f["Science/Y"][50:, :, :]


# combine all frames
y_frame_mean_all = np.mean(y_all, axis=0)

# make full frame nan array and populate with mean frame
y_full_frame = np.zeros((256, 1024)) + np.nan
y_full_frame[v_start:v_end+1, :] = y_frame_mean_all

# remove bad points
# y_full_frame[y_full_frame < -1000.0] = np.nan
# y_full_frame[y_full_frame > 1000.0] = np.nan

# remean
y_full_frame -= np.nanmean(y_full_frame)

# check row means
# y_frame_row_mean = np.nanmean(y_frame_mean, axis=1)
# plt.figure()
# plt.plot(y_frame_row_mean)

plt.figure(figsize=(8, 6))
plt.title("UVIS Atlas Mean Frames")
plt.xlabel("Detector column (spectral direction)")
plt.ylabel("Detector row")
plt.imshow(y_full_frame, aspect="auto", vmin=0, vmax=100)

# cut down the frame to illuminated zone
y_frame_mean = y_full_frame[115:145, 200:900]
# remean
y_frame_mean -= np.nanmean(y_frame_mean)


y_masked = y_frame_mean * mask
y_dark_masked = y_frame_mean * dark_mask

# plt.figure()
# plt.xlabel("Detector column (spectral direction)")
# plt.ylabel("Detector row")
# plt.imshow(y_masked, aspect="auto")
# plt.figure()
# plt.xlabel("Detector column (spectral direction)")
# plt.ylabel("Detector row")
# plt.imshow(y_dark_masked, aspect="auto")

mean_y = np.nanmean(y_masked)
mean_y_dark = np.nanmean(y_dark_masked)
std_y = np.nanstd(y_masked)
std_y_dark = np.nanstd(y_dark_masked)

print("Y mean=", mean_y, std_y, "Y dark=", mean_y_dark, std_y_dark)
print("Y mean=", mean_y, std_y/np.sqrt(np.sum(mask)), "Y dark=", mean_y_dark, std_y_dark/np.sqrt(np.sum(dark_mask)))

counts, bins = np.histogram(np.sort(y_masked[y_masked != 0.0].flatten()), bins=40, range=(-100, 100))
counts_dark, bins_dark = np.histogram(np.sort(y_dark_masked[y_dark_masked != 0.0].flatten()), bins=40, range=(-100, 100))

# plot normalised histograms
plt.figure()
plt.stairs(counts/np.max(counts), bins, label="Atlas", linewidth=2)
plt.stairs(counts_dark/np.max(counts_dark), bins_dark, label="Dark pixels", linewidth=2)
plt.legend()
plt.title("Atlas histogram of counts")
plt.xlabel("Pixel counts")
