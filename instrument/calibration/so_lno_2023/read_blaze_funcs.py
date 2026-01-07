# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:07:14 2024

@author: iant

READ IN BLAZE FUNC H5 MADE DURING STEP 2
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from tools.plotting.colours import get_colours

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")


blaze_h5f = h5py.File(os.path.join(MINISCAN_PATH, "blaze_functions.h5"), "r")


ts = []
blaze_widths = []
blaze_width_rs = []
blaze_width_rs2 = []
blaze_peaks = []
aotfs = []


h5_prefixes = list(blaze_h5f.keys())
orders = [int(s.split("-")[3]) for s in h5_prefixes]

colours = get_colours(max(orders) - 100)

fig1, (ax1a, ax1b) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [20, 1]})

for i, key in enumerate(h5_prefixes):
    print("%i/%i: %s" % (i, len(h5_prefixes), key))
    blaze = blaze_h5f[key]["blaze"][...]
    t = blaze_h5f[key]["ts"][...]
    aotf = blaze_h5f[key]["aotf_col0"][...]

    hr_scaler = blaze.shape[1] / 320.0

    blaze_peak = np.argmax(blaze, axis=1) / hr_scaler

    # skip bad data
    if np.max(blaze_peak) > 230:
        continue

    blaze_norm = np.zeros_like(blaze)
    for i, blaze_spec in enumerate(blaze):
        blaze_spec /= np.max(blaze_spec)
        blaze_norm[i, :] = blaze_spec

        width_l = np.searchsorted(blaze_spec[0:900], 0.6)
        width_l2 = np.searchsorted(blaze_spec[0:900], 0.5)
        width_r = blaze_spec.size - np.searchsorted(blaze_spec[1100:][::-1], 0.6)
        width_r2 = blaze_spec.size - np.searchsorted(blaze_spec[1100:][::-1], 0.5)
        width = width_r - width_l
        width2 = width_r2 - width_l2

        blaze_widths.append(width / hr_scaler)
        blaze_width_rs.append(width_r)
        blaze_width_rs2.append(width_r2)

    blaze_peaks.extend(blaze_peak)

    colour = colours[int(key.split("-")[3]) - 100 - 1]
    ax1a.plot(blaze_norm[::10, :].T, alpha=0.01, color=colour)

    ts.extend(t)
    aotfs.extend(aotf)

fig1.colorbar(ScalarMappable(norm=plt.Normalize(min(orders), max(orders)), cmap="Spectral"), cax=ax1b, label="Order")

ax1a.grid()
ax1a.set_xlabel("Pixel number")


blaze_width_rs = np.asarray(blaze_width_rs)
blaze_width_rs2 = np.asarray(blaze_width_rs2)
aotfs = np.asarray(aotfs)

# plt.figure()
# plt.title("Blaze width FW@0.6M")
# plt.scatter(aotfs, blaze_widths)
plt.figure()
sc = plt.scatter(blaze_width_rs[blaze_width_rs2 < 1600], blaze_width_rs2[blaze_width_rs2 < 1600], c=aotfs[blaze_width_rs2 < 1600], cmap="Spectral")
plt.colorbar(sc)

polyfit = np.polyfit(blaze_width_rs[blaze_width_rs2 < 1600], blaze_width_rs2[blaze_width_rs2 < 1600], 1)
polyvals = np.polyval(polyfit, blaze_width_rs[blaze_width_rs2 < 1600])
plt.plot(blaze_width_rs[blaze_width_rs2 < 1600], polyvals, "k--")

# plt.figure()
# plt.title("Blaze peak pixel")
# plt.scatter(aotfs, blaze_peaks)


# for i in range(50, 1600, 100):
#     plt.plot(blaze_norm[:, i]/np.max(blaze_norm[:, i]))
