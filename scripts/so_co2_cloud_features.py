# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:43:00 2020

@author: iant
"""

from tools.file.hdf5_functions import get_file
import matplotlib.pyplot as plt
import numpy as np
from tools.plotting.colours import get_colours

filename = "20190502_200016_1p0a_SO_I_S"

order = 161
# pixel_high = 128
# pixel_low = 122

# order = 163
# pixel_high = 126
# pixel_low = 131

# order = 164
pixel_high = 128
pixel_low = 122

# order = 165
# pixel_high = 148
# pixel_low = 153



hdf5_filename,hdf5_file = get_file(filename, "hdf5_level_1p0a", 0)

alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:,0]

orders = hdf5_file["Channel/DiffractionOrder"][...]

a = np.where(orders == order)
b = np.where((60 < alts) & (alts < 73))
indices = np.intersect1d(a, b)


y = hdf5_file["Science/Y"][:,:]
x = hdf5_file["Science/X"][indices[0],:]

# plt.plot(y[indices, :].T, alpha=0.5)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
colours = get_colours(len(indices), cmap="plasma")
for i, index in enumerate(indices):
    if i>0:
        prev_index = indices[i-1]
        # plt.plot(y[index, :]/np.max(y[index, :]) - y[index-1, :]/np.max(y[index-1, :]), alpha=0.5, color=colours[i], label="%0.1f" %alts[index])
        # plt.plot(y[index, :] - y[index-1, :], alpha=0.5, color=colours[i], label="%0.1f" %alts[index])
        # plt.plot(y[index, :], alpha=0.5, color=colours[i], label="%0.1f" %alts[index])
        norm = (y[index, :]-y[index, pixel_low])/(y[index, pixel_high]-y[index, pixel_low])
        norm1 = (y[prev_index, :]-y[prev_index, pixel_low])/(y[prev_index, pixel_high]-y[prev_index, pixel_low])
    
        ax2.plot(x, norm-norm1, alpha=0.5, color=colours[i], label="%0.1f" %alts[index])
        ax1.plot(x, norm1, alpha=0.5, color=colours[i], label="%0.1f km" %alts[index])
     
ax1.legend(loc='center left', bbox_to_anchor=(1, 0))
# fig.colorbar(ax1)

ax1.set_title(filename + ": normalised transmittances")
ax2.set_title(filename + ": subtracted from previous")