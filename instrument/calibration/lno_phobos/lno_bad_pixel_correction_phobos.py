# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:59:39 2025

@author: iant
"""

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt


def bad_pixel_correction(y_3d, n_loops=15, plot=[[]]):
    """correct bad pixels in input 3d array of raw spectra with a polyfit, replacing bad values by interpolated fit
    y_3d has dimensions frames x rows x wavelengths and plot is list of [row x frame] index lists"""
    x = np.arange(y_3d.shape[2])
    y_3d_new = np.copy(y_3d)

    # loop through N times, fitting and removing worst pixels
    for loop in range(n_loops):
        for row in range(y_3d_new.shape[1]):
            for frame in range(y_3d_new.shape[0]):
                y = y_3d_new[frame, row, :]
                poly = Polynomial.fit(x, y, 5)
                yfit = poly(x)
                dev = np.abs(y-yfit)
                bad_ix = np.where(dev == np.max(dev))
                y[bad_ix] = yfit[bad_ix]
                y_3d_new[frame, row, :] = y

                if len(plot) > 1 and len(plot[0]) > 1:
                    for plot_ix in plot:
                        if row == plot_ix[0] and frame == plot_ix[1]:
                            if loop == n_loops - 1:
                                fig, ax = plt.subplots()
                                ax.set_title("Bad pixel correction check, row=%i, frame=%i" % (row, frame))
                                ax.plot(y_3d[frame, row, :], alpha=0.3)
                                ax.plot(yfit, alpha=0.6)
                                ax.plot(y_3d_new[frame, row, :], alpha=0.3)

    return y_3d_new

# simplest correction
# bad_pixel_correction(y_3d)
# plot some fits
# bad_pixel_correction(y_3d, n_loops=15, plot=[[0, 2], [1, 4], [3, 2]])
