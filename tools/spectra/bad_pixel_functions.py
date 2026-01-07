# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:39:15 2024

@author: iant

BAD PIXEL REMOVAL FUNCTIONS
1. GET MEAN OF AN ARRAY OF SPECTRA BY REMOVING N POINTS FURTHEST FROM THE MEDIAN VALUE E.G. TO REMOVE BAD PIXELS
"""


import numpy as np
import matplotlib.pyplot as plt


def median_frame_bad_pixel(array, n_px_to_remove=2, plot=False):
    """remove bad pixels from a frame of spectra by finding mean of pixels nearest the median value
    array must be of the form rows (all identical in theory) x columns (not identical)"""

    nrows, ncols = array.shape

    median = np.median(array, axis=0)  # median spectrum
    median_array = np.tile(median, (nrows, 1))

    array_abs_sub = np.abs(array - median_array)

    sorted_ixs_array = np.argsort(array_abs_sub, axis=0)

    good_ixs_array = np.delete(sorted_ixs_array, slice(nrows - n_px_to_remove, nrows), axis=0)

    spectrum = np.zeros(ncols)
    for i in range(ncols):
        spectrum[i] = np.mean(array[good_ixs_array[:, i], i])

    if plot:
        plt.plot(array.T)
        plt.plot(spectrum, "k--")

    return spectrum
