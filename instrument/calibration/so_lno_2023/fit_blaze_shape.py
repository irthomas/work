# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:21:09 2024

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt

# from tools.general.get_minima_maxima import get_local_minima
# from tools.general.progress_bar import progress


def fit_threshold(x, y_norm, loop, start_coeff, degree, verbose, plot):

    # fit a polynomial to the data:
    polyfit = np.polyfit(x, y_norm, degree)

    # make a function based on those polynomial coefficients
    polyfunc = np.poly1d(polyfit)

    # make a lower threshold that is offset below the continuum fit. All points
    # below this fit (i.e. spectral lines) will be excluded from the fit in the
    # next iteration.
    thresh = polyfunc(x) - (start_coeff * (1. / (loop + 1)))

    if plot:
        # plot the original spectrum:
        plt.plot(x, y_norm)
        # overplot the continuum fit
        plt.plot(x, polyfunc(x))
        plt.plot(x, thresh)

    mask = np.where(y_norm > thresh)[0]

    residrms = np.std(y_norm / polyfunc(x))
    if (verbose is True):
        print("i=%i, residual=%0.3g" % (loop, residrms))

    return residrms, mask, polyfit


def fit_blaze_iterative(y, maxrms, start_coeff=0.05, max_loop=10, degree=7, verbose=False, plot=False):

    # centre x values
    x = np.arange(len(y)) - len(y)/2.0
    y_norm = y / max(y)

    mask = np.arange(len(y_norm))
    residrms = maxrms + 1.0

    for loop in range(max_loop):
        residrms, mask_ixs, polyfit = fit_threshold(x[mask], y_norm[mask], loop, start_coeff, degree, verbose, plot)
        mask = mask[mask_ixs]

        if residrms < maxrms:
            # quit early if rms found
            return polyfit

    return polyfit


def fit_blaze(spectrum, max_rms=0.02, degree=7, verbose=False, plot=False):

    x = np.arange(len(spectrum))

    spectrum_coeffs = fit_blaze_iterative(spectrum, max_rms, degree=degree, verbose=verbose, plot=plot)
    x2 = x - min(x) - (max(x) - min(x))/2.
    spectrum_fit = np.polyval(spectrum_coeffs, x2) * np.max(spectrum)

    return spectrum_fit
