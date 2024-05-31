# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:21:09 2024

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.general.get_minima_maxima import get_local_minima
from tools.general.progress_bar import progress


def blazeFit(wav, spec, maxrms, numcalls=10, curcall=0,
             verbose=False, showplot=False):
    """PURPOSE: To fit the continuum of an order of an
    echelle spectrum to model the Blaze Function.
    INPUTS:
    WAV: The wavelength
    SPEC: The spectrum. This should be the same number of elements
    (and corrsepond to) the input wavelength array (wav).
    MAXRMS: The threshold criteria for the fit in normalized rms.
    For example, a threshold of 0.01 will keep iterating until
    the rms of the residuals of dividing the continuum pixels
    by the Blaze Function comes out to 1%.
    NUMCALLS: The maximum number of recursive iterations to execute.
    CURCALL: Store the current iteration for recursive purposes.
    VERBOSE: Set this to True to print out the iteration, residual
    rms and the threshold value.
    SHOWPLOT: Set this to True to produce a plot of the spectrum,
    threshold and continuum at every iteration.
    """

    # center wavelength range about zero:
    wavcent = wav - min(wav) - (max(wav) - min(wav))/2.

    # normalize the spectrum:
    normspec = spec/max(spec)

    # fit a polynomial to the data:
    z = np.polyfit(wavcent, normspec, 7)

    # make a function based on those polynomial coefficients:
    cfit = np.poly1d(z)

    # make a lower threshold that is offset below the continuum fit. All points
    # below this fit (i.e. spectral lines) will be excluded from the fit in the
    # next iteration.
    thresh = cfit(wavcent) - (0.15 * (1. / (curcall + 1)))

    if (showplot is True):
        # plot the original spectrum:
        plt.plot(wavcent, normspec)
        # overplot the continuum fit
        plt.plot(wavcent, cfit(wavcent))
        plt.plot(wavcent, thresh)

    mask = np.where(normspec > thresh)[0]

    residrms = np.std(normspec/cfit(wavcent))
    if (verbose is True):
        print("i=%i, residual=%0.3g" % (curcall, residrms))
        # print('now in iteration {0}'.format(curcall))
        # print('residrms is now {0:.5f}'.format(residrms))
        # print('maxrms is {0})'.format(maxrms))
        # print('z is: {}'.format(z))

    if ((curcall < numcalls) and (residrms > maxrms)):
        z = blazeFit(wavcent[mask], normspec[mask], maxrms,
                     numcalls=numcalls, curcall=curcall+1, verbose=verbose, showplot=showplot)

    # now un-center the wavelength range:
    # if curcall == 0:
        # z[-1] = z[-1] - min(wav) - wavspread/2.

    return z


# smooth = savgol_filter(fits2[0, :], 199, 1)
# plt.plot(smooth)


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


def fit_blaze_iterative(y, maxrms, start_coeff=0.15, max_loop=10, degree=7, verbose=False, plot=False):

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


mins = []

plt.figure()
for i in progress(np.arange(0, 1901, 1)):

    y = arr[i, :]
    x = np.arange(len(y))
    # y_norm = y / np.max(y)

    # y2_coeffs = blazeFit(x, y, 0.02, verbose=True)
    y2_coeffs = fit_blaze_iterative(y, 0.02, verbose=False)
    x2 = x - min(x) - (max(x) - min(x))/2.
    y2 = np.polyval(y2_coeffs, x2) * np.max(y)

    y_norm = y / y2
    plt.plot(y_norm)

    abs_ix_guess = 2180

    # find real minimum
    abs_centre = np.argmin(y_norm[abs_ix_guess-50:abs_ix_guess+50]) + abs_ix_guess-50

    # fit minimum
    abs_polyfit = np.polyfit(x[abs_centre-5:abs_centre+6], y_norm[abs_centre-5:abs_centre+6], 2)

    abs_polyval = np.polyval(abs_polyfit, x[abs_centre-5:abs_centre+6])
    plt.scatter(x[abs_centre-5:abs_centre+6], abs_polyval)

    # minimum of polynomial
    poly_min = abs_polyfit[2] - abs_polyfit[1]**2 / (4*abs_polyfit[0])

    mins.append(poly_min)


plt.figure()
plt.plot(mins)
