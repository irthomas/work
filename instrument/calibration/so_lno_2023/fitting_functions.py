# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:08:53 2024

@author: iant

FIT ASYMMETRIC BLAZE FUNCTION TO DATA
WILL ALSO TRY TO FIT TO ABSORPTION LINES

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from instrument.nomad_lno_instrument_v02 import nu_mp
from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze


# order = 165
# row_ix = 500
# channel = "lno"
# # no_abs_ixs = np.asarray([18, 98, 123, 225, 284, 340, 663, 676, 694, 768, 790, 905, 955, 984, 1029, 1102, 1229, 1243, 1384, 1423, 1475, 1542])
# first_guess = [290., -5.0, 1.01]
# ncols = 1600

# array = array_norms[0]

# spectrum_no_abs = spectrum[no_abs_ixs]


def make_blaze(params, channel, order, ncols):

    [eff_n_px, t, scaler] = params
    print(channel, order, ncols, params)

    px_nus = nu_mp(order, np.arange(ncols)*(eff_n_px/ncols), t)
    blaze = asymmetric_blaze(channel, order, px_nus)*scaler
    return blaze


def min_blaze(params, args):
    # print(params)
    # print(args)

    [spectrum, channel, order] = args

    ncols = len(spectrum)
    blaze = make_blaze(params, channel, order, ncols)
    chisq_px = (spectrum - blaze)**2
    ixs = np.where(spectrum > blaze)[0]  # penalise where spectrum is higher than blaze, so blaze is always on top
    chisq_px[ixs] *= 100.0
    chisq = np.sum(chisq_px)
    return chisq


def fit_blaze_array(array, first_guess, channel, order):
    nrows, ncols = array.shape
    for row in range(nrows):
        spectrum = array[row, :]
        res = minimize(min_blaze, first_guess, args=[spectrum, channel, order])

    print(res.x)
    plt.figure()
    plt.plot(spectrum)
    plt.plot(make_blaze(res.x))

    plt.plot(spectrum / make_blaze(res.x))


def fit_blaze_spectrum(spectrum, first_guess, channel, order):
    res = minimize(min_blaze, first_guess, args=[spectrum, channel, order])

    ncols = len(spectrum)
    print(res.x)
    plt.figure()
    plt.plot(spectrum)
    blaze = make_blaze(res.x, channel, order, ncols)
    plt.plot(blaze)

    plt.plot(spectrum / blaze)
