# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:08:53 2024

@author: iant

FIT BLAZE FUNCTION
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from instrument.nomad_lno_instrument_v02 import nu_mp
from instrument.calibration.so_lno_2023.asymmetric_blaze import asymmetric_blaze


order = 165
row_ix = 500
channel = "lno"
# no_abs_ixs = np.asarray([18, 98, 123, 225, 284, 340, 663, 676, 694, 768, 790, 905, 955, 984, 1029, 1102, 1229, 1243, 1384, 1423, 1475, 1542])
first_guess = [290., -5.0, 1.01]
# ncols = 1600

array = array_norms[0]

# spectrum_no_abs = spectrum[no_abs_ixs]


def make_blaze(params):

    [eff_n_px, t, scaler] = params

    px_nus = nu_mp(order, np.arange(ncols)*(eff_n_px/ncols), t)
    blaze = asymmetric_blaze(channel, order, px_nus)*scaler
    return blaze


def min_blaze(params, spectrum):

    blaze = make_blaze(params)
    chisq_px = (spectrum - blaze)**2 / spectrum
    ixs = np.where(spectrum > blaze)[0]
    chisq_px[ixs] *= 100.0
    chisq = np.sum(chisq_px)
    return chisq


# def fit_blaze(array):
nrows, ncols = array.shape
for row in range(nrows):
    res = minimize(min_blaze, first_guess)


# print(res.x)
# plt.figure()
# plt.plot(spectrum)
# plt.plot(make_blaze(res.x))

# plt.plot(spectrum / make_blaze(res.x))
