# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:48:04 2024

@author: iant

STANDALONE SOLAR LINE SIMULATION
"""


# import os
# from astropy.io import fits

import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# from scipy.ndimage import gaussian_filter
# from scipy.signal import argrelmin


# from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu_mp, nu0_aotf, F_aotf_sinc


from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.baseline_als import baseline_als
# from tools.general.progress_bar import progress


# from instrument.calibration.so_lno_2023.fit_blaze_shape import fit_blaze
from instrument.calibration.so_lno_2023.asymmetric_blaze import get_asym_blaze

from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs, make_ils

px_ixs = np.arange(320.0)
t_mean = 0.0
channel = "lno"

# order 125
aotf_array = [17530]

# plot_solar = True
plot_solar = False


ncols = len(px_ixs)


# find upper/lower nus for high res solar spectrum
min_aotf_khz = np.min(aotf_array)
max_aotf_khz = np.max(aotf_array)

aotf_search_range_khz = np.arange(16000, 32000)
aotf_search_range_nu = nu0_aotf(aotf_search_range_khz)

min_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, min_aotf_khz)]
max_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, max_aotf_khz)]

# get solar spectrum
nu_hr = np.arange(min_nu - 50.0, max_nu + 50.0, 0.001)
solar_hr_raw = get_solar_hr(nu_hr)
# normalise + correct solar with baseline ALS
solar_cont = baseline_als(solar_hr_raw, lam=1e11, p=0.9999)
solar_hr = solar_hr_raw / solar_cont

if plot_solar:
    plt.figure()
    plt.plot(nu_hr, solar_hr_raw)
    plt.plot(nu_hr, solar_cont)
    plt.figure()
    plt.plot(nu_hr, solar_hr)


# from aotf frequency get aotf function
# first get px grid for each order
order_search_range = np.arange(110, 210)
order_search_range_nu = nu_mp(order_search_range, 160.0, 0.0)

# get list of orders within the miniscan
min_order = order_search_range[np.searchsorted(order_search_range_nu, min_nu)-1]
max_order = order_search_range[np.searchsorted(order_search_range_nu, max_nu)]
orders = np.arange(min_order-1, max_order+1)

px_ixs = np.arange(320.0)
px_ixs_hr = np.arange(ncols) / ncols * 320.0


solar_spec_d = {}
for order in orders:
    px_nus = nu_mp(order, px_ixs, t_mean)
    solar_spec_d[order] = {}

    solar_spec = np.zeros_like(px_nus)
    for i, px_nu in enumerate(px_nus):
        solar_ixs = np.searchsorted(nu_hr, [px_nu-0.2, px_nu+0.2])
        solar_spec[i] = np.mean(solar_hr[solar_ixs])

    solar_spec_d[order]["solar_spec"] = solar_spec


# ILS convolution
# sum of ILS for each pixel for T=1
ils_sum = np.zeros(len(px_nus))
ils_sums = np.zeros((len(orders), len(px_nus)))
# sum of ILS for each pixel including absorptions
ils_sums_spectrum = np.zeros((len(orders), len(px_nus)))

for order_ix, order in enumerate(orders):
    # spectral calibration of each pixel in order
    px_nus = nu_mp(order, px_ixs, t_mean)
    # plt.plot(px_nus, solar_spec_d[order]["solar_spec"], label="Order %i" % order)

    # convolve with ILS to make low resolution plot
    # approx value for order
    aotf_nu_centre = min_nu
    ils_d = get_ils_coeffs(channel, aotf_nu_centre)

    for px, px_nu in enumerate(px_nus):

        # get bounding indices of solar grid
        ix_start = np.searchsorted(nu_hr, px_nu - 0.7)
        ix_end = np.searchsorted(nu_hr, px_nu + 0.7)

        width = ils_d["ils_width"][px]  # only ILS width known for LNO
        # displacement = ils_d["ils_displacement"][px]
        # amplitude = ils_d["ils_amplitude"][px]
        displacement = 0.0
        amplitude = 0.0

        nu_grid = nu_hr[ix_start:ix_end] - px_nu
        solar_grid = solar_hr[ix_start:ix_end]

        ils = make_ils(nu_grid, width/6., displacement, amplitude)
        # summed ils without absorption lines - different for different pixels but v. similar for orders
        ils_sum[px] = np.sum(ils)
        ils_sums[order_ix, px] = ils_sum[px]
        # summed ils with absorption lines - changes with pixel and order
        ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_grid)

spectrum = ils_sums_spectrum / ils_sums

plt.figure()
# blaze
blazes = np.zeros((len(orders), len(px_nus)))
for order_ix, order in enumerate(orders):
    px_nus = nu_mp(order, px_ixs, t_mean)
    blaze_func = get_asym_blaze(channel, order, px_nus)
    blazes[order_ix, :] = blaze_func
    plt.plot(px_nus, blaze_func)


# aotf
aotfs = np.zeros((len(orders), len(px_nus)))
for order_ix, order in enumerate(orders):
    # spectral calibration of each pixel in order
    px_nus = nu_mp(order, px_ixs, t_mean)
    aotf_func = F_aotf_sinc(orders[1], px_nus, t_mean)
    aotfs[order_ix, :] = aotf_func
    plt.plot(px_nus, aotf_func, "--")


spectrum_aotf_blaze = (ils_sums_spectrum / ils_sums) * blazes * aotfs


plt.figure()
px_nus = nu_mp(order, px_ixs, t_mean)
plt.plot(np.sum(spectrum_aotf_blaze, axis=0), label="Order %i" % order)
plt.legend()


plt.figure()
for order_ix, order in enumerate(orders):
    # px_nus = nu_mp(order, px_ixs, t_mean)
    px_nus_hr = nu_mp(order, px_ixs_hr, t_mean)

    spectrum_norm = spectrum[order_ix, :] / np.max(spectrum[order_ix, :])

    spectrum_norm_hr = np.interp(px_ixs_hr, px_ixs, spectrum_norm)

    plt.plot(px_nus_hr, spectrum_norm_hr, label="Order %i" % order)

    # if order == 194:
    #     plt.plot(px_nus_hr[10:], scan_array_norm[75, 10:], linestyle="--")
    # if order == 195:
    #     plt.plot(px_nus_hr[10:], scan_array_norm[120, 10:], linestyle="--")


plt.legend()
plt.grid()
