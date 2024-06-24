# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:52:06 2024

@author: iant

CONVERT SOLAR LINE INTO AOTF/PIXEL RANGE
"""


import os
# import h5py
from astropy.io import fits


import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu_mp, nu0_aotf, F_aotf_sinc


from tools.spectra.solar_spectrum import get_solar_hr
from tools.general.progress_bar import progress

from instrument.calibration.so_lno_2023.solar_line_dict import solar_line_dict

from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs, make_ils

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

# channel = "so"
channel = "lno"


# save_ifigs = True
save_ifigs = False


plot = []


for h5_prefix, solar_line_data_all in solar_line_dict.items():  # loop through files
    channel = h5_prefix.split("-")[0].lower()

    # get data from miniscan file
    with fits.open(os.path.join(MINISCAN_PATH, channel, "%s.fits" % h5_prefix)) as hdul:
        keys = [i.name for i in hdul if i.name != "PRIMARY"]
        n_reps = len([i for i, key in enumerate(keys) if "ARRAY" in key])

        arrs = []
        aotfs = []
        ts = []
        for i in range(n_reps):
            arrs.append(hdul["ARRAY%02i" % i].data)
            aotfs.append(hdul["AOTF%02i" % i].data)
            ts.append(hdul["T%02i" % i].data)


aotf_array = aotfs[0]
scan_array = arrs[0]
t_mean = np.mean(ts[0])

# find upper/lower nus for high res solar spectrum
min_aotf_khz = np.min(aotf_array)
max_aotf_khz = np.max(aotf_array)

aotf_search_range_khz = np.arange(18000, 32000)
aotf_search_range_nu = nu0_aotf(aotf_search_range_khz)

min_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, min_aotf_khz)]
max_nu = aotf_search_range_nu[np.searchsorted(aotf_search_range_khz, max_aotf_khz)]

# get solar spectrum
nu_hr = np.arange(min_nu - 50.0, max_nu + 50.0, 0.001)
solar_hr = get_solar_hr(nu_hr)

# plt.plot(nu_hr, solar_hr)

aotf_line = aotf_array[0, :]
scan_line = scan_array[0, :]
# plt.plot(aotf_line, scan_line)


# from aotf frequency get aotf function
# first get px grid for each order
order_search_range = np.arange(110, 210)
order_search_range_nu = nu_mp(order_search_range, 0.0, 0.0)

# get list of orders
min_order = order_search_range[np.searchsorted(order_search_range_nu, min_nu)]
max_order = order_search_range[np.searchsorted(order_search_range_nu, max_nu)]
orders = np.arange(min_order, max_order+1)

px_ixs = np.arange(320.0)


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
    aotf_nu_centre = np.mean(aotf_line)
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

        ils = make_ils(nu_grid, width, displacement, amplitude)
        # summed ils without absorption lines - different for different pixels but v. similar for orders
        ils_sum[px] = np.sum(ils)
        ils_sums[order_ix, px] = ils_sum[px]
        # summed ils with absorption lines - changes with pixel and order
        ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_grid)

spectrum = ils_sums_spectrum / ils_sums

for order_ix, order in enumerate(orders):
    px_nus = nu_mp(order, px_ixs, t_mean)
    plt.plot(px_nus, spectrum[order_ix, :])

# plt.legend()
# plt.grid()


# norm_scan_array = np.zeros_like(scan_array)
# for i in range(aotf_array.shape[0]):
#     norm_scan_array[i, :] = scan_array[i, :]/np.max(scan_array[i, :])


# plt.plot(norm_scan_array[:, 10])

# for i in range(0, aotf_array.shape[0], 49):
#     plt.plot(scan_array[i, :]/np.max(scan_array[i, :]), c=[i/aotf_array.shape[0], i/aotf_array.shape[0], i/aotf_array.shape[0]])


# #check if horizontal strips line up correctly
# for i in range(0, aotf_array.shape[1], 49):
#     plt.plot(scan_array[:, i]/np.max(scan_array[:, i]), c=[i/aotf_array.shape[1], i/aotf_array.shape[1], i/aotf_array.shape[1]])


# convert aotf line to nus in all orders


# import numpy as np

# arr4 = np.zeros_like(arr)

# for i in range(arr.shape[0]):
#     norm=(arr[i,:]/np.max(arr[i,:]))/(smooth/np.max(smooth))
#     # plt.plot(norm)

#     arr4[i, :] = norm

# plt.figure()
# plt.imshow(arr4[:, 50:])

# plt.figure()
# plt.plot(arr4[:, 2130+50])

# plt.figure()
# plt.plot(arr4[:, 50+50])


# 10000.0
