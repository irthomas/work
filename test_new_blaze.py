# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:55:05 2022

@author: iant


SOLAR LINE SIMULATION, COMPARE TO MINISCAN ARRAYS
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter




from instrument.nomad_so_instrument_v03 import m_aotf, aotf_peak_nu, lt22_waven, aotf_func
from tools.spectra.solar_spectrum import get_solar_hr
from tools.file.paths import paths


from test_blaze_function import list_miniscan_data_1p0a, get_miniscan_data_1p0a

# h5_prefix = "20190416_020948-194-1"
# h5_prefix = "20221011_132104-188-8"

h5_prefix = "20201010_113533-188-8"


# aotf_file = "aotf_20211105_155547-191-4.tsv"
aotf_file = "aotf_20201010_113533-188-8.tsv"

n_orders = 3
frame_ix = 0

centre_order = int(h5_prefix.split("-")[1])
t = d2[h5_prefix]["t"]
aotf_freqs = d2[h5_prefix]["aotf"]

aotf_freq = aotf_freqs[frame_ix]

#get orders
orders = np.arange(centre_order - n_orders, centre_order + n_orders + 1)
#get pixel wavenumbers
px_nus = [lt22_waven(i, t) for i in orders]
px_nu_centre = lt22_waven(centre_order, t)


#high res solar line
nu_hr = np.arange(np.min(px_nus[0]) - 5.0, np.max(px_nus[-1]) + 5.0)
solar_spectrum_filename = "pfsolspec_hr.dat"
# solar_spectrum_filename = "Solar_irradiance_ACESOLSPEC_2015.dat"
ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], solar_spectrum_filename)
I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
#normalise
I0_solar_hr /= np.max(I0_solar_hr)

#pre-convolute solar spectrum to approximate level - only for fitting temperature and plotting
# I0_lr = savgol_filter(I0_solar_hr, 99, 1)

#get aotf
aotf_nu = aotf_peak_nu(aotf_freq, t)
aotf_nus, aotf_func = aotf_func(aotf_nu, aotf_range=200.0, step_nu=0.1)

#get aotf from text file
aotf_nus2, aotf_func2 = np.loadtxt(aotf_file, skiprows=1, unpack=True)
# aotf_nus2, aotf_func2 = np.loadtxt(".tsv", skiprows=1, unpack=True)
#shift peak
aotf_nu_max = aotf_nus2[aotf_func2.argmax()]
aotf_nus2 -= aotf_nu_max
#normalise
aotf_func2 /= np.max(aotf_func2)

#get blaze
blaze = np.loadtxt("blaze.tsv", skiprows=1)[:, 1]

# plt.figure()
# for px_nu in px_nus:
#     plt.plot(px_nu, blaze)

# plt.plot(aotf_nus + aotf_nu, aotf_func)
# plt.plot(aotf_nus2 + aotf_nu, aotf_func2)
# plt.plot(nu_hr, I0_solar_hr)


#do order addition
orders_contribs = []
orders_contribs2 = []
for px_nu in px_nus:
    #resample aotf to pixel grid
    aotf_px_nu2 = np.interp(px_nu, aotf_nus2 + aotf_nu, aotf_func2)
    aotf_px_nu = np.interp(px_nu, aotf_nus + aotf_nu, aotf_func)
    #resample solar spectrum to pixel grid
    solar_px_nu = np.interp(px_nu, nu_hr, I0_solar_hr)
    
    #multiply all
    order_spectrum = blaze * aotf_px_nu * solar_px_nu
    order_spectrum2 = blaze * aotf_px_nu2 * solar_px_nu
    
    orders_contribs.append(order_spectrum)
    orders_contribs2.append(order_spectrum2)
    # plt.plot(px_nu, order_spectrum)
    
orders_contribs = np.asfarray(orders_contribs)
orders_contribs2 = np.asfarray(orders_contribs2)

order_contribs = np.sum(orders_contribs, axis=0)
order_contribs2 = np.sum(orders_contribs2, axis=0)
order_contribs /= np.max(order_contribs)
order_contribs2 /= np.max(order_contribs2)

plt.figure()
plt.plot(px_nu_centre, order_contribs, label="Old AOTF, new blaze")
plt.plot(px_nu_centre, order_contribs2, label="%s, new blaze" %aotf_file)

spectrum = d2[h5_prefix]["array_raw"][0, :]
spectrum /= np.max(spectrum)

plt.plot(px_nu_centre, spectrum, label="raw")

spectrum = d2[h5_prefix]["array_corrected"][0, :]
spectrum /= np.max(spectrum)

plt.plot(px_nu_centre, spectrum, label="corrected")

plt.grid()
plt.legend()