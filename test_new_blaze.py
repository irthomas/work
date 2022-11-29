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
h5_prefix = "20221011_132104-188-8"

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


#high res solar line
nu_hr = np.arange(np.min(px_nus[0]) - 5.0, np.max(px_nus[-1]) + 5.0)
solar_spectrum_filename = "pfsolspec_hr.dat"
# solar_spectrum_filename = "Solar_irradiance_ACESOLSPEC_2015.dat"
ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], solar_spectrum_filename)
I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)


#pre-convolute solar spectrum to approximate level - only for fitting temperature and plotting
I0_lr = savgol_filter(I0_solar_hr, 99, 1)

#get aotf
aotf_nu = aotf_peak_nu(aotf_freq, t)
aotf_nus, aotf_func = aotf_func(aotf_nu, aotf_range=200.0, step_nu=0.1)

#get blaze
blaze = np.loadtxt("blaze.tsv", skiprows=1)[:, 1]

plt.figure()
for px_nu in px_nus:
    plt.plot(px_nu, blaze)

plt.plot(aotf_nus + aotf_nu, aotf_func)
plt.plot(nu_hr, I0_solar_hr/np.max(I0_solar_hr))
#do order addition

# for px
