# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:25:39 2021

@author: iant

COMPARE PROPER ILS CONVOLUTION TO SAVGOL FILTER

"""

import numpy as np
import os


from scipy.signal import savgol_filter

import matplotlib.pyplot as plt


from tools.spectra.solar_spectrum import get_solar_hr
from tools.file.paths import paths

from instrument.calibration.so_aotf_ils.simulation_config import ORDER_RANGE, pixels, nu_range, D_NU
from instrument.nomad_so_instrument import nu_grid, F_blaze, nu_mp, spec_res_order


# SOLAR_SPECTRUM = "Solar_irradiance_ACESOLSPEC_2015.dat"
SOLAR_SPECTRUM = "pfsolspec_hr.dat"



temperature = 0.0

dnu = D_NU
order_range = ORDER_RANGE





nu_hr = np.arange(nu_range[0], nu_range[1], dnu)

ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], SOLAR_SPECTRUM)
I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)







Nbnu_hr = len(nu_hr)
NbP = len(pixels)


c_order = int(np.mean(order_range))
spec_res = spec_res_order(c_order)



#old and new blaze functions are functional identical - use 2021 function only
sconv = spec_res/2.355
W_conv = np.zeros((NbP,Nbnu_hr))

iord = c_order

nu_pm = nu_mp(iord, pixels, temperature)
for ip in pixels:
    W_conv[ip,:] += (dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
    
W_conv[W_conv < 1.0e-5] = 0.0 #remove small numbers
I0_hr = I0_solar_hr
I0_p = np.matmul(W_conv, I0_hr)




I0_lr = savgol_filter(I0_solar_hr, 99, 1)


plt.figure()
plt.plot(nu_hr, I0_lr, label="Filter")
plt.plot(nu_pm, I0_p, label="ILS convolution")

plt.title(SOLAR_SPECTRUM)
plt.legend()