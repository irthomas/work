# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:29:48 2020

@author: iant

CONVOLVE GUILLAUME SOLAR SPECTRUM TO LNO RESOLUTION
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tools.spectra.savitzky_golay import savitzky_golay
from tools.file.read_write_hdf5 import write_hdf5_from_dict

RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.normcase('c:\\users\\iant\\documents\\data\\pfm_auxiliary_files\\radiometric_calibration')

window_size1 = 699
window_size2 = 9
order = 1

def load_solar_spectrum():
    
    solar_spectrum_file = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, "irrad_spectrale_1_5_UA_ACE_kurucz.npz")
    
    npzfile = np.load(solar_spectrum_file)
    x = np.asarray(npzfile['arr_0']).squeeze()
    y = np.asarray(npzfile['arr_1']).squeeze()

    return x, y

x, y = load_solar_spectrum()
plt.figure(figsize=(17, 6))
plt.plot(x, y, alpha=0.5, label="Full resolution")

y_smooth = np.zeros_like(y)
y_smooth[:405995] = savitzky_golay(y[:405995], window_size1, order, deriv=0, rate=1)
y_smooth[405995:] = savitzky_golay(y[405995:], window_size2, order, deriv=0, rate=1)

plt.plot(x, y_smooth, label="Convolved to approx. LNO resolution")
plt.xlabel("Wavenumbers cm$^{-1}$")
plt.legend()
plt.tight_layout()

output_dict = {"x":x, "y":y_smooth}

write_hdf5_from_dict("irrad_spectrale_1_5_UA_ACE_kurucz", output_dict, {}, {}, {})