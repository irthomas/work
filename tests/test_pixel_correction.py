# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:55:47 2020

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
from tools.spectra.fit_polynomial import fit_linear_errors


import os
import h5py

#SO pixel correction 
SO_PIXEL_CORRECTION_AUX_FILE = os.path.join("px_correction_windowtop=120.h5")

pixel_correction_dict = {}

with h5py.File(SO_PIXEL_CORRECTION_AUX_FILE) as f:
    bin_indices_str = list(f.keys())
    bin_indices = [int(i) for i in bin_indices_str]
    for bin_index, bin_index_str in zip(bin_indices, bin_indices_str):
        coefficients = f[bin_index_str]["coefficients"][...]
        pixel_correction_dict[bin_index] = coefficients
        



stop()


x = [-13.861301, -8.775637, 1.4414688, 2.8829756, 12.394587]
y = [1.419999999999618,  5.089999999999918, 13.349999999999682, 13.760000000000218, 22.970000000000027]
chi = [5.421346453665266e-07, 1.3810010392667396e-06, 0.00010992595165269564, 6.8476361317001585e-06, 1.443261243359843e-05]
y_err = 1./np.asfarray(chi)


# _, coeffs = fit_linear_errors(x, y, y_err, coeffs=True, error=False)



from scipy.optimize import curve_fit

if type(x) == list:
    x = np.asfarray(x)
if type(y) == list:
    y = np.asfarray(y)
if type(y_err) == list:
    y_err = np.asfarray(y_err)
    
    
def func(x, a, b):
    return a*x + b

poly_fit, pcov = curve_fit(func, x, y, p0=[0.75, 10.0])

poly_val = func(x, *poly_fit)




plt.figure()
plt.scatter(x, y)
plt.plot(x, np.polyval(poly_fit, x))
plt.errorbar(x, y,  np.asfarray(chi)*5000, fmt=".")