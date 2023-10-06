# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:34:00 2023

@author: iant

TEST HR ARRAY CREATION METHODS
"""


import numpy as np
from scipy.interpolate import RectBivariateSpline#RegularGridInterpolator


def make_hr_array(array, aotfs, hr_scaler):
    
    
    array_shape = array.shape
    x = np.arange(array_shape[0])
    y = np.arange(array_shape[1])
    
    interp = RectBivariateSpline(x, y, array)
                                 #bounds_error=False, fill_value=None, method="linear")
    x_hr = np.arange(0, array_shape[0], 1.0/hr_scaler)
    y_hr = np.arange(0, array_shape[1], 1.0/hr_scaler)
    X, Y = np.meshgrid(x_hr, y_hr, indexing='ij')
    
    array_hr = interp(x_hr, y_hr)
    
    #interpolate aotf freqs onto same grid
    aotf_hr = np.interp(x_hr, x, aotfs)

    return array_hr, aotf_hr


#for testing (first run correct_miniscan_diagonals.py)

# arr = array[:, 160:240]
# aotf = aotfs[:]

# array_hr, aotf_hr = make_hr_array(arr, aotf, 10.0)

# plt.figure()
# plt.plot(np.arange(arr.shape[0]), arr[:, 40])

# plt.plot(np.linspace(0, arr.shape[0], array_hr.shape[0]), array_hr[:, 400])
