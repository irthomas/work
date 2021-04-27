# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:52:46 2020

@author: iant

EXPONENTIAL FIT TO ABSORPTION BAND
"""



def fit_gaussian_absorption(x_in, y_in, error=False):
    """fit inverted gaussian to absorption band. Normalise continuum to 1 first"""
    """optional argument to calculate chi sq error"""
    import numpy as np
    from scipy.optimize import curve_fit

    def func(x, a, b, c, d):
        return 1.0 - a * np.exp(-((x - b)/c)**2.0) + d
    
    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    try:
        popt, pcov = curve_fit(func, x_centred, y_in, p0=[0.1, 0.02, 0.25, 0.0])
    except RuntimeError: #curve fit failed to find solution
        if error:
            return 0.0, 0.0, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0
    
    if error:
        y_fit = func(x_centred, *popt)
        chi_squared = np.sum(((y_in - y_fit) / y_fit)**2) #divide by yfit to normalise large and small absorption bands
        
    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
    y_hr = func(x_hr - x_mean, *popt)
    
    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    
    x_min_position = x_hr[min_index]
    
    if error:
        return x_hr, y_hr, x_min_position, chi_squared
    else:
        return x_hr, y_hr, x_min_position
    
