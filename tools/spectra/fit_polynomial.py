# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:44:26 2020

@author: iant

DO POLYFIT
"""

def fit_polynomial(x_in, y_in, degree=1, coeffs=False, error=False, indices=[None]):
    """fit polynomial and return values (and coefficients, optionally).
    Can also give a list of which points should be considered for fit"""
    
    import numpy as np
    
    if indices[0] == None:   
        poly_fit = np.polyfit(x_in, y_in, degree)
    else:
        poly_fit = np.polyfit(x_in[indices], y_in[indices], degree)

    poly_val = np.polyval(poly_fit, x_in)
    if error:
        chi_squared = np.sum(((y_in - poly_val) / poly_val)**2) #divide by yfit to normalise large and small absorption bands

    if coeffs:
        if error:
            return poly_val, poly_fit, chi_squared
        else:
            return poly_val, poly_fit
    else:
        if error:
            return poly_val, chi_squared
        else:
            return poly_val



def fit_linear_errors(x_in, y_in, y_err, coeffs=False, error=False):
    """fit polynomial using curve fit with errors"""
    
    import numpy as np
    from scipy.optimize import curve_fit

    def func(x, a, b):
        return a*x + b

    if type(x_in) == list:
        x_in = np.asfarray(x_in)
    if type(y_in) == list:
        y_in = np.asfarray(y_in)
    if type(y_err) == list:
        y_err = np.asfarray(y_err)

    
    poly_fit, pcov = curve_fit(func, x_in, y_in, p0=[0.75, 10.0], sigma=y_err)
    
    poly_val = func(x_in, *poly_fit)
    if error:
        chi_squared = np.sum(((y_in - poly_val) / poly_val)**2) #divide by yfit to normalise large and small absorption bands
    
    if coeffs:
        if error:
            return poly_val, poly_fit, chi_squared
        else:
            return poly_val, poly_fit
    else:
        if error:
            return poly_val, chi_squared
        else:
            return poly_val
    