# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:44:26 2020

@author: iant

DO POLYFIT
"""

def fit_polynomial(x_in, y_in, degree=1, coeffs=False, indices=[None]):
    """fit polynomial and return values (and coefficients, optionally).
    Can also give a list of which points should be considered for fit"""
    
    import numpy as np
    
    if indices[0] == None:   
        poly_fit = np.polyfit(x_in, y_in, degree)
    else:
        poly_fit = np.polyfit(x_in[indices], y_in[indices], degree)

    poly_val = np.polyval(poly_fit, x_in)
    if coeffs:
        return poly_val, poly_fit
    else:
        return poly_val
