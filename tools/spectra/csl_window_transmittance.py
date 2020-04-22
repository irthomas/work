# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:22:24 2020

@author: iant
"""

def csl_window_transmittance(x_scale):
    """get ground calibration CSL window transmission for given wavenumber range"""
    import numpy as np
    import os
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    data_in = np.loadtxt(os.path.join(dir_path, "sapphire_window.csv"), skiprows=1, delimiter=",")
    wavenumbers_in =  10000. / data_in[:,0]
    transmission_in = data_in[:,1] / 100.0
    
    window_transmission = np.interp(x_scale, wavenumbers_in[::-1], transmission_in[::-1])
    
    return window_transmission

