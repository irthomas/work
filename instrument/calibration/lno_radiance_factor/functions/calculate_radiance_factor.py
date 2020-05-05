# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:31:36 2020

@author: iant
"""

def calculate_radiance_factor(y, synth_solar_spectrum, sun_mars_distance_au):
    """calculate radiance factor from i and f, accounting for sun to mars distance"""
    
    import numpy as np
    
    n_spectra = len(y[:, 0])
    
    #TODO: add conversion factor to account for solar incidence angle
    #TODO: this needs checking. No nadir or so FOV in calculation!
    rSun = 695510.0 # radius of Sun in km
    dSun = sun_mars_distance_au * 1.496e+8 #1AU to km
    angle_solar = np.pi * (rSun / dSun) **2 / 2.0 #why /2.0?
    #do I/F using shifted observation wavenumber scale
    y_rad_fac = y / np.tile(synth_solar_spectrum, [n_spectra, 1]) / angle_solar
        
    return y_rad_fac
