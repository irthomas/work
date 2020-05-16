# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:31:36 2020

@author: iant
"""

def calculate_radiance_factor(y, synth_solar_spectrum, sun_mars_distance_au, mean_incidence_angles_deg):
    """calculate radiance factor from i and f, accounting for sun to mars distance"""
    
    import numpy as np
    
    n_spectra = len(y[:, 0])
    n_pixels = len(y[0, :])
    
    #TODO: this needs checking. No nadir or SO FOV in calculation!
    rSun = 695510.0 # radius of Sun in km
    dSun = sun_mars_distance_au * 1.496e+8 #1AU to km
    angle_solar = np.pi * (rSun / dSun) **2
    
    mean_incidence_angles = mean_incidence_angles_deg * np.pi / 180.0
    
    #do I/F using shifted observation wavenumber scale
    y_rad_fac = y / np.tile(synth_solar_spectrum, [n_spectra, 1]) / angle_solar / np.cos(np.tile(mean_incidence_angles.T, (n_pixels, 1)).T)
        
    return y_rad_fac
