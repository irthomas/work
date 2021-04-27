# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:54:30 2020

@author: iant
"""

def make_synth_solar_spectrum(diffraction_order, observation_wavenumbers, observation_temperature, rad_fac_aux_filepath):
    """read in data from radiance factor calibration table to make synthetic solar spectrum at
    same instrument temperature as the nadir observation"""
    
    import h5py
    import numpy as np
    
    with h5py.File("%s.h5" %rad_fac_aux_filepath, "r") as rad_fac_aux_file:
        
        if "%i" %diffraction_order in rad_fac_aux_file.keys():
        
            #read in coefficients and wavenumber grid
            wavenumber_grid_in = rad_fac_aux_file["%i" %diffraction_order+"/wavenumber_grid"][...]
            coefficient_grid_in = rad_fac_aux_file["%i" %diffraction_order+"/coefficients"][...].T
        else:
            print("Diffraction order %i not found in auxiliary file %s" %(diffraction_order, rad_fac_aux_filepath))
            
    #find coefficients at wavenumbers matching real observation
    corrected_solar_spectrum = np.zeros_like(observation_wavenumbers)
    for pixel_index, observation_wavenumber in enumerate(observation_wavenumbers):
        index = np.abs(observation_wavenumber - wavenumber_grid_in).argmin()
        
        coefficients = coefficient_grid_in[index, :]
        correct_solar_counts = np.polyval(coefficients, observation_temperature)
        corrected_solar_spectrum[pixel_index] = correct_solar_counts
    
    return corrected_solar_spectrum


