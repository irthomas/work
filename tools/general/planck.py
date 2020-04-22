# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:36:29 2020

@author: iant
"""
import numpy as np
import os


def planck(xscale, temp, units): 
    """return planck function with units W/cm2/sr/spectral unit
    for given xscale and temperature"""

    if units=="microns" or units=="um" or units=="wavel":
        c1=1.191042e8
        c2=1.4387752e4
        return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0) / 1.0e4 # m2 to cm2
    elif units=="wavenumbers" or units=="cm-1" or units=="waven":
        c1=1.191042e-5
        c2=1.4387752
        return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2
    else:
        print("Error: Unknown units given")


def planck_solar(xscale): 
    """get solar spectrum in units W/cm2/sr/cm-1 for given xscale in cm-1"""

    SOLAR_SPECTRUM = np.loadtxt("reference_files"+os.sep+"nomad_solar_spectrum_solspec.txt")
    solarRad = np.zeros(len(xscale))
        
    wavenumberInStart = SOLAR_SPECTRUM[0,0]
    wavenumberDelta = 0.005
            
    print("Finding solar radiances in ACE file")
    for pixelIndex,xValue in enumerate(xscale):
        index = np.int((xValue - wavenumberInStart)/wavenumberDelta)
        if index == 0:
            print("Warning: wavenumber out of range of solar file (start of file). wavenumber = %0.1f" %(xValue))
        if index == len(SOLAR_SPECTRUM[:,0]):
            print("Warning: wavenumber out of range of solar file (end of file). wavenumber = %0.1f" %(xValue))
        solarRad[pixelIndex] = SOLAR_SPECTRUM[index,1]
    return solarRad / 1.0e4 #m2 to cm2
            
        

def planck_solar_irradiance(xscale): 
    """don't use. Requires fudge factor"""

    SOLAR_SPECTRUM = np.loadtxt("reference_files"+os.sep+"nomad_solar_spectrum_solspec.txt")
    RADIANCE_TO_IRRADIANCE = 8.77e-5 / 100.0**2 #fudge to make curves match. should be 2.92e-5 on mars, 6.87e-5 on earth
    try:
        solarRad = np.zeros(len(xscale))
    except:
        xscale = [xscale]
        solarRad = np.zeros(len(xscale))
        
    wavenumberInStart = SOLAR_SPECTRUM[0,0]
    wavenumberDelta = 0.005
            
    print("Finding solar radiances in ACE file")
    for pixelIndex,xValue in enumerate(xscale):
        index = np.int((xValue - wavenumberInStart)/wavenumberDelta)
        if index == 0:
            print("Warning: wavenumber out of range of solar file (start of file). wavenumber = %0.1f" %(xValue))
        if index == len(SOLAR_SPECTRUM[:,0]):
            print("Warning: wavenumber out of range of solar file (end of file). wavenumber = %0.1f" %(xValue))
        solarRad[pixelIndex] = SOLAR_SPECTRUM[index,1] / RADIANCE_TO_IRRADIANCE #/ 1.387 #1AU to MCO-1 AU
    return solarRad / 1.0e4 #m2 to cm2
            
