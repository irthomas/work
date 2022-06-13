# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:19:22 2022

@author: iant
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tools.file.paths import paths
from tools.spectra.baseline_als import baseline_als
# from tools.spectra.smooth_hr import smooth_hr
from tools.spectra.savitzky_golay import savitzky_golay
from tools.spectra.fft_zerofilling import fft_hr_nu_spectrum


def uvis_solar_hr(remove_continuum=True):
    
    
    solar_data = np.loadtxt(os.path.join(paths["REFERENCE_DIRECTORY"], "meftah_2018_spectrum.dat"), skiprows=1)
    
    nm = solar_data[:, 0]
    solar_hr = solar_data[:, 1]
    continuum = baseline_als(solar_hr, lam=1000, p=0.95)

    if remove_continuum:
        return nm, solar_hr/continuum
    else:
        return nm, solar_hr
    

def uvis_solar_lr():

    nm, solar = uvis_solar_hr(remove_continuum=False)
    
    #binning by 5
    solar2 = solar[1::].reshape(2040, -1).mean(axis=1)
    nm2 = nm[1::].reshape(2040, -1).mean(axis=1)
    
    cont = baseline_als(solar2, lam=1000, p=0.95)

    return nm2, solar2/cont


def uvis_solar_superhr(zerofilling=10):

    nm, solar_hr = uvis_solar_hr()
    nm_fft, solar_hr_fft = fft_hr_nu_spectrum(nm, solar_hr, zerofilling=zerofilling)

    return nm_fft, solar_hr_fft


def uvis_solar_sg():
    """best simulation of UVIS nadir spectra"""
    
    nm, solar_hr = uvis_solar_hr(remove_continuum=False)
    continuum = baseline_als(solar_hr, lam=100000, p=0.95)
    
    sg = savitzky_golay(solar_hr/continuum, 29, 1)
    
    return nm, sg

# plt.figure()
# nm, solar_hr = uvis_solar_hr()
# plt.plot(nm, solar_hr)
# nm_fft, solar_hr_fft = uvis_solar_superhr()
# plt.plot(nm_fft, solar_hr_fft)
