# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:40:27 2020

@author: iant
"""

def fft_zerofilling(row, filling_amount):
    """apply fft, zero fill by a multiplier then reverse fft 
    to give very high resolution spectrum"""
    import numpy as np

    n_pixels = len(row)
    
    rowrfft = np.fft.rfft(row, len(row))
    rowzeros = np.zeros(n_pixels * filling_amount, dtype=np.complex)
    rowfft = np.concatenate((rowrfft, rowzeros))
    row_hr = np.fft.irfft(rowfft).real #get real component for reversed fft
    row_hr *= len(row_hr)/len(row) #need to scale by number of extra points

    pixels_hr = np.linspace(0, n_pixels, num=len(row_hr))    
    return pixels_hr, row_hr

	
def fft_hr_nu_spectrum(wavenumbers, spectrum, zerofilling=10):
    """apply fft to spectrum to make high res spectrum and wavenumber grid"""
    import numpy as np

    px_hr, abs_hr = fft_zerofilling(spectrum, zerofilling)
    wavenumbers_hr = np.interp(px_hr, np.arange(len(spectrum)), wavenumbers)
    return wavenumbers_hr, abs_hr
