# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:54:14 2021

@author: iant

CALCULATE BEST AOTF CENTRAL WAVENUMBER
"""

cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
ipix  = range(320)
xdat  = np.polyval(cfpixel,ipix)*order
xdat += xdat*np.polyval(ncoeff, tmeds[i])