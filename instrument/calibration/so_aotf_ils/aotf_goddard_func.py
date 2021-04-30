# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:23:57 2021

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt


# from instrument.nomad_so_instrument import nu0_aotf



# # AOTF shape parameters
# aotfwc  = [1.11085173e-06, -8.88538288e-03,  3.83437870e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
# aotfsc  = [2.87490586e-06, -1.65141511e-02,  2.49266314e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
# aotfaf  = [-5.47912085e-07, 3.60576934e-03, -4.99837334e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]

# # Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
# cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])        # Frequency of AOTF [cm-1 from kHz]
# aotfts  = -6.5278e-5                                         # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
# cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
# tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
# blazep  = [0.22,150.8]                                       # Blaze pixel location with order [pixel from order]


aotf_freq = 26614.
temperature = -5.0









def F_aotf_goddard21(m, nu, t, A=None):

    if m != 0.0:
        return [0.0]
    # AOTF shape parameters
    aotfwc  = [1.11085173e-06, -8.88538288e-03,  3.83437870e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [2.87490586e-06, -1.65141511e-02,  2.49266314e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfaf  = [-5.47912085e-07, 3.60576934e-03, -4.99837334e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    
    # Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])        # Frequency of AOTF [cm-1 from kHz]
    aotfts  = -6.5278e-5                                         # AOTF frequency shift due to temperature [relative cm-1 from Celsius]


    def sinc(dx, amp, width, lobe, asym):
    	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2
    	ind = (abs(dx)>width).nonzero()[0]
    	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
    	ind = (dx<=-width).nonzero()[0]
    	if len(ind)>0: sinc[ind] = sinc[ind]*asym
    	return sinc

    nu0 = np.polyval(cfaotf, A)
    nu0 += aotfts * t * nu0

    wd0 = np.polyval(aotfwc, nu0)
    sl0 = np.polyval(aotfsc, nu0)
    af0 = np.polyval(aotfaf, nu0)

    dx = nu - nu0
    F = sinc(dx, 1.0, wd0, sl0, af0)
    
    return F


def F_blaze_goddard21(m, p, t):
    # Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
    blazep  = [0.22,150.8]                                       # Blaze pixel location with order [pixel from order]

    xdat  = np.polyval(cfpixel, p) * m
    dpix = np.polyval(tcoeff, t)
    xdat += dpix * (xdat[-1] - xdat[0]) / 320.0
    
    
    blazep0 = round(np.polyval(blazep, m)) # Center location of the blaze  in pixels
    blaze0 = xdat[blazep0]                    # Blaze center frequency [cm-1]
    blazew = np.polyval(cfpixel, blazep0)      # Blaze width [cm-1]
    dx = xdat - blaze0
    dx[blazep0] = 1.0e-6
    F = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2

    return F
# spec = data[i,:]/(np.max(data[i,:])*blaze)

nu = np.arange(nu0-200, nu0+200, 0.001)
aotf = F_aotf_goddard21(0.0, nu, temperature, A=aotf_freq)

plt.figure()
plt.plot(nu, aotf)





order = round(nu0/(np.polyval(cfpixel,160.0)))
pixels  = np.arange(320.)

blaze = F_blaze_goddard21(order, pixels, temperature)

plt.figure()
plt.plot(pixels, blaze)



