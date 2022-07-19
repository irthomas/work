# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:31 2022

@author: iant
"""

import numpy as np




def F_blaze3(x, blazef, blazew):
    
    dx = x - blazef
    F = np.sinc((dx) / blazew)**2
    return F



def so_cal(aotf_khz, t):
    """
    #FSRpeak = P0 + P1dv + P2dv2 + P3dv3, where dv is vAOTF 3700 cm 1, and P0, P1, P2, P3 are 2.25863468E+01, 9.79270239E 06,  7.20616355E 09, and  1.00162255E 11 respectively 
    #FSRi = vi/m = F0 + F1·i + F2·i2, where i is the pixel number (0 to 319), vi is the frequency at pixel i,
    #m is the order, and the coefficients F0, F1 and F2 are 2.24734E+01, 5.55953E 04, 1.75128E 08 respectively.
    #The order number (m) is simply the AOTF frequency (in wavenumbers) divided by the FSR at that pixel.
    #grooves spacing should expand as FSR’ = FSR·[1+K(T)], where K(T) is a scaling correction factor for temperature T [ºC]. In our analysis of all calibration data from 2018 to 2021 of FSRpeak from the full-scans and the line-positions from the mini-scans, we observe the same K(T) function, which we determine to be: K(T) = K0 + K1T + K2T2, where K0, K1 and K2 are  1.90001923E 04,  2.30708836E 05, and  2.44383699E 07 respectively 
    """

    d = {}
    
    def sinc_gd(dx, width, lobe, asym, offset):
        """new spectral calibration functions Aug/Sep 2021"""
        #goddard version
        sinc = (width * np.sin(np.pi * dx / width)/(np.pi * dx))**2.0
        ind = (abs(dx) > width).nonzero()[0]
        if len(ind) > 0: sinc[ind] = sinc[ind] * lobe
        ind = (dx <= -width).nonzero()[0]
        if len(ind) > 0: sinc[ind] = sinc[ind] * asym
        sinc += offset
        return sinc
    
    
    def F_aotf3(dx, d):
        offset = d["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotfgw"]**2.0))
        sinc = sinc_gd(dx, d["aotfw"], d["aotfs"], d["aotfa"], offset)
        return sinc
       
    
    #AOTF in cm-1
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])                  # Frequency of AOTF [cm-1 from kHz]
    aotff = np.polyval(cfaotf, aotf_khz)
    
    #AOTF t correction
    aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    aotff += aotfts * t
    
    nu_hr = np.arange(aotff - 50.0, aotff + 50.0, 0.001)
    


    #AOTF shape
    #2nd october slack
    aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

    d["aotfw"] = np.polyval(aotfwc, aotff)
    d["aotfs"] = np.polyval(aotfsc, aotff)
    d["aotfa"] = np.polyval(aotfac, aotff)
    d["aotfo"] = np.polyval(aotfoc,aotff)
    d["aotfg"] = np.polyval(aotfgc,aotff)
    d["aotfgw"] = 50. #offset width cm-1

    #make AOTF
    dx = nu_hr - aotff
    d["F_aotf"] = F_aotf3(dx, d)


    # Calibration coefficients
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])            # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency


    blazew =  np.polyval(blazep, aotff - 3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    blazew += blazew * np.polyval(ncoeff, t)        # FSR, corrected for temperature
    d["blazew"] = blazew


    # Frequency of the pixels
    for order in orders:
        d[order] = {}
        pixf = np.polyval(cfpixel,range(320))*order
        pixf += pixf*np.polyval(ncoeff, tempg) + d["blaze_shift"]
        blazef = order*blazew                        # Center of the blaze
        d[order]["pixf"] = pixf
        d[order]["blazef"] = blazef
        # print(order, blazef, blazew)
        
        blaze = F_blaze3(pixf, blazef, blazew)
        d[order]["F_blaze"] = blaze
    
    
        F_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
        nu_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
    
        for i, order in enumerate(d["orders"]):
        
            F_blaze = list(d[order]["F_blaze"])
            F_blazes[i*321:(i+1)*320+i] = F_blaze
            nu_blazes[i*321:(i+1)*320+i] = d[order]["pixf"]
            
        d["F_blazes"] = F_blazes
        d["nu_blazes"] = nu_blazes
        
        d = get_ils_params(d)
        d = blaze_conv(d)
    
    return d
