# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:04:59 2023

@author: iant
"""


import numpy as np


def lt22_p0_shift(t):
    """first pixel (temperature shift) Loic Feb 22"""
    
    p0 = -0.8276 * t #px/°C * T(interpolated)
    return p0



def lt22_waven(order, t, channel="so", coeffs=False):
    """spectral calibration Loic Feb 22. Get pixel wavenumbers from order + temperature"""
    
    px_shifted = np.arange(320.0) + lt22_p0_shift(t)

    cfpixel = {"so":[3.32e-8, 5.480e-4, 22.4701], "lno":[3.32e-8, 5.480e-4, 22.4701]}
    xdat  = np.polyval(cfpixel[channel], px_shifted) * order
    
    if coeffs:
        return cfpixel[channel]
    
    else:
        return xdat


def aotf_peak_nu(aotf_freq, t, channel="so"):
    """Villanueva 2022. Get aotf peak wavenumber from frequency and temperature"""

    cfaotf  = {"so":[1.34082e-7, 0.1497089, 305.0604],                   # Frequency of AOTF [cm-1 from kHz]
               "lno":[9.409476e-8, 0.1422382, 300.67657]} #Old values
    aotfts  = {"so":-6.5278e-5,                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
               "lno":-6.5278e-5}


    aotf_nu  = np.polyval(cfaotf[channel], aotf_freq)
    aotf_nu += aotfts[channel] * t * aotf_nu

    return aotf_nu



def get_orders_nu(channel, orders, t, px_ixs=np.arange(320)):
    
    orders_d = {}
    for order in orders:
        if channel == "so":
            px_nus = lt22_waven(order, t)[px_ixs]
        if channel == "lno":
            px_nus = nu_mp(order, px_ixs, t)
        
        orders_d[order] = {"px_nus":px_nus, "px_ixs":px_ixs}

    return orders_d



"""LNO"""
#updated for LNO from Liuzzi et al. May not be newest coefficients?
G0=300.67657
G1=0.1422382
G2=9.409476e-8
def nu0_aotf(A, G0=G0, G1=G1, G2=G2):
    """aotf frequency to aotf centre in wavenumbers. Liuzzi 2019"""
    nu0 = G0 + A*(G1 + A*G2)
    return nu0

#new values from LNO fullscan occultation analysis Jan 2023
Q0=0.0
Q1=-0.851152813531944
Q2=0.0
def t_p0(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0


#updated for LNO Jan 2023
F0=22.471896082567795
F1=0.0005612789704190822
F2=3.942616548909985e-09
def nu_mp(m, p, t, F0=F0, F1=F1, F2=F2):
    """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
    p0 = t_p0(t)
    f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
    return f
