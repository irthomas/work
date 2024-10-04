# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:26:43 2022

@author: iant

SO calibration following 2022 papers
"""

import numpy as np


"""
relationship between the grating peak FSR and AOTF frequency as a 3rd degree polynomial
FSRpeak = P0 + P1dv + P2dv2 + P3dv3, where dv is vAOTF-3700 cm-1, and P0, P1, P2, P3 are 2.25863468E+01, 9.79270239E-06, -7.20616355E-09, and -1.00162255E-11 respectively 

We investigated the effects of temperature on the grating by analyzing the variability of FSRpeak 
FSR’ = FSR·[1+K(T)], where K(T) is a scaling correction factor for temperature T [ºC]
In our analysis of all calibration data from 2018 to 2021 of FSRpeak from the full-scans and the line-positions from the mini-scans, 
we observe the same K(T) function, which we determine to be: K(T) = K0 + K1T + K2T2, where K0, K1 and K2 are -1.90001923E-04, -2.30708836E-05, and -2.44383699E-07 

Pixel spectral cal from Loic 2022

AOTF central wavenumber + temperature shift from 2021 analysis

"""


def lt22_p0_shift(t):
    """first pixel (temperature shift) Loic Feb 22"""

    p0 = -0.8276 * t  # px/°C * T(interpolated)
    return p0


def lt22_waven(order, t, channel="so", coeffs=False, px_ixs=np.arange(320.0)):
    """spectral calibration Loic Feb 22. Get pixel wavenumbers from order + temperature"""

    px_shifted = px_ixs + lt22_p0_shift(t)

    cfpixel = {"so": [3.32e-8, 5.480e-4, 22.4701], "lno": [3.32e-8, 5.480e-4, 22.4701]}
    xdat = np.polyval(cfpixel[channel], px_shifted) * order

    if coeffs:
        return cfpixel[channel]

    else:
        return xdat


def aotf_peak_nu(aotf_freq, t, channel="so"):
    """Villanueva 2022. Get aotf peak wavenumber from frequency and temperature"""

    cfaotf = {"so": [1.34082e-7, 0.1497089, 305.0604],                   # Frequency of AOTF [cm-1 from kHz]
              "lno": [9.409476e-8, 0.1422382, 300.67657]}  # Old values
    aotfts = {"so": -6.5278e-5,                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
              "lno": -6.5278e-5}

    aotf_nu = np.polyval(cfaotf[channel], aotf_freq)
    aotf_nu += aotfts[channel] * t * aotf_nu

    return aotf_nu


def blaze_peak_nu_order(aotf_nu, t):
    """Villanueva 2022. Get blaze peak wavenumber / order from aotf peak wavenumber and temperature"""

    blaze_peak_coeffs = [-1.00162255E-11, -7.20616355E-09, 9.79270239E-06, 2.25863468E+01]
    blaze_t_shift_coeffs = [-2.44383699E-07, -2.30708836E-05, -1.90001923E-04]

    d_nu = aotf_nu - 3700.0
    fsr_nu = np.polyval(blaze_peak_coeffs, d_nu)

    kt = np.polyval(blaze_t_shift_coeffs, t)

    fsr_t_shifted = fsr_nu * (1.0 + kt)

    return fsr_t_shifted


def aotf_func(aotf_nu, aotf_range=200.0, step_nu=0.1):

    def sinc_gd(dx, width, lobe, asym, offset):
        # goddard version
        sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
        ind = (abs(dx) > width).nonzero()[0]
        if len(ind) > 0:
            sinc[ind] = sinc[ind]*lobe
        ind = (dx <= -width).nonzero()[0]
        if len(ind) > 0:
            sinc[ind] = sinc[ind]*asym
        sinc += offset
        return sinc

    def F_aotf3(dx, width, lobe, asym, offset, offset_width):

        offset = offset * np.exp(-dx**2.0 / (2.0 * offset_width**2.0))
        sinc = sinc_gd(dx, width, lobe, asym, offset)
        return sinc

    # from github
    aotfwc = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01]  # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc = [8.10749274e-07, -3.30238496e-03,  4.08845247e+00]  # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00]  # Asymmetry factor [scaler from AOTF frequency cm-1]
    # aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc = [1.49266526e-07, -9.63798656e-04,  1.60097815e+00]  # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

    width = np.polyval(aotfwc, aotf_nu)
    lobe = np.polyval(aotfsc, aotf_nu)
    asym = np.polyval(aotfac, aotf_nu)
    offset = np.polyval(aotfgc, aotf_nu)
    offset_width = 50.  # offset width cm-1

    nu = np.arange(-1.0 * aotf_range, aotf_range + step_nu, step_nu)
    F_aotf = F_aotf3(nu, width, lobe, asym, offset, offset_width)

    F_aotf /= np.max(F_aotf)

    return nu, F_aotf


"""order to aotf frequency"""
A_aotf = {
    110: 14332, 111: 14479, 112: 14627, 113: 14774, 114: 14921, 115: 15069, 116: 15216, 117: 15363, 118: 15510, 119: 15657,
    120: 15804, 121: 15951, 122: 16098, 123: 16245, 124: 16392, 125: 16539, 126: 16686, 127: 16832, 128: 16979, 129: 17126,
    130: 17273, 131: 17419, 132: 17566, 133: 17712, 134: 17859, 135: 18005, 136: 18152, 137: 18298, 138: 18445, 139: 18591,
    140: 18737, 141: 18883, 142: 19030, 143: 19176, 144: 19322, 145: 19468, 146: 19614, 147: 19761, 148: 19907, 149: 20052,
    150: 20198, 151: 20344, 152: 20490, 153: 20636, 154: 20782, 155: 20927, 156: 21074, 157: 21219, 158: 21365, 159: 21510,
    160: 21656, 161: 21802, 162: 21947, 163: 22093, 164: 22238, 165: 22384, 166: 22529, 167: 22674, 168: 22820, 169: 22965,
    170: 23110, 171: 23255, 172: 23401, 173: 23546, 174: 23691, 175: 23836, 176: 23981, 177: 24126, 178: 24271, 179: 24416,
    180: 24561, 181: 24706, 182: 24851, 183: 24996, 184: 25140, 185: 25285, 186: 25430, 187: 25575, 188: 25719, 189: 25864,
    190: 26008, 191: 26153, 192: 26297, 193: 26442, 194: 26586, 195: 26731, 196: 26875, 197: 27019, 198: 27163, 199: 27308,
    200: 27452, 201: 27596, 202: 27740, 203: 27884, 204: 28029, 205: 28173, 206: 28317, 207: 28461, 208: 28605, 209: 28749,
    210: 28893,
}

"""aotf frequency to nearest order"""


def m_aotf(value):
    res_key, res_val = min(A_aotf.items(), key=lambda x: abs(value - x[1]))
    return res_key


def aotf_freq_to_order(aotf_freq):
    """aotf frequency kHz to nearest order"""
    res_key, res_val = min(A_aotf.items(), key=lambda x: abs(aotf_freq - x[1]))
    return res_key


def aotf_nu_to_order(aotf_nu):  # aotf in khz
    """AOTF wavenumber to nearest diffraction order"""
    order = np.round(np.polyval([-1.33960735E-11, 4.43204703E-02, -1.75036273E-04], aotf_nu))

    if order < 80:  # code to detect dark frames and change order to zero
        order = 0
    return order
