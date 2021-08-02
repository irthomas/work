# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:40:17 2021

@author: iant

IMPROVE SPEED BY MAKING APPROXIMATIONS
REFORMULATE WITH BLAZE/AOTF TOGETHER
"""


import numpy as np
import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
# import scipy.signal as ss
# import lmfit

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist
# from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict
from tools.file.paths import paths

# from tools.sql.get_sql_spectrum_temperature import get_sql_temperatures_all_spectra

from tools.spectra.solar_spectrum import get_solar_hr
# from tools.spectra.baseline_als import baseline_als
# from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
# from tools.spectra.fit_polynomial import fit_polynomial

# from tools.general.get_nearest_index import get_nearest_index

# from instrument.nomad_so_instrument import nu_grid, nu_mp, spec_res_order, F_aotf_goddard18b, t_nu_mp
# from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

# from instrument.nomad_so_instrument import m_aotf as m_aotf_so





"""old functions"""
#new values from mean gradient/offset of best LNO orders (SO should be same): 142, 151, 156, 162, 166, 167, 178, 189, 194
Q0=-10.13785778
Q1=-0.829174444
Q2=0.0
def t_p0(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0


F0=22.473422
F1=5.559526e-4
F2=1.751279e-8
def nu_mp(m, p, t, p0=None, F0=F0, F1=F1, F2=F2):
    """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
    if p0 == None:
        p0 = t_p0(t)
    f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
    return f




"""2021 new functions"""
def F_blaze(p, p0, p_width):
    
    dp = p - p0
    dp[dp == 0.0] = 1.0e-6
    F = (p_width*np.sin(np.pi*dp/p_width)/(np.pi*dp))**2
    
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
    print("blazew=", blazew)
    dx = xdat - blaze0
    dx[blazep0] = 1.0e-6
    F = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2

    return F


def F_aotf(nu, A_nu0, width, lobe, asym):
    """reverse AOTF asymmetry"""
    def sinc(dx, amp, width, lobe, asym):
        # """asymetry switched 
     	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2

     	ind = (abs(dx)>width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*lobe

     	ind = (dx>=width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*asym

     	return sinc

    dx = nu - A_nu0
    F = sinc(dx, 1.0, width, lobe, asym)
    
    return F







file_level = "hdf5_level_0p2a"
regex = re.compile("20190416_020948_0p2a_SO_1_C")

# hdf5_filename = '20190416_020948_0p2a_SO_1_C'


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)
hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]
print(hdf5_filename)




channel = hdf5_filename.split("_")[3].lower()
aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
detector_data_all = hdf5_file["Science/Y"][...]
detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
spectra = np.mean(detector_centre_data, axis=1)

index = 0
spectrum = spectra[index]
A=aotf_freq[index]


F0=22.473422
D_NU = 0.005
ORDER_RANGE = [192, 196]

pixels = np.arange(320)
dnu = D_NU

m = 194
m_range = ORDER_RANGE

t = -10.75794428682947
spec_res = 0.3854474053055081
sconv = spec_res/2.355




"""spectral grid and blaze functions of all orders"""
nu_range = [4309.7670539950705, 4444.765043408191]
nu_hr = np.arange(nu_range[0], nu_range[1], dnu)




"""solar ref"""
ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)

#pre-convolute
I0_lr = savgol_filter(I0_solar_hr, 99, 1)
# pixels_nu = nu_mp(order, pixels, temperature)




"""blaze"""
#pre-compute delta nu per pixel
nu_mp_centre = nu_mp(m, pixels, t, p0=0)
p_dnu = (nu_mp_centre[-1] - nu_mp_centre[0])/320.0


p0 = np.polyval([0.22,150.8], m)
p_width = F0 / p_dnu

F = F_blaze(pixels, p0, p_width)




"""aotf"""
A_nu0 = np.polyval([1.34082e-7, 0.1497089, 305.0604], A)
width  = np.polyval([1.11085173e-06, -8.88538288e-03,  3.83437870e+01], A_nu0)
lobe  = np.polyval([2.87490586e-06, -1.65141511e-02,  2.49266314e+01], A_nu0)
asym  = np.polyval([-5.47912085e-07, 3.60576934e-03, -4.99837334e+00], A_nu0)
G = F_aotf(nu_hr, A_nu0, width, lobe, asym)


#old code
Nbnu_hr = len(nu_hr)
NbP = len(pixels)

W_conv = np.zeros((NbP,Nbnu_hr))
for im in range(m_range[0], m_range[1]+1):
    print("Blaze order %i" %im)
    nu_pm = nu_mp(im, pixels, t)
    W_blaze = F_blaze_goddard21(im, pixels, t)
    for ip in pixels:
        W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
        


# plt.plot(nu_hr, W_conv[[130,150,170,190,210], :].T)


I0_hr = G * I0_solar_hr
I0_p = np.matmul(W_conv, I0_hr)
solar_scaled = I0_p/max(I0_p)


#new code
#for each order
#for each pixel

#get blaze for pixel
#get wavenumber for pixel
#get aotf for pixel
#get LR solar spectrum for pixel

nu_pm_c = nu_mp(m, pixels, t)

solar = np.zeros((NbP))
for im in range(m_range[0], m_range[1]+1):

    nu_pm = nu_mp(im, pixels, t)
    
    F2 = F_blaze(pixels, p0, p_width)
    G2 = F_aotf(nu_pm, A_nu0, width, lobe, asym)
    I0_lr_p = np.interp(nu_pm, nu_hr, I0_lr)
    
    solar += F2 * G2 * I0_lr_p
    
    plt.plot(nu_pm, F2, label="Blaze %i" %im)
    plt.plot(nu_pm, G2, "--", label="AOTF %i" %im)
    plt.plot(nu_pm, I0_lr_p/max(I0_lr_p), label="Solar %i" %im)

plt.plot(nu_pm_c, spectrum/max(spectrum), "r--")

plt.plot(nu_pm_c, solar/max(solar), "k--")
plt.plot(nu_pm_c, solar_scaled/max(solar_scaled), "k:")
plt.legend()



