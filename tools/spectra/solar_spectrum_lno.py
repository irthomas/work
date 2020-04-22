# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:43:33 2020

@author: iant
"""
import numpy as np
from scipy import interpolate

from instrument.nomad_lno_instrument import nu_mp, spec_res_order, F_blaze, F_aotf_goddard19draft




def get_solar_hr(nu_hr, solspec_filepath):
    """get high res solar spectrum interpolated to input nu_hr wavenumber grid"""

    nu_hr_min = np.min(nu_hr)
    nu_hr_max = np.max(nu_hr)
    
    with open(solspec_filepath, "r") as f:
        nu_solar = []
        I0_solar = []
        for line in f:
            nu, I0 = [float(val) for val in line.split()]
            if nu < nu_hr_min - 2.:
                continue
            if nu > nu_hr_max + 2.:
                break
            nu_solar.append(nu)
            I0_solar.append(I0)
    f_solar = interpolate.interp1d(nu_solar, I0_solar)
    I0_solar_hr = f_solar(nu_hr)
    return I0_solar_hr




def nu_hr_grid(diffraction_order, adj_orders, instrument_temperature):
    """make high res wavenumber grid for given diffraction order +- n adjacent orders"""
    nu_hr_min = nu_mp(diffraction_order - adj_orders, 0, instrument_temperature) - 5.
    nu_hr_max = nu_mp(diffraction_order + adj_orders, 320., instrument_temperature) + 5.
    dnu = 0.001
    Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
    nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
    dnu = nu_hr[1]-nu_hr[0]
    return nu_hr, dnu
    
    
def solar_hr_orders(diffraction_order, adj_orders, instrument_temperature, solspec_file):
    """get high res solar spectrum for given diffraction order +- n adjacent orders"""
    nu_hr, dnu = nu_hr_grid(diffraction_order, adj_orders, instrument_temperature)
    I0_solar_hr = get_solar_hr(nu_hr, solspec_file)
    
    return nu_hr, I0_solar_hr, dnu



def lno_solar_spectrum(diffraction_order, instrument_temperature, solspec_file, adj_orders=2):
    """get low res LNO channel simulated solar spectrum for given diffraction order"""
       
    spec_res = spec_res_order(diffraction_order)
    pixels = np.arange(320)
    
    nu_hr, I0_solar_hr, dnu = solar_hr_orders(diffraction_order, adj_orders, instrument_temperature, solspec_file)
    Nbnu_hr = len(nu_hr)
    
    NbP = len(pixels)
    
    W_conv = np.zeros((NbP,Nbnu_hr))
    sconv = spec_res/2.355
    for iord in range(diffraction_order-adj_orders, diffraction_order+adj_orders+1):
        nu_pm = nu_mp(iord, pixels, instrument_temperature)
        W_blaze = F_blaze(iord, pixels, instrument_temperature)
        for ip in pixels:
            W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
    
    W_aotf = F_aotf_goddard19draft(diffraction_order, nu_hr, instrument_temperature)
    I0_hr = W_aotf * I0_solar_hr
    I0_p = np.matmul(W_conv, I0_hr)
    
    return nu_pm, I0_p
    



def lno_solar_line_temperature_shift(diffraction_order, instrument_temperatures, solspec_file, adj_orders=2, cutoff=0.999):
    """unfinished: get LNO spectrum due to temperature induced solar line shift"""
       
    spec_res = spec_res_order(diffraction_order)
    pixels = np.arange(320)
    
    instrument_temperature  = instrument_temperatures[0]
    
    nu_hr, I0_solar_hr, dnu = solar_hr_orders(diffraction_order, adj_orders, instrument_temperature, solspec_file)
    Nbnu_hr = len(nu_hr)
    
    NbP = len(pixels)
    W_conv = np.zeros((NbP,Nbnu_hr))
    sconv = spec_res/2.355
    for iord in range(diffraction_order-adj_orders, diffraction_order+adj_orders+1):
        nu_pm = nu_mp(iord, pixels, instrument_temperature)
        W_blaze = F_blaze(iord, pixels, instrument_temperature)
        for ip in pixels:
            W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
    
    
    
    W_aotf = F_aotf_goddard19draft(diffraction_order, nu_hr, instrument_temperature)
    I0_hr = W_aotf * I0_solar_hr
    I0_p = np.matmul(W_conv, I0_hr)
    
    #plt.plot(nu_hr, I0_solar_hr)
        #plt.plot(nu_hr, I0_hr)
        #plt.plot(nu_hr, W_aotf)
    #    plt.plot(pixels, I0_p)
    
    instrument_temperature  = instrument_temperatures[1]
    
    nu_hr2, I0_solar_hr2, dnu2 = solar_hr_orders(diffraction_order, adj_orders, instrument_temperature, solspec_file)
    Nbnu_hr2 = len(nu_hr2)
    
    NbP2 = len(pixels)
    W_conv2 = np.zeros((NbP2,Nbnu_hr2))
    sconv2 = spec_res/2.355
    for iord in range(diffraction_order-adj_orders, diffraction_order+adj_orders+1):
        nu_pm2 = nu_mp(iord, pixels, instrument_temperature)
        W_blaze2 = F_blaze(iord, pixels, instrument_temperature)
        for ip in pixels:
            W_conv2[ip,:] += (W_blaze2[ip]*dnu2)/(np.sqrt(2.*np.pi)*sconv2)*np.exp(-(nu_hr2-nu_pm2[ip])**2/(2.*sconv2**2))
    
    
    
    W_aotf2 = F_aotf_goddard19draft(diffraction_order, nu_hr2, instrument_temperature)
    I0_hr2 = W_aotf2 * I0_solar_hr2
    I0_p2 = np.matmul(W_conv2, I0_hr2)
    
    ratio = I0_p / I0_p2
    
    polyfit = np.polyval(np.polyfit(pixels, ratio, 5), pixels)
    
    normalisedRatio = ratio/polyfit
    
    return normalisedRatio
    
    
