# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:41:37 2021

@author: iant

MINISCAN AOTF FITTING
FIT BEST PARAMETERS FOR 
"""


import re
import numpy as np
import lmfit



import matplotlib.pyplot as plt

from instrument.nomad_so_instrument import nu_mp


from instrument.calibration.so_aotf_ils.simulation_functions import get_file, get_data_from_file, select_data, fit_temperature#, calc_blaze, aotf_conv


from instrument.calibration.so_aotf_ils.simulation_config import ORDER_RANGE, pixels




regex = re.compile("20190416_020948_0p2a_SO_1_C")


"""spectral grid and blaze functions of all orders"""
nu_range = [4309.7670539950705, 4444.765043408191]
dnu = 0.005
nu_hr = np.arange(nu_range[0], nu_range[1], dnu)


hdf5_file, hdf5_filename = get_file(regex)
d = get_data_from_file(hdf5_file, hdf5_filename)

m_range = ORDER_RANGE

def F_blaze(variables):
    
    dp = pixels - variables["blaze_centre"]
    dp[dp == 0.0] = 1.0e-6
    F = (variables["blaze_width"]*np.sin(np.pi*dp/variables["blaze_width"])/(np.pi*dp))**2
    
    return F


def F_aotf(nu_pm, variables):
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

    dx = nu_pm - A_nu0 - variables["aotf_shift"]
    F = sinc(dx, 1.0, variables["aotf_width"], variables["sidelobe"], variables["asymmetry"]) + variables["offset"]
    
    return F
    # return F/max(F)






def calc(variables, I0=[0]):
    
    
    if I0[0] == 0:
        I0 = I0_lr
    
    solar = np.zeros(len(pixels))
    for im in range(m_range[0], m_range[1]+1):

        nu_pm = nu_mp(im, pixels, t)
        
        F = F_blaze(variables)
        G = F_aotf(nu_pm, variables)
        I0_lr_p = np.interp(nu_pm, nu_hr, I0)
        
        solar += F * G * I0_lr_p
    
    return solar/max(solar)




def fit_resid(params, spectrum_norm, sigma):

    variables = {}
    for key in params.keys():
        variables[key] = params[key].value

    return (calc(variables) - spectrum_norm) / sigma


def trapezium_area(y1, y2, dx=1.0):
    return 0.5 * (y1 + y2) * dx
    

def area_under_curve(curve1, curve2, dx):
    area = 0
    for i in range(len(curve1)-1):
        y1 = curve1[i] - curve2[i]
        y2 = curve1[i+1] - curve2[i+1]
    
        area += trapezium_area(y1, y2)
    return area

    
variables_fit = {"A":[], "A_nu0":[], "solar_line_area":[], "chisq":[]}
# for key, value in param_dict.items():
#     variables_fit[key] = []




for index in range(0, 256, 1):

    d = select_data(d, index)
    d["nu_hr"] = nu_hr
    d = fit_temperature(d, hdf5_file)
    # d = calc_blaze(d)
    
    m = d["centre_order"]
    A = d["aotf_freqs"][index]
    print("A=%0.1f kHz" %A)
    t = d["temperature"]
    I0_lr = d["I0_lr"]
    
    I0_lr_slr = d["I0_lr_slr"]
    
    """compute parameters from fits"""
    """blaze"""
    #pre-compute delta nu per pixel
    nu_mp_centre = nu_mp(m, pixels, d["temperature"], p0=0)
    p_dnu = (nu_mp_centre[-1] - nu_mp_centre[0])/320.0
    
    
    p0 = np.polyval([0.22,150.8], m)
    p_width = 22.473422 / p_dnu
    
    
    """aotf"""
    A_nu0 = np.polyval([1.34082e-7, 0.1497089, 305.0604], A)
    width  = np.polyval([1.11085173e-06, -8.88538288e-03,  3.83437870e+01], A_nu0)
    lobe  = np.polyval([2.87490586e-06, -1.65141511e-02,  2.49266314e+01], A_nu0)
    asym  = np.polyval([-5.47912085e-07, 3.60576934e-03, -4.99837334e+00], A_nu0)
    
    
    #best, min, max
    param_dict = {
        "blaze_centre":[p0, p0-20.0, p0+20.],
        "blaze_width":[p_width, p_width+20., p_width-20.],
        "aotf_width":[width, width-2., width+2.],
        "aotf_shift":[0.0, -10.0, 10.0],
        "sidelobe":[lobe, 0.05, 20.0],
        "asymmetry":[asym, 0.01, 2.0],
        "offset":[0.0, 0.0, 0.3],
        }
    
    
    
    
    
    #set up initial variables
    variables = {}
    for key, value in param_dict.items():
        variables[key] = value[0]

    nu_pm_c = nu_mp(m, pixels, t)
    
    aotf_nu_range = np.arange(nu_range[0], nu_range[1], 0.1)
    
    W_aotf = F_aotf(aotf_nu_range, variables)
    W_blaze = F_blaze(variables)
    
    
    # print("Fitting AOTF and blaze")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, value[0], min=value[1], max=value[2])
       
    sigma = np.ones_like(d["spectrum_norm"])
    smi = d["absorption_pixel"]
    sigma[smi-8:smi+9] = 0.01
    sigma[:50] = 10.
    sigma[300:] = 10.
    
    lm_min = lmfit.minimize(fit_resid, params, args=(d["spectrum_norm"], sigma), method='leastsq')
    chisq = lm_min.chisqr
    print("chisq=", chisq)
    for key in params.keys():
        variables[key] = lm_min.params[key].value
    
    # print(variables)
    
    
    solar_fit = calc(variables)
    solar_fit_slr = calc(variables, I0=I0_lr_slr)
    
    #calculate area between curves

    sl_extent_p = [204, 228]
    sl_extent_p_indices = np.arange(sl_extent_p[0], sl_extent_p[1]+1)
    area = area_under_curve(solar_fit_slr[sl_extent_p_indices], solar_fit[sl_extent_p_indices], 1)
    
    plt.figure()
    # plt.plot(solar_fit-solar_fit_slr)
    # variables_fit["aotf_freq"] = A
    # for key in variables.keys():
    #     variables_fit[key].append(variables[key])
    
    plt.title(A)
    plt.plot(pixels, solar_fit)
    plt.plot(pixels, d["spectrum_norm"])
    plt.fill_between(pixels, y1=solar_fit, y2=solar_fit_slr, linestyle="--", alpha=0.5)
    plt.savefig("%s_%i_%ikHz.png" %(hdf5_filename, index, A))
    
    plt.close()

    variables_fit["A"].append(A)
    variables_fit["A_nu0"].append(A_nu0)
    variables_fit["solar_line_area"].append(area)
    variables_fit["chisq"].append(chisq)
    for key in variables.keys():
        if key not in variables_fit.keys():
            variables_fit[key] = []
        variables_fit[key].append(variables[key])

    # stop()

plt.figure()
plt.plot(variables_fit["A_nu0"], variables_fit["solar_line_area"]/max(variables_fit["solar_line_area"]), label="AOTF")
for key in variables.keys():
    plt.plot(variables_fit["A_nu0"], variables_fit[key] / max(variables_fit[key]), label=key)
plt.legend()

plt.figure()
plt.errorbar(variables_fit["A_nu0"], variables_fit["solar_line_area"]/max(variables_fit["solar_line_area"]), yerr=(variables_fit["chisq"]/np.float64(1000.)))