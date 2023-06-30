# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:20:03 2023

@author: iant

CONVOLUTION FUNCTIONS

"""

import numpy as np
import matplotlib.pyplot as plt


# blaze_from_file = True
blaze_from_file = False

# aotf_from_file = True
aotf_from_file = False


if aotf_from_file:
    from analysis.so_lno_2023.functions.aotf_blaze_ils import get_aotf_file as get_aotf
else:
    from analysis.so_lno_2023.functions.aotf_blaze_ils import get_aotf_sinc_gaussian as get_aotf

if blaze_from_file:
    from analysis.so_lno_2023.functions.aotf_blaze_ils import get_blaze_file as get_blaze
else:
    from analysis.so_lno_2023.functions.aotf_blaze_ils import get_blaze_sinc as get_blaze



from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs
from analysis.so_lno_2023.functions.spectral_cal import get_orders, aotf_peak_nu, nu0_aotf






def get_calibration(channel, aotf_freq, centre_order, orders, nomad_t, plot=False):

    cal_d = {}
    
    
    if channel == "so":
        aotf_nu_centre = aotf_peak_nu(aotf_freq, nomad_t)
    elif channel == "lno":
        aotf_nu_centre = nu0_aotf(aotf_freq)
        
        
    cal_d["aotf"] = get_aotf(channel, aotf_nu_centre=aotf_nu_centre)
    
    # cal_d["aotf"]["aotf_nus"] -= 1.5 #offset wavenumbers 
    aotf_nu_centre = cal_d["aotf"]["aotf_nu_centre"]



    cal_d["ils"] = get_ils_coeffs(channel, aotf_nu_centre)
    cal_d["orders"] = get_orders(channel, orders, nomad_t)
    
    if blaze_from_file:
        blaze = get_blaze(channel)["F_blaze"]
        for order in orders:
            cal_d["orders"][order]["F_blaze"] = blaze
        
    
    
    
    #convolve AOTF function to wavenumber of each pixel in each order
    for order in orders:
        
        px_nus = cal_d["orders"][order]["px_nus"]
        cal_d["orders"][order]["F_aotf"] = np.interp(px_nus, cal_d["aotf"]["aotf_nus"], cal_d["aotf"]["F_aotf"])
 
        if not blaze_from_file:
            cal_d["orders"][order]["F_blaze"] = get_blaze(channel, px_nus, aotf_nu_centre, order, nomad_t)
            
              
    
               
    if plot:
        plt.figure(constrained_layout=True)
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Line transmittance / normalised response")
        plt.plot(cal_d["aotf"]["aotf_nus"], cal_d["aotf"]["F_aotf"], color="k")
        
        for order in cal_d["orders"].keys():
            plt.plot(cal_d["orders"][order]["px_nus"], cal_d["orders"][order]["F_blaze"], label=order)
        plt.legend()
        plt.grid()


    return cal_d