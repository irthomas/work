# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:53:23 2021

@author: iant

CALCULATE ILS PARAMETERS PER PIXEL
ASSUME FIRSTPXIEL=0 TO CALC CM-1
"""

import numpy as np

from instrument.nomad_so_instrument import A_aotf, nu0_aotf


# pixels = np.arange(320)
# p0= 0.0

# order = 189

# G0=305.0604
# G1=0.1497089
# G2=1.34082E-7
# #def nu0_aotf(A, G0=313.91768, G1=0.1494441, G2=1.340818E-7): #SWT13
# def nu0_aotf(A, G0=G0, G1=G1, G2=G2):
#     """aotf frequency to aotf centre in wavenumbers. Update from team telecon Feb 2019"""
#     nu0 = G0 + A*(G1 + A*G2)
#     return nu0

# def nu_mp(m, p, p0):
#     """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
#     F0=22.473422
#     F1=5.559526e-4
#     F2=1.751279e-8
#     f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
#     return f


# nu_px = nu_mp(order, pixels, p0)


def get_ils_params(hdf5_filename, nu_px, save_file=True, order=None):
    """make ils parameter file for a given hdf5 file"""
    if len(nu_px) == 320:
        pixels = np.arange(320)
    else:
        print("Error: incorrect number of pixels")
        return []
    

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    if not order:
        order = int(hdf5_filename[-3:])
    aotf_freq = A_aotf[order]

    A_nu0 = nu0_aotf(aotf_freq)
    
    A_w_nu0 = A_nu0 / rp
    sconv = A_w_nu0/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * A_nu0 #displacement adjusted for wavenumber
    
    if save_file:
        #columns are: nu_p 0.0 sconv 1.0 disp_order sconv amp
        lines = []
        for nu, disp in zip(nu_px, disp_order):
            lines.append("%0.5f, %0.1f, %0.6f, %0.1f, %0.8f, %0.6f, %0.6f\n" %(nu, 0.0, sconv, 1.0, disp, sconv, amp))
        
        with open("%s_ils.txt" %hdf5_filename, "w") as f:
            f.writelines(lines)
    
    else:
        return {"width":np.tile(sconv, len(pixels)), "displacement":disp_order, "amplitude":np.tile(amp, len(pixels))}
        


def get_ils_params2(A_nu0):
    """make ils parameter file for a given hdf5 file"""
    if len(nu_px) == 320:
        pixels = np.arange(320)
    else:
        print("Error: incorrect number of pixels")
        return []
    

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    if not order:
        order = int(hdf5_filename[-3:])
    aotf_freq = A_aotf[order]

    A_nu0 = nu0_aotf(aotf_freq)
    
    A_w_nu0 = A_nu0 / rp
    sconv = A_w_nu0/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * A_nu0 #displacement adjusted for wavenumber
    
    if save_file:
        #columns are: nu_p 0.0 sconv 1.0 disp_order sconv amp
        lines = []
        for nu, disp in zip(nu_px, disp_order):
            lines.append("%0.5f, %0.1f, %0.6f, %0.1f, %0.8f, %0.6f, %0.6f\n" %(nu, 0.0, sconv, 1.0, disp, sconv, amp))
        
        with open("%s_ils.txt" %hdf5_filename, "w") as f:
            f.writelines(lines)
    
    else:
        return {"width":np.tile(sconv, len(pixels)), "displacement":disp_order, "amplitude":np.tile(amp, len(pixels))}
        
