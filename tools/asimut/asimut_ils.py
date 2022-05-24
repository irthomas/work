# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:49:46 2022

@author: iant

MAKE ASIMUT ILS INPUT FILE
"""

import numpy as np


def asimut_ils(aotf_nu, px_nu, ils_filepath):
    """make ils parameter file for a given hdf5 file"""
    pixels = np.arange(320)
    

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    A_w_nu0 = aotf_nu / rp
    sconv = A_w_nu0/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * aotf_nu #displacement adjusted for wavenumber
    
    #columns are: nu_p 0.0 sconv 1.0 disp_order sconv amp
    lines = []
    lines.append("% nu_p 0.0 sconv 1.0 disp_order sconv amp\n")
    for nu, disp in zip(px_nu, disp_order):
        lines.append("%0.5f %0.1f %0.6f %0.1f %0.8f %0.6f %0.6f\n" %(nu, 0.0, sconv, 1.0, disp, sconv, amp))
    
    with open(ils_filepath, "w") as f:
        f.writelines(lines)



