# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:27:43 2022

@author: iant

MAKE ASIMUT AOTF INPUT FILE
"""

# import numpy as np

from instrument.nomad_so_instrument_v03 import aotf_func


def asimut_aotf(aotf_nu, aotf_filepath):
    
    
    relative_nu, F_aotf = aotf_func(aotf_nu, aotf_range=200.0, step_nu=0.1)
    
    lines = []
    lines.append("%% SO AOTF centre %0.5f\n" %aotf_nu)
    lines.append("% nu(cm^-1)   F_aotf\n")

    
    #columns are: relative nu, aotf function
    for nu, F in zip(relative_nu, F_aotf):
        lines.append("%0.5f %0.5f\n" %(nu, F))
    
    with open(aotf_filepath, "w") as f:
        f.writelines(lines)
    