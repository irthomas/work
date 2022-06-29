# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:44:42 2022

@author: iant

MAKE ASIMUT SPECTRAL CALIBRATION COEFFICIENTS


format: "PIX->WN",binning,bin number, (PIXEL1),Param 1, Param 2, … Param 5
"PIX->WN",-1,-1,1.29612000674467,0.0,0.0,3.28702402e-08,0.000548031504,22.470027254747695

"""

import numpy as np


from instrument.nomad_so_instrument_v03 import lt22_p0_shift, lt22_waven


def asimut_wavenb(order, t, wavenb_filepath):
    
    binning = -1
    bin_no = -1
    first_px = lt22_p0_shift(t)
    coeffs = lt22_waven(order, t, channel="so", coeffs=True)
    
    coeffs_5_params = np.zeros(5)
    coeffs_5_params[-1*len(coeffs):] = coeffs
    
    out_str = f""""PIX->WN",{binning},{bin_no},{first_px:#.3f},""" + ",".join(["%0.6e" %i for i in coeffs_5_params])
    
    with open(wavenb_filepath, "w") as f:
        f.writelines(out_str)
