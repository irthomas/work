# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:42:03 2020

@author: iant
"""



def get_solar_hr(nu_hr, solspec_filepath, nu_limit=2.0):
    """get high res solar spectrum interpolated to input nu_hr wavenumber grid"""
    import numpy as np
    from scipy import interpolate

    nu_hr_min = np.min(nu_hr)
    nu_hr_max = np.max(nu_hr)
    
    with open(solspec_filepath, "r") as f:
        nu_solar = []
        I0_solar = []
        for line in f:
            nu, I0 = [float(val) for val in line.split()]
            if nu < nu_hr_min - nu_limit:
                continue
            if nu > nu_hr_max + nu_limit:
                break
            nu_solar.append(nu)
            I0_solar.append(I0)
    f_solar = interpolate.interp1d(nu_solar, I0_solar)
    I0_solar_hr = f_solar(nu_hr)
    return I0_solar_hr

