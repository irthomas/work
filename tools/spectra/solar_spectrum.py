# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:42:03 2020

@author: iant
"""
import os

from tools.file.paths import paths

# SOLAR_SPECTRUM_FILE = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
SOLAR_SPECTRUM_FILE = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "pfsolspec_hr.dat")


def get_solar_hr(nu_hr, solspec_filepath=SOLAR_SPECTRUM_FILE, nu_limit=2.0, interpolate=True):
    """get high res solar spectrum interpolated to input nu_hr wavenumber grid"""
    import numpy as np
    from scipy import interpolate as sciin

    nu_hr_min = np.min(nu_hr)
    nu_hr_max = np.max(nu_hr)

    with open(solspec_filepath, "r") as f:
        nu_solar = []
        I0_solar = []
        for line in f:
            if line[0] == "%":
                continue

            nu, I0 = [float(val) for val in line.split()]
            if nu < nu_hr_min - nu_limit:
                continue
            if nu > nu_hr_max + nu_limit:
                break
            nu_solar.append(nu)
            I0_solar.append(I0)

    # print(len(nu_solar))

    if interpolate:
        f_solar = sciin.interp1d(nu_solar, I0_solar)
        I0_solar_hr = f_solar(nu_hr)
        return I0_solar_hr
    else:
        return [nu_solar, I0_solar]
