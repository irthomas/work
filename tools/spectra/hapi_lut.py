# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:52:01 2023

@author: iant
"""

import os
import h5py


from tools.file.paths import paths




def get_abs_coeff(order, molecule, iso, t):
    lut_filename = "lut_so_%i_%s.h5" %(order, molecule)
    lut_filepath = os.path.join(paths["LOCAL_DIRECTORY"], "lut", lut_filename)
    
    if os.path.exists(lut_filepath):
        with h5py.File(lut_filepath, "r") as h5f:
            ts = [float(f) for f in h5f["%i" %iso].keys()]
            if t in ts:
                coeff = h5f["%i" %iso]["%0.1f" %t][...]
                nu = h5f["nu"]["%i" %iso][...]
            else:
                print("Error: t not found, must be %0.1f-%0.1fK" %(min(ts), max(ts)))
                return [], []
    else:
        print("Error: file not found")
        return [], []
        
    return nu, coeff


def abs_coeff_pt(coeff, p, t):
    
    def volumeConcentration(p,T):
        cBolts = 1.380648813E-16 # erg/K, CGS
        return (p/9.869233e-7)/(cBolts*T) # CGS


    """pressure in atmospheres
    temperature in K"""
    coeff_pt = coeff * volumeConcentration(p, t)
    return coeff_pt


def hapi_transmittance(nu, coeff_pt, path_length_km, t, spec_res=None):
    
    import hapi


    path_length_cm = 100.0 * 1.0e3 * path_length_km #km in cm
    
    nu, trans = hapi.transmittanceSpectrum(nu, coeff_pt, Environment={'T':t, 'l':path_length_cm})
    
    if not spec_res:
        return [], trans

    else:
        nu_conv, trans_conv, i1, i2, slit = hapi.convolveSpectrum(nu, trans, SlitFunction=hapi.SLIT_GAUSSIAN, Resolution=spec_res, AF_wing=0.3)
        return nu_conv, trans_conv



# for testing
# import matplotlib.pyplot as plt

# order = 134
# molecule = "CH4"
# iso = 1
# t = 130.0
# mol_ppmv = 1.0e-3
# p = 0.007 * mol_ppmv * 1.0e-6 #atmospheres to ppm * fraction
# path_length_km = 200.0


# nu, coeff = get_abs_coeff(order, molecule, iso, t)
# coeff_pt = abs_coeff_pt(coeff, p, t)
# _, trans = hapi_transmittance(nu, coeff_pt, path_length_km, t, spec_res=None)
# nu_conv, trans_conv = hapi_transmittance(nu, coeff_pt, path_length_km, t, spec_res=0.15)

# plt.figure()
# plt.plot(nu, trans)
# plt.plot(nu_conv, trans_conv)

# order = 185
# molecule = "CO"
# isos = [1, 2, 3, 4]
# t = 130.0
# mol_ppmv = 1.0
# p = 0.007 * mol_ppmv * 1.0e-6 #atmospheres to ppm * fraction
# path_length_km = 200.0

# plt.figure()
# for iso in isos:
#     nu, coeff = get_abs_coeff(order, molecule, iso, t)
#     coeff_pt = abs_coeff_pt(coeff, p, t)
#     _, trans = hapi_transmittance(nu, coeff_pt, path_length_km, t, spec_res=None)
#     nu_conv, trans_conv = hapi_transmittance(nu, coeff_pt, path_length_km, t, spec_res=0.15)

#     plt.plot(nu, trans)
#     plt.plot(nu_conv, trans_conv)





