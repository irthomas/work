# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:22:54 2023

@author: iant

READ IN SOLAR SPECTRUM
"""

import os
import numpy as np


from tools.file.paths import paths


FILENAMES = {
    "solspec":"nomad_solar_spectrum_solspec.txt",
    "pfs":"pfsolspec_hr.dat",
    "ace":"ace-solar-spectrum.txt",
    }

nu_range = [3000, 5000]

interp_grid = False
name = False


def get_solar(nu_range, name=False, interp_grid=False):
    """get solar spectrum from a reference file. If no name given then solspec is used.
    give a np array for the interp_grid and the solar spectrum will be interpolated onto the grid
    Output is a list of 2 arrays, wavenumbers and solar spectrum"""
    if isinstance(name, str):
        if name in FILENAMES.keys():
            filename = FILENAMES[name]
        else:
            #default to solspec
            filename = FILENAMES["solspec"]
    else:
        #default to solspec
        filename = FILENAMES["solspec"]
    
    filepath = os.path.join(paths["REFERENCE_DIRECTORY"], filename)
    
    
    nu, radiance = np.loadtxt(filepath, unpack=True)
    
    #normalise radiance
    radiance /= np.max(radiance)
    
    
    if isinstance(interp_grid, np.ndarray):
        return [interp_grid, np.interp(interp_grid, nu, radiance)]
    else:
        ixs = np.searchsorted(nu, nu_range)+1
        return [nu[ixs[0]:ixs[-1]], radiance[ixs[0]:ixs[-1]]]





def get_nomad_solar(nu_range, interp_grid=False):
    if np.max(nu_range[1] <= 4430):
        #normal usage
        return get_solar(nu_range, name="solspec", interp_grid=interp_grid)
    else:
        #solspec ends at 4430cm-1. Use pfs above this
        nu_s, solspec = get_solar([nu_range[0], 4430], name="solspec", interp_grid=False)
        nu_p, pfs = get_solar([4430, nu_range[1]], name="pfs", interp_grid=False)
        
        pfs += (0.92 - 0.52)
    
        nu_cat = np.concatenate((nu_s, nu_p))    
        rad_cat = np.concatenate((solspec, pfs))
        
        if isinstance(interp_grid, np.ndarray):
            return [interp_grid, np.interp(interp_grid, nu_cat, rad_cat)]
        else:
            return [nu_cat, rad_cat]
            
    
"""examples"""  
# import matplotlib.pyplot as plt

# plt.figure()
# for i, name in enumerate(FILENAMES.keys()):
#     nu, rad = get_solar(nu_range, name=name, interp_grid=False)
#     plt.plot(nu, rad+(0.25e-6*i), label=name, alpha=0.5)
# plt.legend()


#get extended solar spectrum
# plt.figure()
# nu, rad = get_nomad_solar(nu_range)
# plt.plot(nu, rad)