# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:12:01 2023

@author: iant

MAKE SIMPLE LUT
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


import hapi

from tools.file.paths import paths

# from tools.datasets.get_gem_data import get_gem_tpvmr
# from tools.spectra.hapi_functions import make_table_name, get_abs_coeff, hapi_transmittance
# from tools.general.progress_bar import progress_bar


ISO_IDS = {"H2O":[1, 2, 3, 4, 5, 6, 129], "CO2":[7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122], "O3":[], "CO":[26, 27, 28, 29, 30, 31], "CH4":[32, 33, 34, 35], "HCL":[52, 53, 107, 108]}


# molecule = "CH4"
# isos = [1]
# order = 134
# order = 136


# molecule = "H2O"
# isos = [1,2,3,4]
# order = 134


molecule = "CO"
isos = [1,2,3,4]
# order = 185
# order = 186
order = 194
# order = 195






# list of GEM temperatures and pressures from altitudes
# altitude = 5.0
# myear = 35
# ls = 90.0
# lat = 0.0
# lon = 0.0
# lst = 12.0
# for altitude in [0.0001, 1.0, 10.0, 100.0, 150.0]:
#     #pressure = pressure of atmosphere in atm
#     t, pressure, mol_ppmv, co2_ppmv = get_gem_tpvmr("CO2", altitude, myear, ls, lat, lon, lst)
#     print(altitude, t, pressure, np.log(pressure))
 


#order 134 = 3011.44 3035.44
#order 185 = 183-192 inc
nu_range_d = {
    ("CH4", 134):[3005.0, 3040.0], 
    ("CH4", 136):[3050.0, 3085.0], 
    ("H2O", 134):[2960.0, 3085.0], 
    ("CO" , 185):[4100.0, 4330.0],
    ("CO" , 186):[4100.0, 4330.0],
    ("CO" , 191):[4100.0, 4500.0],
    ("CO" , 192):[4100.0, 4500.0],
    ("CO" , 193):[4100.0, 4500.0],
    ("CO" , 194):[4100.0, 4500.0],
    ("CO" , 195):[4200.0, 4500.0],
}
nu_range = nu_range_d[(molecule, order)]
nu_step = 0.001

component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]


t_range = np.arange(120.0, 250.0, 5.0)


lut_filename = "lut_so_%i_%s.h5" %(order, molecule)
lut_filepath = os.path.join(paths["LOCAL_DIRECTORY"], "lut", lut_filename)



#make file, add attributes
if os.path.exists(lut_filepath):
    print("Error: file already exists")
    sys.exit()

with h5py.File(lut_filepath, "w") as h5f:
    h5f.attrs["nu_range"] = nu_range
    h5f.attrs["nu_step"] = nu_step
    h5f.create_group("nu")


for iso in isos:
    
    with h5py.File(lut_filepath, "a") as h5f:
        h5f.create_group("%i" %iso)

            
    Components=[(component, iso)]

    for t in t_range:
        pressure = 0.007 #choose any value
        table_name = molecule
        
        
        print("iso", iso, "T=", t)
            
        iso_codes = ISO_IDS[molecule]
        hapi.fetch_by_ids(table_name, iso_codes, nu_range[0], nu_range[1])#, ParameterGroups=PARAMETER_GROUPS)
        
        
            
        
            
        #still need to give a pressure for the lineshape calculation, but changing the value does not change anything
        #if not using hitran units, need to give pressure (inefficient)
        #instead, use HITRAN units then multiply by hapi.volumeConcentration(pressure, t)
        #gives same answer within relative error of 6e-13
        nu, coeff = hapi.absorptionCoefficient_Voigt(Components=Components, SourceTables=[table_name], Diluent={'CO2':1.0}, \
                                                    Environment={'T':t,'p':pressure}, WavenumberStep=nu_step, HITRAN_units=True)

        #check nu range has same values each time
        if t == t_range[0]:
            nu_start = nu
            with h5py.File(lut_filepath, "a") as h5f:
                h5f["nu"].create_dataset("%i" %iso, dtype=np.float32, data=nu, compression="gzip", shuffle=True)

        else:
            if np.all(nu != nu_start):
                print("Warning, nu not the same")
                sys.exit()
    
        # plt.plot(nu, coeff, label=t, alpha=0.3)
        with h5py.File(lut_filepath, "a") as h5f:
            h5f["%i" %iso].create_dataset("%0.1f" %t, dtype=np.float32, data=coeff, compression="gzip", shuffle=True)
       

