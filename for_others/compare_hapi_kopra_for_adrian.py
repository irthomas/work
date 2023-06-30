# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:03:19 2023

@author: iant


COMPARE HAPI TO ADRIAN KOPRA RESULTS
"""

import numpy as np
import matplotlib.pyplot as plt


from tools.spectra.hapi_functions import get_gem_tpvmr, get_abs_coeff, hapi_transmittance

from analysis.so_lno_2023.functions.geometry import make_path_lengths


chosen_alt = 80.0

kopra_nus, kopra_trans = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\science\adrian_kopra_hr_trans\transmittance\spec_%ikm.csv" %chosen_alt, skiprows=1, delimiter=",", unpack=True)

alt_in,p_in,t_in,co2_ppmv_in,_,h2o_ppmv_in,_,_,_,_ = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\science\adrian_kopra_hr_trans\ref_atm_dusty_MTP004_orb484I.csv", skiprows=2, delimiter=",", unpack=True)

alt_indices = np.where(alt_in >= chosen_alt)[0]

#limited range
# alt_indices = alt_indices[0:2]

alt_grid = alt_in[alt_indices]
p_grid = p_in[alt_indices] / 1013.25
t_grid = t_in[alt_indices]
co2_ppmv_grid = co2_ppmv_in[alt_indices] * 0.95 #TODO: remove
h2o_ppmv_grid = h2o_ppmv_in[alt_indices] * 0.95 #TODO: remove

path_lengths = make_path_lengths(alt_grid)

clear_hapi = True
# clear_hapi = False


molecules = ["H2O", "CO2"]
# molecules = ["H2O"]

nu_range = [kopra_nus[0], kopra_nus[-1]]
nu_step = (kopra_nus[1] - kopra_nus[0]) * 1.0
isos = [1,2,3,4,5]
# path_length_km = 390.


# t, pressure, mol_ppmv, co2_ppmv = get_gem_tpvmr(molecule, alt, myear, ls, lat, lon, lst, plot=False)
# if chosen_alt == 10.:
#     pressure = 2.4729e+00 / 1013.25
#     t = 230.648
    # mol_ppmv = 9.9913e+05
    # co2_ppmv = 9.9913e+05
    # h20_ppmv = 1.6495e+02
# elif chosen_alt == 80.:
#     pressure = 2.3295e-03 / 1013.25
#     t = 156.623
    # mol_ppmv = 9.9912e+05
    # co2_ppmv = 9.9912e+05
    # h20_ppmv = 8.9800e+01


hapi_trans_grid = {"H2O":[], "CO2":[]}
hapi_nus_d = {}

for layer_ix, (t, pressure, h2o_ppmv, co2_ppmv, path_length) in enumerate(zip(t_grid, p_grid, h2o_ppmv_grid, co2_ppmv_grid, path_lengths)):
    print("layer %i/%i:" %(layer_ix, len(t_grid)), "t:", t, "pressure:", pressure, "mol_ppmv:", h2o_ppmv, "co2_ppmv:", co2_ppmv)

    for molecule in molecules:
        
        if molecule == "H2O":
            mol_ppmv = h2o_ppmv
            if clear_hapi:
                clear = True
            else:
                clear = False
        elif molecule == "CO2":
            mol_ppmv = co2_ppmv
            clear = False
            
        hapi_nus, hapi_abs_coefs = get_abs_coeff(molecule, nu_range, nu_step, \
                                   mol_ppmv, co2_ppmv, t, pressure, clear=clear)
        hapi_nus, hapi_trans = hapi_transmittance(hapi_nus, hapi_abs_coefs, path_length, t, spec_res=None)
        
        if molecule not in hapi_nus_d.keys():
            hapi_nus_d[molecule] = hapi_nus

        hapi_trans_grid[molecule].append(hapi_trans)

for molecule in molecules:
    hapi_trans_grid[molecule] = np.asfarray(hapi_trans_grid[molecule])

    hapi_trans_grid["%s_prod" %molecule] = np.prod(hapi_trans_grid[molecule], axis=0)
    
#CO2 nu range is wider than H2O - interpolate H2O onto CO2
hapi_trans_grid["H2O_interp"] = np.interp(hapi_nus_d["CO2"], hapi_nus_d["H2O"], hapi_trans_grid["H2O_prod"])
hapi_nus = hapi_nus_d["CO2"]


hapi_trans_prod = hapi_trans_grid["CO2_prod"] * hapi_trans_grid["H2O_interp"]

plt.figure(figsize=(15, 8), constrained_layout=True)
plt.title("KOPRA %ikm+" %chosen_alt)
plt.plot(kopra_nus, kopra_trans)
plt.grid()
plt.savefig("kopra_%ikm+.png" %chosen_alt)

plt.figure(figsize=(15, 8), constrained_layout=True)
plt.title("HAPI %ikm+" %chosen_alt)
plt.plot(hapi_nus, hapi_trans_prod)
plt.grid()
plt.savefig("hapi_%ikm+.png" %chosen_alt)


plt.figure(figsize=(15, 8), constrained_layout=True)
plt.title("HAPI %ikm" %chosen_alt)
plt.plot(hapi_nus_d["H2O"], hapi_trans_grid["H2O"][0, :])
plt.plot(hapi_nus_d["CO2"], hapi_trans_grid["CO2"][0, :])
plt.grid()
plt.xlim([3780, 3781])

plt.figure(figsize=(15, 8), constrained_layout=True)
plt.title("HAPI %ikm+ separate" %chosen_alt)
plt.plot(hapi_nus_d["H2O"], hapi_trans_grid["H2O_prod"])
plt.plot(hapi_nus_d["CO2"], hapi_trans_grid["CO2_prod"])
plt.grid()


hapi_kopra_interp = np.interp(kopra_nus, hapi_nus, hapi_trans_prod)

plt.figure(figsize=(15, 8), constrained_layout=True)
plt.title("KOPRA - HAPI %ikm+" %chosen_alt)
plt.plot(kopra_nus, kopra_trans - hapi_kopra_interp)
plt.grid()
plt.savefig("kopra_hapi_diff_%ikm+.png" %chosen_alt)


# plt.xlim([3780, 3781])

# plt.plot(hapi_nus, hapi_trans)

# hapi_trans2 = np.copy(hapi_trans)

# voigt_diff = hapi_trans - hapi_trans2

# plt.figure()
# plt.plot(hapi_nus, voigt_diff)