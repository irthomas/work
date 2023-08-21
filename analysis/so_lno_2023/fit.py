# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:29:24 2023

@author: iant
"""






import numpy as np
# import matplotlib.pyplot as plt


from analysis.so_lno_2023.functions.h5 import read_h5
from analysis.so_lno_2023.calibration import get_calibration
from analysis.so_lno_2023.functions.geometry import make_path_lengths
from analysis.so_lno_2023.forward_model import forward

from tools.datasets.get_gem_data import get_gem_tpvmr
from tools.spectra.baseline_als import baseline_als

from lmfit import minimize, Parameters


# clear_hapi = True
clear_hapi = False



"""SO"""
h5 = "20220301_114833_1p0a_SO_A_I_186"
channel = "so"
ix = 190

molecules = {
    "CO":{"isos":[1,2,3,4,5]},
}


alt_delta = 10.0
alt_max = 100.0

# orders = np.arange(183, 193)
orders = np.arange(184, 190)


mol_scaler = 0.3409


plot = ["fit"]





"""LNO"""










h5_d = read_h5(h5)

y = h5_d["y"][ix]
y_cont = baseline_als(y)
y_flat = y / y_cont
        


centre_order = h5_d["order"]
aotf_freq = h5_d["aotf_freq"]
nomad_t = h5_d["nomad_t"]
if "hr" in plot:
    plot_hr = True
else:
    plot_hr = False
cal_d = get_calibration(channel, aotf_freq, centre_order, orders, nomad_t, plot=plot_hr)



molecule_d = {}
for molecule in molecules.keys():
    isos = molecules[molecule]["isos"]
    nu_range = cal_d["aotf"]["aotf_nu_range"]
    
    alt = h5_d["alts"][ix]
    myear = h5_d["myear"]
    ls = h5_d["ls"]
    lat = h5_d["lats"][ix]
    lon = h5_d["lons"][ix]
    lst = h5_d["lst"][ix]
    
    alt_grid = np.arange(alt, alt_max, alt_delta)
    path_lengths = make_path_lengths(alt_grid)
    path_lengths_km = path_lengths #524.8 #10km grid at 41.85km



    #get gem data, interpolate onto altitude grid        
    ts, pressures, mol_ppmvs, co2_ppmvs = get_gem_tpvmr(molecule, alt_grid, myear, ls, lat, lon, lst, plot=False)
    molecule_d[molecule] = {}
    molecule_d[molecule]["ts"] = ts
    molecule_d[molecule]["pressures"] = pressures
    molecule_d[molecule]["mol_ppmvs"] = mol_ppmvs
    molecule_d[molecule]["co2_ppmvs"] = co2_ppmvs
    molecule_d[molecule]["path_lengths_km"] = path_lengths_km
    molecule_d[molecule]["alt_grid"] = alt_grid
    molecule_d[molecule]["isos"] = isos


# ssd = forward(, mol_scaler)


params = Parameters()
params.add('mol_scaler', value=mol_scaler)

out = minimize(forward, params, args=(channel, cal_d, h5_d, molecule_d, y_flat), max_nfev=1)

out.params.pretty_print()