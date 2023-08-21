# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:48:08 2023

@author: iant

FORWARD MODEL TO FIT TO RAW

"""

import numpy as np
import matplotlib.pyplot as plt


from analysis.so_lno_2023.functions.h5 import read_h5
from analysis.so_lno_2023.calibration import get_calibration
from analysis.so_lno_2023.functions.geometry import make_path_lengths
# from analysis.so_lno_2023.forward_model import forward

from tools.datasets.get_gem_data import get_gem_tpvmr
from tools.spectra.baseline_als import baseline_als
from tools.datasets.get_solar import get_nomad_solar

from analysis.so_lno_2023.functions.deconvolve_hapi_trans import reduce_resolution
from analysis.so_lno_2023.functions.aotf_blaze_ils import make_ils
from tools.spectra.hapi_functions import get_abs_coeff, hapi_transmittance


# clear_hapi = True
clear_hapi = False


nu_step = 0.001 #cm-1
pxs = np.arange(320)
orders = np.arange(184, 190)


plot = ["fit", "cont"]





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

y_raw = h5_d["y_raw"][ix]


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







centre_order = h5_d["order"]


nu_range = cal_d["aotf"]["aotf_nu_range"]

if channel == "so":

    for molecule in molecule_d.keys():

        path_lengths_km = molecule_d[molecule]["path_lengths_km"]
        alt_grid = molecule_d[molecule]["alt_grid"]
        
        isos = molecule_d[molecule]["isos"]
        



        hapi_transs = []
        
        for i in range(len(alt_grid)):
            #loop through altitudes getting transmittance of each layer
            print("%i/%i: altitude %0.2fkm, path length %0.2fkm" %(i+1, len(alt_grid), alt_grid[i], path_lengths_km[i]))
    
            mol_ppmv_scaled =  molecule_d[molecule]["mol_ppmvs"][i] * mol_scaler
        
            print("t:",  molecule_d[molecule]["ts"][i], \
                  "pressure:",  molecule_d[molecule]["pressures"][i], \
                  "mol_ppmv:",  molecule_d[molecule]["mol_ppmvs"][i], \
                  "mol_ppmv_scaled:",  mol_ppmv_scaled, \
                  "co2_ppmv:",  molecule_d[molecule]["co2_ppmvs"][i])
    
            hapi_nus, hapi_abs_coeffs = get_abs_coeff(molecule, nu_range, nu_step, \
                                       mol_ppmv_scaled, molecule_d[molecule]["co2_ppmvs"][i], molecule_d[molecule]["ts"][i], molecule_d[molecule]["pressures"][i], isos=isos, clear=clear_hapi)
            hapi_nus, hapi_trans = hapi_transmittance(hapi_nus, hapi_abs_coeffs, path_lengths_km[i], molecule_d[molecule]["ts"][i], spec_res=None)
        
            #reduce spectral resolution
            hapi_nus_red, hapi_trans_red = reduce_resolution(hapi_nus, hapi_trans, 0.01)
            hapi_nus = hapi_nus_red
            hapi_trans = hapi_trans_red
    
            if "hr" in plot:
                plt.plot(hapi_nus, hapi_trans, label="%0.1f km" %alt_grid[i])
            hapi_transs.append(hapi_trans) #save transmittances of each layer
            
            
        hapi_transs = np.asarray(hapi_transs)
        hapi_trans_total = np.prod(hapi_transs, axis=0) #multiply transmittances together to get total atmos trans
        if "hr" in plot:
            plt.plot(hapi_nus, hapi_trans_total, "k")

        #raw solar spectrum
        solar_hr_nu, solar_hr_rad = get_nomad_solar(nu_range, interp_grid=hapi_nus)


        #convolve AOTF function to wavenumber of each pixel in each order
        
        
        #ILS convolution
        #loop through pixel
        ils_sums = np.zeros((len(orders), len(pxs)))
        ils_sums_spectrum = np.zeros((len(orders), len(pxs)))
        blaze_aotf = np.zeros((len(orders), len(pxs)))

        rel_cont = np.zeros((len(orders), len(pxs)))

        
        for px in pxs:
    
            width = cal_d["ils"]["ils_width"][px]
            displacement = cal_d["ils"]["ils_displacement"][px]
            amplitude = cal_d["ils"]["ils_amplitude"][px]
        
            #loop through order
            for order_ix, order in enumerate(orders):
    
                blaze = cal_d["orders"][order]["F_blaze"][px]
                
                #px central cm-1
                px_nu = cal_d["orders"][order]["px_nus"][px]
                aotf = cal_d["orders"][order]["F_aotf"][px]
                
                #get bounding indices of hapi grid
                ix_start = np.searchsorted(hapi_nus, px_nu - 0.7)
                ix_end = np.searchsorted(hapi_nus, px_nu + 0.7)
                
                #make ILS function on hapi grid
                hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                
                
                ils = make_ils(hapi_grid, width, displacement, amplitude)
                ils_sums[order_ix, px] = np.sum(ils)# * blaze * aotf
                ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_total[ix_start:ix_end] * solar_hr_rad[ix_start:ix_end])# * blaze * aotf

                blaze_aotf[order_ix, px] = blaze * aotf
                
            
        ils_sums_blaze_aotf = ils_sums * blaze_aotf
        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf


        #loop through order
        for order_ix, order in enumerate(orders):
            rel_cont[order_ix, :] = 1.0 - (1.0 - (ils_sums_spectrum[order_ix, :] / ils_sums[order_ix, :])) * (blaze_aotf[order_ix] / np.sum(blaze_aotf[:, :], axis=0))
        
        rel_1_cont = 1.0 - rel_cont
    
        spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)# / np.sum(ils_sums_blaze_aotf, axis=0)

        spectrum_norm = spectrum / np.max(spectrum) * np.max(y_raw)

        ssd = np.sum(np.square(y_raw - spectrum_norm))
        
        if "fit" in plot:
            plt.figure(constrained_layout=True)
            plt.xlabel("Wavenumber cm-1")
            plt.ylabel("SO raw spectrum")
            
        
            plt.plot(cal_d["orders"][centre_order]["px_nus"], y_raw)
            plt.plot(cal_d["orders"][centre_order]["px_nus"], spectrum_norm)
            plt.grid()
            plt.title(ssd)
            plt.savefig(("%0.8f" %ssd).replace(".","p")+".png")
    
        if "cont" in plot:

            plt.figure(constrained_layout=True)
            plt.xlabel("Pixel number")
            plt.ylabel("Contribution from each order")
            rel_1_cont_cumul = np.zeros(len(pxs)) #set to 0 for first bars
            for order_ix, order in enumerate(orders):
                plt.bar(pxs, rel_1_cont[order_ix, :], bottom=rel_1_cont_cumul, label=order)
                rel_1_cont_cumul = rel_1_cont[order_ix, :]
        
            plt.legend()
            plt.grid()

        print("ssd=", ssd)
