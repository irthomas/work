# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:48:08 2023

@author: iant

FORWARD MODEL TO FIT TO RAW

"""



# import os
import numpy as np
import matplotlib.pyplot as plt


# from analysis.so_lno_2023.functions.deconvolve_hapi_trans import reduce_resolution
from analysis.so_lno_2023.functions.aotf_blaze_ils import make_ils
# from tools.spectra.hapi_functions import get_abs_coeff, hapi_transmittance
from tools.datasets.get_solar import get_nomad_solar

from tools.spectra.hapi_lut import get_abs_coeff, abs_coeff_pt, hapi_transmittance


FIG_SIZE = (15, 8)




class forward:
    
    def __init__(self, raw=False):
        
        self.raw = raw #raw signal or transmittance?
        

    def calibrate(self, cal_d):
        self.cal_d = cal_d

        self.centre_order = cal_d["centre_order"]
        self.orders = cal_d["orders"].keys()
    
        self.n_px = len(cal_d["orders"][self.centre_order]["px_nus"])
        self.pxs = np.arange(self.n_px)

        self.nu_range = cal_d["aotf"]["aotf_nu_range"]


    def geometry(self, geom_d):
        self.geom_d = geom_d
        
    def molecules(self, molecule_d):
        self.molecule_d = molecule_d
        
        self.molecules = ", ".join(list(molecule_d.keys()))
        


    
    def forward_so(self, params, plot=[]):
        
        mol_scaler = params["mol_scaler"].value

        path_lengths_km = self.geom_d["path_lengths_km"]
        alt_grid = self.geom_d["alt_grid"]
        
        
        for molecule in self.molecule_d.keys():
            print(molecule, mol_scaler)

            if "hr" in plot:
                plt.figure(figsize=FIG_SIZE, constrained_layout=True)

            
            isos = self.molecule_d[molecule]["isos"]
            
    
    
    
            hapi_transs = []
            
            for i in range(len(alt_grid)):
                print("Altitude %0.2fkm, path length %0.2fkm" %(alt_grid[i], path_lengths_km[i]))
        
                t_raw = self.molecule_d[molecule]["ts"][i]
                #round temperature to nearest 5K
                t = np.round(t_raw / 5.0) * 5.0
                
                p = self.molecule_d[molecule]["pressures"][i] * self.molecule_d[molecule]["mol_ppmvs"][i] / 1.0e6
                for iso in isos:
            
                    # print("t:",  molecule_d[molecule]["ts"][i], \
                          # "pressure:",  molecule_d[molecule]["pressures"][i], \
                          # "mol_ppmv:",  molecule_d[molecule]["mol_ppmvs"][i], \
                          # "mol_ppmv_scaled:",  mol_ppmv_scaled, \
                          # "co2_ppmv:",  molecule_d[molecule]["co2_ppmvs"][i])
            
                    # hapi_nus, hapi_abs_coeffs = get_abs_coeff(molecule, nu_range, nu_step, \
                    #                            mol_ppmv_scaled, molecule_d[molecule]["co2_ppmvs"][i], molecule_d[molecule]["ts"][i], molecule_d[molecule]["pressures"][i], isos=isos, clear=clear_hapi)
                    # hapi_nus, hapi_trans = hapi_transmittance(hapi_nus, hapi_abs_coeffs, path_lengths_km[i], molecule_d[molecule]["ts"][i], spec_res=None)
    
                    hapi_nus, hapi_abs_coeffs = get_abs_coeff(self.centre_order, molecule, iso, t)
                    hapi_abs_coeffs_pt = abs_coeff_pt(hapi_abs_coeffs, p, t) * mol_scaler
                    _, hapi_trans = hapi_transmittance(hapi_nus, hapi_abs_coeffs_pt, path_lengths_km[i], t, spec_res=None)
                
                    #reduce spectral resolution
                    # hapi_nus_red, hapi_trans_red = reduce_resolution(hapi_nus, hapi_trans, 0.01)
                    # hapi_nus = hapi_nus_red
                    # hapi_trans = hapi_trans_red
            
                    # if "hr" in plot:
                    #     plt.plot(hapi_nus, hapi_trans, label="%0.1f km" %alt_grid[i])
                    hapi_transs.append(hapi_trans)
                
                
        #multiply transmittances together to get total atmos trans
        hapi_transs = np.asarray(hapi_transs)
        hapi_trans_total = np.prod(hapi_transs, axis=0)
        if "hr" in plot:
            plt.plot(hapi_nus, hapi_trans_total, "k")
            for order in self.orders:
                px_nus = self.cal_d["orders"][order]["px_nus"]
                plt.fill_betweenx([0,1], px_nus[0], px_nus[-1], alpha=0.3, label="Order %i" %order)
                px_ixs = self.cal_d["orders"][order]["px_ixs"]
                for i in range(0, 320, 50):
                    if i in px_ixs:
                        ix = np.where(i == px_ixs)[0]
                        plt.axvline(px_nus[ix], color="k", linestyle=":", alpha=0.5)
                    
                
            plt.legend(loc="lower left")



        if self.raw:
            #raw solar spectrum
            solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hapi_nus)


        #convolve AOTF function to wavenumber of each pixel in each order
        
        
        #ILS convolution
        #loop through pixel
        ils_sums = np.zeros((len(self.orders), len(self.pxs)))
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        if "cont" in plot:
            rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        
        for px in self.pxs:
    
            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]
        
            #loop through order
            for order_ix, order in enumerate(self.orders):
    
                blaze = self.cal_d["orders"][order]["F_blaze"][px]
                
                #px central cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]
                
                #get bounding indices of hapi grid
                ix_start = np.searchsorted(hapi_nus, px_nu - 0.7)
                ix_end = np.searchsorted(hapi_nus, px_nu + 0.7)
                
                #make ILS function on hapi grid
                hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                
                ils = make_ils(hapi_grid, width, displacement, amplitude)
                ils_sums[order_ix, px] = np.sum(ils)# * blaze * aotf
                
                if self.raw:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_total[ix_start:ix_end] * solar_hr_rad[ix_start:ix_end])# * blaze * aotf
                else:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_total[ix_start:ix_end])# * blaze * aotf


                blaze_aotf[order_ix, px] = blaze * aotf
                
            
        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf
        if self.raw:
            spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)
            spectrum = spectrum_sum / np.max(spectrum_sum)

        else:
            ils_sums_blaze_aotf = ils_sums * blaze_aotf
            spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0) / np.sum(ils_sums_blaze_aotf, axis=0)


        if "cont" in plot:

            # loop through orders
            for order_ix, order in enumerate(self.orders):
                rel_cont[order_ix, :] = 1.0 - (1.0 - (ils_sums_spectrum[order_ix, :] / ils_sums[order_ix, :])) * (blaze_aotf[order_ix] / np.sum(blaze_aotf[:, :], axis=0))
            
            rel_1_cont = 1.0 - rel_cont
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Pixel number")
            plt.ylabel("Contribution from each order")
            rel_1_cont_cumul = np.zeros(len(self.pxs)) #set to 0 for first bars
            for order_ix, order in enumerate(self.orders):
                plt.bar(self.pxs, rel_1_cont[order_ix, :], bottom=rel_1_cont_cumul, label=order)
                rel_1_cont_cumul = rel_1_cont[order_ix, :]
        
            plt.legend()
            plt.grid()
            
        return spectrum
                

    def fit(self, params, y_raw, plot=[]):

        mol_scaler = params["mol_scaler"].value

        
        #normalise raw SO spectrum
        y_raw /= np.max(y_raw)
        
        self.spectrum_norm = self.forward_so(params, plot=plot)
        sum_sq = np.sum(np.square(y_raw - self.spectrum_norm))

        if "fit" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber cm-1")
            plt.ylabel("SO transmittance")
            
        
            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], y_raw, label="SO raw spectrum")
            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], self.spectrum_norm, label="Simulation")
            plt.grid()
            plt.title("%s: %0.4f %0.4f" %(self.molecules, sum_sq, mol_scaler))
            plt.legend()
            # plt.savefig(("%0.8f" %ssd).replace(".","p")+".png")

        
    
        print("sum_sq=", sum_sq)
        return np.square(y_raw - self.spectrum_norm) 


    def forward_toa(self, plot=[]):
        
        hr_nu_grid = self.cal_d["aotf"]["aotf_nus"]

        solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hr_nu_grid)


        #convolve AOTF function to wavenumber of each pixel in each order
        
        
        #ILS convolution
        #loop through pixel
        ils_sums = np.zeros((len(self.orders), len(self.pxs)))
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        # if "cont" in plot:
        #     rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        
        for px in self.pxs:
    
            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]
        
            #loop through order
            for order_ix, order in enumerate(self.orders):
    
                blaze = self.cal_d["orders"][order]["F_blaze"][px]
                
                #px central cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]
                
                #get bounding indices of hapi grid
                ix_start = np.searchsorted(hr_nu_grid, px_nu - 0.7)
                ix_end = np.searchsorted(hr_nu_grid, px_nu + 0.7)
                
                #make ILS function on hapi grid
                hapi_grid = hr_nu_grid[ix_start:ix_end] - px_nu
                
                ils = make_ils(hapi_grid, width, displacement, amplitude)
                ils_sums[order_ix, px] = np.sum(ils)# * blaze * aotf
                
                if self.raw:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_hr_rad[ix_start:ix_end])
                else:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils)


                blaze_aotf[order_ix, px] = blaze * aotf
                
            
        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf
        if self.raw:
            spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)
            spectrum = spectrum_sum / np.max(spectrum_sum)

        else:
            ils_sums_blaze_aotf = ils_sums * blaze_aotf
            spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0) / np.sum(ils_sums_blaze_aotf, axis=0)


        if "aotf" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber")
            plt.ylabel("AOTF / solar radiance")
            plt.plot(hr_nu_grid, solar_hr_rad/np.max(solar_hr_rad), label="Solar spectrum")
            plt.plot(hr_nu_grid, self.cal_d["aotf"]["F_aotf"], label="AOTF function")
            plt.legend()
            plt.grid()
            

            
        return spectrum
