# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:20:10 2021

@author: iant

MINISCAN FITTING
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from tools.plotting.colours import get_colours

from instrument.calibration.so_aotf_ils.simulation_functions import (get_file, get_data_from_file, select_data, fit_temperature,
get_start_params, make_param_dict, calc_spectrum, fit_spectrum, area_under_curve, get_solar_spectrum)
from instrument.calibration.so_aotf_ils.simulation_config import sim_parameters


#AOTF freqs where AOTF is strong: 4380-4390cm-1 = 26560-26640kHz

# line = 4383.5
line = 4276.1
# line = 3787.9

SETUP = False
# SETUP = True


PLOT_RAW = False
# PLOT_RAW = True




if PLOT_RAW:
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_xlabel("Pixel Number")
    ax1.set_ylabel("Normalised Spectra")
    ax2.set_ylabel("Solar Line")


filenames = sim_parameters[line]["filenames"]
for filename in filenames:
    regex = re.compile(filename) #(approx. orders 188-202) in steps of 8kHz



    hdf5_file, hdf5_filename = get_file(regex)
    d = get_data_from_file(hdf5_file, hdf5_filename)
    
    
    solar_spectra = sim_parameters[line]["solar_spectra"].keys()
    for solar_spectrum in solar_spectra:
        d["line"] = line
        d["solar_spectrum"] = solar_spectrum
        d = get_solar_spectrum(d, plot=SETUP)
    
    
        indices = range(len(d["aotf_freqs"]))
        # indices = range(45,55,2)
        # indices = range(72,73)
        # indices = [*range(80,140,1), *range(80+256,140+256,1), *range(80+256*2, 140+256*2, 1), *range(80+256*3, 140+256*3, 1), *range(80+256*4, 140+256*4, 1), *range(80+256*5, 140+256*5, 1)]
        
        colours = get_colours(len(indices))
        
        variables_fit = {"A":[], "A_nu0":[], "solar_line_area":[], "chisq":[], "temperature":[], "t0":[]}
        
        for index, frame_index in enumerate(indices):
            print("frame_index=%i (%i/%i)" %(frame_index, index, len(indices)))
        
            """spectral grid and blaze functions of all orders"""
            
            #get data, fit temperature and get starting function parameters
            d = select_data(d, frame_index)
            d = fit_temperature(d, hdf5_file, plot=SETUP)
            d = get_start_params(d)
            
            param_dict = make_param_dict(d)
            
            
            #sigma for preferentially fitting of certain pixels
            d["sigma"] = np.ones_like(d["spectrum_norm"])
            smi = d["absorption_pixel"]
            d["sigma"][smi-18:smi+19] = 0.01
            # d["sigma"][:100] = 10.
            # d["sigma"][280:] = 10.
            
            
            variables = {}
            for key, value in param_dict.items():
                variables[key] = value[0]
            
            
            variables, chisq = fit_spectrum(param_dict, variables, d)
            print("chisq=", chisq)
        
            spectrum_norm = d["spectrum_norm"]
            
            solar_fit = calc_spectrum(variables, d)
            scalar = 1.0 / max(solar_fit)
            
            solar_fit_norm = solar_fit * scalar
            
            solar_fit_slr = calc_spectrum(variables, d, I0=d["I0_lr_slr"])
            solar_fit_slr_norm = solar_fit_slr * scalar
            
            if PLOT_RAW:
                ax1.plot(spectrum_norm, color=colours[index])
                ax1.plot(solar_fit_norm, color=colours[index], linestyle=":")
                ax1.plot(solar_fit_slr_norm, color=colours[index], linestyle="--")
                ax2.plot(solar_fit_slr_norm - solar_fit_norm, color=colours[index])
            
            
            
            
            area = area_under_curve(solar_fit_slr_norm, solar_fit_norm)
        
            variables_fit["A"].append(d["A"])
            variables_fit["A_nu0"].append(d["A_nu0"])
            variables_fit["solar_line_area"].append(area)
            variables_fit["chisq"].append(chisq)
            variables_fit["t0"].append(d["t0"])
            variables_fit["temperature"].append(d["temperature"])
            for key in variables.keys():
                if key not in variables_fit.keys():
                    variables_fit[key] = []
                variables_fit[key].append(variables[key])
        
        
        variables_fit["solar_line_area"] = np.array(variables_fit["solar_line_area"])  
        variables_fit["chisq"] = np.array(variables_fit["chisq"])
        variables_fit["A_nu0"] = np.array(variables_fit["A_nu0"])
        variables_fit["A"] = np.array(variables_fit["A"])
        variables_fit["t0"] = np.array(variables_fit["t0"])
        variables_fit["temperature"] = np.array(variables_fit["temperature"])
        
        #remove really bad points 
        variables_fit["error"] = variables_fit["chisq"]/1000.
        good_indices = np.where(variables_fit["error"] < 10)[0]
        
        variables_fit["solar_line_area_norm"] = variables_fit["solar_line_area"]/np.max(variables_fit["solar_line_area"][good_indices])
        
        
        # plt.figure()
        # # plt.plot(variables_fit["A_nu0"], variables_fit["solar_line_area_norm"], label="AOTF")
        # for key in variables.keys():
        #     plt.plot(variables_fit["A_nu0"], variables_fit[key] / max(variables_fit[key]), label=key)
        # plt.legend()
        
        # plt.figure()
        # plt.errorbar(variables_fit["A_nu0"], variables_fit["solar_line_area_norm"], yerr=variables_fit["error"], linestyle="")
        
        
        # plt.figure()
        # plt.scatter(range(len(variables_fit["error"])), variables_fit["error"])
        
        plt.figure()
        plt.title("%s:\nAOTF shape from solar line simulation" %hdf5_filename)
        plt.scatter(variables_fit["A_nu0"][good_indices], variables_fit["solar_line_area_norm"][good_indices])
        plt.xlabel("Wavenumber of AOTF peak")
        plt.ylabel("AOTF function (normalised)")
        plt.savefig("%s_%s_aotf_function.png" %(hdf5_filename, solar_spectrum))
        plt.close()
        
        
        x = variables_fit["A_nu0"][good_indices]
        x2 = variables_fit["A"][good_indices]
        y = variables_fit["solar_line_area_norm"][good_indices]
        t = variables_fit["temperature"][good_indices]
        t0 = variables_fit["t0"][good_indices]
        chi = variables_fit["chisq"][good_indices]
        
        np.savetxt("%s_%s_aotf_function.txt" %(hdf5_filename, solar_spectrum), np.array([x2,x,t0,t,y,chi]).T, fmt="%.6f", delimiter=",", header="Frequency,Wavenumber,TGO Temperature,Fitted Temperature,AOTF function,Chi Squared")
        
        
        # param_dict = make_param_dict_aotf(d)
        
        # variables, chisq = fit_aotf(param_dict, x, y)
        # aotf = F_aotf2(variables, x, y)
        
        # plt.figure()
        # plt.scatter(x, y, c=t)
        # plt.plot(x, aotf)
        
        
        # plt.figure()
        # plt.plot(d["nu_hr"], d["I0_lr"])
        # plt.plot(d["nu_hr"], d["I0_lr_slr"])
        
    
