# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:20:10 2021

@author: iant

MINISCAN FITTING
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tools.plotting.colours import get_colours

from instrument.calibration.so_aotf_ils.simulation_functions import (get_file, get_data_from_file, select_data, fit_temperature,
get_start_params, make_param_dict, calc_spectrum, fit_spectrum, area_under_curve, get_solar_spectrum)
from instrument.calibration.so_aotf_ils.simulation_config import sim_parameters

from instrument.calibration.so_aotf_ils.simulation_functions import get_absorption_line_indices

#AOTF freqs where AOTF is strong: 4380-4390cm-1 = 26560-26640kHz

line = 4383.5
# line = 4276.1
# line = 3787.9

SETUP = False
# SETUP = True


PLOT_RAW = False
# PLOT_RAW = True


PLOT_CENTRE_VS_ADJACENT_ORDERS = False
# PLOT_CENTRE_VS_ADJACENT_ORDERS = True


if PLOT_RAW:
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_xlabel("Pixel Number")
    ax1.set_ylabel("Normalised Spectra")
    ax2.set_ylabel("Solar Line")


filenames = sim_parameters[line]["filenames"]
solar_spectra = sim_parameters[line]["solar_spectra"].keys()

print("There are %i filenames and %i solar spectra" %(len(filenames), len(solar_spectra)))


parser = argparse.ArgumentParser(description = "Select filename and solar spectrum indices")
parser.add_argument("-fi", "--filename_index", help = "Select filename index", required = False, type=str)
parser.add_argument("-si", "--solar_index", help = "Select solar spectrum index ACE or PFS", required = False, type=str)
#must be string, otherwise int(0) is same as no argument supplied

args = parser.parse_args()

if args.filename_index:
    print("Filename index:", args.filename_index)
    filenames = [filenames[int(args.filename_index)]]
if args.solar_index:
    print("Solar spectrum suffix:", args.solar_index)
    solar_spectra = [args.solar_index]



for filename in filenames:
    regex = re.compile(filename) #(approx. orders 188-202) in steps of 8kHz



    hdf5_file, hdf5_filename = get_file(regex)
    d = get_data_from_file(hdf5_file, hdf5_filename)
    
    
    
    for solar_spectrum in solar_spectra:
        d["line"] = line
        d["solar_spectrum"] = solar_spectrum
        d = get_solar_spectrum(d, plot=SETUP)
    
    
        # indices = range(len(d["aotf_freqs"]))
        # indices = range(0, len(d["aotf_freqs"]), 50)
        indices = range(0,255,1)
        # indices = get_absorption_line_indices(d)
        # indices = range(0, 101, 20)
        # indices = range(72,73)
        # indices = [*range(80,140,1), *range(80+256,140+256,1), *range(80+256*2, 140+256*2, 1), *range(80+256*3, 140+256*3, 1), *range(80+256*4, 140+256*4, 1), *range(80+256*5, 140+256*5, 1)]
        
        colours = get_colours(len(indices))
        
        variables_fit = {"A":[], "A_nu0":[], "solar_line_area":[], "solar_line_area_c":[], "chisq":[], "temperature":[], "t0":[]}
        
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

            for key in variables.keys():
                if key not in variables_fit.keys():
                    variables_fit[key] = []
                variables_fit[key].append(variables[key])

        
            
            solar_fit = calc_spectrum(variables, d)
            # scalar = 1.0 / max(solar_fit)
            # solar_fit_norm = solar_fit * scalar
            
            solar_fit_slr = calc_spectrum(variables, d, I0=d["I0_lr_slr"])
            # solar_fit_slr_norm = solar_fit_slr * scalar

            pixels_solar_line_area = sim_parameters[line]["pixels_solar_line_area"] #select pixel range i.e. to avoid adjacent order solar lines
            # area = area_under_curve(solar_fit_slr_norm[pixels_solar_line_area], solar_fit_norm[pixels_solar_line_area])
            area = area_under_curve(solar_fit_slr[pixels_solar_line_area], solar_fit[pixels_solar_line_area])
            
            #calculate relative contributions from main order and all adjacent orders
            centre_order_contribution = calc_spectrum(variables, d, order_range=[d["centre_order"]])
            centre_order_contribution_slr = calc_spectrum(variables, d, I0=d["I0_lr_slr"], order_range=[d["centre_order"]])
            order_range = list(range(d["m_range"][0], d["m_range"][1]+1))
            order_range.remove(d["centre_order"])
            non_centre_order_contribution = calc_spectrum(variables, d, order_range=order_range)
            
            if PLOT_CENTRE_VS_ADJACENT_ORDERS:
                    
                plt.plot(d["spectrum_norm"]*max(centre_order_contribution + non_centre_order_contribution), label="Raw")
                plt.plot(centre_order_contribution, label="Centre order contribution")
                plt.plot(non_centre_order_contribution, label="Other order contribution")
                plt.plot(centre_order_contribution + non_centre_order_contribution, label="Total")
                plt.axvline(smi)
                plt.legend(loc="lower right")
                sys.exit()
            
            #now do solar line calc but only for main order - doesn't work
            solar_fit_c = calc_spectrum(variables, d, order_range=[d["centre_order"]])
            # scalar_c = 1.0 / max(solar_fit_c)
            # solar_fit_norm_c = solar_fit_c * scalar_c
            
            solar_fit_slr_c = calc_spectrum(variables, d, I0=d["I0_lr_slr"], order_range=range(d["centre_order"],d["centre_order"]+1))
            # solar_fit_slr_norm_c = solar_fit_slr_c * scalar_c
            
            ratio_centre = centre_order_contribution_slr[smi] / non_centre_order_contribution[smi]

            # area_c = area_under_curve(solar_fit_slr_norm_c[pixels_solar_line_area], solar_fit_norm_c[pixels_solar_line_area]) * ratio_centre
            area_c = area_under_curve(solar_fit_slr_c[pixels_solar_line_area], solar_fit_c[pixels_solar_line_area]) * ratio_centre
            
            
            
            if PLOT_RAW:
                ax1.plot(d["spectrum_norm"] * max(solar_fit_slr), color=colours[index])
                # ax1.plot(solar_fit_norm, color=colours[index], linestyle=":")
                # ax1.plot(solar_fit_slr_norm, color=colours[index], linestyle="--")
                # ax2.plot(solar_fit_slr_norm - solar_fit_norm, color=colours[index])
                # ax1.plot(solar_fit_norm_c, color=colours[index], linestyle=":")
                # ax1.plot(solar_fit_slr_norm_c, color=colours[index], linestyle="--")
                # ax2.plot(solar_fit_slr_norm_c - solar_fit_norm_c, color=colours[index])

                ax1.plot(solar_fit, color=colours[index], linestyle=":")
                ax1.plot(solar_fit_slr, color=colours[index], linestyle="--")
                ax2.plot(solar_fit_slr - solar_fit, color=colours[index])
                # ax1.plot(solar_fit_c, color=colours[index], linestyle="-")
                # ax1.plot(solar_fit_slr_c, color=colours[index], linestyle="-.")
                # ax2.plot(solar_fit_slr_c - solar_fit_c, color=colours[index])
                sys.exit()
            
            
        
            variables_fit["A"].append(d["A"])
            variables_fit["A_nu0"].append(d["A_nu0"])
            variables_fit["solar_line_area"].append(area)
            variables_fit["solar_line_area_c"].append(area_c)
            variables_fit["chisq"].append(chisq)
            variables_fit["t0"].append(d["t0"])
            variables_fit["temperature"].append(d["temperature"])
        
        
        variables_fit["solar_line_area"] = np.array(variables_fit["solar_line_area"])  
        variables_fit["solar_line_area_c"] = np.array(variables_fit["solar_line_area_c"])  
        variables_fit["chisq"] = np.array(variables_fit["chisq"])
        variables_fit["A_nu0"] = np.array(variables_fit["A_nu0"])
        variables_fit["A"] = np.array(variables_fit["A"])
        variables_fit["t0"] = np.array(variables_fit["t0"])
        variables_fit["temperature"] = np.array(variables_fit["temperature"])
        
        #remove really bad points 
        variables_fit["chisq"] = variables_fit["chisq"]/variables_fit["solar_line_area"]
        
        error_scaler = 10000.0
        
        plt.figure(constrained_layout=True)
        plt.title("%s:\nAOTF shape from solar line simulation" %hdf5_filename)
        # plt.scatter(variables_fit["A_nu0"], variables_fit["solar_line_area"])
        plt.errorbar(variables_fit["A_nu0"], variables_fit["solar_line_area"], yerr=variables_fit["chisq"]/error_scaler, ls="none", marker=".")
        plt.errorbar(variables_fit["A_nu0"], variables_fit["solar_line_area_c"], yerr=variables_fit["chisq"]/error_scaler, ls="none", marker=".")
        plt.xlabel("Wavenumber of AOTF peak")
        plt.ylabel("AOTF function (normalised)")
        plt.ylim([-0.1, max([max(variables_fit["solar_line_area"]), max(variables_fit["solar_line_area_c"])])+0.1])
        plt.savefig("%s_%s_%.0f_aotf_function.png" %(hdf5_filename, solar_spectrum, line))
        plt.close()
        
        
        header = ""
        data_out = []
        for name in ["A_nu0", "A", "temperature", "t0", "solar_line_area", "solar_line_area_c", "chisq"]:
            header += "%s," %name
            data_out.append(variables_fit[name])


        for key in param_dict.keys():
            header += "%s," %key
            data_out.append(variables_fit[key])


        np.savetxt("%s_%s_%.0f_aotf_function.txt" %(hdf5_filename, solar_spectrum, line), np.array(data_out).T, fmt="%.6f", delimiter=",", header=header)

        
        # np.savetxt("%s_%s_%.0f_aotf_function.txt" %(hdf5_filename, solar_spectrum, line), np.array([x2,x,t0,t,y,y2,chi]).T, fmt="%.6f", delimiter=",", header="Frequency,Wavenumber,TGO Temperature,Fitted Temperature,AOTF function,AOTF function centre order,Chi Squared")
        
        
        # param_dict = make_param_dict_aotf(d)
        
        # variables, chisq = fit_aotf(param_dict, x, y)
        # aotf = F_aotf2(variables, x, y)
        
        # plt.figure()
        # plt.scatter(x, y, c=t)
        # plt.plot(x, aotf)
        
        
        # plt.figure()
        # plt.plot(d["nu_hr"], d["I0_lr"])
        # plt.plot(d["nu_hr"], d["I0_lr_slr"])
        
    
