# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:18:31 2020

@author: iant

CALCULATE PIXEL TEMPERATURE SHIFTS FROM GROUND CAL DATA

"""
import sys
#import os
#import h5py
import numpy as np
#from datetime import datetime

import matplotlib.pyplot as plt
#from scipy.signal import savgol_filter, butter, lfilter
#from scipy.optimize import curve_fit

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions import make_filelist
from tools.file.filename_lists import getFilenameList

from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption

from tools.general.get_consecutive_indices import get_consecutive_indices
from tools.general.get_minima_maxima import get_local_minima
from tools.plotting.colours import get_colours

from instrument.nomad_lno_instrument import t_p0

from tools.spectra.fit_polynomial import fit_polynomial

from presentations.plot_figures_for_cal_paper_2020_functions import getExternalTemperatureReadings, findOrder, plotTemperatureDependency, applyFilter, fitCurveError

SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True


"""blank"""
fileLevel = "hdf5_level_0p1a"
obspaths = []
model = "PFM"
title = ""





channel = "lno"
#choose what to plot
plot_gradient = True #fig2
#    plot_gradient = False #fig2
plot_spectra = True #fig1
#    plot_spectra = False #fig1
plot_shift = True #temp dep
#    plot_shift = False #temp dep








#fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))
#ax3.grid(True)
#fig4, ax4 = plt.subplots(figsize=(FIG_X, FIG_Y))
#ax3.grid(True)


#GRADIENT_APPROXIMATION = 0.85
SUFFIX = ""
#MATCHING_TEMPERATURES_FOUND = 2
#DELTA_TEMPERATURE = 15.0
#MATCHING_LINES_FOUND = 1
#SUBPIXEL_SEARCH_RANGE = 20
#USE_CSL_TEMPERATURES = True
USE_CSL_TEMPERATURES = False
CSL_TEMPERATURE_COLUMN = 4

order_dicts = {
#134:{"aotf_frequency":18899.3, "pixel_range":[50,300], "nu_range":[3013., 3014.5], "molecule":"ch4"},    
136:{"aotf_frequency":19212.4, "pixel_range":[20,300], "n_stds":0.93, "molecule":"ch4"},    
146:{"aotf_frequency":20771.9, "pixel_range":[50,300], "n_stds":0.93, "molecule":"c2h2"},    
167:{"aotf_frequency":24020.9, "pixel_range":[50,300], "n_stds":0.93, "molecule":"ch4"},    
#190:{"aotf_frequency":27558.5, "pixel_range":[50,300], "nu_range":[3013., 3014.5], "molecule":"ch4"},    
        }


chosen_bin = 12


#STD_CUTOFF = 0.5
#    STD_CUTOFF = 1.0
END_OF_DETECTOR = 292
START_OF_DETECTOR = 10
POLYNOMIAL_RANGE = 20
GENERIC_GRADIENT_STDEV = 0.05
    
    

pixels = np.arange(320)
    
PIXEL_RANGE = [5, 2720]

colours = get_colours(42) #from -20C to +20C


    
order_data_dict = {}

for diffraction_order, order_dict in order_dicts.items():

    order_data_dict[diffraction_order] = {}

#    if plot_gradient: 
#        fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
#        ax2.grid(True)



    order_data_dict[diffraction_order]["mean_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["std_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["n_gradients_all_bins"] = []

    
    
    order_data_dict[diffraction_order]["obspaths_all"] = getFilenameList("ground cal %s cell%s" %(order_dict["molecule"], SUFFIX))
    hdf5Files, hdf5Filenames, _ = make_filelist(order_data_dict[diffraction_order]["obspaths_all"], fileLevel, model=model, silent=True)
    
#    order_data_dict[diffraction_order]["hdf5Filenames"] = hdf5Filenames

    order_data_dict[diffraction_order]["measurement_temperatures"] = []
    order_data_dict[diffraction_order]["colour"] = []
    order_data_dict[diffraction_order]["hdf5_filenames"] = []
    order_data_dict[diffraction_order]["spectra"] = []
    order_data_dict[diffraction_order]["continuum_mean"] = []
    order_data_dict[diffraction_order]["continuum_std"] = []

    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

        detector_data_all = hdf5_file["Science/Y"][...]
        
#        window_top_all = hdf5_file["Channel/WindowTop"][...]
#        binning = hdf5_file["Channel/Binning"][0] + 1
#        integration_time = hdf5_file["Channel/IntegrationTime"][0]
        sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
        measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
        datetimes = hdf5_file["DateTime"][...]
        
        aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
        
        order_data_dict[diffraction_order]["aotf_frequencies_all"] = aotf_frequencies
        
        
        if not sbsf:
            print("%s does not have background subtraction" %hdf5_filename)
            sys.exit()
            
        spectra = detector_data_all[:, chosen_bin, :]
        #basic bad pixel correction
        if chosen_bin == 12:
            spectra[:, 138] = np.mean([spectra[:, 137], spectra[:, 140]], axis=0)
            spectra[:, 139] = np.mean([spectra[:, 137], spectra[:, 140]], axis=0)
        if chosen_bin == 11:
            spectra[:, 284] = np.mean([spectra[:, 283], spectra[:, 285]], axis=0)
            spectra[:, 138] = np.mean([spectra[:, 137], spectra[:, 140]], axis=0)
            spectra[:, 139] = np.mean([spectra[:, 138], spectra[:, 140]], axis=0)
        if chosen_bin == 10:
            spectra[:, 41] = np.mean([spectra[:, 40], spectra[:, 42]], axis=0)
        if chosen_bin == 8:
            spectra[:, 175] = np.mean([spectra[:, 174], spectra[:, 176]], axis=0)

        #selected matching orders in fullscan/miniscan observations
        chosen_order_indices = [i for i, aotf_frequency in enumerate(aotf_frequencies) if (order_dict["aotf_frequency"]-1.1) < aotf_frequency < (order_dict["aotf_frequency"]+1.1)]
        order_data_dict[diffraction_order]["aotf_frequencies_all"] = aotf_frequencies
        

            
        for chosen_order_index in chosen_order_indices:
            
            spectrum = spectra[chosen_order_index, :]
            measurement_time = datetimes[chosen_order_index]
            
            normalised_spectrum = spectrum / np.max(spectrum)
            
            if USE_CSL_TEMPERATURES: #overwrite temperature with one from external file
                measurement_temperature = getExternalTemperatureReadings(measurement_time.decode(), CSL_TEMPERATURE_COLUMN)


            order_data_dict[diffraction_order]["hdf5_filenames"].append(hdf5_filename)
            order_data_dict[diffraction_order]["measurement_temperatures"].append(measurement_temperature)
            order_data_dict[diffraction_order]["colour"].append(colours[int(measurement_temperature)+20])
#            minimum_indices_all.append(minimum_indices) #make list of lists of all absorption minima
            
            
            #remove continuum
            continuum = baseline_als(normalised_spectrum)
            order_data_dict[diffraction_order]["continuum_mean"].append(np.mean(continuum))
            order_data_dict[diffraction_order]["continuum_std"].append(np.std(continuum))

            absorption_spectrum = normalised_spectrum / continuum
            order_data_dict[diffraction_order]["spectra"].append(absorption_spectrum)


        if len(chosen_order_indices) == 0:
            text = "AOTF frequency %0.0f kHz (order %i) %0.1fC not found in file %s" %(order_dict["aotf_frequency"], diffraction_order, measurement_temperature, hdf5_filename)
            aotf_frequency_all = hdf5_file["Channel/AOTFFrequency"][...]
            diffraction_orders = np.asfarray([findOrder(channel, aotf_frequency, silent=True) for aotf_frequency in aotf_frequency_all])
            text += " (%0.0f-%0.0fkHz; orders=%i-%i)" %(min(aotf_frequency_all), max(aotf_frequency_all), min(diffraction_orders), max(diffraction_orders))
            print(text)

        else:
            print("AOTF frequency %0.0f kHz (order %i) %0.1fC found in file %s. Adding to search list" %(order_dict["aotf_frequency"], diffraction_order, measurement_temperature, hdf5_filename))
                        
            

    #sort by temperature
    sort_indices = np.argsort(np.asfarray(order_data_dict[diffraction_order]["measurement_temperatures"]))
    
    for list_name in ["colour", "continuum_mean", "continuum_std", "hdf5_filenames", "measurement_temperatures", "spectra"]:
        order_data_dict[diffraction_order][list_name] = [order_data_dict[diffraction_order][list_name][i] for i in sort_indices]
    
    
            
    if plot_spectra:
        
        fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
        ax1.grid(True)
        ax1.set_title("Diffraction order %i" %diffraction_order)
        #plot horizontal lines for st dev limits
        ax1.axhline(order_dict["n_stds"], color="k", linestyle="--")
        
        order_data_dict[diffraction_order]["minima"] = {}

        for absorption_spectrum, measurement_temperature, colour, continuum_mean, continuum_std, hdf5_filename in zip(
                order_data_dict[diffraction_order]["spectra"], 
                order_data_dict[diffraction_order]["measurement_temperatures"],
                order_data_dict[diffraction_order]["colour"],
                order_data_dict[diffraction_order]["continuum_mean"],
                order_data_dict[diffraction_order]["continuum_std"],
                order_data_dict[diffraction_order]["hdf5_filenames"]
                ):
            ax1.plot(absorption_spectrum, color=colour, linestyle="--", label="%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature))

    
            plt.legend()
        
 

            #find absorption minima
            order_data_dict[diffraction_order]["minima"][measurement_temperature] = {}
            order_data_dict[diffraction_order]["minima"][measurement_temperature]["pixels"] = []
            order_data_dict[diffraction_order]["minima"][measurement_temperature]["chisq"] = []
        
            minimum_indices = get_local_minima(absorption_spectrum)
            for minimum_index in minimum_indices:
                if order_dict["pixel_range"][0] < minimum_index < order_dict["pixel_range"][1]:
                    if absorption_spectrum[minimum_index] < order_dict["n_stds"]:
                        #index below std line
                        absorption_indices = np.arange(minimum_index - 5, minimum_index + 6, 1)
                        #apply gaussian fit
                        fit = fit_gaussian_absorption(absorption_indices, absorption_spectrum[absorption_indices], error=True)
                        if fit[3] < 0.005:
                            ax1.plot(fit[0], fit[1], "k")
                            ax1.axvline(fit[2], color=colour)
                        
                            order_data_dict[diffraction_order]["minima"][measurement_temperature]["pixels"].append(fit[2])
                            order_data_dict[diffraction_order]["minima"][measurement_temperature]["chisq"].append(fit[2])
            
            
#    for measurement_temperature in order_data_dict[diffraction_order]["minima"][measurement_temperature]:
#        pixels = order_data_dict[diffraction_order]["minima"][measurement_temperature]["pixels"]
#        chisqs = order_data_dict[diffraction_order]["minima"][measurement_temperature]["chisq"]

    #find which minima are from the same absorptions
    #first - calculate predicted pixel locations
    first_temperature = list(np.min(order_data_dict[diffraction_order]["minima"].keys()))[0]
    first_pixels = order_data_dict[diffraction_order]["minima"][first_temperature]["pixels"]
    
    #empty array to hold 
    found_absorptions = [[i] for i in first_pixels]
    found_temperatures = [[first_temperature] for i in range(len(first_pixels))]
    coefficients = [[] for i in first_pixels]
    
    first_pixel_shift = t_p0(first_temperature)
    
    for temperature in order_data_dict[diffraction_order]["minima"]:
        if temperature != first_temperature:
            abs_pixel_shift = t_p0(temperature)
            pixel_shift = first_pixel_shift - abs_pixel_shift
            
            #loop through pixels where abs was found at lowest temperature
            for pixel_index, first_pixel in enumerate(first_pixels):
                #find expected pixel due to temperature shift
                expected_pixel = first_pixel + pixel_shift
                
                #search through pixels
                pixels = order_data_dict[diffraction_order]["minima"][temperature]["pixels"]
                for pixel in pixels:
                    if np.abs(expected_pixel - pixel) < 4:
                        print("Match found!")
                        print(temperature, pixel, expected_pixel)
                        found_absorptions[pixel_index].append(pixel)
                        found_temperatures[pixel_index].append(temperature)

    found_pixel_shifts = [[found_absorptions[j][i] - found_absorptions[j][0] for i in range(1, len(found_absorptions[j]))] for j in range(len(found_absorptions))]

    #loop through the absorption shifts found in the order
    #plot pixel shift vs instrument temperature
    for slope_index, (found_pixel_shift, found_temperature) in enumerate(zip(found_pixel_shifts, found_temperatures)):
        _, coeffs = fit_polynomial(found_temperature[1:], found_pixel_shift, coeffs=True)
        
        coefficients[slope_index] = coeffs
        



    if plot_shift: 
        fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y)) #plot pixel shift vs temperature
        shift_colours = get_colours(len(found_pixel_shifts))
        ax2.set_title("Diffraction order %i" %diffraction_order)
        ax2.set_ylabel("Pixel shift")
        ax2.set_xlabel("Instrument Temperature (C)")
        for slope_index, (found_pixel_shift, found_temperature) in enumerate(zip(found_pixel_shifts, found_temperatures)):
            ax2.scatter(found_temperature[1:], found_pixel_shift, color=shift_colours[slope_index], label="Pixel %0.0f" %first_pixels[slope_index])
            ax2.plot(found_temperature[1:], np.polyval(coefficients[slope_index], found_temperature[1:]), c=shift_colours[slope_index])
            ax2.text(0, 10-2*slope_index, coefficients[slope_index], color=shift_colours[slope_index])
        ax2.legend(loc="upper left")
        
    
    order_data_dict[diffraction_order]["shifts"] = {}
    order_data_dict[diffraction_order]["shifts"]["coefficients"] = coefficients
    order_data_dict[diffraction_order]["shifts"]["temperatures"] = found_temperatures
    order_data_dict[diffraction_order]["shifts"]["pixel_shifts"] = found_pixel_shifts
    order_data_dict[diffraction_order]["shifts"]["starting_pixels"] = first_pixels
    
    mean_shift = np.mean([i[0] for i in coefficients])
    order_data_dict[diffraction_order]["mean_shift"] = mean_shift
    



#
##LINEAR
##plot all gradients as a function of pixel number
#ax3.set_title("Mean gradient of temperature pixel shift vs. diffraction order (pixels / C)")
#ax3.set_xlabel("Diffraction Order")
#ax3.set_ylabel("Mean gradient for all matching absorption lines")
#ax3.errorbar(orders, mean_gradient_all_orders, yerr=std_gradient_all_orders, fmt="o", color={"old":"b", "new":"r"}[lno_detector])
#
##write number of gradients 
#for order, mean_gradient, n_gradients in zip(orders, mean_gradient_all_orders, n_gradients_all_orders):
#    if np.isfinite(mean_gradient):
#        ax3.text(order, mean_gradient, "%i" %n_gradients)
#
#    
#fitCurveError(np.asfarray(orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(mean_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(std_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], ax=ax3)
#
##QUADRATIC
##plot all gradients as a function of pixel number
#ax4.set_title("Mean gradient of temperature pixel shift vs. diffraction order")
#ax4.set_xlabel("Nearest diffraction Order")
#ax4.set_ylabel("Mean gradient for all matching absorption lines")
#ax4.errorbar(orders, mean_gradient_all_orders, yerr=std_gradient_all_orders, fmt="o", color={"old":"b", "new":"r"}[lno_detector])
#
##write number of gradients 
#for order, mean_gradient, n_gradients in zip(orders, mean_gradient_all_orders, n_gradients_all_orders):
#    if np.isfinite(mean_gradient):
#        ax4.text(order, mean_gradient, "%i" %n_gradients)
#   
#fitCurveError(np.asfarray(orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(mean_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(std_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], ax=ax4, fit="quadratic")
#
#
