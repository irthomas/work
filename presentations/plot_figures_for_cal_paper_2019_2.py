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

from tools.spectra.fit_polynomial import fit_polynomial, fit_linear_errors

from presentations.plot_figures_for_cal_paper_2020_functions import getExternalTemperatureReadings, findOrder, plotTemperatureDependency, applyFilter, fitCurveError

SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

USE_FIT_ERROR = False
# USE_FIT_ERROR = True


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
USE_CSL_TEMPERATURES = True
# USE_CSL_TEMPERATURES = False
CSL_TEMPERATURE_COLUMN = 4

order_dicts = {
# 134:{"aotf_frequency":18899.3, "pixel_range":[50,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},    
# 136:{"aotf_frequency":19212.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},    
# 146:{"aotf_frequency":20771.9, "pixel_range":[50,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},    
# 167:{"aotf_frequency":24020.9, "pixel_range":[50,320], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
#190:{"aotf_frequency":27558.5, "pixel_range":[50,300], "n_stds":0.93, "n_points":3, "molecule":"co"},    


130:{"aotf_frequency":18271.7, "pixel_range":[20,300], "n_stds":0.9, "n_points":2, "molecule":"ch4"},
# 131:{"aotf_frequency":18428.8, "pixel_range":[20,300], "n_stds":0.85, "n_points":3, "molecule":"ch4"},
# 132:{"aotf_frequency":18585.8, "pixel_range":[20,300], "n_stds":0.8, "n_points":3, "molecule":"ch4"},
# 133:{"aotf_frequency":18742.6, "pixel_range":[20,300], "n_stds":0.6, "n_points":3, "molecule":"ch4"},
# # 134:{"aotf_frequency":18899.3, "pixel_range":[20,300], "n_stds"3, "n_points":3, "molecule":"ch4"},
# 135:{"aotf_frequency":19055.9, "pixel_range":[20,300], "n_stds":0.6, "n_points":3, "molecule":"ch4"},
# 136:{"aotf_frequency":19212.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# 137:{"aotf_frequency":19368.8, "pixel_range":[20,300], "n_stds":0.6, "n_points":3, "molecule":"ch4"},
# 138:{"aotf_frequency":19525.1, "pixel_range":[20,300], "n_stds":0.6, "n_points":3, "molecule":"ch4"},
# 139:{"aotf_frequency":19681.3, "pixel_range":[20,300], "n_stds":0.6, "n_points":3, "molecule":"ch4"},

# 140:{"aotf_frequency":19837.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 141:{"aotf_frequency":19993.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 142:{"aotf_frequency":20149.3, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 143:{"aotf_frequency":20305.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 144:{"aotf_frequency":20460.7, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 145:{"aotf_frequency":20616.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 146:{"aotf_frequency":20771.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 147:{"aotf_frequency":20927.3, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 148:{"aotf_frequency":21082.6, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 149:{"aotf_frequency":21237.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},

# 150:{"aotf_frequency":21393.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 151:{"aotf_frequency":21548.2, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 152:{"aotf_frequency":21703.2, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 153:{"aotf_frequency":21858.2, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 154:{"aotf_frequency":22013.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# 155:{"aotf_frequency":22167.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"c2h2"},
# # 156:{"aotf_frequency":22322.6, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 157:{"aotf_frequency":22477.3, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 158:{"aotf_frequency":22631.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 159:{"aotf_frequency":22786.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},

# # 160:{"aotf_frequency":22940.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 161:{"aotf_frequency":23095.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 162:{"aotf_frequency":23249.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 163:{"aotf_frequency":23404.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 164:{"aotf_frequency":23558.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 165:{"aotf_frequency":23712.6, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# 166:{"aotf_frequency":23866.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 167:{"aotf_frequency":24020.9, "pixel_range":[20,320], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 168:{"aotf_frequency":24175.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 169:{"aotf_frequency":24329.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    

# 170:{"aotf_frequency":24483.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 171:{"aotf_frequency":24637.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 172:{"aotf_frequency":24791.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 173:{"aotf_frequency":24944.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# 174:{"aotf_frequency":25098.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"h2o"},    
# # 175:{"aotf_frequency":25252.6, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 176:{"aotf_frequency":25406.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 177:{"aotf_frequency":25560.2, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 178:{"aotf_frequency":25714.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 179:{"aotf_frequency":25867.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},

# # 180:{"aotf_frequency":26021.5, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 181:{"aotf_frequency":26175.3, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 182:{"aotf_frequency":26329.0, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 183:{"aotf_frequency":26482.7, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# # 184:{"aotf_frequency":26636.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"ch4"},
# 185:{"aotf_frequency":26790.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 186:{"aotf_frequency":26943.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 187:{"aotf_frequency":27097.4, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 188:{"aotf_frequency":27251.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 189:{"aotf_frequency":27404.8, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},

# 190:{"aotf_frequency":27558.5, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 191:{"aotf_frequency":27712.2, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 192:{"aotf_frequency":27865.9, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 193:{"aotf_frequency":28019.6, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 194:{"aotf_frequency":28173.3, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
# 195:{"aotf_frequency":28327.1, "pixel_range":[20,300], "n_stds":0.93, "n_points":3, "molecule":"co"},
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

colours = get_colours(42, cmap="plasma") #from -20C to +20C


    
order_data_dict = {}

for diffraction_order, order_dict in order_dicts.items():

    order_data_dict[diffraction_order] = {}


    order_data_dict[diffraction_order]["mean_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["std_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["n_gradients_all_bins"] = []

    
    
    order_data_dict[diffraction_order]["obspaths_all"] = getFilenameList("ground cal %s cell%s" %(order_dict["molecule"], SUFFIX))
    hdf5Files, hdf5Filenames, _ = make_filelist(order_data_dict[diffraction_order]["obspaths_all"], fileLevel, model=model, silent=True)
    
    order_data_dict["hdf5Filenames"] = {}

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
        
        order_data_dict["hdf5Filenames"][hdf5_filename] = {"aotf_frequencies_all":aotf_frequencies}
        
        
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
            

        #loop through spectra where aotf frequency is close enough to specified value
        for chosen_order_index in chosen_order_indices:
            
            spectrum = spectra[chosen_order_index, :]
            measurement_time = datetimes[chosen_order_index]
            
            normalised_spectrum = spectrum / np.max(spectrum)
            
            if USE_CSL_TEMPERATURES: #overwrite temperature with one from external file
                measurement_temperature = getExternalTemperatureReadings(measurement_time.decode(), CSL_TEMPERATURE_COLUMN)


            order_data_dict[diffraction_order]["hdf5_filenames"].append(hdf5_filename)
            order_data_dict[diffraction_order]["measurement_temperatures"].append(measurement_temperature)
            order_data_dict[diffraction_order]["colour"].append(colours[int(measurement_temperature)+20])
            
            
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
    
    #how many points before/after minima should be included
    n_points = order_dict["n_points"]
    
            
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
        if plot_spectra:
            ax1.plot(absorption_spectrum, color=colour, linestyle="dashed", label="%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature))


        plt.legend()
    
 

        #find absorption minima using gaussian and polynomial
        order_data_dict[diffraction_order]["minima"][measurement_temperature] = {}
        order_data_dict[diffraction_order]["minima"][measurement_temperature]["exp_pixels"] = []
        order_data_dict[diffraction_order]["minima"][measurement_temperature]["exp_chisq"] = []
        order_data_dict[diffraction_order]["minima"][measurement_temperature]["poly_pixels"] = []
        order_data_dict[diffraction_order]["minima"][measurement_temperature]["poly_chisq"] = []
        order_data_dict[diffraction_order]["minima"][measurement_temperature]["spectra"] = [] #copy chosen spectra to  new dict location
    
        minimum_indices = get_local_minima(absorption_spectrum)
        found = False
        for minimum_index in minimum_indices:
            if order_dict["pixel_range"][0] < minimum_index < order_dict["pixel_range"][1]:
                if absorption_spectrum[minimum_index] < order_dict["n_stds"]:
                    #index below std line
                    absorption_indices = np.arange(minimum_index - n_points, minimum_index + (n_points + 1), 1)
                    #apply gaussian fit
                    gauss_fit = fit_gaussian_absorption(absorption_indices, absorption_spectrum[absorption_indices], error=True)
                    if type(gauss_fit[0]) != list:
                    # if gauss_fit[3] < 1:
                        found = True
                        if plot_spectra:
                            # print(gauss_fit[3])
                            ax1.plot(gauss_fit[0], gauss_fit[1], "k")
                            ax1.axvline(gauss_fit[2], color=colour)
                    
                        order_data_dict[diffraction_order]["minima"][measurement_temperature]["exp_pixels"].append(gauss_fit[2])
                        order_data_dict[diffraction_order]["minima"][measurement_temperature]["exp_chisq"].append(gauss_fit[3])


                        #apply polynomial fit if gauss fit found
                        poly_fit = fit_polynomial(absorption_indices, absorption_spectrum[absorption_indices], degree=5, error=True, coeffs=True)
                        x_hr = np.arange(absorption_indices[0], absorption_indices[-1], 0.01)
                        y_hr = np.polyval(poly_fit[1], x_hr)
                        poly_minima = x_hr[y_hr==np.min(y_hr)][0]
                        
                        # if plot_spectra:
                        #     ax1.plot(x_hr, y_hr, "k--")
                        #     ax1.axvline(poly_minima, color=colour, linestyle="--")
                    
                        order_data_dict[diffraction_order]["minima"][measurement_temperature]["poly_pixels"].append(poly_minima)
                        order_data_dict[diffraction_order]["minima"][measurement_temperature]["poly_chisq"].append(poly_fit[2])
                    else:
                        if plot_spectra:
                            ax1.plot(x_hr, y_hr, color="grey", linestyle="dashed")

        if found:
            order_data_dict[diffraction_order]["minima"][measurement_temperature]["spectra"] = absorption_spectrum
        
        

    #find which minima are from the same absorptions
    """first - calculate predicted pixel locations gauss"""
    first_temperature = list(np.min(order_data_dict[diffraction_order]["minima"].keys()))[0]
    first_pixels = order_data_dict[diffraction_order]["minima"][first_temperature]["exp_pixels"]
    first_chisqs = order_data_dict[diffraction_order]["minima"][first_temperature]["exp_chisq"]
    
    #empty array to hold 
    found_absorptions = [[i] for i in first_pixels]
    found_temperatures = [[first_temperature] for i in range(len(first_pixels))]
    found_chisqs = [[i] for i in first_chisqs]
    coefficients = [[] for i in first_pixels]
    
    first_pixel_shift = t_p0(first_temperature)

    order_data_dict[diffraction_order]["minima"][first_temperature]["expected_shift"] = 0.0

    
    for temperature in order_data_dict[diffraction_order]["minima"]:
        
        
        if temperature != first_temperature: #find shift from coldest temperature -> ignore first one
            abs_pixel_shift = t_p0(temperature)
            pixel_shift = first_pixel_shift - abs_pixel_shift
            
            order_data_dict[diffraction_order]["minima"][temperature]["expected_shift"] = pixel_shift
            
            #loop through pixels where abs was found at lowest temperature
            for pixel_index, first_pixel in enumerate(first_pixels):
                #find expected pixel due to temperature shift
                expected_pixel = first_pixel + pixel_shift
                
                
                #search through pixels
                exp_fit_pixels = order_data_dict[diffraction_order]["minima"][temperature]["exp_pixels"]
                chisqs = order_data_dict[diffraction_order]["minima"][temperature]["exp_chisq"]
                for pixel, chisq in zip(exp_fit_pixels, chisqs):
                    if np.abs(expected_pixel - pixel) < 4:
                        # print("Match found!")
                        # print(temperature, pixel, expected_pixel)
                        found_absorptions[pixel_index].append(pixel)
                        found_temperatures[pixel_index].append(temperature)
                        found_chisqs[pixel_index].append(chisq)

    found_pixel_shifts = [[found_absorptions[j][i] - found_absorptions[j][0] if i>0 else 0 for i in range(0, len(found_absorptions[j]))] for j in range(len(found_absorptions))]
    found_shift_errors = [[found_chisqs[j][i] for i in range(0, len(found_absorptions[j]))] for j in range(len(found_absorptions))]


    #loop through the absorption shifts found in the order
    #plot pixel shift vs instrument temperature
    for slope_index, (found_pixel_shift, found_temperature, found_shift_error) in enumerate(zip(found_pixel_shifts, found_temperatures, found_shift_errors)):
        if USE_FIT_ERROR:
            _, coeffs = fit_linear_errors(found_temperature, found_pixel_shift, 1./np.asfarray(found_shift_error), coeffs=True, error=False)
        else:
            _, coeffs = fit_linear_errors(found_temperature, found_pixel_shift, 1./np.asfarray(np.ones(len(found_pixel_shift))), coeffs=True, error=False)
        
        coefficients[slope_index] = coeffs
        



    if plot_shift: 
        fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y)) #plot pixel shift vs temperature
        shift_colours = get_colours(len(found_pixel_shifts), cmap="brg")
        ax2.set_title("Diffraction order %i" %diffraction_order)
        ax2.set_ylabel("Pixel shift")
        ax2.set_xlabel("Instrument Temperature (C)")
        for slope_index, (found_pixel_shift, found_temperature, found_shift_error) in enumerate(zip(found_pixel_shifts, found_temperatures, found_shift_errors)):
            ax2.scatter(found_temperature, found_pixel_shift, color=shift_colours[slope_index], label="Pixel %0.0f" %first_pixels[slope_index])
            ax2.errorbar(found_temperature, found_pixel_shift,  np.asfarray(found_shift_error)*500, fmt=".", color=shift_colours[slope_index], label="Pixel %0.0f" %first_pixels[slope_index], capsize=2)
            ax2.plot(found_temperature, np.polyval(coefficients[slope_index], found_temperature), c=shift_colours[slope_index])
            ax2.text(0, 8-1.5*slope_index, "%0.2fx + %0.1f" %tuple(coefficients[slope_index]), color=shift_colours[slope_index])
        ax2.legend(loc="upper left")
        ax2.text(0, 9.5, "Gaussian")
        
    
    order_data_dict[diffraction_order]["shifts"] = {}
    order_data_dict[diffraction_order]["shifts"]["exp_coefficients"] = coefficients
    order_data_dict[diffraction_order]["shifts"]["exp_temperatures"] = found_temperatures
    order_data_dict[diffraction_order]["shifts"]["exp_pixel_shifts"] = found_pixel_shifts
    order_data_dict[diffraction_order]["shifts"]["exp_starting_pixels"] = first_pixels
    
    mean_shift = np.mean([i[0] for i in coefficients])
    order_data_dict[diffraction_order]["mean_shift"] = mean_shift
    



    """second - calculate predicted pixel locations poly"""
    first_temperature = list(np.min(order_data_dict[diffraction_order]["minima"].keys()))[0]
    first_pixels = order_data_dict[diffraction_order]["minima"][first_temperature]["poly_pixels"]
    first_chisqs = order_data_dict[diffraction_order]["minima"][first_temperature]["poly_chisq"]
    
    #empty array to hold 
    found_absorptions = [[i] for i in first_pixels]
    found_temperatures = [[first_temperature] for i in range(len(first_pixels))]
    found_chisqs = [[i] for i in first_chisqs]
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
                poly_fit_pixels = order_data_dict[diffraction_order]["minima"][temperature]["poly_pixels"]
                chisqs = order_data_dict[diffraction_order]["minima"][temperature]["poly_chisq"]
                for pixel, chisq in zip(poly_fit_pixels, chisqs):
                    if np.abs(expected_pixel - pixel) < 4:
                        # print("Match found!")
                        # print(temperature, pixel, expected_pixel)
                        found_absorptions[pixel_index].append(pixel)
                        found_temperatures[pixel_index].append(temperature)
                        found_chisqs[pixel_index].append(chisq)

    found_pixel_shifts = [[found_absorptions[j][i] - found_absorptions[j][0] if i>0 else 0 for i in range(0, len(found_absorptions[j]))] for j in range(len(found_absorptions))]
    found_shift_errors = [[found_chisqs[j][i] for i in range(0, len(found_absorptions[j]))] for j in range(len(found_absorptions))]


    #loop through the absorption shifts found in the order
    #plot pixel shift vs instrument temperature
    for slope_index, (found_pixel_shift, found_temperature, found_shift_error) in enumerate(zip(found_pixel_shifts, found_temperatures, found_shift_errors)):
        if USE_FIT_ERROR:
            _, coeffs = fit_linear_errors(found_temperature, found_pixel_shift, 1./np.asfarray(found_shift_error), coeffs=True, error=False)
        else:
            _, coeffs = fit_linear_errors(found_temperature, found_pixel_shift, 1./np.asfarray(np.ones(len(found_pixel_shift))), coeffs=True, error=False)
        
        coefficients[slope_index] = coeffs
        



    if plot_shift: 
        # fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y)) #plot pixel shift vs temperature
        # shift_colours = get_colours(len(found_pixel_shifts), cmap="brg")
        # ax2.set_title("Diffraction order %i" %diffraction_order)
        # ax2.set_ylabel("Pixel shift")
        # ax2.set_xlabel("Instrument Temperature (C)")
        for slope_index, (found_pixel_shift, found_temperature, found_shift_error) in enumerate(zip(found_pixel_shifts, found_temperatures, found_shift_errors)):
            ax2.scatter(found_temperature, found_pixel_shift, color=shift_colours[slope_index], label="Pixel %0.0f" %first_pixels[slope_index])
            ax2.errorbar(found_temperature, found_pixel_shift,  np.asfarray(found_shift_error)*500, fmt=".", color=shift_colours[slope_index], label="Pixel %0.0f" %first_pixels[slope_index], capsize=2)
            ax2.plot(found_temperature, np.polyval(coefficients[slope_index], found_temperature), c=shift_colours[slope_index], linestyle="--")
            ax2.text(5, 8-1.5*slope_index, "%0.2fx +  %0.1f" %tuple(coefficients[slope_index]), color=shift_colours[slope_index])
        ax2.legend(loc="upper left")
        ax2.text(5, 9.5, "Polynomial")
        
    
    order_data_dict[diffraction_order]["shifts"]["poly_coefficients"] = coefficients
    order_data_dict[diffraction_order]["shifts"]["poly_temperatures"] = found_temperatures
    order_data_dict[diffraction_order]["shifts"]["poly_pixel_shifts"] = found_pixel_shifts
    order_data_dict[diffraction_order]["shifts"]["poly_starting_pixels"] = first_pixels
    
    mean_shift = np.mean([i[0] for i in coefficients])
    order_data_dict[diffraction_order]["mean_shift"] = mean_shift
    




    ###try shifting spectra to match
    fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y)) #plot pixel shift vs temperature
    for spectrum_index, temperature in enumerate(order_data_dict[diffraction_order]["minima"]):
        expected_shift = order_data_dict[diffraction_order]["minima"][temperature]["expected_shift"]
        
        print(expected_shift)
        
        ax3.plot(pixels - expected_shift, order_data_dict[diffraction_order]["minima"][temperature]["spectra"], label=temperature)
    plt.legend()



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
