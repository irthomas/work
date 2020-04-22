# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:04:53 2019

@author: iant


PLOT FIGURES FOR CALIBRATION PAPER

"""

import os
import h5py
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, lfilter
from scipy.optimize import curve_fit

from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList, printFileNames
from filename_lists_v01 import getFilenameList


SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True


"""blank"""
fileLevel = "hdf5_level_0p1a"
obspaths = []
model = "PFM"
title = ""


"""MCC line scans"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["20160615_224950_0p1a_SO_1", "20160615_233950_0p1a_LNO_1"]
#title = "MCC line scan"


"""temperature spectral calibration"""
fileLevel = "hdf5_level_0p1a"
#obspaths = ["*2015042*LNO"]
#obspaths = getFilenameList("ground cal ch4 cell")
#obspaths = getFilenameList("ground cal c2h2 cell")
#obspaths = getFilenameList("ground cal co2 cell")
#obspaths = getFilenameList("ground cal co cell")

#after detector replacement
#obspaths = getFilenameList("ground cal ch4 cell new")
#obspaths = getFilenameList("ground cal co2 cell new")


##CH4
#obspaths = [
#"20150425_074022_0p1a_LNO_1",
#"20150425_081023_0p1a_LNO_1",
#"20150425_090234_0p1a_LNO_1",
#"20150425_093303_0p1a_LNO_1",
#"20150427_081547_0p1a_LNO_1",
#"20150427_092635_0p1a_LNO_1",
#"20150427_095826_0p1a_LNO_1",
#"20150427_105912_0p1a_LNO_1",
#"20150427_112853_0p1a_LNO_1",
#]
#CO2
#obspaths = [
#"20150425_130615_0p1a_LNO_1",
#"20150425_133627_0p1a_LNO_1",
#"20150427_123133_0p1a_LNO_1",
#"20150427_130112_0p1a_LNO_1",
#"20150427_140318_0p1a_LNO_1",
#"20150427_143253_0p1a_LNO_1",
#"20150427_153701_0p1a_LNO_1",
#"20150427_161333_0p1a_LNO_1",
#]


#title = "temperature calibration"
title = "find absorption lines"
#title = "find absorption lines new"
#title = "plot orders"



"""bb detector illumination"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["20150426_054602_0p1a_LNO_1"]#,"20150426_030851_0p1a_LNO_1"] #150C BB (cold only)
###obspaths = ["20161121_233000_0p1a_LNO_1"] #MCO1 sun only
#title = "bb detector illumination"


"""radiometric calibration"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["20150426_054602_0p1a_LNO_1"]#,"20150426_030851_0p1a_LNO_1"] #150C BB (cold only)
###obspaths = ["20161121_233000_0p1a_LNO_1"] #MCO1 sun only
#title = "radiometric calibration"


#channel={"SO ":"so", "SO-":"so", "LNO":"lno", "UVI":"uvis"}[title[0:3]]
#detector_centre={"so":128, "lno":152, "uvis":0}[channel] #or 152 for lno??




def getExternalTemperatureReadings(utc_string, column_number): #input format 2015 Mar 18 22:41:03.916651
    """get CSL ground calibration temperatures. Input SPICE style datetime, output in Celsius"""
    
    utc_datetime = datetime.strptime(utc_string[:20], "%Y %b %d %H:%M:%S")

    with open(os.path.join(BASE_DIRECTORY, "reference_files", "csl_anc_calibration_temperatures.txt")) as f:
        lines = f.readlines()
        
    utc_timedeltas = []
    temperatures = []
    for line in lines[1:]:
        split_line = line.split("\t")
        utc_timedeltas.append(datetime.strptime(split_line[0], "%d/%m/%Y %H:%M") - utc_datetime)
        temperatures.append(split_line[column_number])
            
    closestIndex = np.abs(utc_timedeltas).argmin()
    if np.min(np.abs(utc_timedeltas)).total_seconds() > 60 * 5:
        print("Error: time delta too high")
    else:
        closestTemperature = np.float(temperatures[closestIndex]) - 273.15
    
    return closestTemperature
    




def splitWindowSteppingByWindow(hdf5_file):
    
    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    
    unique_window_tops = sorted(list(set(window_top_all)))
    window_rows = unique_window_tops[1] - unique_window_tops[0]
    
    window_data = []
    detector_row_numbers = []
    for unique_window_top in unique_window_tops:
        window_frame = detector_data_all[window_top_all == unique_window_top, :, :]
        window_data.append(window_frame)
        
        detector_row_numbers.append(list(range(unique_window_top, unique_window_top + window_rows, binning)))
        
    return window_data, unique_window_tops, detector_row_numbers




def findAbsorptionMinimum(spectrum, continuum_range, plot=False):
    
    continuum_centre = int((continuum_range[3] - continuum_range[0])/2)
    ABSORPTION_WIDTH_INDICES = list(range(continuum_centre - 3, continuum_centre + 3, 1))
        
    pixels = np.arange(320)
    
    continuum_pixels = pixels[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]    
    continuum_spectra = spectrum[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]
    
    #fit polynomial to continuum on either side of absorption band
    coefficients = np.polyfit(continuum_pixels, continuum_spectra, 2)
    continuum = np.polyval(coefficients, pixels[range(continuum_range[0], continuum_range[3])])
    #divide by continuum to get absorption
    absorption = spectrum[range(continuum_range[0], continuum_range[3])] / continuum
    
    #fit polynomial to centre of absorption
    abs_coefficients = np.polyfit(pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES], absorption[ABSORPTION_WIDTH_INDICES], 2)
#    detector_row = illuminated_window_tops[frame_index]+bin_index*binning
    
    if plot:
        plt.plot(pixels[list(range(continuum_range[0], continuum_range[3]))], absorption)
        fitted_absorption = np.polyval(abs_coefficients, pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES])
        plt.plot(pixels[list(range(continuum_range[0],continuum_range[3]))][ABSORPTION_WIDTH_INDICES], fitted_absorption)
    
    absorption_minima = (-1.0 * abs_coefficients[1]) / (2.0 * abs_coefficients[0])
    
    return absorption_minima





def findOrder(channel,aotfFrequency,silent=False): #aotf in khz
    """SO:::::
    AOTF Hz to Order Dec2015: y = 4.34672E-15x2 + 6.68363E-06x + 1.30241E+01 #old version, used only to find order
     
    LNO::::
    AOTF Hz to Order Dec2015: y = 5.120394E-15x2 + 6.251145E-06x + 1.386261E+01 #old version, used only to find order"""

    if channel=="so":
        order= np.polyval([  4.34672E-15,   6.68363E-06,   1.30241E+01],aotfFrequency * 1000.0)
    elif channel=="lno":
        order= np.polyval([  5.120394E-15,   6.251145E-06,   1.386261E+01],aotfFrequency * 1000.0)

    orderRounded = np.round(order)
    if not silent: print("calculated order=%0.2f, rounded to order %0.0f" %(order,orderRounded))
    return orderRounded



    
def plotOrders(hdf5_files, hdf5Filenames, orders):

    CHOSEN_BIN = 11
    SILENT = True
#    SILENT = False
    
    PLOT = True
#    PLOT = False
    
    for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):

        channel = hdf5_filename.split("_")[3].lower()

        detector_data_all = hdf5_file["Science/Y"][...]
        aotf_frequency_all = hdf5_file["Channel/AOTFFrequency"][...]
        
        diffraction_orders = np.asfarray([findOrder(channel, aotf_frequency, silent=SILENT) for aotf_frequency in aotf_frequency_all])
        
        for order in orders:
            indices = np.where(diffraction_orders == order)[0]

            if len(indices) > 0:
                text = "Order %i found in file %s" %(order, hdf5_filename)
                if PLOT: plt.figure()
                if PLOT: plt.title(hdf5_filename)

                for index in indices:
                    spectrum = detector_data_all[index, CHOSEN_BIN, :]
                    if PLOT: plt.plot(spectrum, label="Order %i (%0.1fkHz)" %(diffraction_orders[index], aotf_frequency_all[index]))
                if PLOT: plt.legend()
            else:
                text = "Order %i not found in file %s" %(order, hdf5_filename)
        
            max_order = max(list(set(diffraction_orders)))
            min_order = min(list(set(diffraction_orders)))
            text += " (min order = %i, max order = %i)" %(min_order, max_order)
            print(text)



def plotTemperatureDependency(temperatures, values, colour, ax=None, title="", xlabel="", ylabel="", plot=True):
    
    def rmse(fit, values):
        return np.sqrt(((fit - values) ** 2).mean())

    if plot:
        if not ax:
            plt.scatter(temperatures, values, color=colour)
            if title == "":
                plt.title("Pixel shift due to temperature")
            else:
                plt.title(title)
            if xlabel == "":
                plt.xlabel("Temperature (C)")
            else:
                plt.xlabel(xlabel)
            if ylabel == "":
                plt.ylabel("Pixel shift")
            else:
                plt.ylabel(ylabel)
            if len(temperatures) > 1:
                coeffs = np.polyfit(temperatures, values, 1)
                fit = np.polyval(coeffs, temperatures)
                rms = rmse(fit, values)
    
                plt.plot(temperatures, fit, color=colour, label="Gradient = %0.3g" %coeffs[0])
                plt.legend()
    
        else:
            ax.scatter(temperatures, values, color=colour)
            if title == "":
                ax.set_title("Pixel shift due to temperature")
            else:
                ax.set_title(title)
            if xlabel == "":
                ax.set_xlabel("Temperature (C)")
            else:
                ax.set_xlabel(xlabel)
            if ylabel == "":
                ax.set_ylabel("Pixel shift")
            else:
                ax.set_ylabel(ylabel)
            if len(temperatures) > 1:
                coeffs = np.polyfit(temperatures, values, 1)
                fit = np.polyval(coeffs, temperatures)
                rms = rmse(fit, values)
                
                ax.plot(temperatures, fit, color=colour, label="Gradient = %0.3g" %coeffs[0])
                ax.legend()
    else:
        if len(temperatures) > 1:
            coeffs = np.polyfit(temperatures, values, 1)
            fit = np.polyval(coeffs, temperatures)
            rms = rmse(fit, values)
        

    return coeffs[0], rms




def fitCurveError(xdata, ydata, ystd, ax, fit="linear"):
    """find and plot linear fit taking into account the errors"""

    xplotdata = np.arange(np.min(xdata), np.max(xdata)+1, 0.1)
    def func(x, a, b):
         return a * x + b
     
    def func2(x, a, b, c):
         return a * x**2 + b * x + c

    if fit=="linear":
        popt, pcov = curve_fit(func, xdata, ydata, p0=(-1.0e-7, 3.0e-5))
        ax.plot(xplotdata, func(xplotdata, *popt), "k--")
    elif fit=="quadratic":
        popt, pcov = curve_fit(func2, xdata, ydata, p0=(-5.0e-7, 2.0e-4, -1.0e-2))
        ax.plot(xplotdata, func2(xplotdata, *popt), "k--")
    print(popt)
    #calculate r_squared
#        chi_sq = np.sum(((ydata - func(xdata, *popt)) ** 2.0) / func(xdata, *popt))
#        plt.annotate("Chi-squared value = %0.2f" %chi_sq, xy=(160, 0.8), xytext=(0.1, 0.9), textcoords='axes fraction')




if title == "MCC line scan":
    """make vertical detector plots where sun is seen to determine slit position and time when in centre"""
    DETECTOR_V_CENTRE = 201
    DETECTOR_CUTOFF = 20000
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model)

    
    for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
        
        channel = hdf5_filename.split("_")[3].lower()

        cmap = plt.get_cmap('Set1')
        window_data, unique_window_tops, detector_row_numbers = splitWindowSteppingByWindow(hdf5_file)
        nColours = 0
        illuminated_window_tops = []
        for window_frames, unique_window_top in zip(window_data, unique_window_tops):
#            print(np.max(window_frames[:, :, DETECTOR_V_CENTRE]))
            if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
                nColours += 1
                illuminated_window_tops.append(unique_window_top)
        colours = [cmap(i) for i in np.arange(nColours)/nColours]


        row_max_value = []
        row_number = []
        for window_index, (window_frames, detector_row_number) in enumerate(zip(window_data, detector_row_numbers)):
            if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
                vertical_slices = window_frames[:, :, DETECTOR_V_CENTRE]

                row_number.extend(detector_row_number)
                row_max_value.extend(np.max(vertical_slices, axis=0))


        plt.figure(figsize=(FIG_X - 4, FIG_Y + 2))
        plt.xlabel("Relative instrument sensitivity for pixel number %i" %DETECTOR_V_CENTRE)
        plt.ylabel("Detector row number")
        plt.title(channel.upper()+" "+title+": sun detector illumination")
        plt.tight_layout()
        plt.grid(True)
        #normalise

        colour_index = -1
        for window_index, (window_frames, detector_row_number) in enumerate(zip(window_data, detector_row_numbers)):
            if np.max(window_frames[:, :, DETECTOR_V_CENTRE]) > DETECTOR_CUTOFF:
                vertical_slices = window_frames[:, :, DETECTOR_V_CENTRE]

                colour_index += 1
                
#                plt.scatter(detector_row_number, np.max(vertical_slices, axis=0), color=colours[colour_index])
                for slice_index, vertical_slice in enumerate(vertical_slices):
                    if slice_index == 0:
                        plt.plot(vertical_slice/np.max(row_max_value), detector_row_number, color=colours[colour_index], label="Vertical rows %i-%i" %(np.min(detector_row_number), np.max(detector_row_number)))
                    else:
                        plt.plot(vertical_slice/np.max(row_max_value), detector_row_number, color=colours[colour_index])
        
        row_numbers = np.arange(np.min(row_number), np.max(row_number))
        row_max_values = np.interp(row_numbers, row_number, row_max_value)
        
        smooth = savgol_filter(row_max_values[row_max_values > DETECTOR_CUTOFF], 9, 5)
        smooth_rows = row_numbers[row_max_values > DETECTOR_CUTOFF]
#        plt.scatter(row_numbers, row_max_values)
        plt.plot(smooth/np.max(row_max_value), smooth_rows, linewidth=5, color="k", label="Instrument sensitivity")
        plt.legend(loc="upper right")
        
        if SAVE_FIGS: 
            plt.savefig(BASE_DIRECTORY+os.sep+channel+"_"+title.replace(" ","_")+"_vertical_columns_on_detector_where_sun_is_seen.png")


#        """check detector smile"""
            
        if channel == "so":
            continuum_range = [210,215,223,228]
            signal_minimum = 200000
            bad_pixel_rows = [114]
            binning=1


        if channel == "lno":
            continuum_range = [203,209,217,223]
            signal_minimum = 300000
            bad_pixel_rows = [106, 108, 112, 218]
            binning=2

        
        #first loop through each window frame, finding minima of chosen absorption (if frame is illuminated)
        absorption_minima=[]
        detector_rows=[]
        for window_frames, unique_window_top, detector_row_list in zip(window_data, unique_window_tops, detector_row_numbers):
            if unique_window_top in illuminated_window_tops:
                for window_frame in window_frames:
                    for spectrum, detector_row_number in zip(window_frame, detector_row_list):
                        if spectrum[200] > signal_minimum:
                            if detector_row_number not in bad_pixel_rows:
                                detector_rows.append(detector_row_number)
                                absorption_minimum = findAbsorptionMinimum(spectrum, continuum_range)
                                absorption_minima.append(absorption_minimum)
                            
        
        plt.figure(figsize = (FIG_X - 4, FIG_Y + 2))
        plt.scatter(absorption_minima, detector_rows, marker="o", linewidth=0, alpha=0.5)

        fit_coefficients = np.polyfit(detector_rows,absorption_minima,1)
        fit_line = np.polyval(fit_coefficients,detector_rows)

        plt.plot(fit_line, detector_rows, "k", label="Line of best fit, min=%0.1f, max=%0.1f" %(np.min(fit_line), np.max(fit_line)))
        plt.legend()
        plt.xlabel("Peak absorption pixel number, determined from quadratic fit")
        plt.ylabel("Detector row")
        plt.title(channel.upper()+" "+title+" Detector smile: Quadratic fits to absorption line")
        plt.tight_layout()
        plt.grid(True)
        if SAVE_FIGS: 
            plt.savefig(BASE_DIRECTORY + os.sep+channel+"_"+title.replace(" ", "_") + "_detector_smile.png")



if title == "temperature calibration":
    
    """old version"""
    
    
    orders = []
    gradients = []

#    for molecule in ["ch4", "c2h2", "co2", "co"]: #orders 134, 146, 167, 190
    for molecule in ["co"]: #orders 134, 146, 167, 190
        
        
    
        
        AOTF_FREQUENCIES = {"ch4":[18899.326], "c2h2":[20771.9], "co2":[24020.9], "co":[27558.5]}[molecule]
        ORDERS = {"ch4":[134], "c2h2":[146], "co2":[167], "co":[190]}[molecule]
        ABSORPTION_COEFFS1 = {"ch4":[[0.85, 251.0]], "c2h2":[[0.804, 202.8]], "co2":[[0.874, 201.9]], "co":[[0.878, 196.0], [0.924, 105.0]]}[molecule] #before detector swap
        ABSORPTION_COEFFS2 = {"ch4":[1.11, 234.0], "c2h2":[1.11, 234.0], "co2":[1.11, 234.0], "co":[1.11, 234.0]}[molecule] #after detector swap
    
        obspaths = getFilenameList("ground cal %s cell" %molecule)
        hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model)
        
        diffraction_order = ORDERS[0]
        orders.append(diffraction_order)
        lineNumber = 1

        def checkSignal(normalised_spectrum):
            return True


#        if molecule == "ch4":
#        
#        def checkSignal(normalised_spectrum):
#            if normalised_spectrum[80] < 0.5: #check if Q branch is there (i.e. CH4 cell in position)
#                return True
#            else:
#                return False
#    
#    if molecule == "c2h2":
#        AOTF_FREQUENCIES = [20771.9] #order 146
#        ABSORPTION_COEFFS1 = [0.804, 202.8]
#        ABSORPTION_COEFFS2 = [1.11, 234.0]
#
#        def checkSignal(normalised_spectrum):
#            if normalised_spectrum[0] < 0.2: #check if signal
#                return True
#            else:
#                return False
#
#    if molecule == "co2":
#        AOTF_FREQUENCIES = [24020.9] #order 167
#        ABSORPTION_COEFFS1 = [0.874, 201.9]
#        ABSORPTION_COEFFS2 = [1.11, 234.0]
#
#        def checkSignal(normalised_spectrum):
#            if normalised_spectrum[0] < 0.2: #check if signal
#                return True
#            else:
#                return False
#
#    if molecule == "co":
#        AOTF_FREQUENCIES = [27558.5] #order 190
#        ABSORPTION_COEFFS1 = [0.878, 196.0]
#        ABSORPTION_COEFFS2 = [1.11, 234.0]
#
#        def checkSignal(normalised_spectrum):
#            if normalised_spectrum[0] < 0.2: #check if signal
#                return True
#            else:
#                return False

    
#    DETECTOR_REPLACMENT_OFFSET = 21.0
    
        pixels = np.arange(320)
        
        CHOSEN_BIN = 11
    
    
        nColours = 50 #from -20C to +30C
        cmap = plt.get_cmap('plasma')
        colours = [cmap(i) for i in np.arange(nColours)/nColours]
    
    
        fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
        ax1.grid(True)
        
        temperatures1 = []
        temperatures2 = []
        absorption_minima1 = []
        absorption_minima2 = []
        
        for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
            spectrum_found = False
            
            plt.title("%s gas cell: spectra of order %s for different instrument temperatures" %(molecule.upper(), diffraction_order))
            plt.xlabel("Pixel number")
            plt.ylabel("Normalised gas cell spectrum")
            channel = hdf5_filename.split("_")[3].lower()
    
    
            detector_data_all = hdf5_file["Science/Y"][...]
            window_top_all = hdf5_file["Channel/WindowTop"][...]
            binning = hdf5_file["Channel/Binning"][0] + 1
            integration_time = hdf5_file["Channel/IntegrationTime"][0]
            sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
            measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
            
            aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
            
            if sbsf == 1:
                spectra = detector_data_all[:, CHOSEN_BIN, :]
                
                for spectrum, aotf_frequency in zip(spectra, aotf_frequencies):
                    if not spectrum_found:
                        if np.any([(frequency-1.1) < aotf_frequency < (frequency+1.1) for frequency in AOTF_FREQUENCIES]):
                            normalised_spectrum = spectrum / np.max(spectrum)
                            
                            if checkSignal(normalised_spectrum):
                                
                                if hdf5_filename[5:7] == "42": #if after detector replacement
                                    spare_detector = True
                                    linestyle = "--"
                                    #find pixel with absorption minimum
                                    #absorption minimum varies with temperature - use approx calibration
                                    approx_centre = int(ABSORPTION_COEFFS2[lineNumber][0] * measurement_temperature + ABSORPTION_COEFFS2[lineNumber][1])
                                else:
                                    spare_detector = False
                                    linestyle = "-"
                                    #find pixel with absorption minimum
                                    #absorption minimum varies with temperature - use approx calibration
                                    approx_centre = int(ABSORPTION_COEFFS1[lineNumber][0] * measurement_temperature + ABSORPTION_COEFFS1[lineNumber][1])
    #                                print(approx_centre)
                                plt.scatter(approx_centre, normalised_spectrum[approx_centre], color=colours[int(measurement_temperature)+20], linewidth=0)
#                                ax1.plot(normalised_spectrum, color=colours[int(measurement_temperature)+20], linestyle=linestyle, label="%s; %0.0fkHz; %0.1fC" %(hdf5_filename[0:15], aotf_frequency, measurement_temperature))
                                ax1.plot(normalised_spectrum, color=colours[int(measurement_temperature)+20], linestyle=linestyle, label="%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature))
                                print("\"%s\", #%s %0.1fC" %(hdf5_filename, molecule.upper(), measurement_temperature))
                                spectrum_found = True
                                
                                minimum_range = [approx_centre-5, approx_centre+5]
                                
                                
                                minimum_position = np.argmin(spectrum[range(minimum_range[0], minimum_range[1])])
                                minimum_pixel = pixels[[range(minimum_range[0], minimum_range[1])]][minimum_position]
                                #define fitting range from this
                                continuum_range = [minimum_pixel-12, minimum_pixel-7, minimum_pixel+7, minimum_pixel+12]
                                
                                absorption_minimum = findAbsorptionMinimum(spectrum, continuum_range, plot=False)
                                plt.axvline(x=absorption_minimum, linestyle=linestyle, color=colours[int(measurement_temperature)+20])
                                
                                if spare_detector:
                                    temperatures2.append(measurement_temperature)
                                    absorption_minima2.append(absorption_minimum)
                                else:
                                    temperatures1.append(measurement_temperature)
                                    absorption_minima1.append(absorption_minimum)
    
    
        ax1.legend()
        
        plt.figure()
        
        if len(temperatures1) > 0:
            gradient1, _ = plotTemperatureDependency(temperatures1, absorption_minima1, "r")
#        if len(temperatures2) > 0:
#            gradient2, _ = plotTemperatureDependency(temperatures2, absorption_minima2, "g")
    
        gradients.append(gradient1)

    plt.figure()
    plt.title("Temperature shift dependency versus order")
    plt.xlabel("Diffraction order")
    plt.ylabel("Gradient of shift vs temperature relation")
    plt.scatter(orders, gradients, color="b")
    coeffs = np.polyfit(orders, gradients, 1)
    plt.plot(np.arange(min(orders), max(orders)), np.polyval(coeffs, np.arange(min(orders), max(orders))), color="b", label="Gradient = %0.3g" %coeffs[0])
    plt.legend()
        





def applyFilter(spectrum, plot=False):
    """input raw spectrum. Output filtered spectrum"""
    # Filter requirements.
    order = 3
    fs = 35.0       # sample rate, Hz
    cutoff = 0.5 #3.667  # desired cutoff frequency of the filter, Hz

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    
    if plot:
        plt.subplots(figsize=(14,10), sharex=True)
        plt.subplot(2, 1, 1)
    
    
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    pixel_in = np.arange(len(spectrum))
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    
    # Filter the data, and plot both the original and filtered signals.
    dataFit = butter_lowpass_filter(spectrum, cutoff, fs, order)
    
    pixelInterp = np.arange(pixel_in[0], pixel_in[-1]+1.0, 0.1)
    dataInterp = np.interp(pixelInterp, pixel_in, spectrum)
    dataFitInterp = np.interp(pixelInterp, pixel_in, dataFit)
    
    
    
    firstPoint = 200
    pixelInterp = pixelInterp[firstPoint:]
    dataInterp = dataInterp[firstPoint:]
    dataFitInterp = dataFitInterp[firstPoint:]
    
    nPoints = len(dataInterp)
    
    
    #chi = [chisquare(data[0:(319-index)] - y[index:319])[0]**2 for index in np.arange(0, 20, 1)]
    #minIndex = np.argmin(chi)-1
    
    chi = np.asfarray([np.sum((dataInterp[0:(nPoints-index)] - dataFitInterp[index:(nPoints)])**2) / (nPoints - index) \
                       for index in np.arange(0, 1000, 1)])
    minIndex = np.argmin(chi)-1
    
    
    
    if plot:
        plt.plot(pixelInterp, dataInterp, 'b-', label='data')
        plt.plot(pixelInterp[0:(nPoints-minIndex)], dataFitInterp[minIndex:(nPoints)], 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    
    x = pixelInterp[0:(nPoints-minIndex)]
    y = dataInterp[0:(nPoints-minIndex)]/dataFitInterp[minIndex:(nPoints)]
    
    
    
    if plot:
        plt.subplot(2, 1, 2)
        plt.plot(x, y, label="residual")
        plt.ylim([0.95, 1.02])
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return x, y







if title == "find absorption lines":

    channel = "lno"
    #choose what to plot
#    plot_gradient = True #fig2
    plot_gradient = False #fig2
#    plot_spectra = True #fig1
    plot_spectra = False #fig1
#    plot_shift = True #temp dep
    plot_shift = False #temp dep

    fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))
    ax3.grid(True)
    fig4, ax4 = plt.subplots(figsize=(FIG_X, FIG_Y))
    ax3.grid(True)

    for lno_detector in ["old", "new"]:
    
        if lno_detector == "old": #after detector replacement
    #        AOTF_FREQUENCIES = [18899.326, 20771.9, 24020.9, 27558.5]
            AOTF_FREQUENCIES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_all_full_range.csv"), delimiter=",")[:,0]
            PIXEL_RANGES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_all_full_range.csv"), delimiter=",")[:,2:4]
    #        AOTF_FREQUENCIES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_ch4.csv"), delimiter=",")[:,0]
    #        PIXEL_RANGES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_ch4.csv"), delimiter=",")[:,2:4]
            GRADIENT_APPROXIMATION = 0.85
            SUFFIX = ""
            MATCHING_TEMPERATURES_FOUND = 2
            DELTA_TEMPERATURE = 15.0
            MATCHING_LINES_FOUND = 1
            SUBPIXEL_SEARCH_RANGE = 20
            
    #        USE_CSL_TEMPERATURES = False
            USE_CSL_TEMPERATURES = True
            CSL_TEMPERATURE_COLUMN = 4
    
        elif lno_detector == "new":
    #        AOTF_FREQUENCIES = [18340.0, 22948.0]
            AOTF_FREQUENCIES = [18304., 18340., 18459., 19360., 19390., 22928., 22948., 23950.]
            PIXEL_RANGES = [5, 2750] * len(AOTF_FREQUENCIES)
    #        with h5py.File(os.path.join(r"C:\Users\iant\Documents\DATA\hdf5_copy\hdf5_level_0p1a\2015\04\27", "20150427_171623_0p1a_LNO_1.h5")) as f: 
    #            AOTF_FREQUENCIES=f["Channel/AOTFFrequency"][22:30]
    #        AOTF_FREQUENCIES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_all_full_range.csv"), delimiter=",")[:,0]
            PIXEL_RANGES = np.loadtxt(os.path.join(BASE_DIRECTORY, "aotf_temperature_dependency_search_frequencies_all_full_range.csv"), delimiter=",")[:,2:4]
    
            
            GRADIENT_APPROXIMATION = 0.9
            SUFFIX = " new"
            MATCHING_TEMPERATURES_FOUND = 1
            DELTA_TEMPERATURE = 10.0
            MATCHING_LINES_FOUND = 0
            SUBPIXEL_SEARCH_RANGE = 20
    
            USE_CSL_TEMPERATURES = False
            CSL_TEMPERATURE_COLUMN = 3
    
        CHOSEN_BINS = [12]
    #    CHOSEN_BINS = range(8,15,1)
        STD_CUTOFF = 0.5
    #    STD_CUTOFF = 1.0
        END_OF_DETECTOR = 292
        START_OF_DETECTOR = 10
        POLYNOMIAL_RANGE = 20
        pixels = np.arange(320)
        
        GENERIC_GRADIENT_STDEV = 0.05
        
        
        
    
    
        orders = []
        mean_gradient_all_orders = []
        std_gradient_all_orders = []
        n_gradients_all_orders = []
        
        
    
        for AOTF_FREQUENCY, PIXEL_RANGE in zip(AOTF_FREQUENCIES, PIXEL_RANGES):
    
            DIFFRACTION_ORDER = findOrder(channel, AOTF_FREQUENCY, silent=True)
            if DIFFRACTION_ORDER < 140:
                molecule = "ch4"
            elif DIFFRACTION_ORDER < 155:
                molecule = "c2h2"
            elif DIFFRACTION_ORDER < 180:
                molecule = "co2"
            elif DIFFRACTION_ORDER < 200:
                molecule = "co"
            else:
                print("Error: molecule unknown")
                continue
            
            mean_gradient_all_bins = []
            std_gradient_all_bins = []
            n_gradients_all_bins = []
        
            
            
            obspaths = getFilenameList("ground cal %s cell%s" %(molecule, SUFFIX))
            hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model, silent=True)
            
            orders.append(DIFFRACTION_ORDER)
       
            
            if plot_gradient: 
                fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
                ax2.grid(True)
            
            nColours = len(CHOSEN_BINS)
            cmap = plt.get_cmap('Spectral')
            bin_colours = [cmap(i) for i in np.arange(nColours)/nColours]
    
            nColours = 50 #from -20C to +30C
            cmap = plt.get_cmap('plasma')
            colours = [cmap(i) for i in np.arange(nColours)/nColours]
    
            print(molecule.upper())
            for bin_index, chosen_bin in enumerate(CHOSEN_BINS):
            
            
                if plot_spectra: fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
                if plot_spectra: ax1.grid(True)
                
                temperatures_all = []
                minimum_indices_all = []
                for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
                    spectrum_found = False
                    
                    if plot_spectra:
                        plt.title("%s gas cell: spectra of order %s for different instrument temperatures (bin %i)" %(molecule.upper(), DIFFRACTION_ORDER, chosen_bin))
                        plt.xlabel("Pixel number")
                        plt.ylabel("Normalised gas cell transmittance")
            
            
                    detector_data_all = hdf5_file["Science/Y"][...]
                    
                    window_top_all = hdf5_file["Channel/WindowTop"][...]
                    binning = hdf5_file["Channel/Binning"][0] + 1
                    integration_time = hdf5_file["Channel/IntegrationTime"][0]
                    sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
                    measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
                    datetimes = hdf5_file["DateTime"][...]
                    
                    aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
                    
                    
                    
                    if sbsf == 1:
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
        
                        
                        for spectrum, aotf_frequency, measurement_time in zip(spectra, aotf_frequencies, datetimes):
                            if not spectrum_found:
                                if (AOTF_FREQUENCY-1.1) < aotf_frequency < (AOTF_FREQUENCY+1.1):
                                    normalised_spectrum = spectrum / np.max(spectrum)
                                    
                                    if USE_CSL_TEMPERATURES: #overwrite temperature with one from external file
                                        measurement_temperature = getExternalTemperatureReadings(measurement_time.decode(), CSL_TEMPERATURE_COLUMN)
                                    
                                    filtered_grid, filtered_spectrum = applyFilter(normalised_spectrum) #detect absorption lines by fitting filter to remove continuum
                                    #returns high resolution x and filtered spectrum
        
                                    filtered_std = np.std(filtered_spectrum)
                                    filtered_mean = np.mean(filtered_spectrum)
                                    indices = np.where(filtered_spectrum < (filtered_mean - filtered_std * STD_CUTOFF))[0] #where absorptions outside of st dev
                                    
                                    linestyle = "-"
    #                               if plot_spectra:  ax1.plot(normalised_spectrum, color=colours[int(measurement_temperature)+20], linestyle=linestyle, label="%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature))
                                    if plot_spectra: ax1.plot(filtered_grid, filtered_spectrum, color=colours[int(measurement_temperature)+20], linestyle=linestyle, label="%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature))
        
                                    #plot horizontal lines for st dev limits
                                    if plot_spectra: ax1.axhline(filtered_mean - filtered_std * STD_CUTOFF, color=colours[int(measurement_temperature)+20], linestyle="--")
                                    if plot_spectra: ax1.axhline(filtered_mean + filtered_std * STD_CUTOFF, color=colours[int(measurement_temperature)+20], linestyle="--")
                                    
                                    #this bit sorts indices for each absorption into separate lists
                                    spl = [0]+[i for i in range(1,len(indices)) if indices[i]-indices[i-1]>1]+[None]
                                    consecutive_indices_groups = [indices[b:e] for (b, e) in [(spl[i-1],spl[i]) for i in range(1,len(spl))]]
                                    
                                    minimum_indices = []
                                    for consecutive_indices in consecutive_indices_groups: #loop through indices for each absorption
                                        if len(consecutive_indices) > 10: #check sufficient points
                                            #fit polynomial to find minimum
                                            
                                            #find minimum, then plot +- 10 points from that
                                            absorption_index = np.argmin(filtered_spectrum[consecutive_indices]) + consecutive_indices[0]
                                            absorption_indices = range(absorption_index-POLYNOMIAL_RANGE, absorption_index+POLYNOMIAL_RANGE+1,1)
                                            
                                            #check that points are not past end of detector or before start of detector
                                            if not np.any(np.asfarray(absorption_indices) > PIXEL_RANGE[1]) and not np.any(np.asfarray(absorption_indices) < PIXEL_RANGE[0]):
                                                spectrum_found = True #set to True so one spectrum plotted from one file
                                            
                                                coefficients = np.polyfit(filtered_grid[absorption_indices], filtered_spectrum[absorption_indices], 2)
                                                polyfit = np.polyval(coefficients, filtered_grid[absorption_indices])
                                                minimum_index = np.argmin(polyfit) + absorption_indices[0]
            
                                                if plot_spectra:
                                                    plt.plot(filtered_grid[absorption_indices], polyfit, color=colours[int(measurement_temperature)+20], linestyle="--")
                                                    plt.scatter(filtered_grid[minimum_index], filtered_spectrum[minimum_index], color="k")
                                                minimum_indices.append(minimum_index)
                        if not spectrum_found:
                            text = "AOTF frequency %0.0f kHz (order %i) %0.1fC not found in file %s" %(AOTF_FREQUENCY, findOrder(channel, AOTF_FREQUENCY, silent=True), measurement_temperature, hdf5_filename)
                            aotf_frequency_all = hdf5_file["Channel/AOTFFrequency"][...]
                            diffraction_orders = np.asfarray([findOrder(channel, aotf_frequency, silent=True) for aotf_frequency in aotf_frequency_all])
                            text += " (%0.0f-%0.0fkHz; orders=%i-%i)" %(min(aotf_frequency_all), max(aotf_frequency_all), min(diffraction_orders), max(diffraction_orders))
                            print(text)
    
                        else:
                            print("AOTF frequency %0.0f kHz (order %i) %0.1fC found in file %s. Adding to search list" %(AOTF_FREQUENCY, findOrder(channel, AOTF_FREQUENCY, silent=True), measurement_temperature, hdf5_filename))
                            temperatures_all.append(measurement_temperature)
                            minimum_indices_all.append(minimum_indices) #make list of lists of all absorption minima
                                        
                            if plot_spectra:
                                plt.legend()
                    else:
                        print("%s does not have background subtraction" %hdf5_filename)
            
        
                """from list of temperatures and minimum absorption indices for each line in each file, match together same absorptions at each temperature"""
                starting_index = np.argmin(temperatures_all) #start with lowest temperature
                starting_temperature = temperatures_all.pop(starting_index)
                starting_minimum_indices = minimum_indices_all.pop(starting_index)
                
                matching_temperatures_all = []
                matching_pixel_numbers_all = []
                starting_pixels = []
                temperature_shift_gradients = []
            
            
                if plot_shift: 
                    plt.figure(figsize=(FIG_X, FIG_Y)) #plot pixel shift vs temperature
                cmap = plt.get_cmap('Spectral')
                nColours = len(starting_minimum_indices)
                shift_colours = [cmap(i) for i in np.arange(nColours)/nColours]
                for absorption_line_index, absorption_index in enumerate(starting_minimum_indices): #loop through absorption lines in first file
                    
                    starting_pixel = filtered_grid[absorption_index] #first pixel value
                    matching_temperatures = [starting_temperature] #add values for lowest temperature to list for later
                    matching_pixel_numbers = [starting_pixel] #add values for lowest temperature to list for later
                    if starting_pixel < END_OF_DETECTOR: #avoid end of detector. Start of detector is less important -> using lowest temperature as reference spectra
                        for temperature, minimum_indices in zip(temperatures_all, minimum_indices_all): #loop through other files
                        
                            approx_pixel = starting_pixel + GRADIENT_APPROXIMATION * (temperature - starting_temperature) #from lowest temperature absorption, calculate approx location of line at other temperatures
    
                            approx_pixel_index = np.abs(filtered_grid - approx_pixel).argmin() #convert to high res grid
                            
    #                        print("%0.1f, %0.1f, %0.1f" %(starting_pixel, temperature, approx_pixel))
                            if plot_spectra: ax1.axvline(approx_pixel, color=colours[int(temperature)+20], linestyle="-", alpha=0.3) #plot expected position of absorption line
                            
                            matching_absorption_found = 0
                            for minimum_index in minimum_indices: #loop through absorption indices in other file until one is found
                                if approx_pixel_index in range(minimum_index-SUBPIXEL_SEARCH_RANGE, (minimum_index+SUBPIXEL_SEARCH_RANGE+1), 1): #if calculated index close to index in other file
                                    if matching_absorption_found != 0: #if line around found print warning
                                        print("Warning: line already found")
                                    matching_absorption_found = filtered_grid[minimum_index]
                                    matching_temperatures.append(temperature)
                                    matching_pixel_numbers.append(filtered_grid[minimum_index]) #store pixel value and temperature
                                    
                                    if plot_spectra: ax1.axvline(filtered_grid[minimum_index], color=colours[int(temperature)+20], linestyle="--", alpha=0.7) #plot position of found absorption line
                            if matching_absorption_found:
                                print("Searching for line at %0.1f. Line found for pixel %0.1f" %(filtered_grid[approx_pixel_index], matching_absorption_found))
                            else:
                                print("Searching for line at %0.1f. Line not found" %(filtered_grid[approx_pixel_index]))
                               
                        matching_temperatures = np.asfarray(matching_temperatures)
                            
                        delta_temperature = np.max(matching_temperatures) - np.min(matching_temperatures)
                        if len(matching_temperatures) > MATCHING_TEMPERATURES_FOUND: #if sufficient points are matched to make line of best fit
                            if delta_temperature > DELTA_TEMPERATURE: #if temperature range is sufficiently large
                                gradient, _ = plotTemperatureDependency(matching_temperatures, matching_pixel_numbers - matching_pixel_numbers[0], shift_colours[absorption_line_index], title="Pixel shift due to temperature (order %i, bin %i)" %(DIFFRACTION_ORDER, chosen_bin), plot=plot_shift)
                    
                                matching_temperatures_all.append(matching_temperatures) #for all absorption bands. for debuging only
                                matching_pixel_numbers_all.append(matching_pixel_numbers)
                
                                starting_pixels.append(starting_pixel) #store first pixel number (lowest temperature) for each absorption band on detector
                                temperature_shift_gradients.append(gradient) #store gradients for each absorption line
            
                if len(temperature_shift_gradients) > MATCHING_LINES_FOUND: #if more than X gradients have been calculated from absorption lines in the given bin
                    mean_temperature_shift_gradient = np.mean(temperature_shift_gradients) #mean of each abs line pixel shift vs temperature gradient
                    if len(temperature_shift_gradients) > 1: #get stdev if more than 1 point
                        std_temperature_shift_gradient = np.std(temperature_shift_gradients) #mean of each abs line pixel shift vs temperature gradient
                    else: #else use generic (large) value
                        std_temperature_shift_gradient = GENERIC_GRADIENT_STDEV
                    mean_gradient_all_bins.append(mean_temperature_shift_gradient)
                    std_gradient_all_bins.append(std_temperature_shift_gradient)
                    n_gradients_all_bins.append(len(temperature_shift_gradients))
                    print("Bin %i mean pixel shift gradient = %0.3g" %(chosen_bin, mean_temperature_shift_gradient))
                    if len(temperature_shift_gradients) > 1: #need more than one point to calculate gradient vs pixel
                        #plot gradient variation across detector for this bin
                        if plot_gradient: 
                            gradient, gradient_std = plotTemperatureDependency(starting_pixels, temperature_shift_gradients, bin_colours[bin_index], ax=ax2, title="Variation in gradient across detector (order %i)" %DIFFRACTION_ORDER, xlabel="Position on detector (pixel number)", ylabel="Gradient of pixel shift vs temperature", plot=True)
                        else:
                            gradient, gradient_std = plotTemperatureDependency(starting_pixels, temperature_shift_gradients, bin_colours[bin_index], title="Variation in gradient across detector", xlabel="Position on detector (pixel number)", ylabel="Gradient of pixel shift vs temperature", plot=False)
                        
                        print("Bin %i gradient = %0.3g" %(chosen_bin, gradient))
            mean_gradient_all_orders.append(np.mean(mean_gradient_all_bins))
            std_gradient_all_orders.append(np.mean(std_gradient_all_bins))
            n_gradients_all_orders.append(np.sum(n_gradients_all_bins))
    #        mean_gradients_all_std.append(np.mean)
            print("###Order %i all bins mean pixel shift gradient = %0.3g" %(DIFFRACTION_ORDER, np.mean(mean_gradient_all_bins)))
    
        #LINEAR
        #plot all gradients as a function of pixel number
        ax3.set_title("Mean gradient of temperature pixel shift vs. diffraction order (pixels / C)")
        ax3.set_xlabel("Diffraction Order")
        ax3.set_ylabel("Mean gradient for all matching absorption lines")
        ax3.errorbar(orders, mean_gradient_all_orders, yerr=std_gradient_all_orders, fmt="o", color={"old":"b", "new":"r"}[lno_detector])
        
        #write number of gradients 
        for order, mean_gradient, n_gradients in zip(orders, mean_gradient_all_orders, n_gradients_all_orders):
            if np.isfinite(mean_gradient):
                ax3.text(order, mean_gradient, "%i" %n_gradients)
    
            
        fitCurveError(np.asfarray(orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(mean_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(std_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], ax=ax3)
        
        #QUADRATIC
        #plot all gradients as a function of pixel number
        ax4.set_title("Mean gradient of temperature pixel shift vs. diffraction order")
        ax4.set_xlabel("Nearest diffraction Order")
        ax4.set_ylabel("Mean gradient for all matching absorption lines")
        ax4.errorbar(orders, mean_gradient_all_orders, yerr=std_gradient_all_orders, fmt="o", color={"old":"b", "new":"r"}[lno_detector])
        
        #write number of gradients 
        for order, mean_gradient, n_gradients in zip(orders, mean_gradient_all_orders, n_gradients_all_orders):
            if np.isfinite(mean_gradient):
                ax4.text(order, mean_gradient, "%i" %n_gradients)
           
        fitCurveError(np.asfarray(orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(mean_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], np.asfarray(std_gradient_all_orders)[np.isfinite(mean_gradient_all_orders)], ax=ax4, fit="quadratic")
    




if title == "plot orders":
    
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model)
    
    
#    orders = list(range(145, 186, 10)) #general
#    orders = [190]#list(range(180, 190, 1))
    orders = [130]
    
    plotOrders(hdf5Files, hdf5Filenames, orders)







if title == "bb detector illumination":
    
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model)
    
    PIXEL_START = 160
    PIXEL_END = 240

    for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
        spectrum_found = False
        
#            plt.title("%s gas cell: spectra of order %s for different instrument temperatures" %(molecule.upper(), diffraction_order))
#            plt.xlabel("Pixel number")
#            plt.ylabel("Normalised gas cell spectrum")
        channel = hdf5_filename.split("_")[3].lower()


        detector_data_all = hdf5_file["Science/Y"][...]
        window_top = hdf5_file["Channel/WindowTop"][0]
        binning = hdf5_file["Channel/Binning"][0] + 1
        integration_time = hdf5_file["Channel/IntegrationTime"][0]
        sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
        measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
        
        aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]

        #check if BB filling FOV
        #sum centre of detector for each row
        plt.figure(figsize=(FIG_X - 4, FIG_Y + 2))
        plt.xlabel("Relative blackbody signal (average of pixels %i-%i)" %(PIXEL_START, PIXEL_END))
        plt.ylabel("Detector row number")
        plt.title(channel.upper()+" "+title+": blackbody detector illumination")

        for index in [12, 22, 32, 42, 52]:
            order = findOrder(channel, aotf_frequencies[index])
            
            detector_data_rows = np.mean(detector_data_all[index, :, PIXEL_START:PIXEL_END], axis=1)
            normalised_detector_data_rows = detector_data_rows / np.max(detector_data_rows)
            
            detector_row_numbers = list(range(window_top + int(binning / 2), window_top + 144 + int(binning / 2), binning))

            

            plt.plot(normalised_detector_data_rows, detector_row_numbers, label="Order %i" %order)
#            plt.plot(smooth/np.max(row_max_value), smooth_rows, linewidth=5, color="k", label="Instrument sensitivity")
        plt.tight_layout()
        plt.grid(True)
        plt.legend()




if title == "radiance per pixel":
    
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, model=model)
    
    for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
        if "20150426_054602" in hdf5_filename:
            bb_temp = 423.0
        elif "20150426_030851" in hdf5_filename:
            bb_temp = 423.0
        elif "20150427_010422" in hdf5_filename:
            bb_temp = 423.0
    
    opticsWavenumbers, cslWindowTransmission = opticalTransmission(csl_window="only")
    cslWindowInterp = np.interp(aotfFunctionWavenumbers, opticsWavenumbers, cslWindowTransmission)
    blackbodyFunction = blackbodyFunctionPlanck * cslWindowInterp         
    