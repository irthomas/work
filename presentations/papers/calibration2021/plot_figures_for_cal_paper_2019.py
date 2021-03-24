# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:04:53 2019

@author: iant


PLOT FIGURES FOR CALIBRATION PAPER

"""

import os
#import h5py
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, lfilter
from scipy.optimize import curve_fit

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions import make_filelist
from tools.file.filename_lists import getFilenameList


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

    with open(os.path.join(paths["BASE_DIRECTORY"], "reference_files", "csl_anc_calibration_temperatures.txt")) as f:
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







if title == "MCC line scan":
    """make vertical detector plots where sun is seen to determine slit position and time when in centre"""
    DETECTOR_V_CENTRE = 201
    DETECTOR_CUTOFF = 20000
    hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)

    
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
            plt.savefig(paths["BASE_DIRECTORY"]+os.sep+channel+"_"+title.replace(" ","_")+"_vertical_columns_on_detector_where_sun_is_seen.png")


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
            plt.savefig(paths["BASE_DIRECTORY"] + os.sep+channel+"_"+title.replace(" ", "_") + "_detector_smile.png")



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
        hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)
        
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
        









if title == "plot orders":
    
    hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)
    
    
#    orders = list(range(145, 186, 10)) #general
#    orders = [190]#list(range(180, 190, 1))
    orders = [130]
    
    plotOrders(hdf5Files, hdf5Filenames, orders)







if title == "bb detector illumination":
    
    hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)
    
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
    
    hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)
    
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
    