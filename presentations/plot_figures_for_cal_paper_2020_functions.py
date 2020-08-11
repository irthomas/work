# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:24:47 2020

@author: iant

FUNCTIONS FOR CALCULATING PIXEL TEMPERATURE SHIFTS FROM GROUND CAL DATA
"""


import os
#import h5py
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, lfilter
from scipy.optimize import curve_fit

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from tools.file.paths import paths
#from tools.file.hdf5_functions import make_filelist
#from tools.file.filename_lists import getFilenameList



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



