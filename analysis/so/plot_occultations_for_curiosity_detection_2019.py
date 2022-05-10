# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Thu Feb  7 14:43:38 2019

@author: iant


"""


import os
#import h5py
import numpy as np
#import numpy.linalg as la
#import gc
#from scipy import stats
#import scipy.optimize
import re

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

from plot_simulations_v01 import plotSimulation, getSimulation, getSOAbsorptionPixels, getSimulationDataNew

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames
#from analysis_functions_v01b import write_log
#from filename_lists_v01 import getFilenameList

#if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
#    print("Running on windows")
#    import spiceypy as sp



#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""

#title = "make simulations"
title = "curiosity detection"

#obspaths = re.compile("201906.*SO.*_(133|134|135|136).*")
#obspaths = re.compile("201906.*SO.*_[IE]_.*136.*")
obspaths = re.compile("(20190618_105903|20190621_015027|20190623_180452).*SO.*_(133|134|135|136).*")
#obspaths = re.compile("20190618_105903.*SO.*_(133|134|135|136).*")
fileLevel = "hdf5_level_0p3k"


def splitIntoBins(data_in, n_bins):
    nSpectra = data_in.shape[0]
    data_out = []
    for index in range(n_bins):
        binRange = range(index, nSpectra, n_bins)
        if data_in.ndim ==2:
            data_out.append(data_in[binRange, :])
        else:
            data_out.append(data_in[binRange])
            
    return data_out



def polynomialFit(array_in, order_in):
    arrayShape = array_in.shape
    if len(arrayShape) == 1:
        nElements = array_in.shape[0]
    
    return np.polyval(np.polyfit(range(nElements), array_in, order_in), range(nElements))




def convertToTransmittance(hdf5_file, hdf5_filename, bin_index, silent=False, top_of_atmosphere=80.0):
    
    SIMPLE_MEAN = True
    TOP_OF_ATMOSPHERE = top_of_atmosphere #above this altitude is sun only. Plot polynomial sun spectrum over this range
    TEST_ALTITUDE = top_of_atmosphere #km. check extrapolation at this altitude to define error
    MAX_PERCENTAGE_ERROR = 2.0 #% if transmittance error greater than this then discard file
    NBINS = 4
    POLYFIT_DEGREE = 2
    PIXEL_NUMBER_START = 190 #centre range of detector
    PIXEL_NUMBER_END = 210 #centre range of detector
    PIXEL_NUMBER = 160 #centre of detector
    
#    DETECTOR_DATA_FIELD = "Y"
    ALTITUDE_FIELD = "TangentAltAreoid"
    
    #get info from filename
    obspath_split = hdf5_filename.split("_")
    obs_type = obspath_split[5]
    diffraction_order = obspath_split[6]

    if not silent: print("Reading in IR file %s" %hdf5_filename)
    
    outputDict = {"error":False}
    
    alt_data = np.mean(hdf5_file["Geometry/Point0/%s" %ALTITUDE_FIELD][...], axis=1)
    sbsf = hdf5_file["Channel/BackgroundSubtraction"][0] #find sbsf flag (just take first value)
    if sbsf == 0:
        detector_data_light = hdf5_file["Science/Y"][...]
        detector_data_dark = hdf5_file["Science/BackgroundY"][...]
    
        if detector_data_light.shape[0] == (detector_data_dark.shape[0] * 5):
            print("File has 5x light frames than dark")
            outputDict["5x_lights"] = True
            n_pixels = len(detector_data_light[0, :])
            detector_data_dark = np.reshape(np.repeat(np.reshape(detector_data_dark, [-1, NBINS*n_pixels]), 5, axis=0), [-1, n_pixels])
        elif (detector_data_light.shape[0] * 5) == (detector_data_dark.shape[0]):
            print("File has 5x dark frames than dark")
            outputDict["5x_darks"] = True
            n_pixels = len(detector_data_light[0, :])
            detector_data_light = np.reshape(np.repeat(np.reshape(detector_data_light, [-1, NBINS*n_pixels]), 5, axis=0), [-1, n_pixels])
        detector_data = detector_data_light - detector_data_dark
        
    elif sbsf == 1:
        if not silent: print("Background already subtracted")
        detector_data = hdf5_file["Science/Y"][...]
        detector_data_dark = np.zeros_like(detector_data)
#    get lat/lons
    lat_data = np.mean(hdf5_file["Geometry/Point0/Lat"][...], axis=1)
    lon_data = np.mean(hdf5_file["Geometry/Point0/Lon"][...], axis=1)
    
    #get extra data
    spectral_resolution = hdf5_file["Channel/SpectralResolution"][0][0]
    print("spectral_resolution = %0.2f" %spectral_resolution)
    sensor1Temperature = hdf5_file["Housekeeping/SENSOR_1_TEMPERATURE_SO"][...]
    #get instrument temperature from aotf temperature measurements (ignore first 2 values - usually wrong)
    measurementTemperature = np.mean(sensor1Temperature[2:10])    
    
    outputDict["spec_res"] = spectral_resolution
    outputDict["temperature"] = measurementTemperature
    
    
    #get data for desired bin only
    detector_data_split = splitIntoBins(detector_data, NBINS)[bin_index]
    detector_dark_split = splitIntoBins(detector_data_dark, NBINS)[bin_index]
    alt_split = splitIntoBins(alt_data, NBINS)[bin_index]
    lat_split = splitIntoBins(lat_data, NBINS)[bin_index]
    lon_split = splitIntoBins(lon_data, NBINS)[bin_index]
    
    detector_data_valid = detector_data_split[alt_split > -100.0, :]
    detector_dark_valid = detector_dark_split[alt_split > -100.0, :]
    alt_valid = alt_split[alt_split > -100.0]
    lat_valid = lat_split[alt_split > -100.0]
    lon_valid = lon_split[alt_split > -100.0]
    
    sort_indices = np.argsort(alt_valid)
    
    detector_data = detector_data_valid[sort_indices, :]
    detector_dark = detector_dark_valid[sort_indices, :]
    alt = alt_valid[sort_indices]
    lat = lat_valid[sort_indices]
    lon = lon_valid[sort_indices]
    
    
        
#    find start/end lat/lons for label
    index_50 = np.abs(alt - 50.0).argmin()
    index_0 = np.abs(alt - 0.0).argmin()
    lat_range = [lat[index_50], lat[index_0]]        
    lon_range = [lon[index_50], lon[index_0]]        
        
    #get indices for top of atmosphere
    top_indices = np.where(alt>TOP_OF_ATMOSPHERE)[0]
    if len(top_indices) < 10:
        print("Error: Insufficient points above %i. n points = %i" %(TOP_OF_ATMOSPHERE, len(top_indices)))
        print(hdf5_filename)
        
    if SIMPLE_MEAN:
        #data is sorted altitude ascending
        detector_data_sun_mean = np.mean(detector_data[top_indices[:10], :], axis=0)
        detector_data_transmittance = detector_data / detector_data_sun_mean
        index_atmos_top = np.abs(alt - TEST_ALTITUDE).argmin()
        y_error = np.abs((detector_data_sun_mean[PIXEL_NUMBER] - detector_data[index_atmos_top,PIXEL_NUMBER])/detector_data[index_atmos_top,PIXEL_NUMBER])
    else:
        detector_data_sun = detector_data[top_indices,:]
        detector_data_sun_mean = np.mean(detector_data_sun, axis=0)
        detector_data_polyfit = np.polyfit(alt[top_indices], np.mean(detector_data_sun[:, PIXEL_NUMBER_START:PIXEL_NUMBER_END], axis=1)/ np.mean(detector_data_sun_mean[PIXEL_NUMBER_START:PIXEL_NUMBER_END]), POLYFIT_DEGREE)
        detector_data_polyval = np.polyval(detector_data_polyfit, alt)
        detector_data_extrap = np.zeros_like(detector_data)
        for pixel_number in range(320):
            detector_data_extrap[:,pixel_number] = detector_data_polyval*detector_data_sun_mean[pixel_number]
        
        index_atmos_top = np.abs(alt - TEST_ALTITUDE).argmin()
        
        detector_data_transmittance = detector_data / detector_data_extrap
        y_error = np.abs((detector_data_extrap[index_atmos_top,PIXEL_NUMBER] - detector_data[index_atmos_top,PIXEL_NUMBER])/detector_data[index_atmos_top,PIXEL_NUMBER])
    y_data = detector_data_transmittance
    y_data_raw = detector_data
    y_dark_raw = detector_dark
        
    
    if not silent: print("%ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
    if y_error * 100 > MAX_PERCENTAGE_ERROR:
        print("Warning: error too large. %ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
        print(hdf5_filename)
        outputDict["error"] = True
    else:
        x = hdf5_file["Science/X"][...]
        ls = hdf5_file["Geometry/LSubS"][0,0]
        x_data = x[0]
        obsDatetime = datetime.strptime(hdf5_filename[0:15], '%Y%m%d_%H%M%S')
#        lst,terminator = getCorrectLST(hdf5_file, silent=silent)
        if not silent: print("lat_mean = %0.1f" %np.mean(lat))
        label_out = hdf5_filename[0:15]+"_"+obs_type+" lat=%0.0f to %0.0f, lon=%0.0f to %0.0f" %(lat_range[0],lat_range[1],lon_range[0],lon_range[1])
        label_out = hdf5_filename[0:15]+"_"+obs_type+" order %s" %diffraction_order
    
        outputDict["x"] = x_data
        outputDict["y"] = y_data
        outputDict["y_raw"] = y_data_raw
        outputDict["y_dark"] = y_dark_raw
        outputDict["alt"] = alt
        outputDict["order"] = int(diffraction_order)
        outputDict["lat"] = lat
        outputDict["lon"] = lon
        outputDict["ls"] = ls
        outputDict["sbsf"] = sbsf
        outputDict["lat_range"] = lat_range
        outputDict["lon_range"] = lon_range
        outputDict["obs_datetime"] = obsDatetime
        #"lst":lst, "terminator":terminator, \
        outputDict["label"] = label_out
        
        
        
    return outputDict









CHOSEN_FILENAMES = ["20190618_105903", "20190621_015027", "20190623_180452"]
POLYFIT_DEGREE = 3

#####BEGIN SCRIPTS########
if title == "curiosity detection":
    for CHOSEN_FILENAME in CHOSEN_FILENAMES:
        
        hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
    #    DETECTOR_REGION = np.arange(160,250)
        CHOSEN_TRANSMITTANCE = 0.30
        CENTRE_PIXEL = 200 #use 150 if cutting off first 50 pixels
        
        obsDicts = []
        bin_index = 1
        yChosenNoWater = []
        
        #loop through, taking y spectra nearest T=0.3 from each order and storing pixel values where no water lines are present
        for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        
            obsDict = convertToTransmittance(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=65.0) #use shape method, returns dictionary
            if not obsDict["error"]:
                if CHOSEN_FILENAME in hdf5_filename:
                    obsDicts.append(obsDict)
                #ignore first 50 pixels
                y = obsDict["y"]#[:, 50:]
                order = obsDict["order"]
                temperature = np.round(obsDict["temperature"])
                wavenumbers = obsDict["x"]#[50:] #choose limited region of spectrum for analysis
                not_water_pixels_strict, water_pixels_strict, combined_spectrum_strict = getSOAbsorptionPixels(order, temperature, cutoff=0.995) #for removing water/ch4 pixels
                not_water_pixels, water_pixels, combined_spectrum = getSOAbsorptionPixels(order, temperature, cutoff=0.98) #for polynomial fit only

    #           plt.plot(alt, obsDict["y"][:,200]) #check TOA altitude
                 
                index = np.abs(y[:, CENTRE_PIXEL] - CHOSEN_TRANSMITTANCE).argmin()
                fit = np.polyval(np.polyfit(wavenumbers[not_water_pixels], y[index, not_water_pixels], POLYFIT_DEGREE), wavenumbers)
                
#                if order == 135:
#                    plt.figure()
#                    plt.plot(wavenumbers, y[index, :])
#                    plt.scatter(wavenumbers[not_water_pixels], y[index, not_water_pixels])
##                    plt.plot(wavenumbers, fit)
#                    stop()
                
                yNormalised = y[index, :] - fit + 1.0
                yNoWater = np.copy(yNormalised)
                yNoWater[water_pixels_strict] = np.nan
                yChosenNoWater.append(yNoWater)

#                if order == 135:
#                    plt.figure()
#                    plt.plot(wavenumbers, yNormalised)
#                    plt.plot(wavenumbers, combined_spectrum / 100.0 + 0.99)
#                    plt.scatter(wavenumbers[not_water_pixels], yNormalised[not_water_pixels] / 100.0 + 0.99)
#                    
#                    stop()

    
        yChosenNoWater = np.asfarray(yChosenNoWater)
        yMean = np.nanmean(yChosenNoWater, axis=0)
        
        cmap = plt.get_cmap('Set1')
        colours = [cmap(i) for i in np.arange(len(obsDicts))/len(obsDicts)]
        
        
        fig1, (ax1,ax2) = plt.subplots(figsize=(FIG_X + 6, FIG_Y + 3), nrows=2, sharex=True)
#        fig1, ax1 = plt.subplots(figsize=(FIG_X + 6, FIG_Y + 3))
        
        altitudes = []
        for dictIndex, obsDict in enumerate(obsDicts):
            y = obsDict["y"]#[:, 50:]
            x = obsDict["x"]#[50:]
            alt = obsDict["alt"]
            order = obsDict["order"]
            index = np.abs(y[:, CENTRE_PIXEL] - CHOSEN_TRANSMITTANCE).argmin()
            print("transmittance = %0.2f @ %0.1fkm" %(y[index, CENTRE_PIXEL], alt[index]))
            altitudes.append(alt[index])

            temperature = np.round(obsDict["temperature"])
            wavenumbers = obsDict["x"]#[50:] #choose limited region of spectrum for analysis
#                water_transmittance = getSimulation(wavenumbers, "so", "H2O") #old simulations in wavenumber, not pixels
#                water_pixels = np.where(water_transmittance < 0.9995)[0]
#                not_water_pixels = np.where(water_transmittance > 0.9999)[0]
            not_water_pixels_strict, water_pixels_strict, combined_spectrum_strict = getSOAbsorptionPixels(order, temperature, cutoff=0.995) #for removing water/ch4 pixels
            not_water_pixels, water_pixels, combined_spectrum = getSOAbsorptionPixels(order, temperature, cutoff=0.98) #for polynomial fit only

#            fit = polynomialFit(y[index, :], POLYFIT_DEGREE) #cubic fit to remove slope
            fit = np.polyval(np.polyfit(x[not_water_pixels], y[index, not_water_pixels], POLYFIT_DEGREE), wavenumbers)
            yNormalised = y[index, :] - fit + 1.0 #keep as 1.0
            yCorrected = yNormalised / yMean
            ax1.plot(x, yNormalised, color=colours[dictIndex], alpha=0.3) #plot uncorrected
            ax1.plot(x, yCorrected, color=colours[dictIndex], label=obsDict["label"])
        
#            water_transmittance = getSimulation(x, "so", "H2O %i" %order, new=True)
#            methane_transmittance = getSimulation(x, "so", "CH4 %i" %order, new=True)
            vmr = 10.0
            alt = 15.0
            water_x, water_transmittance = getSimulationDataNew("so", "H2O", order, vmr, alt, temperature)
            vmr = 0.25 / 1e3
            methane_x, methane_transmittance = getSimulationDataNew("so", "CH4", order, vmr, alt, temperature)
            ax2.plot(x, water_transmittance, color=colours[dictIndex], linestyle="--")
            if dictIndex == 0:
                ax1.plot(x, methane_transmittance, color="k", label="CH4 250pptv")
            else:
                ax1.plot(x, methane_transmittance, color="k")
            
#            ax2.plot(x, getSimulation(x, "so", "H2O"))#old simulations in wavenumber, not pixels
            
        ax1.legend(loc="lower left")
        ax2.set_title("H2O simulation")
        ax1.set_title("Transmittance before/after fixed pattern noise removal, altitude = %0.1f-%0.1fkm" %(min(altitudes), max(altitudes)))
        ax1.set_ylabel("Normalised transmittance (shape removed)")
        ax2.set_xlabel("Wavenumber cm-1")
        if SAVE_FIGS:
            fig1.savefig(os.path.join(BASE_DIRECTORY, "CH4_search_%s" %CHOSEN_FILENAME), dpi=300)



