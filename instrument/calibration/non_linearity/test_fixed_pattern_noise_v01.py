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



SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


#regex = re.compile("201906.*SO.*_(133|134|135|136).*")
#regex = re.compile("201906.*SO.*_[IE]_.*136.*")
#regex = re.compile("(20190618_105903|20190621_015027|20190623_180452).*SO.*_(133|134|135|136).*")
#regex = re.compile("20190618_105903.*SO.*_(133|134|135|136).*")
#regex = re.compile("20190618_105903.*SO.*_(133|134|135|136).*")
#regex = re.compile("(20190625_233600|20190622_230012|20190620_195653|20190617_223039|20190615_180901).*SO.*")
regex = re.compile("20190625_233600.*SO.*")
fileLevel = "hdf5_level_1p0a"


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
    #    get lat/lons
    lat_data = np.mean(hdf5_file["Geometry/Point0/Lat"][...], axis=1)
    lon_data = np.mean(hdf5_file["Geometry/Point0/Lon"][...], axis=1)
    sbsf = hdf5_file["Channel/BackgroundSubtraction"][0] #find sbsf flag (just take first value)

    if "_0p" in hdf5_filename:
        if sbsf == 0:
            detector_data_light = hdf5_file["Science/Y"][...]
            detector_data_dark = hdf5_file["Science/BackgroundY"][...]
            if detector_data_light.shape[0] == (detector_data_dark.shape[0] * 5):
                print("File has 5x light frames than dark")
                outputDict["5x_lights"] = True
                n_pixels = len(detector_data_light[0, :])
                detector_data_dark = np.reshape(np.repeat(np.reshape(detector_data_dark, [-1, NBINS*n_pixels]), 5, axis=0), [-1, n_pixels]) #this is now correct
            elif (detector_data_light.shape[0] * 5) == (detector_data_dark.shape[0]):
                print("File has 5x dark frames than dark")
                outputDict["5x_darks"] = True
                n_pixels = len(detector_data_light[0, :])
                detector_data_light = np.reshape(np.repeat(np.reshape(detector_data_light, [-1, NBINS*n_pixels]), 5, axis=0), [-1, n_pixels]) #this is now correct
            detector_data = detector_data_light - detector_data_dark
            
        elif sbsf == 1:
            if not silent: print("Background already subtracted")
            detector_data = hdf5_file["Science/Y"][...]
            detector_data_dark = np.zeros_like(detector_data)
        
       
        
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

    else:
        y_error = 0
        indbin = hdf5_file["Channel"]["IndBin"][...]
        bin_indices = np.where(indbin == bin_index)[0]
        y_data = hdf5_file["Science"]["YMean"][bin_indices, :]
        y_data_raw = hdf5_file["Science"]["YUnmodified"][bin_indices, :]
        y_dark_raw = 0.0
        alt = alt_data[bin_indices]
        lat = lat_data[bin_indices]
        lon = lon_data[bin_indices]
    
#    find start/end lat/lons for label
    index_50 = np.abs(alt - 50.0).argmin()
    index_0 = np.abs(alt - 0.0).argmin()
    lat_range = [lat[index_50], lat[index_0]]        
    lon_range = [lon[index_50], lon[index_0]]        

    #correct bad pixel in bin2
    if bin_index == 2:
        y_data[:, 269] = (y_data[:, 268] + y_data[:, 270])/2.0
        y_data[:, 84] = (y_data[:, 83] + y_data[:, 85])/2.0
        
    
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

        #get extra data
        spectral_resolution = hdf5_file["Channel/SpectralResolution"][0]
    #    print("spectral_resolution in file = %0.2f" %spectral_resolution)
    #    sensor1Temperature = hdf5_file["Housekeeping/SENSOR_1_TEMPERATURE_SO"][...]
        #get instrument temperature from aotf temperature measurements (ignore first 2 values - usually wrong)
    #    measurementTemperature = np.mean(sensor1Temperature[2:10])
        measurementTemperature = hdf5_file["Channel/MeasurementTemperature"][0]
        
        outputDict["spec_res"] = spectral_resolution
        outputDict["temperature"] = measurementTemperature

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









#CHOSEN_FILENAMES = ["20190618_105903", "20190621_015027", "20190623_180452"]
POLYFIT_DEGREE = 3
CENTRE_PIXEL = 180
CHOSEN_PIXELS = [180, 182, 183, 186]
CHOSEN_TRANSMITTANCE_RANGE = [0.1, 0.2]
#CHOSEN_TRANSMITTANCE_RANGE = [0.2, 0.3]
#CHOSEN_TRANSMITTANCE_RANGE = [0.3, 0.4]
#CHOSEN_TRANSMITTANCE_RANGE = [0.4, 0.5]

BINS = [0,1,2,3]
yChosenNoWater = []
obsDictsBins = []

hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
#loop through, taking y spectra nearest T=0.3 from each order and storing pixel values where no water lines are present
for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

    obsDicts = []
    for bin_index in BINS:
        obsDict = convertToTransmittance(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=65.0) #use shape method, returns dictionary
        if not obsDict["error"]:
            obsDicts.append(obsDict)
    obsDictsBins.append(obsDicts)


cmap = plt.get_cmap('Set1')
colours = [cmap(i) for i in np.arange(len(BINS))/len(BINS)]

for obsDicts in obsDictsBins:
    fig1, ax1 = plt.subplots(figsize=(FIG_X + 6, FIG_Y))
    for binIndex, obsDict in enumerate(obsDicts):


        y = obsDict["y"]#[:, 50:]
#        y_raw = obsDict["y_raw"]
        order = obsDict["order"]
        temperature = np.round(obsDict["temperature"])
        wavenumbers = obsDict["x"]#[50:] #choose limited region of spectrum for analysis
        pixels = np.arange(0.0, len(wavenumbers))
        not_water_pixels, water_pixels, combined_spectrum = getSOAbsorptionPixels(order, temperature, delta_t=1.0, cutoff=0.999) #for polynomial fit only
#
#        """first plot y light and y dark together"""
#        fig3, (ax3, ax3b) = plt.subplots(figsize=(FIG_X + 6, FIG_Y + 3), nrows=2, sharex=True)
#        for pixel in CHOSEN_PIXELS:
#            ax3.plot(obsDict["y_raw"][:, pixel] - obsDict["y_dark"][:, pixel])
#            ax3b.plot(obsDict["y_dark"][:, pixel])
        
    
    
#        """takes indices in range and plots residual for each bin"""
        indices = np.where((y[:, CENTRE_PIXEL] > CHOSEN_TRANSMITTANCE_RANGE[0]) & (y[:, CENTRE_PIXEL] < CHOSEN_TRANSMITTANCE_RANGE[1]))[0]
        print("nspectra=%i" %len(indices))
        print("temperature=%0.1f" %temperature)
    
        for index in indices:
            fit = np.polyval(np.polyfit(pixels[not_water_pixels], y[index, not_water_pixels], POLYFIT_DEGREE), pixels)
                
            yNormalised = y[index, :] - fit #this is the residual
            pixels[water_pixels] = np.nan
            yNormalised[water_pixels] = np.nan
    
            ax1.plot(pixels, yNormalised - (binIndex / 100), color=colours[binIndex], alpha=0.5)
        ax1.set_xlim([0,320])
        ax1.set_ylabel("Residual transmittance")
        ax1.set_xlabel("Pixel Number")
        ax1.set_title("Fixed pattern noise during occultation, H2O-free pixels")

        """finds residual of all frames first"""
        yNormalised = np.zeros_like(y)
        for frame_index in range(len(y[:,0])):
            fit = np.polyval(np.polyfit(pixels[not_water_pixels], y[frame_index, not_water_pixels], POLYFIT_DEGREE), pixels)
            yNormalised[frame_index,:] = y[frame_index, :] - fit #this is the residual
            yNormalised[frame_index, water_pixels] = np.nan
#
#        
        fig2, (ax2,ax2b) = plt.subplots(figsize=(FIG_X + 6, FIG_Y + 3), nrows=2, sharex=True)
        for pixel in CHOSEN_PIXELS:
            ax2.plot(y[:, pixel], yNormalised[:, pixel], label="Bin %i pixel %i" %(BINS[bin_index], pixel))
            
            #remove sun region
            atmos_indices = np.where((y[:, pixel] > 0.01) & (y[:, pixel] < 0.98))[0]
            
            
            fit = np.polyval(np.polyfit(y[atmos_indices, pixel], yNormalised[atmos_indices, pixel], 1), y[atmos_indices, pixel])
            ax2.plot(y[atmos_indices, pixel], fit, linestyle="--")
            
#            ax2.plot(yNormalised[:, pixel], label="Pixel %i" %pixel)
#            ax2b.plot(y[:, CENTRE_PIXEL])
        ax2.set_title(obsDict["label"])
        ax2.legend()


