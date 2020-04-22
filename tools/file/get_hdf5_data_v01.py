# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:31:21 2019

@author: iant
"""

import numpy as np
from datetime import datetime


def getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=65.0):
    """Convert the selected bin to transmittance and output a dictionary
    Inputs:
        hdf5_file = an open level 1.0a hdf5 file
        hdf5_filename = filename (not path)
        bin_index = bin number i.e. 0, 1, 2 or 3
        silent = remove verbose message
        top_of_atmosphere = tangent altitude of Sun region
    Output:
        a dictionary containing selected fields
        y_mean = calculated transmittance
        x = wavenumbers
        alt = tangent altitude areoid
        label = a label for adding to a legend
        etc.
    """
    
    outputDict = {}
    outputDict["error"] = False

    TEST_ALTITUDE = top_of_atmosphere
    ALTITUDE_FIELD = "TangentAltAreoid"
    MAX_PERCENTAGE_ERROR = 2.0
    TRANSMITTANCE_FIELD = "Y"
    DETECTOR_DATA_FIELD = "YUnmodified"
    DETECTOR_DATA_ERROR = "YErrorNorm"
        
    #get info from filename
    obspath_split = hdf5_filename.split("_")
    obs_type = obspath_split[5]
    diffraction_order = obspath_split[6]
    
    if not silent: print("Reading in IR file %s" %hdf5_filename)
    
    bins = hdf5_file["Science/Bins"][:, 0]
    uniqueBins = sorted(list(set(bins)))
    binIndex = np.where(bins == uniqueBins[bin_index])[0]
      
    transmittance = hdf5_file["Science/%s" %TRANSMITTANCE_FIELD][binIndex, :]
    detector_data = hdf5_file["Science/%s" %DETECTOR_DATA_FIELD][binIndex, :]
    detector_error = hdf5_file["Science/%s" %DETECTOR_DATA_ERROR][binIndex, :]
    lon = hdf5_file["Geometry/Point0/Lon"][binIndex, 0]
    lat = hdf5_file["Geometry/Point0/Lat"][binIndex, 0]
    alt = hdf5_file["Geometry/Point0/%s" %ALTITUDE_FIELD][binIndex, 0]
    
  
    wavenumbers = hdf5_file["Science/X"][0, :]
           
        
    #get indices for top of atmosphere
    top_indices = np.where(alt>top_of_atmosphere)[0]
    if len(top_indices) < 10:
        print("Error: Insufficient points above %i. n points = %i" %(top_of_atmosphere, len(top_indices)))
        print(hdf5_filename)
        outputDict["error"] = True
        
    #data is sorted altitude ascending
    detector_data_sun_mean = np.mean(detector_data[top_indices[:10], :], axis=0)
    detector_data_transmittance = detector_data / detector_data_sun_mean
    
    #check error
    index_atmos_top = np.abs(alt - TEST_ALTITUDE).argmin()
    y_error = np.abs((detector_data_sun_mean[200] - detector_data[index_atmos_top, 200]) / detector_data[index_atmos_top, 200])
    if not silent: print("%ikm error = %0.2f" %(TEST_ALTITUDE, y_error * 100) + r"%")
    
    #find start/end lat/lons for label
    index_toa = np.abs(alt - top_of_atmosphere).argmin()
    index_0 = np.abs(alt - 0.0).argmin()
    lat_range = [lat[index_toa], lat[index_0]]        
    lon_range = [lon[index_toa], lon[index_0]]        
    label_full_out = hdf5_filename[0:15]+"_"+obs_type+" lat=%0.0f to %0.0f, lon=%0.0f to %0.0f" %(lat_range[0],lat_range[1],lon_range[0],lon_range[1])
    label_out = hdf5_filename[0:15]+"_"+obs_type+" order %s" %diffraction_order
    
#    #correct bad pixel in bin2
#    if bin_index == 2:
#        detector_data_transmittance[:, 269] = (detector_data_transmittance[:, 268] + detector_data_transmittance[:, 270])/2.0
#        detector_data_transmittance[:, 84] = (detector_data_transmittance[:, 83] + detector_data_transmittance[:, 85])/2.0
        
    
    if y_error * 100 > MAX_PERCENTAGE_ERROR:
        print("Warning: error too large. %ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
        print(hdf5_filename)
        outputDict["error"] = True

    #get extra data
#    spectral_resolution = hdf5_file["Channel/SpectralResolution"][0]
    measurementTemperature = hdf5_file["Channel/MeasurementTemperature"][0]
    firstPixel = hdf5_file["Channel/FirstPixel"][0]
    ls = hdf5_file["Geometry/LSubS"][0, 0]
    lst = hdf5_file["Geometry/Point0/LST"][0, 0]
    obsDatetime = datetime.strptime(hdf5_filename[0:15], '%Y%m%d_%H%M%S')

    outputDict["hdf5_filename"] = hdf5_filename
    
    outputDict["alt"] = alt
    outputDict["x"] = np.tile(wavenumbers, [len(binIndex), 1])
    outputDict["y_mean"] = detector_data_transmittance
    outputDict["y"] = transmittance
    outputDict["y_raw"] = detector_data
    outputDict["y_error"] = detector_error
    outputDict["order"] = int(diffraction_order)
    outputDict["temperature"] = measurementTemperature
    outputDict["first_pixel"] = firstPixel
    outputDict["obs_datetime"] = obsDatetime
    outputDict["label"] = label_out
    outputDict["label_full"] = label_full_out
    outputDict["bin_index"] = np.asarray([bin_index] * len(binIndex))
    outputDict["ls"] = np.tile(ls, len(binIndex))
    outputDict["lst"] = np.tile(lst, len(binIndex))
    outputDict["longitude"] = lon
    outputDict["latitude"] = lat

    return outputDict













def getLevel0p2Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=65.0):
    
    outputDict = {}
    outputDict["error"] = False

    TEST_ALTITUDE = top_of_atmosphere
    ALTITUDE_FIELD = "TangentAltAreoid"
    MAX_PERCENTAGE_ERROR = 2.0
    DETECTOR_DATA_FIELD = "Y"
        
    #get info from filename
    obspath_split = hdf5_filename.split("_")
    obs_type = obspath_split[5]
    diffraction_order = obspath_split[6]
    
    if not silent: print("Reading in IR file %s" %hdf5_filename)
    
    bins = hdf5_file["Science/Bins"][:, 0]
    uniqueBins = sorted(list(set(bins)))
    binIndex = np.where(bins == uniqueBins[bin_index])[0]
    alt = hdf5_file["Geometry/Point0/%s" %ALTITUDE_FIELD][binIndex, 0]

    altitude_indices = np.argsort(alt)
      
    detector_data = hdf5_file["Science/%s" %DETECTOR_DATA_FIELD][binIndex, :][altitude_indices, :]
    lon = hdf5_file["Geometry/Point0/Lon"][binIndex, 0][altitude_indices]
    lat = hdf5_file["Geometry/Point0/Lat"][binIndex, 0][altitude_indices]
    alt = hdf5_file["Geometry/Point0/%s" %ALTITUDE_FIELD][binIndex, 0][altitude_indices]
    
  
        
    #get indices for top of atmosphere
    top_indices = np.where(alt>top_of_atmosphere)[0]
    if len(top_indices) < 10:
        print("Error: Insufficient points above %i. n points = %i" %(top_of_atmosphere, len(top_indices)))
        print(hdf5_filename)
        outputDict["error"] = True
        
    #data is sorted altitude ascending
    detector_data_sun_mean = np.mean(detector_data[top_indices[:10], :], axis=0)
    detector_data_transmittance = detector_data / detector_data_sun_mean
    
    #check error
    index_atmos_top = np.abs(alt - TEST_ALTITUDE).argmin()
    y_error = np.abs((detector_data_sun_mean[200] - detector_data[index_atmos_top, 200]) / detector_data[index_atmos_top, 200])
    if not silent: print("%ikm error = %0.2f" %(TEST_ALTITUDE, y_error * 100) + r"%")
    
    #find start/end lat/lons for label
    index_50 = np.abs(alt - 50.0).argmin()
    index_0 = np.abs(alt - 0.0).argmin()
    lat_range = [lat[index_50], lat[index_0]]        
    lon_range = [lon[index_50], lon[index_0]]        
    label_out = hdf5_filename[0:15]+"_"+obs_type+" lat=%0.0f to %0.0f, lon=%0.0f to %0.0f" %(lat_range[0],lat_range[1],lon_range[0],lon_range[1])
    label_out = hdf5_filename[0:15]+"_"+obs_type+" order %s" %diffraction_order
    
    #correct bad pixel in bin2
    if bin_index == 2:
        detector_data_transmittance[:, 269] = (detector_data_transmittance[:, 268] + detector_data_transmittance[:, 270])/2.0
        detector_data_transmittance[:, 84] = (detector_data_transmittance[:, 83] + detector_data_transmittance[:, 85])/2.0
        
    
    if y_error * 100 > MAX_PERCENTAGE_ERROR:
        print("Warning: error too large. %ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
        print(hdf5_filename)
        outputDict["error"] = True

    #get extra data
    measurementTemperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_SO"][2:10])
    firstPixel = np.polyval([-7.299039e-1, -6.267734], measurementTemperature)
    
    wavenumbers = np.polyval([1.751279e-8, 5.559526e-4, 22.473422], np.arange(320)+firstPixel) * float(diffraction_order)

    ls = hdf5_file["Geometry/LSubS"][0, 0]
    lst = hdf5_file["Geometry/Point0/LST"][0, 0]
    obsDatetime = datetime.strptime(hdf5_filename[0:15], '%Y%m%d_%H%M%S')

    outputDict["hdf5_filename"] = hdf5_filename
    
    outputDict["alt"] = alt
    outputDict["x"] = np.tile(wavenumbers, [len(binIndex), 1])
    outputDict["y_mean"] = detector_data_transmittance
    outputDict["y"] = detector_data_transmittance
    outputDict["y_raw"] = detector_data
    outputDict["y_error"] = detector_data_transmittance / 1000.0
    outputDict["order"] = int(diffraction_order)
    outputDict["temperature"] = measurementTemperature
    outputDict["first_pixel"] = firstPixel
    outputDict["obs_datetime"] = obsDatetime
    outputDict["label"] = label_out

    outputDict["ls"] = np.tile(ls, len(binIndex))
    outputDict["lst"] = np.tile(lst, len(binIndex))
    outputDict["longitude"] = lon
    outputDict["latitude"] = lat

    return outputDict





