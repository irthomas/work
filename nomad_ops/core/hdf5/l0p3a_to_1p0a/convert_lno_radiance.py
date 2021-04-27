# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:58:54 2020

@author: iant
"""

import logging
import os.path

import h5py
import numpy as np
from datetime import datetime


from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.running_mean import running_mean

from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, \
    LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME, \
    NA_VALUE, RUNNING_MEAN, REMOVE_NEGATIVES, HDF5_TIME_FORMAT



__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian . thomas @ aeronomie .be"


logger = logging.getLogger( __name__ )





def convert_lno_radiance(hdf5_basename, hdf5FileIn, errorType):
    """Radiometrically calibrate LNO. Not for fullscans"""

    
    #get Y data
    yIn = hdf5FileIn["Science/Y"][...]
    ##get X data
    #xIn = hdf5FileIn["Science/X"][0, :]
    
    #get observation start time and diffraction order/ AOTF
    observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]
    diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...]
    aotfFrequencies = hdf5FileIn["Channel/AOTFFrequency"][...]
    
    integrationTimes = hdf5FileIn["Channel/IntegrationTime"][...]
    bins = hdf5FileIn["Science/Bins"][...]
    nAccumulations = hdf5FileIn["Channel/NumberOfAccumulations"][...]
    
    integrationTime = np.float(integrationTimes[0]) / 1.0e3 #milliseconds to seconds
    nAccumulation = np.float(nAccumulations[0])/2.0 #assume LNO nadir background subtraction is on
    binning = np.float(bins[0,1] - bins[0,0]) + 1.0 #binning starts at zero
    nBins = 1.0 #Science/Bins reflects the real binning
    
    measurementSeconds = integrationTime * nAccumulation
    measurementPixels = binning * nBins
#    logger.info("integrationTime = %0.3f, light nAccumulation = %i, binning = %i, measurementSeconds = %0.1f" %(integrationTime, nAccumulation, binning, measurementSeconds))

    
    #check that all aotf freqs are the same (they should be for this function)
    error = False
    if (aotfFrequencies == aotfFrequencies[0]).all():
        diffractionOrder = diffractionOrders[0]
    else:
        logger.error("Error: AOTF frequencies are not the same in file %s. Use another function for fullscan or calibrations." %hdf5_basename)
        error = True
    
    if not error:
        yBinnedNorm = yIn / measurementSeconds / measurementPixels #scale to counts per second per pixel
        #calculate standard deviation - remove continuum shape to leave random error on first 50 pixels only
        #find first spectrum which is non nan
        not_nan_index = min([i for i, value in enumerate(yBinnedNorm[:, 0]) if not np.isnan(value)])

        yFitted = np.polyval(np.polyfit(range(50), yBinnedNorm[not_nan_index, 0:50], 2), range(50))
        yStd = np.std(yBinnedNorm[not_nan_index, 0:50] - yFitted)
        
        if RUNNING_MEAN:
            """apply running mean with n=10"""
            yIntermediate = running_mean(yBinnedNorm, 10)
            Y = np.copy(yIntermediate)
        else:
            """don't apply running mean with n=10 and remove negatives"""
            yIntermediate = yBinnedNorm
            Y = np.copy(yIntermediate)
            
        if REMOVE_NEGATIVES:
            """remove negatives from data"""
            negativesFound = False
            for spectrumIndex, spectrum in enumerate(yIntermediate):
                if np.min(spectrum) < 0.0:
                    negativesFound = True
                    negativeIndices = np.where(spectrum < 0.0)[0]
                    Y[spectrumIndex, negativeIndices] = 0.0
            if negativesFound:
                logger.info("Warning: negative Y values found in file. Replaced by zeroes")
        
        nSpectra = Y.shape[0] #calculate size
        
        
        """calibrate"""
        #read in data from channel calibration table - do radiance calibration
        logger.info("Opening radiometric calibration file %s.h5 for reading" %LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME)
        radiometric_calibration_table = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME)
        with h5py.File("%s.h5" % radiometric_calibration_table, "r") as calibrationFile:
        
            #convert times to numerical timestamps
            calibrationTimes = list(calibrationFile.keys())
            
            
            calibrationTimestamps = [datetime.strptime(calibrationTime, HDF5_TIME_FORMAT) for calibrationTime in calibrationTimes]
            measurementTimestamp = datetime.strptime(observationStartTime.decode(), HDF5_TIME_FORMAT)
 
            #find which observation corresponds to calibration measurement time
            timeIndex = max([i for i, v in enumerate(calibrationTimestamps) if v < measurementTimestamp])
            calibration_time = calibrationTimes[timeIndex]
            hdf5Group = calibration_time
            logger.info("hdf5Group=%s" %hdf5Group)
            
            calibrationDiffractionOrders = calibrationFile[hdf5Group]["DiffractionOrder"][:]
            diffractionOrderIndex = np.abs(calibrationDiffractionOrders - diffractionOrder).argmin()
            
            CountsPerRadianceFit = calibrationFile[hdf5Group]["CountsPerRadianceFit"][diffractionOrderIndex, :]
            CountsPerRadianceAtWavenumberFit = calibrationFile[hdf5Group]["CountsPerRadianceAtWavenumberFit"][diffractionOrderIndex, :]
        
            table_creation_datetime = calibrationFile.attrs["DateCreated"]
        
        
#        logger.info("Radiance calibration")
            
        """radiance calibration"""
        y_radiance = Y / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
        y_radiance_simple = Y / np.tile(CountsPerRadianceAtWavenumberFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
    
        if errorType == "Per Pixel":
            y_error = yStd / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
            snr = y_radiance / y_error
    
        elif errorType == "One Value":
            logger.info("One value error calculation not yet implemented")
            y_error = np.zeros_like(y_radiance) + np.float(NA_VALUE)
            snr = np.zeros_like(y_radiance) + np.float(NA_VALUE)
        else:
            logger.info("No error calculation selected")
            y_error = np.zeros_like(y_radiance) + np.float(NA_VALUE)
            snr = np.zeros_like(y_radiance) + np.float(NA_VALUE)

           
    radiance_cal_dict = {}
    radiance_cal_dict["Science/YRadiance"] = {"data":y_radiance, "dtype":np.float32, "compression":True}
    radiance_cal_dict["Science/YRadianceSimple"] = {"data":y_radiance_simple, "dtype":np.float32, "compression":True}
    radiance_cal_dict["Science/YRadianceError"] = {"data":y_error, "dtype":np.float32, "compression":True}
    radiance_cal_dict["Science/SNRRadiance"] = {"data":snr, "dtype":np.float32, "compression":True}
    radiance_cal_dict["Science/YNormalisedCounts"] = {"data":Y, "dtype":np.float32, "compression":True}
    radiance_cal_dict["Science/YUnmodified"] = {"data":yIn, "dtype":np.float32, "compression":True}

    calib_ref = "Radiance calibration using table %s created on %s" %(LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME, table_creation_datetime)
    error_ref = "%s radiance error calculated using table %s created on %s" %(errorType, LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME, table_creation_datetime)
    radiance_refs = {"calib_ref":calib_ref, "error_ref":error_ref}

    return radiance_cal_dict, radiance_refs
