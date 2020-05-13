# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:31:46 2020

@author: iant
"""

import logging
import os.path

import h5py
import numpy as np
import spiceypy as sp
#from scipy import interpolate
#import re
#import matplotlib.pyplot as plt

from instrument.calibration.lno_radiance_factor.config import \
    NOMAD_TMP_DIR, LNO_FLAGS_DICT, RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, \
    LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME, \
    PFM_AUXILIARY_FILES, THUMBNAIL_DIRECTORY, SAVE_FILES, NA_VALUE, DATASETS_TO_BE_REMOVED, \
    ATTRIBUTES_TO_BE_REMOVED, trans, generics

from instrument.calibration.lno_radiance_factor.functions.check_flags import check_flags


logger = logging.getLogger( __name__ )



def convert_lno_fullscan_radiance(hdf5file_path):
    """not yet implemented. Fill with zeros"""
    """Radiometrically calibrate LNO fullscans"""

    hdf5FileIn = h5py.File(hdf5file_path, "r")
    hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
#    hdf5FileOut = h5py.File(hdf5FilepathOut, "w")
    #get observation start time and diffraction order/ AOTF
    observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]

    
    #read in data from channel calibration table
    logger.info("Opening radiometric calibration file %s.h5 for reading", LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME)
    radiometric_calibration_table = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME)
    with h5py.File("%s.h5" % radiometric_calibration_table, "r") as calibrationFile:

        #convert times to numerical timestamps
        calibrationTimes = list(calibrationFile.keys())
        calibrationTimestamps = np.asfarray([sp.utc2et(calibrationTime) for calibrationTime in calibrationTimes])
        measurementTimestamp = sp.utc2et(observationStartTime)
        
        #find which observation corresponds to calibration measurement time
        timeIndex = np.max(np.where(measurementTimestamp>calibrationTimestamps))
        calibrationTime = calibrationTimes[timeIndex]
        hdf5Group = calibrationTime
        logger.info("hdf5Group=%s", hdf5Group)
        tableCreationDatetime = calibrationFile.attrs["DateCreated"]
#       tableComments = calibrationFile.attrs["Comments"]


#    diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...]
#    aotfFrequencies = hdf5FileIn["Channel/AOTFFrequency"][...]


    #apply running mean with n=10
    yIntermediate = hdf5FileIn["Science/Y"][...]
    #fill with zeroes for the time being
    YOut = np.zeros_like(yIntermediate)
#    nSpectra = Y.shape[0]
    YError = np.zeros_like(YOut) + np.float(NA_VALUE)
    SNR = np.zeros_like(YOut) + np.float(NA_VALUE)


    



    YCalibRef = "CalibrationTime=%s; File=%s" %(calibrationTime,tableCreationDatetime)
    YErrorRef = "CalibrationTime=%s; File=%s" %(calibrationTime,tableCreationDatetime)

    YTypeFlagOld = hdf5FileIn["Science/YTypeFlag"]
    YTypeFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_TYPE_FLAG"]
    YUnitFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_UNIT_FLAG"]
    YErrorFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_ERROR_FLAG"]

    logger.info("Writing new datasets to output file")
    with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:

        generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
        for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            if dset_path in DATASETS_TO_BE_REMOVED:
                continue
            dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            hdf5FileIn.copy(dset_path, dest)
    
        hdf5FileOut.attrs["YCalibRef"] = YCalibRef
        hdf5FileOut.attrs["YErrorRef"] = YErrorRef
    
        hdf5FileOut.create_dataset("Science/Y", dtype=np.float, data=YOut, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YError", dtype=np.float, data=YError, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.int, data=YTypeFlag, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.int, data=YUnitFlag, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.int, data=YErrorFlag, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/SNR", dtype=np.float, data=SNR, compression="gzip", shuffle=True)

    hdf5FileIn.close()

    return hdf5FilepathOut


