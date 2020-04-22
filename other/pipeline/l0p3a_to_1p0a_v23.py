# -*- coding: utf-8 -*-

#TESTING=True
TESTING=False



import logging
import os.path

import h5py
import numpy as np
import spiceypy as sp
#from scipy import interpolate
import re

import matplotlib.pyplot as plt

logger = logging.getLogger( __name__ )

if not TESTING:
    import matplotlib
    matplotlib.use('Agg')
    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
    from nomad_ops.config import ROOT_STORAGE_PATH
    import nomad_ops.core.hdf5.generic_functions as generics
    #from pipeline.pipeline_config_v04 import NA_VALUE, NA_STRING
    from nomad_ops.core.hdf5.l0p3a_to_1p0a import l0p3a_to_1p0a_v23_Transmittance as trans
    from nomad_ops.core.hdf5.l0p3a_to_1p0a.lno_rad_fac_functions import NADIR_DICT, correctSpectralShift, getCorrectedSolarSpectrum, getReferenceSpectra, checkNadirSpectra
    from nomad_ops.core.hdf5.l0p3a_to_1p0a.build_lno_solar_atmospheric_orders_v01 import nu_mp
    SAVE_FILES = True
else:
    from lno_rad_fac_functions import NADIR_DICT, correctSpectralShift, getCorrectedSolarSpectrum, getReferenceSpectra, checkNadirSpectra
    from build_lno_solar_atmospheric_orders_v01 import nu_mp
    PFM_AUXILIARY_FILES = r"C:\Users\iant\Documents\DATA\pfm_auxiliary_files"
    NOMAD_TMP_DIR = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\tmp")
    ROOT_STORAGE_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
    trans = []
    generics = []
    SAVE_FILES = False
    logging.basicConfig(level=logging.INFO)



__project__   = "NOMAD"
__author__    = "Ian Thomas, Roland Clairquin, Loïc Trompet"
__contact__   = "ian.thomas@oma.be"




#============================================================================================
# 5. CONVERT HDF5 LEVEL 0.3A TO LEVEL 1.0A
# RADIOMETRIC CALIBRATION OF LNO CHANNEL:
# NADIR AND LIMB
# NADIR FULLSCAN
# SO FULLSCAN
#============================================================================================
# DONE:
# SWAP Y AND YRADIANCE
# ADD FIT COEFFICIENTS TO FILE
# TO DO:
# ADD RADIOMETRIC CALIBRATION ERROR
# ADD FILE ATTRIBUTE CALIBRATION REFERENCES
# ADD TEMPERATURE DB DATA
# USE ONLY MATCHING ABSORPTIONS TO CALCULATE N FITS AND ERROR IN FILE AND ON PLOT
#============================================================================================



VERSION = 80
OUTPUT_VERSION = "1.0A"

"""set constants"""
NA_VALUE = -999

FIG_X = 15
FIG_Y = 8

#RUNNING_MEAN = True
RUNNING_MEAN = False

#REMOVE_NEGATIVES = True
REMOVE_NEGATIVES = False


"""set paths to calibration files"""
LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME = "LNO_Radiometric_Calibration_Table_v03"
#LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v01"
LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v02"

RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration")


"""specify datasets/attributes to be modified/removed"""
DATASETS_TO_BE_REMOVED = [
    "Science/X",
    "Science/Y",
    "Science/YTypeFlag",
    "Science/YUnitFlag",
    "Science/YError",
    "Science/YErrorFlag",
    "Science/YUnmodified",
    ]

ATTRIBUTES_TO_BE_REMOVED = ["YCalibRef", "YErrorRef"]



LNO_FLAGS_DICT = {
#    "Y_UNIT_FLAG":4, # 0 = NONE; 1 = RADIANCE FACTOR; 2 = RADIANCE; 3 = BRIGHTNESS TEMPERATURE; 4 = RADIANCE AND RADIANCE FACTOR
#    "Y_TYPE_FLAG":5, #0 = NONE; 1 = RADIANCE; 2 = ; 3 = RADIANCE FACTOR; 4 = BRIGHTNESS TEMPERATURE; 5 = RADIANCE AND RADIANCE FACTOR
#    "Y_ERROR_FLAG":2, #0 = NONE; 1 = ONE VALUE; 2 = PER PIXEL
    "Y_UNIT_FLAG":2, # 0 = NONE; 1 = RADIANCE FACTOR; 2 = RADIANCE; 3 = BRIGHTNESS TEMPERATURE; 4 = RADIANCE AND RADIANCE FACTOR
    "Y_TYPE_FLAG":1, #0 = NONE; 1 = RADIANCE; 2 = ; 3 = RADIANCE FACTOR; 4 = BRIGHTNESS TEMPERATURE; 5 = RADIANCE AND RADIANCE FACTOR
    "Y_ERROR_FLAG":2, #0 = NONE; 1 = ONE VALUE; 2 = PER PIXEL
    }



def checkFlags(flagsDict):
    """check if combination of flags is correct"""
    calibrationType = ""
    if flagsDict["Y_UNIT_FLAG"] == 0 and flagsDict["Y_TYPE_FLAG"] == 0: #no calibration
        calibrationType = "None"
    elif flagsDict["Y_UNIT_FLAG"] == 1 and flagsDict["Y_TYPE_FLAG"] == 3: #radiance factor
        calibrationType = "Radiance Factor"
    elif flagsDict["Y_UNIT_FLAG"] == 2 and flagsDict["Y_TYPE_FLAG"] == 1: #radiance in W/cm2/sr/cm-1
        calibrationType = "Radiance"
    elif flagsDict["Y_UNIT_FLAG"] == 3 and flagsDict["Y_TYPE_FLAG"] == 4: #brightness temperature in K
        calibrationType = "Brightness Temperature"
    elif flagsDict["Y_UNIT_FLAG"] == 4 and flagsDict["Y_TYPE_FLAG"] == 5: #radiance in W/cm2/sr/cm-1 and radiance factor together in file
        calibrationType = "Radiance & Radiance Factor"

    errorType = ""
    if flagsDict["Y_ERROR_FLAG"] == 0:
        errorType = "None"
    elif flagsDict["Y_ERROR_FLAG"] == 1:
        errorType = "One Value"
    if flagsDict["Y_ERROR_FLAG"] == 2:
        errorType = "Per Pixel"

    if calibrationType == "":
        logger.error("Error: Calibration type unknown for Y_UNIT_FLAG = %i and Y_TYPE_FLAG = %i", flagsDict["Y_UNIT_FLAG"], flagsDict["Y_TYPE_FLAG"])
    elif errorType == "":
        logger.error("Error: Error type unknown for Y_ERROR_FLAG = %i", flagsDict["Y_ERROR_FLAG"])

    return calibrationType, errorType



def prepare_nadir_fig_tree(figName):
    
    channel=figName.split('_')[3]
    
    # Move to config
    PATH_TRANS_LINREG_FIG = os.path.join(ROOT_STORAGE_PATH, "thumbnails_1p0a_radfac", channel)  
    
    m = re.match("(\d{4})(\d{2})(\d{2}).*", figName)
    year = m.group(1)
    month = m.group(2)
#    path_fig = os.path.join(PATH_TRANS_LINREG_FIG)
    path_fig = os.path.join(PATH_TRANS_LINREG_FIG, year, month)
    if not os.path.isdir(path_fig):
            os.makedirs(path_fig, exist_ok=True)
    return os.path.join(path_fig, figName)




def runningMean(detector_data, n_spectra_to_mean):
    """make a running mean of n data points. detector data output has same length as input"""
    nSpectra = detector_data.shape[0]
    running_mean_data = np.zeros_like(detector_data)#[0:(-1*(n_spectra_to_mean-1)), :])
    
    runningIndicesCentre = [np.asarray(range(startingIndex, startingIndex+n_spectra_to_mean)) for startingIndex in range(0, (nSpectra-n_spectra_to_mean)+1)]
    runningIndicesStart = [np.asarray([0] * index + list(range(0, 10 - index))) for index in range(5, 0, -1)]
    runningIndicesEnd = [np.asarray(list(range(nSpectra - n_spectra_to_mean + index, nSpectra)) + [nSpectra - 1] * index) for index in range(1,5)]
    runningIndices = runningIndicesStart + runningIndicesCentre + runningIndicesEnd
    for rowIndex,indices in enumerate(runningIndices):
        running_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return running_mean_data






def doFullscanRadiometricCalibration(hdf5file_path):
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





def doRadiometricCalibration(hdf5file_path):
    """Radiometrically calibrate LNO. Not for fullscans"""

    hdf5FileIn = h5py.File(hdf5file_path, "r")
    hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
    hdf5_basename = os.path.basename(hdf5file_path).split(".")[0]
    
    #check combination of flags is valid
    calibrationType, errorType = checkFlags(LNO_FLAGS_DICT)
    
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
        yFitted = np.polyval(np.polyfit(range(50),yBinnedNorm[0,0:50],2),range(50))
        yStd = np.std(yBinnedNorm[0,0:50] - yFitted)
        
        if RUNNING_MEAN:
            """apply running mean with n=10"""
            yIntermediate = runningMean(yBinnedNorm, 10)
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
                    Y[spectrumIndex,negativeIndices] = 0.0
            if negativesFound:
                logger.info("Warning: negative Y values found in file. Replaced by zeroes")
        
        nSpectra = Y.shape[0] #calculate size
        
        """correct spectral shift"""
        #get temperature
        #TODO: replace by temperature db
        sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
        observationTemperature = np.mean(sensor1Temperature[2:10])
        
        #get sun-mars distance
        sun_mars_distance = hdf5FileIn["Geometry/DistToSun"][0,0] #in AU. Take first value in file only
        
        
        """recalculate x data"""
        xIn = nu_mp(diffractionOrder, np.arange(320.0), observationTemperature)
    
        
        """calibrate"""
        #read in data from channel calibration table - do radiance calibration
        logger.info("Opening radiometric calibration file %s.h5 for reading" %LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME)
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
            logger.info("hdf5Group=%s" %hdf5Group)
            
            calibrationDiffractionOrders = calibrationFile[hdf5Group]["DiffractionOrder"][:]
            diffractionOrderIndex = np.abs(calibrationDiffractionOrders - diffractionOrder).argmin()
            
            CountsPerRadianceFit = calibrationFile[hdf5Group]["CountsPerRadianceFit"][diffractionOrderIndex, :]
            CountsPerRadianceAtWavenumberFit = calibrationFile[hdf5Group]["CountsPerRadianceAtWavenumberFit"][diffractionOrderIndex, :]
        
            tableCreationDatetime = calibrationFile.attrs["DateCreated"]
        
        
        if calibrationType == "None":
            logger.warning("No calibration applied")
            XOut = xIn
            YOut = Y
            SNR = np.zeros_like(YOut) + np.float(NA_VALUE)

        elif calibrationType == "Radiance":
            logger.info("Radiance calibration")
            XOut = xIn
            YOut = Y / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
            YOutAtWavenumber = Y / np.tile(CountsPerRadianceAtWavenumberFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
        
            if errorType == "Per Pixel":
                YError = yStd / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
                SNR = YOut / YError
        
            elif errorType == "One Value":
                logger.info("One value error calculation not yet implemented")
                YError = np.zeros_like(YOut) + np.float(NA_VALUE)
                SNR = np.zeros_like(YOut) + np.float(NA_VALUE)
            else:
                logger.info("No error calculation selected")
                YError = np.zeros_like(YOut) + np.float(NA_VALUE)
                SNR = np.zeros_like(YOut) + np.float(NA_VALUE)
        
        
        
        elif calibrationType == "Radiance Factor":
            logger.error("Radiance factor calibration not yet implemented")
            XOut = xIn
            YOut = Y
            YOutAtWavenumber = Y
            YError = np.zeros_like(YOut) + np.float(NA_VALUE)
            SNR = np.zeros_like(YOut) + np.float(NA_VALUE)
            return []
        
        elif calibrationType == "Radiance & Radiance Factor":
            logger.info("Radiance & radiance factor calibration")
            
            """radiance calibration"""
            YOut = Y / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
            YOutAtWavenumber = Y / np.tile(CountsPerRadianceAtWavenumberFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
        
            if errorType == "Per Pixel":
                YError = yStd / np.tile(CountsPerRadianceFit, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
                SNR = YOut / YError
        
            elif errorType == "One Value":
                logger.info("One value error calculation not yet implemented")
                YError = np.zeros_like(YOut) + np.float(NA_VALUE)
                SNR = np.zeros_like(YOut) + np.float(NA_VALUE)
            else:
                logger.info("No error calculation selected")
                YError = np.zeros_like(YOut) + np.float(NA_VALUE)
                SNR = np.zeros_like(YOut) + np.float(NA_VALUE)
        

            
            #TODO: add conversion factor to account for solar incidence angle
            #TODO: this needs checking. No nadir or so FOV in calculation!
            rSun = 695510.0 # radius of Sun in km
            dSun = sun_mars_distance * 1.496e+8 #1AU to km
            angleSolar = np.pi * (rSun / dSun) **2 / 2.0 #why /2.0?
            
            """first - check if diffraction order is in the dictionary"""
            error = False
            if diffractionOrder not in NADIR_DICT.keys():
                error = True
                logger.warning("Diffraction order %i not in dictionary", diffractionOrder)
            
            if not error:
                """then check reference spectra, get position of minima"""
                #get solar/atmospheric absorption reference spectra
                fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
                ax1a.grid(True)
                ax1b.grid(True)
                nu_hr, normalised_reference_spectrum, true_wavenumber_minima = getReferenceSpectra(diffractionOrder, ax1b)

           
                #now check nadir data
                error = False
                observation_wavenumber_minima, validIndices, chi_sq_fit_all, incidence_angle = checkNadirSpectra(xIn, yBinnedNorm, diffractionOrder, hdf5FileIn, ax1a, ax1b)
                if chi_sq_fit_all[0] == 0:
                    error = True
    
                if error:
                    figName=hdf5_basename + "_raw_error.png"
                    fig1.suptitle("ERROR: %s (incidence angle = %0.1f)" %(hdf5_basename, incidence_angle))
                else:
                    figName=hdf5_basename + "_raw.png"
                    fig1.suptitle("%s (incidence angle = %0.1f)" %(hdf5_basename, incidence_angle))
                fig_dest_path = prepare_nadir_fig_tree(figName)
                plt.savefig(fig_dest_path, bbox_inches='tight') 
                plt.close()


            if not error:            
                #correct spectral shift
                observation_wavenumbers, chi_sq_fit_matching = correctSpectralShift(xIn, observation_wavenumber_minima, true_wavenumber_minima, chi_sq_fit_all)

                
                #make solar fullscan data
                corrected_solar_spectrum = getCorrectedSolarSpectrum(diffractionOrder, observation_wavenumbers, observationTemperature)

                #do I/F using shifted observation wavenumber scale
                YRadFac = Y / np.tile(corrected_solar_spectrum, [nSpectra, 1]) / angleSolar

                error = False
                if len(chi_sq_fit_matching) == 0:
                    error = True
                    logger.warning("No matching absorptions found in reference and nadir spectra")
                else:
                    Criteria = chi_sq_fit_matching
                logger.info("Chi squared fit matches %i points, error = %0.3e" %(len(chi_sq_fit_matching), np.mean(chi_sq_fit_matching)))


            if not error:            
                mean_radfac = np.mean(YRadFac[validIndices, :], axis=0)
                molecule = NADIR_DICT[diffractionOrder][4]
            
            
                plt.figure(figsize=(FIG_X, FIG_Y))
                plt.plot(observation_wavenumbers, mean_radfac, "k")
                
                if molecule == "Solar":
                    plt.plot(nu_hr, normalised_reference_spectrum * np.mean(mean_radfac), "b--")
                else:
                    plt.plot(nu_hr, normalised_reference_spectrum * np.mean(mean_radfac), "c--")

                #plt.plot(convolved_solar_wavenumbers, normalised_solar_spectrum * np.mean(mean_radfac), "g--")
                plt.xlabel("Wavenumbers cm-1")
                plt.ylabel("Radiance factor")
                plt.title("Radiance factor after spectral correction %s" %hdf5_basename)
                plt.grid()
                plt.text(np.min(nu_hr)+10.0, np.mean(mean_radfac)*0.75, "chi sq fit (%i points) error = %.3e" %(len(chi_sq_fit_matching), np.mean(chi_sq_fit_matching)))
            
                figName=hdf5_basename + "_radfac.png"
                fig_dest_path = prepare_nadir_fig_tree(figName)
                plt.savefig(fig_dest_path, bbox_inches='tight') 
                plt.close()
                
                XOut = np.tile(observation_wavenumbers, [nSpectra, 1])

           

    if not error:  #write out file          
        YCalibRef = "%s Calibration, CalibrationTime=%s; File=%s" %(calibrationType, calibrationTime, tableCreationDatetime)
        YErrorRef = "%s Error Type, CalibrationTime=%s; File=%s" %(errorType, calibrationTime, tableCreationDatetime)
        logger.info("%s" %YCalibRef)
        
        YTypeFlagOld = hdf5FileIn["Science/YTypeFlag"]
        YTypeFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_TYPE_FLAG"]
        YUnitFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_UNIT_FLAG"]
        YErrorFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_ERROR_FLAG"]
                
        if SAVE_FILES:
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
    
                hdf5FileOut.create_dataset("Science/X", dtype=np.float32, data=XOut, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.int16, data=YTypeFlag, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.int16, data=YUnitFlag, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.int16, data=YErrorFlag, compression="gzip", shuffle=True)
    
                hdf5FileOut.create_dataset("Science/YNormCounts", dtype=np.float32, data=Y, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YUnmodified", dtype=np.float32, data=yIn, compression="gzip", shuffle=True)
                
                if calibrationType == "Radiance & Radiance Factor":
                    hdf5FileOut.create_dataset("Science/Y", dtype=np.float32, data=YRadFac, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YRadiance", dtype=np.float32, data=YOut, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YRadianceAtWavenumber", dtype=np.float32, data=YOutAtWavenumber, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Criteria/LineFit/NumberOfLinesFit", dtype=np.int16, data=len(Criteria))
                    hdf5FileOut.create_dataset("Criteria/LineFit/ChiSqError", dtype=np.float32, data=Criteria, compression="gzip", shuffle=True)
    
                elif calibrationType == "Radiance":
                    hdf5FileOut.create_dataset("Science/Y", dtype=np.float32, data=YOut, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YAtWavenumber", dtype=np.float32, data=YOutAtWavenumber, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YError", dtype=np.float32, data=YError, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/SNR", dtype=np.float32, data=SNR, compression="gzip", shuffle=True)

    hdf5FileIn.close()
    
    if error:
        logger.error("Error found: Not saving LNO file")
        return []
    else:
        return hdf5FilepathOut



def convert(hdf5file_path):
    logger.info("convert: %s", hdf5file_path)


    with h5py.File(hdf5file_path, "r") as hdf5FileIn:
        if not TESTING:
            observationType = generics.getObservationType(hdf5FileIn)
            channel, channelType = generics.getChannelType(hdf5FileIn)
        else:
            observationType = "D"
            channel = "lno"

    if channel == "uvis":
        logger.info("UVIS file: skipping (%s)", hdf5file_path)
        return []

    if observationType is None:
        return []

    #if science measurement, do spectral calibration
    #E and I do not pass here
    if observationType in ["S"]:
        """TRANSMITTANCE ESTIMATION"""
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
        return trans.doTransmittanceCalibrationForS(hdf5file_path, hdf5FilepathOut)

    elif observationType in ["D"]:#,"N","L"]:
        logger.info("Radiometrically calibrating LNO normal science observation")
        logger.info("Y_UNIT_FLAG=%s, Y_TYPE_FLAG=%s, Y_ERROR_FLAG=%s", LNO_FLAGS_DICT["Y_UNIT_FLAG"],LNO_FLAGS_DICT["Y_TYPE_FLAG"],LNO_FLAGS_DICT["Y_ERROR_FLAG"])


        """RADIANCE CALIBRATION"""
        hdf5FilepathOut = doRadiometricCalibration(hdf5file_path)
        
        if len(hdf5FilepathOut) == 0:
            return []



    elif observationType in ["F"]:
        logger.info("Radiometrically calibrating LNO fullscan observation. Not yet implemented")
        logger.info("Y_UNIT_FLAG=%s, Y_TYPE_FLAG=%s, Y_ERROR_FLAG=%s", LNO_FLAGS_DICT["Y_UNIT_FLAG"],LNO_FLAGS_DICT["Y_TYPE_FLAG"],LNO_FLAGS_DICT["Y_ERROR_FLAG"])
#        """RADIANCE CALIBRATION"""
#        hdf5FilepathOut = doFullscanRadiometricCalibration(hdf5file_path)
        return []


    else:
        logger.info("File %s of type %s not yet calibrated by pipeline - copying uncalibrated datasets to new file", hdf5file_path, observationType)
        hdf5FileIn = h5py.File(hdf5file_path, "r")
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
        hdf5FileOut = h5py.File(hdf5FilepathOut, "w")
        generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION)
        for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            hdf5FileIn.copy(dset_path, dest)
        hdf5FileOut.close()
        hdf5FileIn.close()
    return [hdf5FilepathOut]



if TESTING:

    
    """this part gets observation files for testing"""

#    diffractionOrder = 118
#    diffractionOrder = 120
#    diffractionOrder = 126
#    diffractionOrder = 130
#    diffractionOrder = 133
#    diffractionOrder = 142
#    diffractionOrder = 151
#    diffractionOrder = 156


#    diffractionOrder = 162
#    diffractionOrder = 163
#    diffractionOrder = 167
#    diffractionOrder = 168
#    diffractionOrder = 169
    diffractionOrder = 189
#    diffractionOrder = 194
#    diffractionOrder = 196
    
    
    from database_functions_v01 import obsDB, makeObsDict
    dbName = "lno_0p3a"
    db_obj = obsDB(dbName)
    #CURIOSITY = -4.5895, 137.4417
    
    
    #min_lat, max_lat, min_lon, max_lon, max_incidence_angle, min_temperature, max_temperature, max_orders
    obsSearchDict = {
            118:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            120:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            126:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            130:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            133:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            142:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            151:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            156:[-90, 90, -180, 180, 90, -30, 30, 4], #none


            162:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
            163:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
            164:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            166:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            167:[-15, 5, 127, 147, 90, -30, 30, 4], #good
            168:[-15, 5, 127, 147, 90, -30, 30, 4], #good
            169:[-15, 5, 127, 147, 90, -30, 30, 4], #good

            173:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            174:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            178:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            179:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            180:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            182:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            184:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            189:[-15, 5, 127, 147, 90, -30, 30, 4], #good
            194:[-90, 90, 127, 147, 90, -30, 30, 4], #good
            195:[-90, 90, -180, 180, 90, -30, 30, 4], #none
            196:[-90, 90, 127, 147, 90, -30, 30, 4], #could be improved
            }


#    bestOrderMolecules = {118:"CO2", 120:"CO2", 126:"CO2", 130:"CO2", 133:"H2O", 142:"CO2", 151:"CO2", 156:"CO2", 160:"CO2", 162:"CO2", \
#                          163:"CO2", 164:"CO2", 166:"CO2", 167:"H2O", 168:"H2O", 169:"H2O", 173:"H2O", 174:"H2O", 178:"H2O", 179:"H2O", \
#                          180:"H2O", 182:"H2O", 184:"CO", 189:"CO", 194:"CO", 195:"CO", 196:"CO"}

    
    if diffractionOrder in obsSearchDict.keys():
        min_lat, max_lat, min_lon, max_lon, max_incidence_angle, min_temperature, max_temperature, max_orders = obsSearchDict[diffractionOrder]
    else:
        print("Error: diffraction order %i not in dictionary" %diffractionOrder)
    
    
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < %i AND latitude > %i AND longitude < %i AND longitude > %i AND n_orders < %i AND incidence_angle < %i AND temperature > %i AND temperature < %i AND diffraction_order == %i" \
                                     %(max_lat, min_lat, max_lon, min_lon, max_orders, max_incidence_angle, min_temperature, max_temperature, diffractionOrder))
    
    obsDict = makeObsDict("lno", searchQueryOutput)
    db_obj.close()
    #plt.figure()
    #plt.scatter(obsDict["longitude"], obsDict["latitude"])
    
    n_files = len(set(obsDict["filename"]))
    print("%i LNO files found" %n_files)
    
    #load spiceypy kernels
    BASE_DIRECTORY = ROOT_STORAGE_PATH
    KERNEL_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Documents", "DATA", "local_spice_kernels", "kernels", "mk")
    METAKERNEL_NAME = "em16_plan.tm"
    print("KERNEL_DIRECTORY=%s, METAKERNEL_NAME=%s" %(KERNEL_DIRECTORY, METAKERNEL_NAME))
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)



    for hdf5file_path in set(obsDict["filepath"]):
        convert(hdf5file_path)
