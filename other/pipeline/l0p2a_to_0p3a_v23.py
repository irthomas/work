# -*- coding: utf-8 -*-

#TESTING=True
TESTING=False



import logging
import os.path

import h5py
import numpy as np
#import spiceypy as sp
from datetime import datetime, timedelta


if not TESTING:
    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
    import nomad_ops.core.hdf5.generic_functions as generics
    from nomad_ops.core.pipeline.extras.heaters_temp import get_temperature_range
    
    
__project__   = "NOMAD"
__author__    = "Ian Thomas & Roland Clairquin"
__contact__   = "roland.clairquin@oma.be"


#============================================================================================
# 5. CONVERT HDF5 LEVEL 0.2A TO LEVEL 0.3A
# SPECTRAL CALIBRATION OF SO AND LNO CHANNELS
#============================================================================================

logger = logging.getLogger( __name__ )

VERSION = 80
OUTPUT_VERSION = "0.3A"

"""set constants"""
NA_VALUE = -999

"""if above this temperature, set quality flag to indicate science could be impacted"""
MAXIMUM_LNO_SCIENCE_TEMPERATURE = 10.0

"""does counting start from 0 or 1?"""
FIRST_PIXEL = 0 #everything but matlab


"""add Goddard AOTF offset?"""
GODDARD_OFFSET = False

"""set paths to calibration files"""
LNO_SPECTRAL_CALIBRATION_TABLE_NAME = "LNO_Spectral_Calibration_Table_v07"
SO_SPECTRAL_CALIBRATION_TABLE_NAME = "SO_Spectral_Calibration_Table_v07"

SPECTRAL_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "spectral_calibration")



"""specify datasets/attributes to be modified/removed"""
DATASETS_TO_BE_REMOVED = set([
    "Science/X",
    "Channel/Pixel1",
    "Science/XUnitFlag",
    "Name"
    ])

ATTRIBUTES_TO_BE_REMOVED = ["XCalibRef"]

#remove Name, DateTime, Timestamp and replace DateTime with Geometry/ObservationDateTime and Timestamp with Geometry/ObservationEphemerisTime
DATASETS_TO_BE_REPLACED = {"DateTime":"Geometry/ObservationDateTime", "Timestamp":"Geometry/ObservationEphemerisTime"}

SO_FLAGS_DICT = {
"X_UNIT_FLAG":1,
"ILS_FLAG":3,
#"AOTF_FUNCTION_FLAG":0, #don't set as 5 for now, LNO coefficients only included. 
"AOTF_FUNCTION_FLAG":6, #don't set as 5 for now, LNO coefficients only included. 
"BLAZE_FUNCTION_FLAG":3,
"AOTF_FUNCTION_X_RANGE_START":-200.0,
"AOTF_FUNCTION_X_RANGE_STOP":200.0,
"AOTF_FUNCTION_X_RANGE_STEP":0.1,
}



LNO_FLAGS_DICT = {
"X_UNIT_FLAG":1,
"ILS_FLAG":3,
"AOTF_FUNCTION_FLAG":5, #AOTF function can't be 0 for LNO (no Goddard parameters yet). At present, AOTFFunction should be renamed AOTFBandwidth.
"BLAZE_FUNCTION_FLAG":3,
"AOTF_FUNCTION_X_RANGE_START":-200.0,
"AOTF_FUNCTION_X_RANGE_STOP":200.0,
"AOTF_FUNCTION_X_RANGE_STEP":0.1,
}


FORMAT_STR_SECONDS = "%Y %b %d %H:%M:%S.%f"



#get TGO temperature readouts for this observation
def get_tgo_readouts(beg_datetimestring, end_datetimestring, delta_minutes=10.0):
#    beg_datetimestring = "2018 Mar 26 21:44:31.879169"
#    end_datetimestring = "2018 Mar 26 22:44:31.879169"
    beg_dt = datetime.strptime(beg_datetimestring.decode(), FORMAT_STR_SECONDS) - timedelta(minutes=delta_minutes)
    end_dt = datetime.strptime(end_datetimestring.decode(), FORMAT_STR_SECONDS) + timedelta(minutes=delta_minutes)
#    headers = ["datetime", "so_nominal", "lno_nominal", "so_redundant", "lno_redundant", "uvis_nominal"]
#    return temperature_data
    temperature_db_data = get_temperature_range(beg_dt, end_dt)
    
    datetimestring = []
    so_nom = []
    lno_nom = []
    so_red = []
    lno_red = []
    uvis_nom = []
    for temperature_db_row in temperature_db_data:
        datetimestring.append(datetime.strftime(temperature_db_row[0], FORMAT_STR_SECONDS).encode())
        so_nom.append(temperature_db_row[1])
        lno_nom.append(temperature_db_row[2])
        so_red.append(temperature_db_row[3])
        lno_red.append(temperature_db_row[4])
        uvis_nom.append(temperature_db_row[5])
    return {"TemperatureDateTime":datetimestring, 
            "NominalSO":np.asfarray(so_nom), 
            "NominalLNO":np.asfarray(lno_nom), 
            "RedundantSO":np.asfarray(so_red), 
            "RedundantLNO":np.asfarray(lno_red),
            "NominalUVIS":np.asfarray(uvis_nom)}


def writeTemperatureQualityFlag(hdf5_file_out, channel, measurement_temperature):
    """write high temperature warning quality flag to file"""
    if channel == "lno" and measurement_temperature > MAXIMUM_LNO_SCIENCE_TEMPERATURE:
        high_temperature = True
    else:
        high_temperature = False

    if high_temperature:
        hdf5_file_out.create_dataset("QualityFlag/HighInstrumentTemperature", dtype=np.int, data=1)
    else:
        hdf5_file_out.create_dataset("QualityFlag/HighInstrumentTemperature", dtype=np.int, data=0)
    return



def getBlazeFunction(flagsDict, coefficientDict, aotfFrequency):
    
#    if flagsDict["BLAZE_FUNCTION_FLAG"] == 0:
#        
#        """calculate using X_RANGE grid in pixels"""
#        #get p0 (central pixel of order)
#        p0 = np.polyval(orderToBlazeFunctionFSRCentreCoefficients[3:6],diffractionOrder)
#        #get FSR (in pixels)
#        wpPixels = np.polyval(orderToBlazeFunctionFSRCentreCoefficients[0:3],diffractionOrder)
#        #calculate blaze function in pixels
#        blazeFunction = func_blaze(BLAZE_FUNCTION_X_RANGE, p0, wpPixels)

#        """calculate using range of x values covering central diffraction order"""
#        p0 = np.polyval(orderToBlazeFunctionP0Coefficients,diffractionOrder)
#        wp = pixelOrderToWavenumberCoefficients[2]
#        x = np.polyval(pixelOrderToWavenumberCoefficients,pixelValues) * diffractionOrder
#        xStep = BLAZE_FUNCTION_X_RANGE[1]-BLAZE_FUNCTION_X_RANGE[0]
#        xGrid = np.arange(np.min(x),np.max(x),xStep)
#        blazeFunction = func_blaze(xGrid+p0, p0, wp)
#
#        """calculate using pixel range"""
#        p0 = np.polyval(orderToBlazeFunctionP0Coefficients,diffractionOrder)
#        xWavenumbers = np.polyval(pixelOrderToWavenumberCoefficients,pixelValues) * diffractionOrder
#        wp = pixelOrderToWavenumberCoefficients[2] / (xWavenumbers[160]-xWavenumbers[159])
#        x = pixelValues
#        blazeFunction = func_blaze(x, p0, wp)
#        blazeFunction = np.insert(blazeFunction,[0,0,0],[BLAZE_FUNCTION_X_RANGE[0],BLAZE_FUNCTION_X_RANGE[-1],BLAZE_FUNCTION_X_RANGE[1]-BLAZE_FUNCTION_X_RANGE[0]])

#    elif flagsDict["BLAZE_FUNCTION_FLAG"] == 1:
#        blazeFunction = np.asfarray([blazeFunctionTheta,blazeFunctionGamma,blazeFunctionAlphaB,blazeFunctionSigma])

    if flagsDict["BLAZE_FUNCTION_FLAG"] == 3:
        #copy coefficients directly from table
        blazeFunction = coefficientDict["BlazeFunction"]
    
    return blazeFunction

  
def getAOTFFunction(flagsDict, coefficientDict, aotfFrequency):


#    def func_aotf(x, x0, i0, w, iG, sigmaG): #Goddard model 2017
#        """AOTF shape"""
#        x0 = x0+0.0001 #fudge to stop infinity at peak
#
#        fsinc = (i0 * w**2.0 * (np.sin(np.pi * (x - x0) / w))**2.0) / (np.pi**2.0 * (x - x0)**2.0)
#        fgauss = iG * np.exp(-1.0 * (x - x0)**2.0 / sigmaG**2.0)
#        f = fsinc + fgauss #slant not included
#        return f/np.max(f) #slant not included. normalised
#
#
#
    if flagsDict["AOTF_FUNCTION_FLAG"] == 0: #make AOTF shape
        
        """calculate aotf function from order and high resolution wavenumber grid covering all required orders"""
        c1 = coefficientDict["AOTFWnCoefficients"]
        c2 = coefficientDict["AOTFSincWidthCoefficients"]
        c3 = coefficientDict["AOTFSidelobeRatioCoefficients"]
        c4 = coefficientDict["AOTFOffsetCoefficients"]
        c5 = coefficientDict["AOTFCentreTemperatureShiftCoefficients"]
        t = coefficientDict["CalibrationTemperature"]
        aotfCentre = np.polyval(c1, aotfFrequency)
        aotfWidth = np.polyval(c2, aotfCentre)
        aotfSideLobe = np.polyval(c3, aotfCentre)
        aotfOffset = np.polyval(c4, aotfCentre)
        aotfShift = np.polyval(c5, aotfCentre*t)
        
        
        xStart = flagsDict["AOTF_FUNCTION_X_RANGE_START"]
        xStop = flagsDict["AOTF_FUNCTION_X_RANGE_STOP"]
        xStep = flagsDict["AOTF_FUNCTION_X_RANGE_STEP"]
        xRange = np.arange(xStart, xStop + xStep, xStep)
    
        aotfCentre += aotfShift
        aotfFunction = np.sinc(xRange / aotfWidth)**2
        aotfFunction[np.abs(xRange) >= aotfWidth] *= aotfSideLobe

#        print(aotfCentre)
#        print(aotfWidth)
#        print(aotfSideLobe)
    
        if GODDARD_OFFSET:
            """add aotf offset (straylight leak) and scale to renormalise peak to 1"""
            aotfFunction = aotfOffset + (1. - aotfOffset) * aotfFunction
#        else:
#            aotfFunction = aotfFunction / np.max(aotfFunction)
        
        aotfFunction = np.insert(aotfFunction,[0,0,0],[xRange[0],xRange[-1],xRange[1] - xRange[0]])
        aotfCentralWavenb = aotfCentre
#
#    elif flagsDict["AOTF_FUNCTION_FLAG"] == 2: #sinc2 (old version)
#        aotfFunction = aotfBandwidth
#
    elif flagsDict["AOTF_FUNCTION_FLAG"] == 5: #sinc2 + gaussian (goddard analysis) LNO only
        w = 18.188122
        sigmaG = 12.181137
        ig_i0 = 0.589821
        """Calculations"""
        ig = ig_i0 / (1.0 + ig_i0)
        i0 = 1.0 - ig
        aotfFunction = np.asfarray([i0, w, ig, sigmaG])

        c1 = coefficientDict["AOTFWnCoefficients"]
        aotfCentre = np.polyval(c1, aotfFrequency)
        aotfCentralWavenb = aotfCentre
        
    elif flagsDict["AOTF_FUNCTION_FLAG"] == 6: #calculate centre, read AOTF from file
        
        if aotfFrequency == 0: #if dark return NA
            return NA_VALUE, NA_VALUE

        c1 = coefficientDict["AOTFWnCoefficients"]
        c2 = coefficientDict["AOTFCentreTemperatureShiftCoefficients"]
        t = coefficientDict["CalibrationTemperature"]
        
        aotfCentre = np.polyval(c1, aotfFrequency)
        aotfShift = np.polyval(c2, aotfCentre*t)
        aotfFunction = aotfCentre + aotfShift #aotf centre in cm-1
        aotfCentralWavenb = aotfFunction
    
    return aotfFunction, aotfCentralWavenb


def getSpectralResolution(flagsDict, coefficientDict, aotfFrequency):
    """Spectral resolution"""
    
    if flagsDict["ILS_FLAG"] == 3:
        
        if aotfFrequency == 0: #if dark return NA
            return NA_VALUE

        c1 = coefficientDict["AOTFWnCoefficients"]
        c2 = coefficientDict["AOTFCentreTemperatureShiftCoefficients"]
        t = coefficientDict["CalibrationTemperature"]
        
        aotfCentre = np.polyval(c1, aotfFrequency)
#        print(f"aotfCentre = {aotfCentre}")
        aotfShift = np.polyval(c2, aotfCentre*t)
        aotfCentre += aotfShift #aotf centre in cm-1
#        print(f"aotfCentre = {aotfCentre}")

        c3 = coefficientDict["ResolvingPowerCoefficients"]
        resolvingPower = np.polyval(c3, aotfCentre)
        spectralResolution = aotfCentre / resolvingPower
    
    return spectralResolution


def getX(flagsDict, coefficientDict, aotfFrequency):

    if flagsDict["X_UNIT_FLAG"] == 1:

        if aotfFrequency == 0: #if dark return NA
            return np.zeros(320) + NA_VALUE, NA_VALUE
        
        #calculate pixel shift based on Goddard analysis and temperature sensor 1.
        c0 = coefficientDict["AOTFOrderCoefficients"]


#        c1 = coefficientDict["FirstPixelCoefficients"]
    
        """use new coefficients"""
        Q0=-10.13785778
        Q1=-0.829174444
        Q2=0.0
    
        c1 = [Q2, Q1, Q0]
        
        c2 = coefficientDict["PixelSpectralCoefficients"]
        t = coefficientDict["CalibrationTemperature"]

        diffractionOrder = np.round(np.polyval(c0, aotfFrequency))
        firstPixelValue = np.polyval(c1, t)
        pixelValues = np.arange(FIRST_PIXEL, 320 + FIRST_PIXEL, 1) + firstPixelValue #apply temperature shift
        x = np.polyval(c2, pixelValues) * diffractionOrder
        
#        else:
#            """if dark, just return NA_VALUES"""
#            x = np.zeros(320) + NA_VALUE
    #        aotfFunction = np.zeros_like(aotfFunction) + NA_VALUE
    #        blazeFunction = np.zeros_like(blazeFunction) + NA_VALUE
    #        spectralResolution = NA_VALUE
    #    return(aotfFunction,blazeFunction,spectralResolution,x)

    return x, firstPixelValue


def getCoefficients(hdf5FileIn, channel, measurementTemperature):
    
    #read in data from channel calibration table
    if channel=="so":
        spectral_calibration_table = os.path.join(SPECTRAL_CALIBRATION_AUXILIARY_FILES, SO_SPECTRAL_CALIBRATION_TABLE_NAME)
        calibrationFile = h5py.File("%s.h5" % spectral_calibration_table, "r")
#        logger.info("Opening spectral calibration file %s.h5 for reading", spectral_calibration_table)
#        sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_SO"][...]
    elif channel=="lno":
        spectral_calibration_table = os.path.join(SPECTRAL_CALIBRATION_AUXILIARY_FILES, LNO_SPECTRAL_CALIBRATION_TABLE_NAME)
        calibrationFile = h5py.File("%s.h5" % spectral_calibration_table, "r")
#        logger.info("Opening spectral calibration file %s.h5 for reading", spectral_calibration_table)
#        sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]

    #get instrument temperature from aotf temperature measurements (ignore first 2 values - usually wrong)
#    measurementTemperature = np.mean(sensor1Temperature[2:10])    

    #read in observation time
    if "Geometry/ObservationDateTime" in hdf5FileIn.keys():
        observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]
    else:
        observationStartTime = hdf5FileIn["DateTime"][0]

    
    #convert times to numerical timestamps
    calibrationTimes = list(calibrationFile.keys())
    calibrationTimestamps = np.asfarray([datetime.timestamp(datetime.strptime(calibrationTime, FORMAT_STR_SECONDS)) for calibrationTime in calibrationTimes])
    measurementTimestamp = datetime.timestamp(datetime.strptime(observationStartTime.decode(), FORMAT_STR_SECONDS))

    #find which observation corresponds to calibration measurement time
    timeIndex = np.max(np.where(measurementTimestamp>calibrationTimestamps))
    hdf5Group = calibrationTimes[timeIndex]
    #print("hdf5Group=%s" %hdf5Group)

    #read in correct coefficients
    fileKeys = calibrationFile[hdf5Group].keys()
    
    coefficientDict = {}
    for key in fileKeys:
        coefficientDict[key] = calibrationFile[hdf5Group][key][...]

    coefficientDict["CalibrationFileCreated"] = calibrationFile.attrs["DateCreated"]
    coefficientDict["CalibrationTime"] = hdf5Group
    coefficientDict["CalibrationTemperature"] = measurementTemperature

    calibrationFile.close()
    
    return coefficientDict


def getDiffractionOrders(coefficientDict, aotfFrequencies):
    """get orders from aotf"""
    diffractionOrdersCalculated = [np.int(np.round(np.polyval(coefficientDict["AOTFOrderCoefficients"], aotfFrequency))) for aotfFrequency in aotfFrequencies]
    #set darks to zero
    diffractionOrders = np.asfarray([diffractionOrder if diffractionOrder > 50 else 0 for diffractionOrder in diffractionOrdersCalculated])
    return diffractionOrders




def convert(hdf5file_path):
#    logger.info("convert: %s", hdf5file_path)

#    hdf5_basename = os.path.basename(hdf5file_path).split(".")[0]
    hdf5FileIn = h5py.File(hdf5file_path, "r")
    hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))

    channel, channelType = generics.getChannelType(hdf5FileIn)
    if channel == "uvis":
        logger.info("UVIS file: skipping (%s)", hdf5file_path)
        return []

    #read in observation start and end times
    if "Geometry/ObservationDateTime" in hdf5FileIn.keys():
        observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]
        observationEndTime = hdf5FileIn["Geometry/ObservationDateTime"][-1,-1]
    else:
        observationStartTime = hdf5FileIn["DateTime"][0]
        observationEndTime = hdf5FileIn["DateTime"][-1]

    error = False
    temperatureDictionary = get_tgo_readouts(observationStartTime, observationEndTime)
    temperatureData = temperatureDictionary["Nominal%s" %channel.upper()]
    if len(temperatureData)>0:
        measurementTemperature = np.mean(temperatureData)
    else:
        logger.error("No TGO temperatures available for %s. Skipping file generation", hdf5file_path)
        measurementTemperature = NA_VALUE
        error = True

    if error:
        return []
    
    """function to find temperature- and time- dependent spectral calibration coefficients, do raw value calculations, and write to file"""
    flagsDict = {"so":SO_FLAGS_DICT, "lno":LNO_FLAGS_DICT}[channel]
    coefficientDict = getCoefficients(hdf5FileIn, channel, measurementTemperature)

#    logger.info("AOTF_FUNCTION_FLAG=%s, BLAZE_FUNCTION_FLAG=%s", flagsDict["AOTF_FUNCTION_FLAG"], flagsDict["BLAZE_FUNCTION_FLAG"])

    XCalibRef = "CalibrationTime=%s; CalibrationTemperature=%0.1f; CalibrationFileCreated=%s" \
                % (coefficientDict["CalibrationTime"], coefficientDict["CalibrationTemperature"], coefficientDict["CalibrationFileCreated"])
#    logger.info("Spectral calibration XCalibRef: %s" % XCalibRef)



    aotfFrequencies = hdf5FileIn["Channel/AOTFFrequency"][...]
    #check to catch if diffraction order in file (not true for non-standard science measurements e.g. fullscans)
    if "Channel/DiffractionOrder" in hdf5FileIn.keys():
        diffractionOrdersFound=True
        diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...]
    else:
        diffractionOrdersFound=False
        diffractionOrders = getDiffractionOrders(coefficientDict, aotfFrequencies)
        

    #check if all AOTF freqs same
    if (aotfFrequencies == aotfFrequencies[0]).all():
        aotfFrequenciesSame = True
        logger.info("All AOTF frequences are the same. Performing simple spectral calibration")
    else:
        aotfFrequenciesSame = False
        

    ydimensions = hdf5FileIn["Science/Y"].shape
    nSpectra = ydimensions[0]



    if aotfFrequenciesSame:
        aotfFrequency = aotfFrequencies[0]

        aotfFunction, aotfCentralWavenb = getAOTFFunction(flagsDict, coefficientDict, aotfFrequency)
        blazeFunction = getBlazeFunction(flagsDict, coefficientDict, aotfFrequency)
        spectralResolution = getSpectralResolution(flagsDict, coefficientDict, aotfFrequency)
        x, firstPixel = getX(flagsDict, coefficientDict, aotfFrequency)


        aotfCentralWavenbTable = np.tile(aotfCentralWavenb, (nSpectra))
        aotfFunctionTable = np.tile(aotfFunction, (nSpectra, 1))
        blazeFunctionTable = np.tile(blazeFunction, (nSpectra, 1))
        spectralResolutionTable = np.tile(spectralResolution, (nSpectra))
        firstPixelTable = np.tile(firstPixel, (nSpectra))
        xTable = np.tile(x, (nSpectra, 1))

    else:
        """for fullscans, run through each spectrum"""
        """make list then convert to np array"""
        logger.info("AOTF frequences are not the same. Calibrating each line separately")
        aotfCentralWavenbTable = []
        aotfFunctionTable = []
        blazeFunctionTable = []
        spectralResolutionTable = []
        xTable = []
        firstPixelTable = []

        for spectrumIndex,aotfFrequency in enumerate(aotfFrequencies):
            aotfFunction, aotfCentralWavenb = getAOTFFunction(flagsDict, coefficientDict, aotfFrequency)
            blazeFunction = getBlazeFunction(flagsDict, coefficientDict, aotfFrequency)
            spectralResolution = getSpectralResolution(flagsDict, coefficientDict, aotfFrequency)
            x, firstPixel = getX(flagsDict, coefficientDict, aotfFrequency)

            aotfCentralWavenbTable.append(aotfCentralWavenb)
            aotfFunctionTable.append(aotfFunction)
            blazeFunctionTable.append(blazeFunction)
            spectralResolutionTable.append(spectralResolution)
            xTable.append(x)
            firstPixelTable.append(firstPixel)

        aotfCentralWavenbTable = np.asfarray(aotfCentralWavenbTable)
        aotfFunctionTable = np.asfarray(aotfFunctionTable)
        blazeFunctionTable = np.asfarray(blazeFunctionTable)
        spectralResolutionTable = np.asfarray(spectralResolutionTable)
        xTable = np.asfarray(xTable)
        firstPixelTable = np.asfarray(firstPixelTable)



    """make other coefficient tables for adding to file"""
    AOTFWnCoefficients = np.tile(coefficientDict["AOTFWnCoefficients"],(nSpectra,1))
    PixelSpectralCoefficients = np.tile(coefficientDict["PixelSpectralCoefficients"],(nSpectra,1))
    AOTFOrderCoefficients = np.tile(coefficientDict["AOTFOrderCoefficients"],(nSpectra,1))
    MeasurementTemperature = np.tile(coefficientDict["CalibrationTemperature"],(nSpectra,1))

    """make flags"""
    XUnitFlag = np.zeros((nSpectra)) + flagsDict["X_UNIT_FLAG"]
    ILSFlag = np.zeros((nSpectra)) + flagsDict["ILS_FLAG"]
    AOTFFunctionFlag = np.zeros((nSpectra)) + flagsDict["AOTF_FUNCTION_FLAG"]
    BlazeFunctionFlag = np.zeros((nSpectra)) + flagsDict["BLAZE_FUNCTION_FLAG"]




    """write to output file"""
    with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:

        generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
        
        #don't copy all datasets to new file
        for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            if dset_path in DATASETS_TO_BE_REMOVED: #don't copy
                continue
            elif dset_path in DATASETS_TO_BE_REPLACED.keys(): #don't copy
                replacement_dset_path = DATASETS_TO_BE_REPLACED[dset_path] #if found, get path to replacement data using dictionary
    
                dset_1 = hdf5FileIn[replacement_dset_path][...] #get data from replacement dataset
                if dset_path in ["DateTime", "Timestamp"]: #replace these datasets by observation date time and ephemeris time to avoid confusion
                    dsetcopy=np.array(dset_1[:, 0]) #just take starting column
                hdf5FileOut.create_dataset(dset_path, dtype=dset_1.dtype, data=dsetcopy, compression="gzip", shuffle=True) #write new data to original path
                continue
    
            dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            hdf5FileIn.copy(dset_path, dest)

        hdf5FileOut.attrs["XCalibRef"] = XCalibRef
        
    
        hdf5FileOut.create_dataset("Channel/FirstPixel", dtype=np.float,
                                data=firstPixelTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/X", dtype=np.float,
                                data=xTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/XUnitFlag", dtype=np.int,
                                data=XUnitFlag, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
    
        hdf5FileOut.create_dataset("Channel/MeasurementTemperature", dtype=np.float,
                                data=MeasurementTemperature, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/PixelSpectralCoefficients", dtype=np.float,
                                data=PixelSpectralCoefficients, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/AOTFWnCoefficients", dtype=np.float,
                                data=AOTFWnCoefficients, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/AOTFOrderCoefficients", dtype=np.float,
                                data=AOTFOrderCoefficients, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/SpectralResolution", dtype=np.float,
                                data=spectralResolutionTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/ILSFlag", dtype=np.int,
                                data=ILSFlag, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/AOTFFunction", dtype=np.float,
                                data=aotfFunctionTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/AOTFFunctionFlag", dtype=np.int,
                                data=AOTFFunctionFlag, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        #TODO: delete these
#        hdf5FileOut.create_dataset("Channel/AOTFBandwidth", dtype=np.float,
#                                data=aotfFunctionTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
#        hdf5FileOut.create_dataset("Channel/AOTFBandwidthFlag", dtype=np.int,
#                                data=AOTFFunctionFlag, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/BlazeFunction", dtype=np.float,
                                data=blazeFunctionTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/BlazeFunctionFlag", dtype=np.int,
                                data=BlazeFunctionFlag, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Channel/AOTFCentralWavenb", dtype=np.float,
                    data=aotfCentralWavenbTable, fillvalue=NA_VALUE, compression="gzip", shuffle=True)


    
        if not diffractionOrdersFound:
            hdf5FileOut.create_dataset("Channel/DiffractionOrder", dtype=np.int,
                                    data=diffractionOrders, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
    
    
        """while instrument temperature is being checked, also set relevant quality flag"""
        writeTemperatureQualityFlag(hdf5FileOut, channel, coefficientDict["CalibrationTemperature"])

        #add tgo temperatures to file
        for key, values in temperatureDictionary.items():
            if "DateTime" not in key:
                if len(values) > 0:
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype=np.float32,
                                               data=values, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                else: #if no data for given time period
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype=np.float32, data=-999.0)

            else:
                if len(values) > 0:
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype="S27",
                                               data=values, compression="gzip", shuffle=True)
                else: #if no data for given time period
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype="S27", data=["-999".encode()])



    hdf5FileIn.close()
    return [hdf5FilepathOut]

   

#if TESTING:

    #load spiceypy kernels if required
#    KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
#    METAKERNEL_NAME = "em16_ops.tm"
#    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
#    print(sp.tkvrsn("toolkit"))
#    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)

#    channel = "so"
#    convert(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\20180421_202111_0p2a_SO_2_E_190.h5"))
    
#convert(os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p2a/2018/05/21/20180521_034316_0p2a_SO_1_I_190.h5"))
