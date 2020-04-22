# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:02:58 2018

@author: iant

RADIOMETRIC CALIBRATION V03

APPLY MOST RECENT SPECTRAL CALIBRATIONS TO LNO BLACKBODY DATASETS

"""


import os
import h5py
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList, printFileNames

######LIST OF DIRECTORIES#######
if os.path.exists(os.path.normcase(r"X:\linux\Data")):
    PFM_AUXILIARY_FILES = r"X:\projects\NOMAD\data\pfm_auxiliary_files"
elif os.path.exists(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")):
    PFM_AUXILIARY_FILES = r"C:\Users\iant\Dropbox\NOMAD\Python\data\pfm_auxiliary_files"


"""set paths to calibration files"""
LNO_SPECTRAL_CALIBRATION_TABLE=os.path.join(PFM_AUXILIARY_FILES,"spectral_calibration","LNO_Spectral_Calibration_Table_v04")
SO_SPECTRAL_CALIBRATION_TABLE=os.path.join(PFM_AUXILIARY_FILES,"spectral_calibration","SO_Spectral_Calibration_Table_v04")
UVIS_SPECTRAL_CALIBRATION_TABLE=os.path.join(PFM_AUXILIARY_FILES,"spectral_calibration","UVIS_Spectral_Calibration_Table_v03")
LNO_FLAG_FILE = os.path.join(PFM_AUXILIARY_FILES,"spectral_calibration","LNO_Spectral_Calibration_Flags_v03")
SO_FLAG_FILE = os.path.join(PFM_AUXILIARY_FILES,"spectral_calibration","SO_Spectral_Calibration_Flags_v03")


LNO_RADIOMETRIC_CALIBRATION_TABLE=os.path.join(PFM_AUXILIARY_FILES,"radiometric_calibration","LNO_Radiometric_Calibration_Table_v02")

SOLAR_SPECTRUM = np.loadtxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_solar_spectrum_solspec.txt")
RADIANCE_TO_IRRADIANCE = 8.77e-5 / 100.0**2 #fudge to make curves match. should be 2.92e-5 on mars, 6.87e-5 on earth
"""theta = tan(diameter of sun / distance to sun) = tan(1.39e6 / 2.28e08km)
   solid angle = 2 * pi * (1-cos(theta/2))"""



SAVE_FIGS = False
#SAVE_FIGS = True

#SAVE_FILES = False
SAVE_FILES = True

NA_VALUE = -999


"""pfm ground/inflight cal"""
fileLevel = "hdf5_level_0p1a"
#obspaths = ["20150426_054602_0p1a_LNO_1","20150426_030851_0p1a_LNO_1","20150427_010422_0p1a_LNO_1", "20161121_233000_0p1a_LNO_1"] #150C BB + MCO1 sun
#obspaths = ["20150426_054602_0p1a_LNO_1","20150426_030851_0p1a_LNO_1","20161121_233000_0p1a_LNO_1"] #150C BB (cold only) + MCO1 sun
obspaths = ["20150426_054602_0p1a_LNO_1","20150426_030851_0p1a_LNO_1"] #150C BB (cold only)
#obspaths = ["20161121_233000_0p1a_LNO_1"] #MCO1 sun only
model = "PFM"
title = "PFM Ground Calibration"
channel = "lno"


"""flight spare bb calibration"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["20150904_231759_LNO_1","20150905_005006_LNO_1","20150905_013134_LNO_1","20150905_031732_LNO_1","20150905_040057_LNO_1","20150905_052528_LNO_1","20150906_034503_LNO_1",\
#   "20150907_190444_LNO_1","20150907_195100_LNO_1","20150907_212248_LNO_1","20150907_221000_LNO_1","20150907_234545_LNO_1","20150908_003506_LNO_1",\
#   "20150909_223132_LNO_1","20150909_232514_LNO_1","20150910_010414_LNO_1","20150910_015625_LNO_1","20150910_033745_LNO_1","20150910_043039_LNO_1"]
#model = "FS"
#title = "FS Ground Calibration"


"""inflight cal only"""
#obspaths = ["20161121_233000_0p1a_LNO_1"]
#fileLevel = "hdf5_level_0p1a"
#model = "flight"
#title = "PFM Inflight Calibration"







############FUNCTIONS#############
def closestIndex(array, value_to_find):
    return np.abs(array-value_to_find).argmin()



def normalise(array_in):
    """normalise input array"""
    return array_in/np.max(array_in)



def getInstTempFromFile(hdf5_file):
    """get instrument temperature from given file"""
    if isinstance(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"], h5py.Dataset):
        instrument_temp = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
    elif isinstance(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_SO"], h5py.Dataset):
        instrument_temp = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
    else:
        print("Error: Temperature HSK not found in file")
    return instrument_temp

def getBBTempFromFile(hdf5_filename, hdf5_file):
    GET_TEMP_FROM_FILE = False #OGSE data not included in new files
    
    if GET_TEMP_FROM_FILE:
        """get blackbody temperature from Target attribute in given file"""
        attr_value = get_hdf5_attribute(hdf5_file, "Target")
        if "lackbody" in str(attr_value):
    #        print(attr_value)
            bb_temp_string = str(attr_value).split("lackbody at ")[1].split("C")[0].replace("\\r","")
            try:
                bb_temp = float(bb_temp_string) + 273.0
            except ValueError:
                print("Warning: BB Temp is not a float")
        elif "Mars" in str(attr_value):
            bb_temp = 5796.0
    else:
        if "20150426_054602" in hdf5_filename:
            bb_temp = 423.0
        elif "20150426_030851" in hdf5_filename:
            bb_temp = 423.0
        elif "20150427_010422" in hdf5_filename:
            bb_temp = 423.0
        elif "20150904_231759" in hdf5_filename:
            bb_temp = 423.0
        elif "20150905_005006" in hdf5_filename:
            bb_temp = 423.0
        elif "20150905_013134" in hdf5_filename:
            bb_temp = 423.0
        elif "20150905_031732" in hdf5_filename:
            bb_temp = 423.0
        elif "20150905_040057" in hdf5_filename:
            bb_temp = 423.0 - 20.0
        elif "20150905_052528" in hdf5_filename:
            bb_temp = 423.0 - 40.0
        elif "20150906_034503" in hdf5_filename:
            bb_temp = 423.0
        elif "20150907_190444" in hdf5_filename:
            bb_temp = 423.0
        elif "20150907_195100" in hdf5_filename:
            bb_temp = 423.0 - 20.0
        elif "20150907_212248" in hdf5_filename:
            bb_temp = 423.0 - 40.0
        elif "20150907_221000" in hdf5_filename:
            bb_temp = 423.0
        elif "20150907_234545" in hdf5_filename:
            bb_temp = 423.0
        elif "20150908_003506" in hdf5_filename:
            bb_temp = 423.0
        elif "20150909_223132" in hdf5_filename:
            bb_temp = 423.0
        elif "20150909_232514" in hdf5_filename:
            bb_temp = 423.0
        elif "20150910_010414" in hdf5_filename:
            bb_temp = 423.0 - 20.0
        elif "20150910_015625" in hdf5_filename:
            bb_temp = 423.0 - 20.0
        elif "20150910_033745" in hdf5_filename:
            bb_temp = 423.0 - 40.0
        elif "20150910_043039" in hdf5_filename:
            bb_temp = 423.0 - 40.0
        elif "2016" in hdf5_filename:
            bb_temp = 5796.0
        
    return bb_temp





def opticalTransmission(csl_window=False):
    #0:Wavelength, 1:Lens ZnSe, 2:Lens Si, 3: Lens Ge, 4:AOTF, 5:Par mirror, 6:Planar miror, 7:Detector, 8:Cold filter, 9:Window transmission function
    #10:CSL sapphire window
    optics_all = np.loadtxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_optics_transmission.csv", skiprows=1, delimiter=",")
    if csl_window==False:
        optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9])
    elif csl_window=="only":
        optics_transmission_total = optics_all[:,10]
    else:
        optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9]) * (optics_all[:,10])
    optics_wavenumbers =  10000. / optics_all[:,0]
    return optics_wavenumbers, optics_transmission_total


def planck(xscale, temp, units): #planck function W/cm2/sr/spectral unit
    if units=="microns" or units=="um" or units=="wavel":
        c1=1.191042e8
        c2=1.4387752e4
        return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0) / 1.0e4 # m2 to cm2
    elif units=="wavenumbers" or units=="cm-1" or units=="waven":
        c1=1.191042e-5
        c2=1.4387752
        return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2
    elif units=="solar radiance cm-1" or units=="solar cm-1":
        try:
            solarRad = np.zeros(len(xscale))
        except:
            xscale = [xscale]
            solarRad = np.zeros(len(xscale))
            
        wavenumberInStart = SOLAR_SPECTRUM[0,0]
        wavenumberDelta = 0.005
                
        print("Finding solar radiances in ACE file")
        for pixelIndex,xValue in enumerate(xscale):
#            index = closestIndex(dataIn[:,0], xValue)
            index = np.int((xValue - wavenumberInStart)/wavenumberDelta)
            if index == 0:
                print("Warning: wavenumber out of range of solar file (start of file). wavenumber = %0.1f" %(xValue))
            if index == len(SOLAR_SPECTRUM[:,0]):
                print("Warning: wavenumber out of range of solar file (end of file). wavenumber = %0.1f" %(xValue))
#            print(pixelIndex)
#            print(index)
#            print(len(solarRad))
#            print(len(SOLAR_SPECTRUM[:,0]))
            solarRad[pixelIndex] = SOLAR_SPECTRUM[index,1] / RADIANCE_TO_IRRADIANCE #/ 1.387 #1AU to MCO-1 AU
        return solarRad / 1.0e4 #m2 to cm2
            
        
    else:
        print("Error: Unknown units given")



def showFullFrame03A(hdf5File, hdf5Filename, frame_numbers, plot_figs=True):
#    fileIndices = list(range(len(hdf5Files)))
    DETECTOR_ROWS_TO_SUM = np.arange(1,24)

#    for fileIndex in fileIndices:
#        hdf5File = hdf5Files[fileIndex]
        #get data from file
#    print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
    detectorData = get_dataset_contents(hdf5File, "Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    
    
    aotfFrequencies = get_dataset_contents(hdf5File, "AOTFFrequency")[0]
    integrationTimes = get_dataset_contents(hdf5File, "IntegrationTime")[0]
    binnings = get_dataset_contents(hdf5File, "Binning")[0]
    nAccumulations = get_dataset_contents(hdf5File, "NumberOfAccumulations")[0]
    nSpectra = detectorData.shape[0]
    
    integrationTime = np.float(integrationTimes[0]) / 1.0e3 #microseconds to seconds
    nAccumulation = np.float(nAccumulations[0])/2.0 #assume LNO nadir background subtraction is on
    binning = np.float(binnings[0]) + 1.0 #binning starts at zero
    
    print("integrationTimeFile = %i" %integrationTimes[0])
    print("integrationTime = %0.2f" %integrationTime)
    print("nAccumulation = %i" %nAccumulation)
    print("binning = %i" %binning)
    
    #normalise to 1s integration time per pixel
    detectorData = detectorData / (integrationTime * nAccumulation) / binning

    """correct obvious detector offset"""
    if hdf5Filename == "20150426_054602_0p1a_LNO_1":
        detectorData[86,:] = detectorData[86,:] - 1.05 #fudge to correct bad point


    #get mean of vertical bins. Data already converted to counts per pixel, so just take mean of all well-illuminated bins
    detectorDataMean = np.mean(detectorData[:, DETECTOR_ROWS_TO_SUM, :], axis=1)
    print("File contains %i detector frames" %nSpectra)

    if plot_figs:
        for frame_number in frame_numbers:
            detectorFrame = detectorData[frame_number, :, :]
            
            plt.figure(figsize=(FIG_X, FIG_Y))
            plt.imshow(detectorFrame, aspect = 5)
            plt.title(hdf5Filename+": position of illumination on detector frame %i" %(frame_number))
            if SAVE_FIGS:
                plt.savefig(hdf5Filename+"_position_of_illumination_on_detector.png")
        plt.figure(figsize=(FIG_X, FIG_Y))
        
        plt.title(hdf5Filename+": illumination intensity per vertically averaged spectrum")
        plt.xlabel("Pixel Number")
        plt.ylabel("Detector counts (bg subtracted)")
        for spectrumIndex, spectrum in enumerate(detectorDataMean):
            plt.plot(spectrum, label=aotfFrequencies[spectrumIndex])
        plt.legend()
        if SAVE_FIGS:
            plt.savefig(hdf5Filename+"_illumination_intensity_per_spectrum.png")

#    return aotfFrequencies[np.where(aotfFrequencies>0)[0]], detectorDataMean[np.where(aotfFrequencies>0)[0],:]
    return aotfFrequencies, detectorDataMean, [integrationTime, nAccumulation, binning]






"""function to read calibration parameters from txt file"""
def readFlagFile(channel):
    if channel=="so":
        flagFile = "%s.txt" %SO_FLAG_FILE
    elif channel=="lno":
        flagFile = "%s.txt" %LNO_FLAG_FILE
    print("Opening spectral calibration file %s for reading", flagFile)

    with open(flagFile) as f:
        for index,line in enumerate(f):
            content = line.strip('\n')
            if "X_UNIT_FLAG=" in content:
                X_UNIT_FLAG = int(content.split("=")[1].strip())
            elif "ILS_FLAG=" in content:
                ILS_FLAG = int(content.split("=")[1].strip())
            elif "AOTF_BANDWIDTH_FLAG=" in content:
                AOTF_BANDWIDTH_FLAG = int(content.split("=")[1].strip())
            elif "BLAZE_FUNCTION_FLAG=" in content:
                BLAZE_FUNCTION_FLAG = int(content.split("=")[1].strip())

            elif "AOTF_BANDWIDTH_X_RANGE_START=" in content:
                AOTF_BANDWIDTH_X_RANGE_START = float(content.split("=")[1].strip())
            elif "AOTF_BANDWIDTH_X_RANGE_STOP=" in content:
                AOTF_BANDWIDTH_X_RANGE_STOP = float(content.split("=")[1].strip())
            elif "AOTF_BANDWIDTH_X_RANGE_STEP=" in content:
                AOTF_BANDWIDTH_X_RANGE_STEP = float(content.split("=")[1].strip())

            elif "BLAZE_FUNCTION_X_RANGE_START=" in content:
                BLAZE_FUNCTION_X_RANGE_START = float(content.split("=")[1].strip())
            elif "BLAZE_FUNCTION_X_RANGE_STOP=" in content:
                BLAZE_FUNCTION_X_RANGE_STOP = float(content.split("=")[1].strip())
            elif "BLAZE_FUNCTION_X_RANGE_STEP=" in content:
                BLAZE_FUNCTION_X_RANGE_STEP = float(content.split("=")[1].strip())
                
            """overwrite flags"""
            AOTF_BANDWIDTH_FLAG = 0
            BLAZE_FUNCTION_FLAG = 0
                
    
    AOTF_BANDWIDTH_X_RANGE=np.arange(AOTF_BANDWIDTH_X_RANGE_START,AOTF_BANDWIDTH_X_RANGE_STOP+AOTF_BANDWIDTH_X_RANGE_STEP,AOTF_BANDWIDTH_X_RANGE_STEP)
    BLAZE_FUNCTION_X_RANGE=np.arange(BLAZE_FUNCTION_X_RANGE_START,BLAZE_FUNCTION_X_RANGE_STOP+BLAZE_FUNCTION_X_RANGE_STEP,BLAZE_FUNCTION_X_RANGE_STEP)
    
    return X_UNIT_FLAG,ILS_FLAG,AOTF_BANDWIDTH_FLAG,BLAZE_FUNCTION_FLAG, AOTF_BANDWIDTH_X_RANGE,BLAZE_FUNCTION_X_RANGE
           




"""function to calculate values that can vary by order"""
def getCalibrationValues(aotfFrequency,diffractionOrder,aotfToWavenumberCoefficients, aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG, aotfBandwidth, orderToBlazeFunctionFSRCentreCoefficients, \
           pixelOrderToWavenumberCoefficients, pixel1Value, blazeFunctionTheta, blazeFunctionGamma, blazeFunctionAlphaB, blazeFunctionSigma, orderToSpectralResolutionCoefficients, \
           AOTF_BANDWIDTH_X_RANGE, BLAZE_FUNCTION_X_RANGE, AOTF_BANDWIDTH_FLAG, BLAZE_FUNCTION_FLAG):


    pixelValues = np.arange(320) + pixel1Value #apply temperature shift



    """AOTF shape"""
    def func_aotf(x, x0, i0, w, iG, sigmaG): #Goddard model 2017
        x0 = x0+0.0001 #fudge to stop infinity at peak
        
        fsinc = (i0 * w**2.0 * (np.sin(np.pi * (x - x0) / w))**2.0) / (np.pi**2.0 * (x - x0)**2.0)
        fgauss = iG * np.exp(-1.0 * (x - x0)**2.0 / sigmaG**2.0)
        f = fsinc + fgauss #slant not included
        return f/np.max(f) #slant not included. normalised
    
    
    
    if AOTF_BANDWIDTH_FLAG==0: #make AOTF shape
        aotfCentreWavenumber = np.polyval(aotfToWavenumberCoefficients, aotfFrequency)
        aotfFunction = func_aotf(AOTF_BANDWIDTH_X_RANGE+aotfCentreWavenumber, aotfCentreWavenumber, aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG)
    
    elif AOTF_BANDWIDTH_FLAG==2: #sinc2 (old version)
        aotfFunction = aotfBandwidth
    
    elif AOTF_BANDWIDTH_FLAG==5: #sinc2 + gaussian (goddard analysis)
        aotfFunction = np.asfarray([aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG])
    
    
    
    """Blaze function"""
    def func_blaze(p, p0, wp):
        p0 = p0 + 0.0001 #to stop infinity at peak
        blaze = (wp * (np.sin(np.pi * (p - p0) / wp))**2.0) / (np.pi**2.0 * (p - p0)**2.0)
        return blaze/np.max(blaze)
    
    
    if BLAZE_FUNCTION_FLAG==0: #make AOTF shape
        """calculate using X_RANGE grid in pixels"""
        #get p0 (central pixel of order)
        p0 = np.polyval(orderToBlazeFunctionFSRCentreCoefficients[3:6],diffractionOrder)
        #get FSR (in pixels)
        wpPixels = np.polyval(orderToBlazeFunctionFSRCentreCoefficients[0:3],diffractionOrder)
        #calculate blaze function in pixels
        blazeFunction = func_blaze(BLAZE_FUNCTION_X_RANGE, p0, wpPixels)

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
    
    elif BLAZE_FUNCTION_FLAG==1:
        blazeFunction = np.asfarray([blazeFunctionTheta,blazeFunctionGamma,blazeFunctionAlphaB,blazeFunctionSigma])
    
    elif BLAZE_FUNCTION_FLAG==2:
        #copy coefficients directly from table
        blazeFunction = orderToBlazeFunctionFSRCentreCoefficients
    
    
    
    
    """Spectral resolution"""
    spectralResolution = np.polyval(orderToSpectralResolutionCoefficients,diffractionOrder)


    """check if not dark"""
    if aotfFrequency > 0: #if not dark frame
        x = np.polyval(pixelOrderToWavenumberCoefficients, pixelValues) * diffractionOrder
    else:
        """if dark, just return NA_VALUES"""
        x = np.zeros(320) + NA_VALUE
        aotfFunction = np.zeros_like(aotfFunction) + NA_VALUE
        blazeFunction = np.zeros_like(blazeFunction) + NA_VALUE
        spectralResolution = NA_VALUE
        

    return(aotfFunction,blazeFunction,spectralResolution,x)





"""function to find temperature- and time- dependent spectral calibration coefficients, do raw value calculations, and write to file"""
def doSpectralCalibration(hdf5FileIn, channel):

    X_UNIT_FLAG,ILS_FLAG,AOTF_BANDWIDTH_FLAG,BLAZE_FUNCTION_FLAG, AOTF_BANDWIDTH_X_RANGE,BLAZE_FUNCTION_X_RANGE = readFlagFile(channel)
    print("AOTF_BANDWIDTH_FLAG=%s, BLAZE_FUNCTION_FLAG=%s", AOTF_BANDWIDTH_FLAG, BLAZE_FUNCTION_FLAG)

    
    #read in data from channel calibration table
    if channel=="lno":
        calibrationFile = h5py.File("%s.h5" % LNO_SPECTRAL_CALIBRATION_TABLE, "r")
        print("Opening spectral calibration file %s.h5 for reading",
                    LNO_SPECTRAL_CALIBRATION_TABLE)
        #use sensor 1 instead of AOTF temperature
        sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
    elif channel=="so":
        calibrationFile = h5py.File("%s.h5" % SO_SPECTRAL_CALIBRATION_TABLE, "r")
        print("Opening spectral calibration file %s.h5 for reading",
                    SO_SPECTRAL_CALIBRATION_TABLE)
        #use sensor 1 instead of AOTF temperature
        sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_SO"][...]
    elif channel=="uvis":
        calibrationFile = h5py.File("%s.h5" % UVIS_SPECTRAL_CALIBRATION_TABLE, "r")
        print("Opening spectral calibration file %s.h5 for reading",
                    UVIS_SPECTRAL_CALIBRATION_TABLE)




        
    #check to catch if diffraction order in file (not true for non-standard science measurements e.g. fullscans)
    diffractionOrdersFound=False
    if "Channel/DiffractionOrder" in hdf5FileIn.keys():
        diffractionOrdersFound=True
        diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...]
        
    aotfFrequencies = hdf5FileIn["Channel/AOTFFrequency"][...]
    
    ydimensions = hdf5FileIn["Science/Y"].shape
    nSpectra = ydimensions[0]
    
    
    #get instrument temperature from aotf temperature measurements (ignore first 2 values - usually wrong)
    measurementTemperature = np.mean(sensor1Temperature[2:10])
#    measurementTemperature = np.mean(aotfTemperatures[2:10])
    
    #convert times to numerical timestamps
    calibrationTimes = list(calibrationFile.keys())
    
    #find which observation corresponds to calibration measurement time
    timeIndex = 0
    hdf5Group = calibrationTimes[timeIndex]
    
    #find which observation corresponds to calibration temperature
    #no temperature dependency now in coefficents. Instead Pixel1 modified to shift spectrum
#    calibrationTemperatures = calibrationFile[hdf5Group]["Temperature"][...]
#    temperatureIndex = np.max(np.where(measurementTemperature>calibrationTemperatures))
#    hdf5Temperature = calibrationTemperatures[temperatureIndex]
    hdf5Temperature = measurementTemperature
    
    print("hdf5Group=%s" %hdf5Group)
    print("hdf5Temperature=%s" %hdf5Temperature)
    
    """now read in correct coefficients"""
    aotfBandwidth = calibrationFile[hdf5Group]["AOTFBandwidth"][...]
#    pixelValues = calibrationFile[hdf5Group]["X"][...]
    pixel1Coefficients = calibrationFile[hdf5Group]["Pixel1Coefficients"][...]
#    pixelOrderToWavenumberCoefficients = calibrationFile[hdf5Group]["PixelOrderToWavenumberCoefficients"][temperatureIndex]
    pixelOrderToWavenumberCoefficients = calibrationFile[hdf5Group]["PixelOrderToWavenumberCoefficients"][...]
    aotfToWavenumberCoefficients = calibrationFile[hdf5Group]["AOTFToWavenumberCoefficients"][...]
#    wavenumberToAotfCoefficients = calibrationFile[hdf5Group]["WavenumberToAOTFCoefficients"][...]
    aotfToOrderCoefficients = calibrationFile[hdf5Group]["AOTFToOrderCoefficients"][...]
#    orderToAotfCoefficients = calibrationFile[hdf5Group]["OrderToAOTFCoefficients"][...]
    aotfFunctionI0 = calibrationFile[hdf5Group]["AOTFFunctionI0Coefficient"][...]
    aotfFunctionIg = calibrationFile[hdf5Group]["AOTFFunctionIgCoefficient"][...]
    aotfFunctionW = calibrationFile[hdf5Group]["AOTFFunctionWCoefficient"][...]
    aotfFunctionSigmaG = calibrationFile[hdf5Group]["AOTFFunctionSigmaGCoefficient"][...]
    orderToSpectralResolutionCoefficients = calibrationFile[hdf5Group]["OrderToSpectralResolutionCoefficients"][...]
    orderToBlazeFunctionFSRCentreCoefficients = calibrationFile[hdf5Group]["OrderToBlazeFunctionFSRCentreCoefficients"][...]
    blazeFunctionTheta = calibrationFile[hdf5Group]["BlazeFunctionTheta"][...]
    blazeFunctionGamma = calibrationFile[hdf5Group]["BlazeFunctionGamma"][...]
    blazeFunctionAlphaB = calibrationFile[hdf5Group]["BlazeFunctionAlphaB"][...]
    blazeFunctionSigma = calibrationFile[hdf5Group]["BlazeFunctionSigma"][...]
    
    tableCreationDatetime = calibrationFile.attrs["DateCreated"]
    
    calibrationFile.close()
  
    #calculate pixel shift based on Goddard analysis and temperature sensor 1.
    pixel1Value = np.polyval(pixel1Coefficients, measurementTemperature)
    
    
    XCalibRef = "CalibrationTime=%s; CalibrationTemperature=%0.1f; CalibrationFileCreated=%s" % (hdf5Group,hdf5Temperature,tableCreationDatetime)
    print("Spectral calibration time: %s, calibration temperature: %0.1f (%0.2f)", \
                hdf5Group, hdf5Temperature, measurementTemperature)

    """check if all AOTF freqs same"""
    if (aotfFrequencies == aotfFrequencies[0]).all():
        print("All AOTF frequences are the same. Performing simple spectral calibration")
        aotfFrequency=aotfFrequencies[0]
        
        """if diffraction order found in file (i.e. normal science measurement) then take first value only"""
        if diffractionOrdersFound:
            diffractionOrder=diffractionOrders[0]
        else:
            """else calculate diffraction order from aotf frequency (for calibration measurements at a single aotf freq only)"""
            diffractionOrder = np.int(np.round(np.polyval(aotfToOrderCoefficients,aotfFrequency)))
            if diffractionOrder < 50:
                diffractionOrder = 0
            diffractionOrdersOut=np.tile(diffractionOrder,(nSpectra,1))

        aotfFunction,blazeFunction,spectralResolution,x = getCalibrationValues(aotfFrequency,diffractionOrder,aotfToWavenumberCoefficients, aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG, aotfBandwidth, orderToBlazeFunctionFSRCentreCoefficients, \
                           pixelOrderToWavenumberCoefficients, pixel1Value, blazeFunctionTheta, blazeFunctionGamma, blazeFunctionAlphaB, blazeFunctionSigma, orderToSpectralResolutionCoefficients, \
                           AOTF_BANDWIDTH_X_RANGE, BLAZE_FUNCTION_X_RANGE, AOTF_BANDWIDTH_FLAG, BLAZE_FUNCTION_FLAG)
        
        """add start,stop,step values to aotf and blaze functions raw values"""
        if AOTF_BANDWIDTH_FLAG == 0:
            aotfFunction = np.insert(aotfFunction,[0,0,0],[AOTF_BANDWIDTH_X_RANGE[0],AOTF_BANDWIDTH_X_RANGE[-1],AOTF_BANDWIDTH_X_RANGE[1]-AOTF_BANDWIDTH_X_RANGE[0]])
        if BLAZE_FUNCTION_FLAG == 0:
            blazeFunction = np.insert(blazeFunction,[0,0,0],[BLAZE_FUNCTION_X_RANGE[0],BLAZE_FUNCTION_X_RANGE[-1],BLAZE_FUNCTION_X_RANGE[1]-BLAZE_FUNCTION_X_RANGE[0]])
        
        
    
        aotfFunctionTable=np.tile(aotfFunction,(nSpectra,1))
        blazeFunctionTable=np.tile(blazeFunction,(nSpectra,1))
        spectralResolutionTable=np.tile(spectralResolution,(nSpectra,1))
        xTable=np.tile(x,(nSpectra,1))
    
    else:
        """for fullscans, run through each spectrum"""
        """make list then convert to np array"""
        print("AOTF frequences are not the same. Calibrating each line separately")
        aotfFunctionTable=[]
        blazeFunctionTable=[]
        spectralResolutionTable=[]
        xTable=[]
        diffractionOrdersOut=[]
        
        for spectrumIndex,aotfFrequency in enumerate(aotfFrequencies):
            diffractionOrder = np.int(np.round(np.polyval(aotfToOrderCoefficients,aotfFrequency)))
            if diffractionOrder < 50:
                diffractionOrder = 0
            diffractionOrdersOut.append(diffractionOrder)
#            print("%0.1f,%i" %(aotfFrequency,diffractionOrder))
            aotfFunction,blazeFunction,spectralResolution,x = getCalibrationValues(aotfFrequency,diffractionOrder,aotfToWavenumberCoefficients, aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG, aotfBandwidth, orderToBlazeFunctionFSRCentreCoefficients, \
                           pixelOrderToWavenumberCoefficients, pixel1Value, blazeFunctionTheta, blazeFunctionGamma, blazeFunctionAlphaB, blazeFunctionSigma, orderToSpectralResolutionCoefficients, \
                           AOTF_BANDWIDTH_X_RANGE, BLAZE_FUNCTION_X_RANGE, AOTF_BANDWIDTH_FLAG, BLAZE_FUNCTION_FLAG)
            
            """add start,stop,step values to aotf and blaze functions raw values"""
            if AOTF_BANDWIDTH_FLAG == 0:
                aotfFunction = np.insert(aotfFunction,[0,0,0],[AOTF_BANDWIDTH_X_RANGE[0],AOTF_BANDWIDTH_X_RANGE[-1],AOTF_BANDWIDTH_X_RANGE[1]-AOTF_BANDWIDTH_X_RANGE[0]])
            if BLAZE_FUNCTION_FLAG == 0:
                blazeFunction = np.insert(blazeFunction,[0,0,0],[BLAZE_FUNCTION_X_RANGE[0],BLAZE_FUNCTION_X_RANGE[-1],BLAZE_FUNCTION_X_RANGE[1]-BLAZE_FUNCTION_X_RANGE[0]])

            aotfFunctionTable.append(aotfFunction)
            blazeFunctionTable.append(blazeFunction)
            spectralResolutionTable.append(spectralResolution)
            xTable.append(x)
            
        aotfFunctionTable = np.asfarray(aotfFunctionTable)
        blazeFunctionTable = np.asfarray(blazeFunctionTable)
        spectralResolutionTable = np.asfarray(spectralResolutionTable)
        xTable = np.asfarray(xTable)
        diffractionOrdersOut = np.asarray(diffractionOrdersOut)
    
    """make other coefficient tables for adding to file"""
#    AOTFWnCoefficients = aotfToWavenumberCoefficients
#    WnAOTFCoefficients = wavenumberToAotfCoefficients
#    PixelSpectralCoefficients = pixelOrderToWavenumberCoefficients
#    AOTFOrderCoefficients = aotfToOrderCoefficients
#    OrderAOTFCoefficients = orderToAotfCoefficients
    
    
    """make flags"""
#    Pixel1 = pixel1Value
#    XUnitFlag = X_UNIT_FLAG
#    ILSFlag = ILS_FLAG
#    AOTFBandwidthFlag = AOTF_BANDWIDTH_FLAG
#    BlazeFunctionFlag = BLAZE_FUNCTION_FLAG
    



    print(XCalibRef)
#    print("Pixel1")
#    print(Pixel1)
#    print("xTable")
#    print(xTable)
#    print("XUnitFlag")
#    print(XUnitFlag)
#    print("PixelSpectralCoefficients")
#    print(PixelSpectralCoefficients)
#    print("AOTFWnCoefficients")
#    print(AOTFWnCoefficients)
#    print("WnAOTFCoefficients")
#    print(WnAOTFCoefficients)
#    print("AOTFOrderCoefficients")
#    print(AOTFOrderCoefficients)
#    print("OrderAOTFCoefficients")
#    print(OrderAOTFCoefficients)
#    print("spectralResolutionTable")
#    print(spectralResolutionTable)
#    print("ILSFlag")
#    print(ILSFlag)
#    print("aotfFunctionTable")
#    print(aotfFunctionTable)
#    print("AOTFBandwidthFlag")
#    print(AOTFBandwidthFlag)
#    print("blazeFunctionTable")
#    print(blazeFunctionTable)
#    print("BlazeFunctionFlag")
#    print(BlazeFunctionFlag)
#    print("diffractionOrdersOut")
#    print(diffractionOrdersOut)
    
    return xTable, BLAZE_FUNCTION_X_RANGE, blazeFunctionTable, AOTF_BANDWIDTH_X_RANGE, aotfFunctionTable, diffractionOrdersOut, aotfToWavenumberCoefficients
    
    


def countsToRadiance(hdf5File, hdf5Filename, channel, frame_numbers, nAdjacentOrders, plot_intermediate_figs=True, plot_main_fig=True):
    if plot_main_fig:
        fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
    
    #list range of adjacent orders to check
    orderOffsets = range(-1*nAdjacentOrders,nAdjacentOrders+1,1)

    #define colours for each order
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(len(orderOffsets))/len(orderOffsets)]

    #define colours for eachframe to plot
    cmap = plt.get_cmap('jet')
    frameColours = [cmap(i) for i in np.arange(len(frame_numbers))/len(frame_numbers)]

    #plot one frame from each file
    showFullFrame03A(hdf5File, hdf5Filename, [20], plot_figs=True)

    #get data from file
    aotfFrequenciesIn, detectorDataMeanIn, [integrationTime, nAccumulation, binning] = showFullFrame03A(hdf5File, hdf5Filename, frame_numbers, plot_figs=False)
    blackbodyTemperature = getBBTempFromFile(hdf5Filename, hdf5File)
    
    wavenumbersIn, blazeFunctionX, blazeFunctionYIn, aotfFunctionX, aotfFunctionYIn, diffractionOrdersIn, aotfToWavenumberCoefficients = doSpectralCalibration(hdf5File, channel)

    #now remove dark frames from all datasets
    nonDarkIndices = np.where(diffractionOrdersIn>0)[0]
    aotfFrequencies = aotfFrequenciesIn[nonDarkIndices]
    detectorDataMean = detectorDataMeanIn[nonDarkIndices]
    wavenumbers = wavenumbersIn[nonDarkIndices]
    blazeFunctionY = blazeFunctionYIn[nonDarkIndices]
    aotfFunctionY = aotfFunctionYIn[nonDarkIndices]
    diffractionOrders = diffractionOrdersIn[nonDarkIndices]
    
    #next remove x start/step/end values from Y datasets
    blazeFunctionY = blazeFunctionY[:, 3:]
    aotfFunctionY = aotfFunctionY[:, 3:]
    
    
    
    #close file
#        hdf5File.close()
    

    diffractionOrdersAll = []
    detectorCountsAll = []
    countsPerRadianceAll = []
    polyCountsPerRadianceAll = []
    for frame_index, frame_number in enumerate(frame_numbers):
        
        #calculate AOTF centre wavenumber
        aotfCentre = np.polyval(aotfToWavenumberCoefficients, aotfFrequencies[frame_number])
        #shift AOTF xscale to real wavenumber values
        aotfFunctionWavenumbers = aotfFunctionX + aotfCentre
        #calculate planck function for blackbody
        if blackbodyTemperature == 5796.0:
            print(np.min(aotfFunctionWavenumbers))
            print(np.max(aotfFunctionWavenumbers))
            blackbodyFunction = planck(aotfFunctionWavenumbers, blackbodyTemperature, "solar cm-1")
#            blackbodyFunction = planck(aotfFunctionWavenumbers, blackbodyTemperature, "cm-1")
        else:
            blackbodyFunctionPlanck = planck(aotfFunctionWavenumbers, blackbodyTemperature, "wavenumbers")
            """remove CSL window transmission from signal"""
            opticsWavenumbers, cslWindowTransmission = opticalTransmission(csl_window="only")
            cslWindowInterp = np.interp(aotfFunctionWavenumbers, opticsWavenumbers, cslWindowTransmission)
            blackbodyFunction = blackbodyFunctionPlanck * cslWindowInterp         
            
        peakBlackbodyFunction = np.max(blackbodyFunction)
            
#        blackbodyFunctionNorm = normalise(blackbodyFunction)
        
        if plot_intermediate_figs:
            plt.figure(figsize=(FIG_X, FIG_Y))
            for offset in orderOffsets:
                plt.plot(blazeFunctionX, blazeFunctionY[frame_number+offset,:], label="Order %i" %(diffractionOrders[frame_number+offset]))
            plt.xlabel("Pixel Number")
            plt.ylabel("Blaze Function")
            plt.legend()
            
            plt.figure(figsize=(FIG_X, FIG_Y))
            plt.plot(aotfFunctionWavenumbers, aotfFunctionY[frame_number,:]*peakBlackbodyFunction, label="AOTF Function", color="k")
            plt.plot(aotfFunctionWavenumbers, blackbodyFunction, label="Blackbody", color="gray", linestyle="--")
            for offsetIndex, offset in enumerate(orderOffsets):
                colour = colours[offsetIndex]
                plt.plot(wavenumbers[frame_number+offset,:], blazeFunctionY[frame_number+offset,:]*peakBlackbodyFunction, label="Order %i" %(diffractionOrders[frame_number+offset]), color=colour)
        
            plt.xlabel("Wavenumber")
            plt.ylabel("AOTF or Blaze Function")
            plt.title("Centre order = %i AOTF frequency = %i kHz" %(diffractionOrders[frame_number], aotfFrequencies[frame_number]))
    
        #now loop through pixels, matching to closest planck and AOTF transmission to each pixel
        aotfFunctionFrame = aotfFunctionY[frame_number, :]
    
        #empty array for summing contributions per pixel
        sumPerPixel = np.zeros_like(wavenumbers[0,:])
        #loop through adjacent orders and pixel numbers
        for offsetIndex, offset in enumerate(orderOffsets):
            aotfBlazeBBOrder = np.zeros_like(wavenumbers[0,:])
            for pixelNumber in range(320):
                colour = colours[offsetIndex]
            
                #find closest planck value
                pixelWavenumber = wavenumbers[frame_number+offset,pixelNumber]
                pixelBlazeFunction = blazeFunctionY[frame_number+offset,pixelNumber]
                
                wavenumberIndex = closestIndex(aotfFunctionWavenumbers, pixelWavenumber)
                blackbodyValue = blackbodyFunction[wavenumberIndex]
                aotfValue = aotfFunctionFrame[wavenumberIndex]
                
                aotfBlazeBB = aotfValue * pixelBlazeFunction * blackbodyValue
                aotfBlazeBBOrder[pixelNumber] = aotfBlazeBB
                sumPerPixel[pixelNumber] += aotfBlazeBB
            
            if plot_intermediate_figs:
                plt.scatter(wavenumbers[frame_number+offset,:], aotfBlazeBBOrder, c=colour, marker="x", label="Combined AOTF Blaze Radiance")
        if plot_intermediate_figs:
            plt.scatter(wavenumbers[frame_number,:], sumPerPixel, c="k", marker="o", label="Combined AOTF Blaze Radiance all orders")
            plt.legend()
    
        detectorDataOrder = detectorDataMean[frame_number,:]
        countsPerRadiance = detectorDataOrder / sumPerPixel
        POLYFIT_PIXEL_RANGE = [50, 320]
        POLYFIT_DEGREE = 6
        polyfitCoefficients = np.polyfit(range(POLYFIT_PIXEL_RANGE[0], POLYFIT_PIXEL_RANGE[1]), countsPerRadiance[POLYFIT_PIXEL_RANGE[0]:POLYFIT_PIXEL_RANGE[1]], POLYFIT_DEGREE)
        polyCountsPerRadiance = np.polyval(polyfitCoefficients, range(320))
        
        if plot_intermediate_figs:
            plt.figure(figsize=(FIG_X, FIG_Y))
            plt.plot(normalise(detectorDataOrder), label="Vertically Averaged Detector Frame")
            plt.plot(normalise(sumPerPixel), label="Calculated Radiance Hitting Each Pixel")
            plt.xlabel("Pixel Number")
            plt.ylabel("Normalised detector data / expected radiance")
            plt.legend()
    
        frameColour = frameColours[frame_index]
        diffractionOrdersAll.append(diffractionOrders[frame_number])
        detectorCountsAll.append(detectorDataOrder)
        countsPerRadianceAll.append(countsPerRadiance)
        polyCountsPerRadianceAll.append(polyCountsPerRadiance)
        if plot_main_fig:
#            print(detectorDataOrder)
#            print(sumPerPixel)
            ax1.plot(countsPerRadiance, label="Order %i" %(diffractionOrders[frame_number]), color=frameColour)
            ax1.plot(polyCountsPerRadiance, color=frameColour, linestyle=":")
            ax1.set_xlabel("Pixel Number")
            ax1.set_ylabel("Counts per unit radiance (per ms per pixel per accumulation")
#            ax1.legend()
            ax1.set_title("%s T=%iK - IT=%ims NAcc=%i Binning=%i" %(hdf5Filename, blackbodyTemperature, integrationTime, nAccumulation, binning))
            ax1.set_ylim([0.0,1.5e8])
    return np.asarray(diffractionOrdersAll), np.asarray(detectorCountsAll), np.asarray(countsPerRadianceAll), np.asarray(polyCountsPerRadianceAll)


nAdjacentOrders = 3 #add together contributions from these adjacent orders (on each side of centre)

hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, model=model)

frame_numbers = range(9, 100) #calculate for these frames i.e. these diffraction orders within the fullscan file
fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))

#for calculating the radiance per mean counts, often values are incorrect at edge of detector (blaze function is too low). Choose centre only
#average just 50 pixels in centre
STARTING_PIXEL = 160
ENDING_PIXEL = 240

countsPerRadianceAll = []
countsPerRadianceMeanAll = []
fittedCountsPerRadianceAll = []
for index in range(len(hdf5Filenames)): #loop through files
    bbTemperature = getBBTempFromFile(hdf5Filenames[index], hdf5Files[index])
    instTemp = getInstTempFromFile(hdf5Files[index])
    if -30 < instTemp < -10:
        colour="b"
    elif -10 < instTemp < 0:
        colour="c"
    elif 0 < instTemp < 10:
        colour="g"
    elif 10 < instTemp < 30:
        colour="r"

    if bbTemperature == 383.0:
        bbText = "%iK Blackbody %0.1fC NOMAD" %(bbTemperature, instTemp)
    elif bbTemperature == 403.0:
        bbText = "%iK Blackbody %0.1fC NOMAD" %(bbTemperature, instTemp)
    elif bbTemperature == 423.0:
        bbText = "%iK Blackbody %0.1fC NOMAD" %(bbTemperature, instTemp)
    elif bbTemperature == 5796.0:
        bbText = "Sun Pointing %0.1fC NOMAD" %(instTemp)
        colour = "k"


    print("%iK BB" %bbTemperature)
    #calculate counts per unit radiance
#    diffractionOrdersMeasured, countsMeasured, countsPerRadianceMeasured = countsToRadiance(hdf5Files[index], hdf5Filenames[index], "lno", frame_numbers, nAdjacentOrders, plot_intermediate_figs=True, plot_main_fig=True)
    diffractionOrdersMeasured, countsMeasured, countsPerRadianceMeasured, fittedCountsPerRadianceMeasured = countsToRadiance(hdf5Files[index], hdf5Filenames[index], "lno", frame_numbers, nAdjacentOrders, plot_intermediate_figs=False, plot_main_fig=True)
    
    #calculate radiance per mean counts at centre of detector
    countsPerRadianceMean = np.asarray([np.mean(countsPerRadianceMeasuredPerOrder[STARTING_PIXEL:ENDING_PIXEL]) for countsPerRadianceMeasuredPerOrder in countsPerRadianceMeasured])
    ax2.scatter(diffractionOrdersMeasured, countsPerRadianceMean, label="%s %s" %(hdf5Filenames[index],bbText), c=colour, alpha=0.5, linewidth=0)

    #or calculate mean counts only and plot versus counts per radiance (look for trend suggesting offsets in data)
#    countsMean = np.asarray([np.mean(countsMeasuredPerOrder) for countsMeasuredPerOrder in countsMeasured])
#    ax2.scatter(countsMean, countsPerRadianceMean, label="%s %iK" %(hdf5Filenames[index],bbTemperature))

    countsPerRadianceAll.append(countsPerRadianceMeasured)
    countsPerRadianceMeanAll.append(countsPerRadianceMean)
    fittedCountsPerRadianceAll.append(fittedCountsPerRadianceMeasured)
ax2.set_xlabel("Diffraction order")
ax2.set_ylabel("Radiance per count per pixel per second at centre of detector")
ax2.set_title("Radiometric Calibration of LNO: Comparison of Blackbody and Solar Fullscans")
meanCountsPerRadiancePerOrder = np.mean(np.asarray(countsPerRadianceMeanAll), axis=0)
stdCountsPerRadiancePerOrder = np.std(np.asarray(countsPerRadianceMeanAll), axis=0)
meanFittedCountsPerRadiance = np.mean(np.asarray(fittedCountsPerRadianceAll), axis=0)
ax2.plot(diffractionOrdersMeasured, meanCountsPerRadiancePerOrder+stdCountsPerRadiancePerOrder, label=title+"Mean+St Dev") #overplot mean for all observations of this run
ax2.plot(diffractionOrdersMeasured, meanCountsPerRadiancePerOrder-stdCountsPerRadiancePerOrder, label=title+"Mean-St Dev") #overplot mean for all observations of this run
ax2.legend()



"""ignore - noise calculated in data, not in calibration file"""
#NOISE_POLYNOMIAL_DEGREE = 5
#
#noiseInCounts = np.zeros_like(meanFittedCountsPerRadiance)
#noiseInRadiance = np.zeros_like(meanFittedCountsPerRadiance)
#noiseCountsSingleValue = np.zeros_like(meanFittedCountsPerRadiance[:,0])
#noiseRadianceSingleValue = np.zeros_like(meanFittedCountsPerRadiance[:,0])
#
#for orderIndex in range(len(meanFittedCountsPerRadiance[:,0])):
#    noiseInCounts[orderIndex,:] = meanFittedCountsPerRadiance[orderIndex,:] - np.polyval(np.polyfit(range(320), meanFittedCountsPerRadiance[orderIndex,:], NOISE_POLYNOMIAL_DEGREE), range(320))
#    noiseCountsSingleValue[orderIndex] = np.std(noiseInCounts[orderIndex,:])
#
#    noiseInRadiance[orderIndex,:] = noiseCountsSingleValue[orderIndex] / meanFittedCountsPerRadiance[orderIndex,:]
    


"""prepare output for writing to calibration look up file"""
pixels = np.arange(320)

yRadianceFactorCounts = np.ones((np.max(diffractionOrdersMeasured)+1,len(pixels))) #array of error values lookup table

yErrorRadiances = np.ones((np.max(diffractionOrdersMeasured)+1,len(pixels))) #array of error values lookup table
yErrorRadianceFactors = np.ones((np.max(diffractionOrdersMeasured)+1,len(pixels))) #array of error values lookup table
yErrorSingleValue = 1.0 #array of error values lookup table

yRadiancesToCounts = np.ones((np.max(diffractionOrdersMeasured)+1,len(pixels))) * -999.0

yRadiancesToCounts[diffractionOrdersMeasured,:] = meanFittedCountsPerRadiance

diffractionOrdersOut = np.arange(0,np.max(diffractionOrdersMeasured)+1)


#aotfToWavenumberCoefficients = np.asarray([9.4094759e-08, 1.4223820e-01, 3.0067657e+02])
#plt.figure(figsize=(FIG_X, FIG_Y))
#for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames):
#    bbTemperature = getBBTempFromFile(hdf5File)
#    aotfFrequencies, detectorDataMean, parameters = showFullFrame03A(hdf5File, hdf5Filename, [0], plot_figs=False)
#    detectorCountsMean = np.asarray([np.mean(detectorDataMeanPerOrder) for detectorDataMeanPerOrder in detectorDataMean])
#
#    wavenumberOrderCentres = np.polyval(aotfToWavenumberCoefficients, aotfFrequencies[np.asarray(frame_numbers)])
#    planckOrders = planck(wavenumberOrderCentres, bbTemperature, "cm-1")
#    
##    plt.scatter(aotfFrequencies[np.asarray(frame_numbers)], detectorCountsMean[np.asarray(frame_numbers)], label=hdf5Filename+" %i" %getBBTempFromFile(hdf5File))
#    plt.scatter(aotfFrequencies[np.asarray(frame_numbers)], detectorCountsMean[np.asarray(frame_numbers)]/planckOrders, label=hdf5Filename+" %i" %getBBTempFromFile(hdf5File))
##    plt.scatter(aotfFrequencies[np.asarray(frame_numbers)], planckOrders, label=hdf5Filename+" %i" %getBBTempFromFile(hdf5File))
#plt.legend()

#plt.figure(figsize=(FIG_X, FIG_Y))
#for bbtemp in [383,403,423]: plt.scatter(aotfFrequencies[np.asarray(frame_numbers)],)







if SAVE_FILES:
    
    """write to coefficient file"""
    title = "%s_Radiometric_Calibration_Table" %channel.upper()
    
    outputFilename = "%s" %(title.replace(" ","_"))
    
    
    calibrationTimes = []
    
    #make arrays of coefficients for given calibration date
    calibrationTimes.append(b"2015 JAN 01 00:00:00.000")
    #at present, values don't change over time. Therefore copy values for dates 2 and 3
    calibrationTimes.append(b"2016 JAN 01 00:00:00.000")
    calibrationTimes.append(b"2017 JAN 01 00:00:00.000")
    
    
    #now write to file
    #open file for writing
    hdf5File = h5py.File(os.path.join(BASE_DIRECTORY,outputFilename+".h5"), "w")
    
    #loop manually through calibration times. Not expecting many calibrations over time!
    calibrationTime = calibrationTimes[0]
    hdf5Group1 = hdf5File.create_group(calibrationTime)
    hdf5Group1.create_dataset("DiffractionOrder",data=diffractionOrdersOut,dtype=np.float)
    hdf5Group1.create_dataset("Pixels",data=pixels,dtype=np.float)
    if channel=="lno": hdf5Group1.create_dataset("YRadiancesToCounts144",data=yRadiancesToCounts,dtype=np.float)
    if channel=="lno": hdf5Group1.create_dataset("YRadianceFactorCounts144",data=yRadianceFactorCounts,dtype=np.float)
    hdf5Group1.create_dataset("YErrorRadiances144",data=yErrorRadiances,dtype=np.float)
    hdf5Group1.create_dataset("YErrorRadianceFactors144",data=yErrorRadianceFactors,dtype=np.float)
    hdf5Group1.create_dataset("YErrorSingleValue144",data=yErrorSingleValue,dtype=np.float)
    
    
    
    calibrationTime = calibrationTimes[1]
    hdf5Group2 = hdf5File.create_group(calibrationTime)
    hdf5Group2.create_dataset("DiffractionOrder",data=diffractionOrdersOut,dtype=np.float)
    hdf5Group2.create_dataset("Pixels",data=pixels,dtype=np.float)
    if channel=="lno": hdf5Group2.create_dataset("YRadiancesToCounts144",data=yRadiancesToCounts,dtype=np.float)
    if channel=="lno": hdf5Group2.create_dataset("YRadianceFactorCounts144",data=yRadianceFactorCounts,dtype=np.float)
    hdf5Group2.create_dataset("YErrorRadiances144",data=yErrorRadiances,dtype=np.float)
    hdf5Group2.create_dataset("YErrorRadianceFactors144",data=yErrorRadianceFactors,dtype=np.float)
    hdf5Group2.create_dataset("YErrorSingleValue144",data=yErrorSingleValue,dtype=np.float)
    
    
    
    calibrationTime = calibrationTimes[2]
    hdf5Group3 = hdf5File.create_group(calibrationTime)
    hdf5Group3.create_dataset("DiffractionOrder",data=diffractionOrdersOut,dtype=np.float)
    hdf5Group3.create_dataset("Pixels",data=pixels,dtype=np.float)
    if channel=="lno": hdf5Group3.create_dataset("YRadiancesToCounts144",data=yRadiancesToCounts,dtype=np.float)
    if channel=="lno": hdf5Group3.create_dataset("YRadianceFactorCounts144",data=yRadianceFactorCounts,dtype=np.float)
    hdf5Group3.create_dataset("YErrorRadiances144",data=yErrorRadiances,dtype=np.float)
    hdf5Group3.create_dataset("YErrorRadianceFactors144",data=yErrorRadianceFactors,dtype=np.float)
    hdf5Group3.create_dataset("YErrorSingleValue144",data=yErrorSingleValue,dtype=np.float)
    
    
    
    
    
    if channel=="lno":
        comments = "Dummy calibration at present for testing purposes"
    elif channel=="so":
        comments = "No SO radiometric calibration at present"
    hdf5File.attrs["Comments"] = comments
    hdf5File.attrs["DateCreated"] = str(datetime.now())
    hdf5File.close()
    


