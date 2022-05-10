# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Tue May  8 12:33:13 2018

@author: iant


##############SET UP INSTRUCTIONS################

1. ADD REQUIRED LIBRARIES. MOST ARE INSTALLED BY DEFAULT EXCEPT PYSFTP (SPICE KERNELS ARE NOT REQUIRED)

2. ADD YOUR HOME DIRECTORY TO THE LIST OF DIRECTORIES E.G. 

elif os.path.exists(os.path.normcase(r"<PATH>")):
    DATA_DIRECTORY = os.path.normcase(r"<PATH>")
    DIRECTORY_STRUCTURE = False
    FIG_X = 18
    FIG_Y = 9
    SEARCH_DATASTORE = True
    DATASTORE_SERVER = ["tethys.oma.be", "iant"]
    DATASTORE_DIRECTORY = r"/ae/data1/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5" #FOR OFFICIAL DATASTORE
#    DATASTORE_DIRECTORY = r"/ae/projects4/NOMAD/Data/db_test/test/iant/hdf5" #FOR DB_TEST FOLDER


3A. IF YOU HAVE A KNOWN LIST OF FILENAMES, ADD THEM TO A LIST CALLED "obspaths"
SELECT THE FILE LEVEL E.G. fileLevel = "hdf5_level_0p3a"
IF THE FILES AREN'T FOUND IN THE DATA_DIRECTORY, THEY WILL BE DOWNLOADED FROM THE SERVER (INPUT YOUR BIRA PASSWORD WHEN REQUESTED)


3B. IF YOU DON'T KNOW THE FILENAMES, YOU CAN SET A SEARCH STRING E.G. obspaths = ["*2018*_0p3a_*LNO*_D_169"]
THE FIRST CHARACTER MUST A * . THEN ALL SUBSEQUENT * INDICATE SEARCH STRINGS
IN THE EXAMPLE ABOVE, THE SEARCH STRINGS ARE "2018", "_0p3a_", "LNO" and "_D_169"


4. MAKE A LIST OF FILEPATHS AND DOWNLOAD FILES IF NECESSARY USING THE COMMAND:
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel)

IF YOU WANT TO SEARCH FOR AN ATTRIBUTE IN THE FILE, PASS AN ADDITIONAL ARGUMENT E.G. FIND FILES WHERE NBINS = 12:
searchAttributes = {"NBins":12} 
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes)

IF YOU WANT TO SEARCH FOR GREATER THAN/LESS THAN/MIN/MAX OF A DATASET IN THE FILE, PASS AN ADDITIONAL ARGUMENT
E.G. FIND FILES WHERE THE MINIMUM OF Geometry/Point0/SunSZA IS LESS THAN 5.0 DEGREES
searchDatasetsMinMax = {"SunSZA":["min","lt",5.0,"Geometry/Point0"]}
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes, search_datasets_min_max=searchDatasetsMinMax)

E.G. FIND FILES WHERE THE MAXIMUM Geometry/Point0/Lat IS GREATER THAN 25.0 DEGREES
searchDatasetsMinMax = {"Lat":["max","gt",25.0,"Geometry/Point0"]}
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes, search_datasets_min_max=searchDatasetsMinMax)

NOTE THAT A FILE WILL ONLY BE RETURNED IF ALL ATTRIBUTES AND DATASETS SATISFY THE GIVEN CONDITIONS!


5. PLOT ALL FILES USING THE COMMAND
plotEachFigure03A(hdf5Files, hdf5Filenames, titles)

TO ACTIVATE THE BAD PIXEL CORRECTION (NOTE: IT MAY ALREADY BE APPLIED TO DATA IN THE DB_TEST FOLDER!), PASS THE ARGUMENT
plotEachFigure03A(hdf5Files, hdf5Filenames, titles, bad_pixel_correction=True)


TO PLOT DISCRETE OR RUNNING MEANS OF FRAMES, PASS THE ARGUMENT
plotEachFigure03A(hdf5Files, hdf5Filenames, titles, mean="running")
OR
plotEachFigure03A(hdf5Files, hdf5Filenames, titles, mean="discrete")
BY DEFAULT, N_SPECTRA_TO_MEAN = 20


6. TO SAVE FIGURES, SET SAVE_FIGS = True

"""


import os
#import h5py
import numpy as np
#import numpy.linalg as la
#import gc

#import bisect
#from scipy.optimize import curve_fit, leastsq
#from mpl_toolkits.basemap import Basemap


#from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v03 import get_dataset_contents, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList

#from filename_lists_v01 import getFilenameList

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
    print("Running on windows")
    import spiceypy as sp
    from PIL import Image
    from plot_simulations_v01 import findSimulations, getSimulationData, getOrderSimulation
    PFM_AUXILIARY_FILES = os.path.join(BASE_DIRECTORY, "data", "pfm_auxiliary_files")

    #load spiceypy kernels if required
    KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
    #KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
    METAKERNEL_NAME = "em16_plan_win.tm"
    METAKERNEL_NAME = "em16_ops_win.tm"
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)

#SAVE_FIGS = False
SAVE_FIGS = True




####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""

#all nadirs order 189 April
#obspaths = ["*201804*_0p3a_*LNO*_D_190"]
#fileLevel = "hdf5_level_0p3a"

#all nadirs order 189
#obspaths = ["*D_189"]
#obsTitles = ["All nadirs order 189"]
#fileLevel = "hdf5_level_0p3a"
#fileLevel = "hdf5_level_0p1e"


#obspaths = ["*20180421_203916_0p1a_LNO"]
#obspaths = ["*20180428_232453_0p1a_LNO"]
#obsTitles = ["20180421_203916"]
#fileLevel = "hdf5_level_0p1a"


#limb measurements
#obspaths = ["*_L_"]
#obsTitles = ["Limb"]
#fileLevel = "hdf5_level_0p1e"

#all order 189
#obspaths = ["20180401_152047_0p3a_LNO_1_D_189",
#"20180403_153234_0p3a_LNO_1_D_189",
#"20180403_193328_0p3a_LNO_1_D_189",
#"20180406_205446_0p3a_LNO_1_D_189",
#"20180407_123900_0p3a_LNO_1_D_189",
#"20180409_154455_0p3a_LNO_1_D_189",
#"20180410_112344_0p3a_LNO_1_D_189",
#"20180412_103258_0p3a_LNO_1_D_189",
#"20180412_142848_0p3a_LNO_1_D_189",
#"20180417_023215_0p3a_LNO_1_D_189",
#"20180417_181522_0p3a_LNO_1_D_189",
#"20180419_192216_0p3a_LNO_1_D_189",
#"20180420_031342_0p3a_LNO_1_D_189",
#"20180421_203916_0p3a_LNO_1_D_189",
#"20180423_214614_0p3a_LNO_1_D_189",
#"20180424_113122_0p3a_LNO_1_D_189",
#"20180426_103030_0p3a_LNO_1_D_189",
#"20180426_222748_0p3a_LNO_1_D_189",
#"20180428_232453_0p3a_LNO_1_D_189",
#"20180429_132004_0p3a_LNO_1_D_189"]
#obsTitles = obspaths
#fileLevel = "hdf5_level_0p3a"

#all 2 subd low beta angle measurements
#obspaths = ["*2018*_0p3a_*LNO*_D_169"]
#obsTitles = obspaths
#fileLevel = "hdf5_level_0p3a"
#searchAttributes = {"NBins":12} #note that a file must match all attributes and min max dataset conditions!
#searchDatasetsMinMax = {"SunSZA":["min","lt",5.0,"Geometry/Point0"]} #"min","max","lt","gt". No group name = "".

#all 2 subd low beta angle measurements
#obspaths = ["20180416_144459_0p3a_LNO_1_D_121",
#        "20180416_144459_0p3a_LNO_1_D_169",
#        "20180417_023215_0p3a_LNO_1_D_167",
#        "20180417_023215_0p3a_LNO_1_D_189",
#        "20180418_020647_0p3a_LNO_1_D_167",
#        "20180418_020647_0p3a_LNO_1_D_169",
#        "20180418_095818_0p3a_LNO_1_D_134",
#        "20180418_095818_0p3a_LNO_1_D_136",
#        "20180418_155157_0p3a_LNO_1_D_121",
#        "20180418_155157_0p3a_LNO_1_D_169"]
#obsTitles = obspaths
#fileLevel = "hdf5_level_0p3a"


#LNO straylight anomalies
#obspaths = ["20180421_203916_0p3a_LNO_1_D_121","20180422_042041_0p3a_LNO_1_D_121","20180424_072536_0p3a_LNO_1_D_119","20180428_232453_0p3a_LNO_1_D_189","20180429_225934_0p3a_LNO_1_D_121"]
#obsTitles = ["LNO Anomalies"] *5
#fileLevel = "hdf5_level_0p3a"

#"""LNO detector offsets"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["20180423_155230_0p1a_LNO_1"] #"20180423_155230_0p3a_LNO_1_D_168"
#obsTitles = ["LNO Detector Offsets"]


"""LNO coverage maps"""
#title = "LNO Coverage Map"
#fileLevel = "hdf5_level_0p3a"
#obspaths = ["*2018*_0p3a_LNO_*D_121"]
#obspaths = ["*2018*_0p3a_LNO_*D_134"]
#obspaths = ["*2018*_0p3a_LNO_*D_167"]
#obspaths = ["*2018*_0p3a_LNO_*D_168"]
#obspaths = ["*2018*_0p3a_LNO_*D_169"]
#obspaths = ["*2018*_0p3a_LNO_*D_190"]
#obspaths = ["*2018*_0p3a_LNO_*L_"] #change sza to tangent alt
#obspaths = ["*2018*_0p3a_LNO_*N_"]

#obspaths = ["*2018042*_0p3a_LNO_*D_"]


"""UVIS coverage maps"""
title = "UVIS Coverage Map"
fileLevel = "hdf5_level_0p2a"
obspaths = ["*20180*_0p2a_UVIS_D"] #update when reprocessing is done!
#obspaths = ["*201811*_0p2a_UVIS_D"]


"""LNO LEVEL 1.0A TESTING"""
#LNO level 1.0A testing
#obspaths = ["20180422_003456_1p0a_LNO_1_D_167"]
#fileLevel = "hdf5_level_1p0a"

#find LNO nadir test files. Orders 167-169, 189-191, low SZA and 2 or 3 subdomains
#obspaths = ["*2018060*_0p3a_*LNO*_D_191"]
#obspaths = ["*2018*_0p3a_*LNO*_D_191"]
#fileLevel = "hdf5_level_0p3a"
#searchAttributes = {"NBins":8} #note that a file must match all attributes and min max dataset conditions!
#searchDatasetsMinMax = {"SunSZA":["min","lt",5.0,"Geometry/Point0"]} #"min","max","lt","gt". No group name = "".

#plot LNO nadir test files. Orders 167-169, 189-191, low SZA and 2 or 3 subdomains
#obspaths = ["20180608_144207_1p0a_LNO_1_D_167",
#            "20180609_082251_1p0a_LNO_1_D_167",
#            "20180608_104620_1p0a_LNO_1_D_168",
#            "20180609_082251_1p0a_LNO_1_D_169",
#            "20180608_144207_1p0a_LNO_1_D_189",
#            "20180609_180214_1p0a_LNO_1_D_190"]
#fileLevel = "hdf5_level_1p0a"



#check individual files - LNO offset/straylight testing
#obspaths = ["20180421_203916_0p1e_LNO_1_D_168", r"tmp\20180421_203916_0p1d_LNO_1_D_168"] #LNO straylight
#obspaths = ["20180422_042041_0p1e_LNO_1_D_169", r"tmp\20180422_042041_0p1d_LNO_1_D_169"] #LNO straylight
#obspaths = ["20180424_072536_0p1e_LNO_1_D_168", r"tmp\20180424_072536_0p1d_LNO_1_D_168"] #LNO straylight
#obspaths = ["20180422_042041_0p1e_LNO_1_D_169", r"tmp\20180422_042041_0p1d_LNO_1_D_169"] #LNO straylight



#obspaths = ["20180422_102424_0p1e_LNO_1_D_167", r"tmp\20180422_102424_0p1d_LNO_1_D_167"] #LNO good observation
#obspaths = [r"tmp\20180422_102424_0p1e_LNO_1_D_167"] #LNO good observation
#obspaths = ["20180423_155230_0p1e_LNO_1_D_168", r"tmp\20180423_155230_0p1d_LNO_1_D_168"] #LNO with offsets
#obsTitles = ["Before modification", "After modification"]
#fileLevel = "hdf5_level_0p1e"
#DIRECTORY_STRUCTURE = False
#DATA_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")



"""calibrate level 0.3a files and plot on albedo map"""
#title = "radiance albedo"
#fileLevel = "hdf5_level_0p3a"
##obspaths = ["20180608_004700_1p0a_LNO_1_D_171"]
#obspaths = ["*20180608*_0p3a_LNO_*D_134"]





def normalise(array_in):
    """normalise input array"""
    return array_in/np.max(array_in)


def correctBadPixelFromData(detector_data):
    POLYNOMIAL_ORDER = 5
    N_STANDARD_DEVIATIONS = 2
    MODE = "Linear"
    nPixels = detector_data.shape[1]
    obsStandardDeviation = np.nanstd(detector_data, axis=0)
    polynomialValues = np.polyval(np.polyfit(range(nPixels), obsStandardDeviation, POLYNOMIAL_ORDER),range(nPixels))
    
    obsPolynomialDifference = obsStandardDeviation - polynomialValues
    polynomialDifferenceStandardDeviation = np.std(obsPolynomialDifference)
    maxDeviation = polynomialDifferenceStandardDeviation * N_STANDARD_DEVIATIONS
    badPixels = list(np.where(obsPolynomialDifference > maxDeviation)[0])
    
    print("BadPixels found: "+"%i "*len(badPixels) %tuple(badPixels))
    detector_data_new = np.copy(detector_data)
    for badPixel in badPixels:
        #if pixel on edge of detector, ignore. if not, find adjacent non-bad pixels
        if badPixel+1 != nPixels and badPixel-1 >= 0:
            pixel1ToInterpolate = badPixel-1
            if pixel1ToInterpolate in badPixels:
                pixel1ToInterpolate -= 1
            pixel2ToInterpolate = badPixel+1

            if pixel2ToInterpolate in badPixels:
                pixel2ToInterpolate += 1
            for rowIndex,detectorRow in enumerate(detector_data):
                if MODE == "Linear":
                    detector_data_new[rowIndex,badPixel] = np.polyval(np.polyfit([pixel1ToInterpolate,pixel2ToInterpolate],[detectorRow[pixel1ToInterpolate],detectorRow[pixel2ToInterpolate]],1),badPixel)
                else:
                    print("Error: MODE not defined")

    return detector_data_new

def runningMean(detector_data, n_spectra_to_mean):
    nSpectra = detector_data.shape[0]
    running_mean_data = np.zeros_like(detector_data[0:(-1*(n_spectra_to_mean-1)), :])
    runningIndices = [np.asarray(range(startingIndex, startingIndex+n_spectra_to_mean)) for startingIndex in range(0, (nSpectra-n_spectra_to_mean)+1)]
    for rowIndex,indices in enumerate(runningIndices):
        running_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return running_mean_data

def discreteMean(detector_data, n_spectra_to_mean):
    nSpectra = detector_data.shape[0]
    nSpectraNew = int(nSpectra/n_spectra_to_mean)
    
    discrete_mean_data = np.zeros_like(detector_data[0:nSpectraNew, :])
    discreteIndices = [np.asarray(range(startingIndex, startingIndex+n_spectra_to_mean)) for startingIndex in range(0, nSpectraNew*n_spectra_to_mean, n_spectra_to_mean)]
    for rowIndex,indices in enumerate(discreteIndices):
        discrete_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return discrete_mean_data


def doRadiometricCalibration(hdf5_file, hdf5_filename, silent=False):
    """Radiometrically calibrate LNO"""
    import h5py
    print("Calibrating file %s" %hdf5_filename)

    LNO_RADIOMETRIC_CALIBRATION_TABLE=os.path.join(PFM_AUXILIARY_FILES,"radiometric_calibration","LNO_Radiometric_Calibration_Table_v02")
    RUNNING_MEAN = False
    REMOVE_NEGATIVES = False
    
    #read in data from channel calibration table
    calibrationFile = h5py.File("%s.h5" % LNO_RADIOMETRIC_CALIBRATION_TABLE, "r")

    #get observation start time and diffraction order/ AOTF
    observationStartTime = hdf5_file["Geometry/ObservationDateTime"][0,0]
    diffractionOrders = hdf5_file["Channel/DiffractionOrder"][...]
    aotfFrequencies = hdf5_file["Channel/AOTFFrequency"][...]

    integrationTimes = hdf5_file["Channel/IntegrationTime"][...]
    bins = hdf5_file["Science/Bins"][...]
    nAccumulations = hdf5_file["Channel/NumberOfAccumulations"][...]
    
    integrationTime = np.float(integrationTimes[0]) / 1.0e3 #milliseconds to seconds
    nAccumulation = np.float(nAccumulations[0])/2.0 #assume LNO nadir background subtraction is on
    binning = np.float(bins[0,1] - bins[0,0]) + 1.0 #binning starts at zero

    #check that all aotf freqs are the same (they should be for this function)
    if (aotfFrequencies == aotfFrequencies[0]).all():
        diffractionOrder = diffractionOrders[0]
    else:
        print("Error: AOTF frequencies are not the same. Use another function for fullscan or calibrations")
        

    #convert times to numerical timestamps
    calibrationTimes = list(calibrationFile.keys())
    calibrationTimestamps = np.asfarray([sp.utc2et(calibrationTime) for calibrationTime in calibrationTimes])
    measurementTimestamp = sp.utc2et(observationStartTime)
    
    #find which observation corresponds to calibration measurement time
    timeIndex = np.max(np.where(measurementTimestamp>calibrationTimestamps))
    calibrationTime = calibrationTimes[timeIndex]
    hdf5Group = calibrationTime

    yIn = hdf5_file["Science/Y"][...] / (integrationTime * nAccumulation) / binning #scale to counts per second per pixel
    
    yFitted = np.polyval(np.polyfit(range(50),yIn[0,0:50],2),range(50))
    yStd = np.std(yIn[0,0:50] - yFitted)
    
    if RUNNING_MEAN:
        """apply running mean with n=10"""
        yIntermediate = runningMean(yIn, 10)
        Y = np.copy(yIntermediate)
        
        if REMOVE_NEGATIVES:
            """remove negatives from data"""
            negativesFound = False
            for spectrumIndex, spectrum in enumerate(yIntermediate):
                if np.min(spectrum) < 0.0:
                    negativesFound = True
                    negativeIndices = np.where(spectrum < 0.0)[0]
                    Y[spectrumIndex,negativeIndices] = 0.0
    else:
        """don't apply running mean with n=10 and remove negatives"""
        yIntermediate = yIn
        Y = np.copy(yIntermediate)
        
        if REMOVE_NEGATIVES:
            """remove negatives from data"""
            negativesFound = False
            for spectrumIndex, spectrum in enumerate(yIntermediate):
                if np.min(spectrum) < 0.0:
                    negativesFound = True
                    negativeIndices = np.where(spectrum < 0.0)[0]
                    Y[spectrumIndex,negativeIndices] = 0.0


    if REMOVE_NEGATIVES:
        if negativesFound:
            print("Warning: negative Y values found in file. Replaced by zeroes")

    
    yCalibration = calibrationFile[hdf5Group]["YRadiancesToCounts144"][diffractionOrder,:] #e.g. 4 counts = 1 unit of radiance
    
    calibrationFile.close()

    nSpectra = Y.shape[0]
    YOut = Y / np.tile(yCalibration, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
    YError = yStd / np.tile(yCalibration, [nSpectra, 1]) #multiple counts measured by radiance/counts from calibration
    
    SNR = YOut / YError

    return YOut, YError, SNR



def plot01ASpectra(hdf5_file, indices=[-1], plot=True):
    """plot LNO binned level 0.1A data (before offset correction etc.)"""
    detectorData = get_dataset_contents(hdf5_file, "Y")[0]
    aotfFrequencies = get_dataset_contents(hdf5_file, "AOTFFrequency")[0]
    if not plot:
        return detectorData, aotfFrequencies

    wavenumbers = np.arange(320)
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(detectorData.shape[0])/detectorData.shape[0]]

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(FIG_X, FIG_Y))
    if indices[0] == -1:
        for spectrumIndex,spectrum in enumerate(detectorData):
            ax.plot(wavenumbers, spectrum, alpha=0.5, color=colours[spectrumIndex])
    else:
        for spectrumIndex in indices:
            ax.plot(wavenumbers, np.mean(detectorData[spectrumIndex,:,:], axis=0), alpha=1.0, color=colours[spectrumIndex])
#            ax.plot(wavenumbers, np.mean(detectorData[spectrumIndex,:,:], axis=0), alpha=1.0, color=colours[spectrumIndex])
        
    ax.set_title("Raw spectrum")
    return detectorData, aotfFrequencies




def plotOnOneFigure03A(hdf5Files, obspaths, titles):
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'ytick.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'small'})
    plt.rcParams.update({'font.size': '8'})
    fileIndices = list(range(len(obspaths)))

    fig, axesArray = plt.subplots(3, 3, figsize=(FIG_X, FIG_Y))

    for fileIndex in fileIndices:
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
        detectorData = get_dataset_contents(hdf5File, "Y")[0]
        wavenumberData = get_dataset_contents(hdf5File, "X")[0]
        aotfTemperature = np.mean(get_dataset_contents(hdf5File, "AOTF_TEMP_LNO")[0][2:10])
        exponentData = get_dataset_contents(hdf5File, "Exponent")[0]
        szaPoint0 = get_dataset_contents(hdf5File, "SunSZA", chosen_group="Geometry/Point0")[0]
        nBins = get_hdf5_attribute(hdf5File, "NBins")

        hdf5File.close()
#        nSpectra = detectorData.shape[0]
        wavenumbers = wavenumberData[0]
        exponentMin = np.min(exponentData)
        exponentMax = np.max(exponentData)
        minSza = np.min(szaPoint0)

        axesArray.flatten()[fileIndex].set_title(titles[fileIndex]+": counts vs. time")
#        plt.title(obspaths[fileIndex])
        plt.ylabel("Detector counts (no bg subtraction)")
        axesArray.flatten()[fileIndex].annotate("AOTF Temperature: %0.1fC\nNumber of Bins per Spectrum: %i\nExponent Min Max Values: %i-%i\nMinimum SZA: %0.1fdegrees"\
                     %(aotfTemperature, nBins, exponentMin, exponentMax, minSza), [0.02, 0.70], xycoords="axes fraction")
        spectrumMean = []
        for spectrum in detectorData:
            if np.mean(spectrum) > 100 and np.mean(spectrum[0:20]) < 100 and np.max(spectrum) < 800:
#            if np.mean(spectrum) > 100 and np.mean(spectrum[0:20]) < 100:
                spectrumMean.append(spectrum)
                axesArray.flatten()[fileIndex].plot(wavenumbers, spectrum, alpha=0.3)
        axesArray.flatten()[fileIndex].plot(wavenumbers, np.mean(np.asfarray(spectrumMean), axis=0), "k")
        axesArray.flatten()[fileIndex].set_ylim([-50, 500])

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

def plotEachFigure03A(hdf5Files,obspaths,titles, bad_pixel_correction=False, mean=None):
    N_SPECTRA_TO_MEAN = 20

    fileIndices = list(range(len(hdf5Files)))

    for fileIndex in fileIndices:
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
        
        if bad_pixel_correction:
            detectorData = correctBadPixelFromData(get_dataset_contents(hdf5File, "Y")[0])
        else:
            detectorData = get_dataset_contents(hdf5File, "Y")[0]
            
        if mean == "running":
            detectorData = runningMean(detectorData, N_SPECTRA_TO_MEAN)

        if mean == "discrete":
            detectorData = discreteMean(detectorData, N_SPECTRA_TO_MEAN)

        wavenumberData = get_dataset_contents(hdf5File, "X")[0]
        aotfTemperature = np.mean(get_dataset_contents(hdf5File, "AOTF_TEMP_LNO")[0][2:10])
        exponentData = get_dataset_contents(hdf5File, "Exponent")[0]
        szaPoint0 = get_dataset_contents(hdf5File, "SunSZA", chosen_group="Geometry/Point0")[0]
        nBins = get_hdf5_attribute(hdf5File, "NBins")
        hdf5File.close()
        wavenumbers = wavenumberData[0]
#        wavenumbers = range(320)
        exponentMin = np.min(exponentData)
        exponentMax = np.max(exponentData)
        minSza = np.min(szaPoint0)

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(titles[fileIndex]+": counts vs. time")
        plt.ylabel("Corrected detector counts DN per ms")
        plt.annotate("AOTF Temperature: %0.1fC\nNumber of Bins per Spectrum: %i\nExponent Min Max Values: %i-%i\nMinimum SZA: %0.1fdegrees"\
                     %(aotfTemperature, nBins, exponentMin, exponentMax, minSza), [0.02, 0.70], xycoords="axes fraction")
        spectrumMean = []
        for spectrum in detectorData:
            spectrumMean.append(spectrum)
            plt.plot(wavenumbers, spectrum, alpha=0.3)
        spectrumMean = np.asfarray(spectrumMean)
        plt.plot(wavenumbers, np.nanmean(spectrumMean, axis=0), "k")
#        plt.set_ylim([-50, 500])
        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig(obspaths[fileIndex])
    
    return wavenumbers,detectorData

def plotAnomalies03A(hdf5Files,obspaths,titles):
    fileIndices = list(range(len(hdf5Files)))

    for fileIndex in fileIndices:
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
        obsDatetime = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        detectorData = get_dataset_contents(hdf5File, "Y")[0]
        wavenumberData = get_dataset_contents(hdf5File, "X")[0]
        aotfTemperature = np.mean(get_dataset_contents(hdf5File, "AOTF_TEMP_LNO")[0][2:10])
        exponentData = get_dataset_contents(hdf5File, "Exponent")[0]
        szaPoint0 = get_dataset_contents(hdf5File, "SunSZA", chosen_group="Geometry/Point0")[0]
        nBins = get_hdf5_attribute(hdf5File, "NBins")
        hdf5File.close()
#        nSpectra = detectorData.shape[0]
        wavenumbers = wavenumberData[0]
        exponentMin = np.min(exponentData)
        exponentMax = np.max(exponentData)
        minSza = np.min(szaPoint0)

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(titles[fileIndex]+": counts vs. time")
#        plt.title(obspaths[fileIndex])
        plt.ylabel("Detector counts (no bg subtraction)")
        plt.annotate("AOTF Temperature: %0.1fC\nNumber of Bins per Spectrum: %i\nExponent Min Max Values: %i-%i\nMinimum SZA: %0.1fdegrees"\
                     %(aotfTemperature, nBins, exponentMin, exponentMax, minSza), [0.02, 0.70], xycoords="axes fraction")
        spectrumMean = []
#        annotateLoop = 0.0
        for spectrumIndex,spectrum in enumerate(detectorData):
            if np.max(spectrum) > 800 or np.min(spectrum) < -500:
#            if np.mean(spectrum) > 100 and np.mean(spectrum[0:20]) < 100:
                spectrumMean.append(spectrum)
                plt.plot(wavenumbers, spectrum, alpha=0.3, label="%s" %np.str(obsDatetime[spectrumIndex][0][0:23]))
#                annotateLoop += 0.1
#                plt.annotate("%s" %obsDatetime[spectrumIndex], [0.3, 1.0-annotateLoop], xycoords="axes fraction")
        plt.plot(wavenumbers, np.nanmean(np.asfarray(spectrumMean), axis=0), "k")
#        plt.set_ylim([-50, 500])
        plt.legend()
        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig("%s_anomalies.png"%obspaths[fileIndex])




def polynomialFit(array_in, order_in):
    arrayShape = array_in.shape
    if len(arrayShape) == 1:
        nElements = array_in.shape[0]
    elif len(arrayShape) == 2:
        nElements = array_in.shape[1]
    return np.polyval(np.polyfit(range(nElements), array_in, order_in), range(nElements))


def plotNormalised03A(hdf5_files, titles, mean_multiplier, error_offset, bad_pixel_correction=False, mean=None):
    N_STANDARD_DEVIATIONS = 2.0
    POLYNOMIAL_ORDER = 15
    N_SPECTRA_TO_MEAN = 20
    
    fileIndices = list(range(len(hdf5Files)))

    for fileIndex in fileIndices:
        hdf5File = hdf5_files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
        if bad_pixel_correction:
            detectorData = correctBadPixelFromData(get_dataset_contents(hdf5File, "Y")[0])
        else:
            detectorData = get_dataset_contents(hdf5File, "Y")[0]
            
        if mean == "running":
            detectorData = runningMean(detectorData, N_SPECTRA_TO_MEAN)

        if mean == "discrete":
            detectorData = discreteMean(detectorData, N_SPECTRA_TO_MEAN)


#        obsDatetime = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        wavenumberData = get_dataset_contents(hdf5File, "X")[0]
#        aotfTemperature = np.mean(get_dataset_contents(hdf5File, "AOTF_TEMP_LNO")[0][2:10])
#        exponentData = get_dataset_contents(hdf5File, "Exponent")[0]
#        szaPoint0 = get_dataset_contents(hdf5File, "SunSZA", chosen_group="Geometry/Point0")[0]
#        nBins = get_hdf5_attribute(hdf5File, "NBins")
        hdf5File.close()
#        nSpectra = detectorData.shape[0]
        wavenumbers = wavenumberData[0]
        
        detectorData = detectorData[:,20:]
        wavenumbers = wavenumbers[20:]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(FIG_X, FIG_Y))
        dividedSpectra = []
        for spectrum in detectorData:
            ax1.plot(wavenumbers, spectrum, label="Raw spectrum")

        good_indices = np.where(np.mean(detectorData, axis=1) > np.mean(detectorData, axis=(0,1)) * mean_multiplier)[0]
        
        for spectrum in detectorData[good_indices, :]:
            polynomialValues = polynomialFit(spectrum, POLYNOMIAL_ORDER)
            dividedSpectra.append(spectrum/polynomialValues)
            ax2.plot(wavenumbers, spectrum/polynomialValues, label="Normalised spectrum")
        
        dividedSpectra = np.asfarray(dividedSpectra)
        dividedSpectraStd = np.std(dividedSpectra, axis=0)
        
        errorBoundsMax = np.ones(len(wavenumbers))+dividedSpectraStd*N_STANDARD_DEVIATIONS
        errorBoundsMin = np.ones(len(wavenumbers))-dividedSpectraStd*N_STANDARD_DEVIATIONS
        
        polynomialErrorMax = polynomialFit(errorBoundsMax, POLYNOMIAL_ORDER)+error_offset
        polynomialErrorMin = polynomialFit(errorBoundsMin, POLYNOMIAL_ORDER)+error_offset
        ax2.fill_between(wavenumbers, polynomialErrorMin, polynomialErrorMax, alpha=0.5, label="Measured error (2 x standard deviations)")
        
        ax1.set_title(titles[fileIndex]+": LNO nadir spectra (%i frame mean) and error estimation" %N_SPECTRA_TO_MEAN)
        ax1.set_ylabel("LNO nadir detector counts")
        ax2.set_ylabel("LNO nadir normalised spectra")
        ax2.set_xlabel("Wavenumbers (cm-1)")
        plt.legend()
        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig("%s_spectra_error.png" %titles[fileIndex])


def plotIR10A(hdf5_file, hdf5_filename, plot_error=True):
    
    detectorData = get_dataset_contents(hdf5_file, "Y")[0]
    detectorError = get_dataset_contents(hdf5_file, "YError")[0]
    wavenumberData = get_dataset_contents(hdf5_file, "X")[0]
    hdf5_file.close()
    wavenumbers = wavenumberData[0]

    cmap = plt.get_cmap('plasma')
    colours = [cmap(i) for i in np.arange(detectorData.shape[0])/detectorData.shape[0]]

    if plot_error:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(FIG_X, FIG_Y))
        for spectrum in detectorData:
            ax1.plot(wavenumbers, spectrum, alpha=0.5)
        for spectrum in detectorError:
            ax2.plot(wavenumbers, spectrum, alpha=0.5)
        ax1.set_title("%s raw spectrum" %hdf5_filename)
        ax2.set_title("Spectrum error")
    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(FIG_X, FIG_Y))
        for spectrumIndex,spectrum in enumerate(detectorData):
            ax.plot(wavenumbers, spectrum, alpha=0.5, color=colours[spectrumIndex])
        ax.set_title("%s raw spectrum" %hdf5_filename)
    return wavenumbers
    

def plotSimulation(wavenumbers, molecules, offset=0.0, scalar=1.0,plot_pixels=False):
    """plot simulations from Ann Carine (needs wavenumbers input)"""
    filesToRead = findSimulations(molecules)
    #plt.figure(figsize=(FIG_X, FIG_Y))
    for fileToRead in filesToRead:
        moleculeName, moleculeWavenumbers, moleculeTransmittance, moleculeSolarTransmittance = getSimulationData(fileToRead)
        moleculeOrderWavenumbers, moleculeOrderTransmittance = getOrderSimulation(wavenumbers, moleculeWavenumbers, moleculeTransmittance)
#        moleculeOrderTransmittanceScaled = (moleculeOrderTransmittance - 1.0) / 100.0 + 0.995
        
        if plot_pixels:
            transmittancePixel = np.interp(wavenumbers, moleculeOrderWavenumbers, moleculeOrderTransmittance)
            plt.plot(range(len(wavenumbers)), transmittancePixel, label=moleculeName + " simulation", alpha=0.5)
        else:
        #    plt.plot(moleculeOrderWavenumbers, moleculeOrderTransmittanceScaled, label=moleculeName)
            plt.plot(moleculeOrderWavenumbers, moleculeOrderTransmittance*scalar+offset, label=moleculeName + " simulation")#, alpha=0.5)
    #plt.savefig("%s.png" %ax2.get_title().replace(" ","_")) 


def getCoverage(hdf5_file, hdf5_filename):
#    LNO_PIXEL_NUBER=160
    CHECK_VALID=False #check y valid flag and only return good data?
    obsTypeLetter = hdf5_filename.split("_")[5]
    
    outputDict = {}
    if CHECK_VALID:
        yValidFlag = get_dataset_contents(hdf5_file, "YValidFlag")[0][:].astype(bool)

        outputDict["lon"] = get_dataset_contents(hdf5_file, "Lon", chosen_group="Geometry/Point0")[0][yValidFlag, 0]
        outputDict["lat"] = get_dataset_contents(hdf5_file, "Lat", chosen_group="Geometry/Point0")[0][yValidFlag, 0]
        if obsTypeLetter in ["D", "N"]:
            outputDict["sza"] = get_dataset_contents(hdf5_file, "SunSZA", chosen_group="Geometry/Point0")[0][yValidFlag, 0]
        elif obsTypeLetter in ["L"]:
            outputDict["tangentalt"] = get_dataset_contents(hdf5_file, "TangentAlt", chosen_group="Geometry/Point0")[0][yValidFlag, 0]
        outputDict["ls"] = get_dataset_contents(hdf5_file, "LSubS")[0][yValidFlag, 0]
        outputDict["order"] = get_dataset_contents(hdf5_file, "DiffractionOrder")[0][yValidFlag]
    #    detector_data = get_dataset_contents(hdf5_file, "Y")[0][yValidFlag, LNO_PIXEL_NUBER]
    else:
        outputDict["lon"] = get_dataset_contents(hdf5_file, "Lon", chosen_group="Geometry/Point0")[0][:, 0]
        outputDict["lat"] = get_dataset_contents(hdf5_file, "Lat", chosen_group="Geometry/Point0")[0][:, 0]
        if obsTypeLetter in ["D", "N"]:
            outputDict["sza"] = get_dataset_contents(hdf5_file, "SunSZA", chosen_group="Geometry/Point0")[0][:, 0]
        elif obsTypeLetter in ["L"]:
            outputDict["tangentalt"] = get_dataset_contents(hdf5_file, "TangentAlt", chosen_group="Geometry/Point0")[0][:, 0]
        outputDict["ls"] = get_dataset_contents(hdf5_file, "LSubS")[0][:, 0]
        outputDict["order"] = get_dataset_contents(hdf5_file, "DiffractionOrder")[0][:]
    #    detector_data = get_dataset_contents(hdf5_file, "Y")[0][:, LNO_PIXEL_NUBER]

    return outputDict

def plotCoverageMaps(hdf5_files, hdf5_filenames, plot_type={"colour":"sza"}):
#    lonsAll = []
    obsTypeLetter = hdf5_filenames[0].split("_")[5]

    latsAll = []
    lsAll = []
    colourAll = []
    diffractionOrderAll = []
    
    for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        outputDict = getCoverage(hdf5File, hdf5Filename)
#        lonsAll.extend(outputDict["lon"])
        latsAll.extend(outputDict["lat"])
        lsAll.extend(outputDict["ls"])
        diffractionOrderAll.extend(outputDict["order"])
        
        if plot_type["colour"] == "sza":
            colourAll.extend(outputDict["sza"])
        elif plot_type["colour"] == "order":
            colourAll = diffractionOrderAll
        elif plot_type["colour"] == "tangentalt":
            colourAll.extend(outputDict["tangentalt"])

    
#    lonsArray = np.asfarray(lonsAll)
    latsArray = np.asfarray(latsAll)
    lsArray = np.asfarray(lsAll)
    colourArray = np.asfarray(colourAll)
    diffractionOrderArray = np.asfarray(diffractionOrderAll)
    
    fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))

    if obsTypeLetter in ["D"]:
        diffractionOrderString = "Order %i" %diffractionOrderArray[0]
        obsTypeString = "LNO Dayside"
    elif obsTypeLetter in ["N"]:
        diffractionOrderString = "All Orders"
        obsTypeString = "LNO Nightside"
    elif obsTypeLetter in ["L"]:
        diffractionOrderString = "All Orders"
        obsTypeString = "LNO Limb"

    size=10
    alpha=0.6
    if plot_type["colour"] == "sza":
        colourbarLabel = "Solar Zenith Angle (deg)"
    elif plot_type["colour"] == "order":
        ax.set_ylim([-5,5])
        size=20
        alpha=1.0
        colourbarLabel = "Diffraction Order"
        diffractionOrderString = "All Orders"
    elif plot_type["colour"] == "tangentalt":
        colourbarLabel = "Tangent Altitude (km)"

    plot = ax.scatter(lsArray, latsArray, c=colourArray, cmap=plt.cm.jet, marker='o', s=size, linewidth=0, alpha=alpha)
    cbar = fig.colorbar(plot)
    cbar.set_label(colourbarLabel, rotation=270, labelpad=20)
        
    
    ax.set_xlabel("Ls (deg)")
    ax.set_ylabel("Latitude (deg)")
    
    
    ax.set_title("%s %s (%i files)" %(obsTypeString, diffractionOrderString, len(hdf5_filenames)))
    
    if SAVE_FIGS:
        plt.savefig(BASE_DIRECTORY + os.sep + "%s_%s.png" %(obsTypeString.replace(" ","_"), diffractionOrderString.replace(" ","_")))
#        plt.savefig(BASE_DIRECTORY + os.sep + "LNO_night_coverage_vs_SZA_order_%i.png" %diffractionOrderArray[0])
#        plt.savefig(BASE_DIRECTORY + os.sep + "LNO_coverage_vs_tangent_altitude_order_%i.png" %diffractionOrderArray[0])



#####BEGIN SCRIPT########
#molecules = ["CO2real","H2Oreal","CH4","HDO","CO"]
#hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
#hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes)
#hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes, search_datasets_min_max=searchDatasetsMinMax)
#hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_datasets_min_max=searchDatasetsMinMax)
#plotEachFigure03A(hdf5Files, hdf5Filenames, titles)
#plotEachFigure03A(hdf5Files, hdf5Filenames, titles, bad_pixel_correction=True, mean="running")


#for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames):
#    wavenumbers = plotIR10A(hdf5File,hdf5Filename, plot_error=False)
#    wavenumbers = plotIR10A(hdf5File, hdf5Filename, plot_error=True)
#    plt.savefig(BASE_DIRECTORY+os.sep+"%s.png" %hdf5Filename) 
#plotSimulation(wavenumbers, molecules, offset=0.0, scalar=250.0)

#wavenumbers, detector_data = plotEachFigure03A(hdf5Files, hdf5Filenames, titles, bad_pixel_correction=True, mean="discrete")
#plotAnomalies03A(hdf5Files, hdf5Filenames, titles)

#plotNormalised03A(hdf5Files, titles, 1.0, 0.015, bad_pixel_correction=True, mean="discrete")




"""LNO detector offset images"""
#hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
#detectorData, aotfFrequencies = plot01ASpectra(hdf5Files[0], plot=False)
#detectorDataBinned = np.mean(detectorData[np.arange(201,310,3),:,:], axis=1)
#plt.figure(figsize=(FIG_X-2, FIG_Y-2))
#plt.plot(runningMean(detectorDataBinned.T, 5))
#plt.title(obsTitles[0])
#plt.xlabel("Pixel Number")
#plt.ylabel("Detector Counts")
#
#offsets = np.mean(detectorDataBinned[:,0:50], axis=1)
#plt.figure(figsize=(FIG_X-2, FIG_Y-2))
#plt.plot(runningMean(detectorDataBinned.T, 5) - offsets)
#plt.title("LNO Detector Offsets\nAfter Subtraction of First 50 Pixel Offset")
#plt.xlabel("Pixel Number")
#plt.ylabel("Detector Counts")
#
#plt.figure



"""LNO coverage map"""
#if title == "LNO Coverage Map":
#    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, silent=True)
#    plotCoverageMaps(hdf5Files, hdf5Filenames, plot_type={"colour":"sza"})



"""LNO radiances versus TES albedo"""
if title == "radiance albedo":
    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
    
        
    
    #read in TES file
    im = Image.open(os.path.join(BASE_DIRECTORY,"reference_files","Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"))
    albedoMap = np.array(im)
    
    fig1, ax1 = plt.subplots(1)
    albedoPlot = plt.imshow(albedoMap, extent = [-180,180,-90,90])
    cbar = plt.colorbar(albedoPlot)
    colorbarLabel = "TES Albedo"
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
    
    
    mapLons = np.arange(-180.0, 180.0, 0.125)
    mapLats = np.arange(-90.0, 90.0, 0.125)
    
    def getAlbedo(lons_in, lats_in, albedo_map):
        lonIndexFloat = np.asarray([int(np.round((180.0 + lon) * 8.0)) for lon in lons_in])
        latIndexFloat = np.asarray([int(np.round((90.0 - lat) * 8.0)) for lat in lats_in])
        lonIndexFloat[lonIndexFloat==2880] = 0
        latIndexFloat[latIndexFloat==1440] = 0
        albedos_out = np.asfarray([albedo_map[lat, lon] for lon, lat in zip(lonIndexFloat, latIndexFloat)])
        return albedos_out
    
    
    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
    
    
    xAxis = "SZA"
    yAxis = "Signal"
    cAxis = "Albedo"
    #cAxis = "Time"
    
    fig2, ax2 = plt.subplots(1)
    for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
        yValidFlag = get_dataset_contents(hdf5_file, "YValidFlag")[0][:].astype(bool)
        lon = get_dataset_contents(hdf5_file, "Lon", chosen_group="Geometry/Point0")[0][yValidFlag, :]
        lat = get_dataset_contents(hdf5_file, "Lat", chosen_group="Geometry/Point0")[0][yValidFlag, :]
        
        lonStart = lon[:,0]
        lonEnd = lon[:,1]
        latStart = lat[:,0]
        latEnd = lat[:,1]
    
        y, yError, snr = doRadiometricCalibration(hdf5_file, hdf5_filename)
        
        y = get_dataset_contents(hdf5_file, "Y")[0][yValidFlag, :]
        sza = get_dataset_contents(hdf5_file, "SunSZA", chosen_group="Geometry/Point0")[0][yValidFlag, 0]
        et = get_dataset_contents(hdf5_file, "ObservationEphemerisTime")[0][yValidFlag, 0]
        
        albedos = getAlbedo(lonStart, latStart, albedoMap)
    #    albedos = getAlbedo(lonEnd, latEnd, albedoMap)
        
        signal = np.mean(y[:,160:240], axis=1)
    #    ax1.scatter(lonStart, latStart, c=signal, cmap=plt.cm.jet, marker='o', linewidth=0)
        ax1.scatter(lonStart, latStart, c=signal, cmap=plt.cm.jet, marker='o', linewidth=0)
        
    #    for spectrumIndex, spectrum in enumerate(y):
    #        szaPlot = plt.plot(range(320), spectrum/signal[spectrumIndex], c=plt.cm.jet(sza[spectrumIndex]/90.0), alpha=0.2)
    
    #    szaPlot = ax2.scatter(et, signal*1e6, c=sza, cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.2)
    #    szaPlot = ax2.scatter(et, albedos, c=signal*1e6, cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.2)
        
    #    validIndices = albedos>0.3
        validIndices = albedos>0.0
        
        
        if sum(validIndices) > 0:
    
            plotDict = {"SZA":sza[validIndices], "Signal":signal[validIndices], "Albedo":albedos[validIndices], "Time":et[validIndices]}
            labelDict = {"SZA":"Solar zenith angle", "Signal":"Radiance (approx. calibration)", "Albedo":"TES Albedo", "Time":"Ephemeris time"}
        
        #    szaPlot = ax2.scatter(plotDict[xAxis][albedos>0.3], plotDict[yAxis][albedos>0.3], c=plotDict[cAxis][albedos>0.3], cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.7)
            szaPlot = ax2.scatter(plotDict[xAxis], plotDict[yAxis], c=plotDict[cAxis], cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.7)
    
    cbar = plt.colorbar(szaPlot)
    
    ax2.set_xlabel(labelDict[xAxis])
    ax2.set_ylabel(labelDict[yAxis])
    cbar.set_label(labelDict[cAxis], rotation=270, labelpad=20)
    
    #mapLons = np.arange(-180.0, 180.0, 1.0)
    #mapLats = np.arange(-90.0, 90.0, 1.0)
    #mapAlbedo = np.ones((len(mapLats), len(mapLons)))
    #for lonPoint in mapLons:
    #    mapAlbedo[:,int(lonPoint)] = getAlbedo((np.ones(180) * lonPoint), mapLats, albedoMap) 
    #
    #plt.figure()
    #plt.imshow(mapAlbedo, extent = [-180,180,-90,90])    
        
        
if title == "UVIS Coverage Map":

    plot = False
#    plot = True

#    name = "uvis_griddedBins5x5_apr_may"
    name = "uvis_griddedBins10x10_apr_may"
#    region1Lon = [-10.0, -5.0]
#    region1Lat = [-25.0, -20.0]
    region1Lon = [-12.0, -2.0]
    region1Lat = [-25.0, -15.0]
    region1Name = "Dark region"

#    region2Lon = [-150.0, -145.0]
#    region2Lat = [-25.0, -20.0]
    region2Lon = [-145.0, -135.0]
    region2Lat = [8.0, 18.0]
    region2Name = "Light region"

    if not plot:
        import h5py

        hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, open_files=False)

        gridLat = 5.0
        gridLon = 5.0
        gridLons = np.arange(-180.0, 180.0, gridLon)
        gridLats = np.arange(-90.0, 90.0, gridLat)
        griddedBins = np.zeros([len(gridLons), len(gridLats)]) + np.nan

        region1Filenames = []
        region2Filenames = []
        
        acceptedFilenames = []
        acceptedFrames = []
        nAccepted = 0
        nRejected = 0
        nGoodFramesAccepted = 0
        nGoodFramesRejected = 0
        for file_index, (hdf5_filepath, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
            if "201804" in hdf5_filename or "201805" in hdf5_filename:
#            if "201811" in hdf5_filename or "201812" in hdf5_filename:
                hdf5_file = h5py.File(hdf5_filepath, "r")
                hStarts = hdf5_file["Channel/HStart"][...]
                hStart = hStarts[0]
                hEnd = hdf5_file["Channel/HEnd"][0]
                nFrames = len(hStarts)
    #            vStart = hdf5_file["Channel/VStart"][0]
    #            vEnd = hdf5_file["Channel/VEnd"][0]

                if hStart == 0 and hEnd == 1047: #check full spectrum
                    incidenceAngleIn = hdf5_file["Geometry/Point0/IncidenceAngle"][...]
                    minIncidenceAngle = np.min(incidenceAngleIn)

                    if minIncidenceAngle < 80: #check incidence angle
                        lonsIn = hdf5_file["Geometry/Point0/Lon"][...]
                        latsIn = hdf5_file["Geometry/Point0/Lat"][...]
                        if np.all(lonsIn > -900) and np.all(latsIn > -900): #check all points on planet
                            flagRegister = hdf5_file["Channel/ReverseFlagAndDataTypeFlagRegister"][...]
                            yPeak = hdf5_file["Science/Y"][:, 120:220, 800:1000]
                            yMax = np.max(yPeak, axis=(1,2))
                            validIndices = [True if (flag == 4 and pixel < 65535) else False for flag,pixel in zip(flagRegister, yMax)]
                            lonsMean = np.mean(lonsIn[validIndices], axis=1)
                            latsMean = np.mean(latsIn[validIndices], axis=1)

                            nSaturated = len(validIndices) - len(lonsMean)
                            nGoodFramesAccepted += len(lonsMean)
                            nGoodFramesRejected += nSaturated

                            if len(lonsMean) == 0:
                                nRejected += 1
                                print("File %i rejected (all frames saturated)" %file_index)
                            else:
                                nAccepted += 1
                                print("File %i %s accepted: %i/%i frames valid" %(file_index, hdf5_filename, len(lonsMean), len(validIndices)))
                                acceptedFilenames.append(hdf5_filename)
                                acceptedFrames.append(validIndices)
                                gridIndex = [[int(np.floor((lonMean+180.0)/gridLon)), int(np.floor((90.0-latMean)/gridLat))] for lonMean,latMean in zip(lonsMean, latsMean)]
                                for lonIndex, latIndex in gridIndex:
                                    if np.isnan(griddedBins[lonIndex, latIndex]):
                                        griddedBins[lonIndex, latIndex] = 0
                                    else:
                                        griddedBins[lonIndex, latIndex] += 1.0
                                
                                for lon, lat in zip(lonsMean, latsMean):
                                    if region1Lon[0] < lon < region1Lon[1]:
                                        if region1Lat[0] < lat < region1Lat[1]:
                                            print("%s: %s, (%0.1f, %0.1f)" %(region1Name, hdf5_filename, lon, lat))
                                            region1Filenames.append(hdf5_filename)
                                    if region2Lon[0] < lon < region2Lon[1]:
                                        if region2Lat[0] < lat < region2Lat[1]:
                                            print("%s: %s, (%0.1f, %0.1f)" %(region2Name, hdf5_filename, lon, lat))
                                            region2Filenames.append(hdf5_filename)
                                
                                
                        else:
                            nRejected += 1
                            print("File %i rejected (not pointing to planet)" %file_index)

                    else:
                        nRejected += 1
                        text = "File %i rejected: " %file_index
                        if not minIncidenceAngle < 80:
                            text += "minIncidenceAngle = %0.1f, " %minIncidenceAngle
                        print(text)

                else:
                    nRejected += 1
                    nGoodFramesRejected += nFrames
                    text = "File %i rejected (%i frames): " %(file_index, nFrames)
                    if hStart != 0:
                        text += "HStart=%i, " %hStart
                    if hEnd != 1047:
                        text += "HEnd=%i, " %hEnd
                    print(text)
        print("Files accepted = %i, files rejected = %i" %(nAccepted, nRejected))
        print("Frames accepted = %i, frames rejected (not including off-nadir or high incidence angles) = %i" %(nGoodFramesAccepted, nGoodFramesRejected))
        np.savetxt("griddedBins5x5_apr_may.txt", griddedBins, fmt="%0.0f")
#        np.savetxt("griddedBins5x5_nov_dec.txt", griddedBins, fmt="%0.0f")
        
        print(region1Name)
        for filename in region1Filenames:
            print(filename)
        print(region2Name)
        for filename in region2Filenames:
            print(filename)

    else:
        from matplotlib.patches import Rectangle
        #read in TES file
        im = Image.open(os.path.join(BASE_DIRECTORY,"reference_files","Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"))
        albedoMap = np.array(im)

        fig1, ax1 = plt.subplots(1, figsize=(18,10))
        albedoPlot = plt.imshow(albedoMap, extent = [-180,180,-90,90], cmap="binary_r")
        griddedBinsIn = np.loadtxt(name+".txt")
#        griddedBinsIn = np.loadtxt("griddedBins5x5_nov_dec.txt")
        plt.imshow(griddedBinsIn.T, interpolation=None, alpha=0.6, extent=[-180, 180, -90, 90])
        ax1.add_patch(Rectangle((region1Lon[0], region1Lat[0]), 5, 5, linewidth=2,edgecolor='r',facecolor='none'))
        ax1.add_patch(Rectangle((region2Lon[0], region2Lat[0]), 5, 5, linewidth=2,edgecolor='r',facecolor='none'))
        plt.title("UVIS 5x5deg coverage for April & May 2018 for all unsaturated frames \n Unbinned full spectrum measurements where min incidence angle in file < 80 degrees")
#        plt.title("UVIS 5x5deg coverage for November & December 2018 for all unsaturated frames \n Unbinned full spectrum measurements where min incidence angle in file < 80 degrees")
        cbar = plt.colorbar()
        colorbarLabel = "Number of frames in each bin"
        cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
        plt.savefig(name+".png")