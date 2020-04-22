# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Thu Feb  7 14:43:38 2019

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
    DATASTORE_DIRECTORY = r"/ae/projects4/NOMAD/Data/db_test/test/iant/hdf5" #FOR DB_TEST FOLDER. TANGENT ALTIUTUDES ARE NOT CORRECT IN OFFICIAL DATA!!


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
searchDatasetsMinMax = [["SunSZA","min","lt",5.0,"Geometry/Point0"]]
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes, search_datasets_min_max=searchDatasetsMinMax)

E.G. FIND FILES WHERE THE MAXIMUM Geometry/Point0/Lat IS GREATER THAN 25.0 DEGREES
searchDatasetsMinMax = [["Lat","max","gt",25.0,"Geometry/Point0"]]
hdf5Files,hdf5Filenames,titles = makeFileList(obspaths,fileLevel, search_attributes=searchAttributes, search_datasets_min_max=searchDatasetsMinMax)

NOTE THAT A FILE WILL ONLY BE RETURNED IF ALL ATTRIBUTES AND DATASETS SATISFY THE GIVEN CONDITIONS!


5. PLOT ALL NORMALISED FILES + TOP OF ATMOSPHERE ERROR UNCERTAINTY AT ALTITUDES OF 150, 160, 175 AND 200KM USING THE COMMAND:
plotIRNormalised03A(hdf5Files, hdf5Filenames, titles, [150.0,160.0,175.0,200.0], bg_subtracted=True)
SET THE ARGUMENT bg_subtracted IF FILES HAVE ONBOARD BACKGROUND SUBTRACTION.


6. TO SAVE FIGURES, SET SAVE_FIGS = True

"""


import os
#import h5py
import numpy as np
#import numpy.linalg as la
#import gc
#from scipy import stats
#import scipy.optimize

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

from hdf5_functions_v03 import get_dataset_contents
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, getFile, makeFileList#, printFileNames
from analysis_functions_v01b import write_log
from filename_lists_v01 import getFilenameList

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
    print("Running on windows")
    import spiceypy as sp



#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


"""plot dust vertical profiles"""
#title = "dust vertical profiles"
title = "dust altitude vs ls"
fileLevel = "hdf5_level_0p3a"
#obspaths = getFilenameList("dust order 121")
obspaths = getFilenameList("dust order 121 apr-sep")
#obspaths = getFilenameList("dust order 134")
#obspaths = getFilenameList("dust order 190 apr-sep")

#obspaths = ["*201804*_0p3a_SO*_121"]

#obspaths = ["20180423_153405_0p3a_SO_1_E_121"]
#20180502_133902_0p3a_SO_1_E_121
#obspaths = ["20180628_130836_0p3a_SO_1_I_121"] #odd jumps (bad bin?)



"""make CH4 corrected SNR spectra for paper"""
#fileLevel = "hdf5_level_0p3a"

#title = "ch4 paper 134 apr"
#title = "ch4 paper 134 aug"
#title = "ch4 paper 135 aug"
#title = "detector correction"
#obspaths = getFilenameList(title)

"""write h20 ch4 simulations to files"""
#title == "make simulations"""


"""plot occultation map"""
#title = "occultation map"
#fileLevel = "hdf5_level_0p3a"
#obspaths = ["*20180*_0p3a_SO*_134"]


if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
    #load spiceypy kernels if required
    KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
    #KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
    METAKERNEL_NAME = "em16_plan_win.tm"
    METAKERNEL_NAME = "em16_ops_win.tm"
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)


def writeOutput(file_name, lines_to_write):
    """function to write output to a log file"""
#    global OUTPUT_PATHS
    outFile = open(BASE_DIRECTORY + os.sep + file_name, 'w')
    for line_to_write in lines_to_write:
        outFile.write(line_to_write+'\n')
    outFile.close()
#    print(line_to_write)



def getCorrectLST(hdf5_file, silent=False):
#    import spiceypy as sp

    def etToLSTHours(et, lon): #lon in degrees
        lst_spice = sp.et2lst(et, 499, lon/sp.dpr(), "PLANETOCENTRIC")
        lst_hours = lst_spice[0] + lst_spice[1] / 60.0 + lst_spice[2] / 3600.0
        return lst_hours

    et = hdf5_file["Geometry/ObservationEphemerisTime"][:,0]
    et_mean = np.mean(et[et > -999.0])
    et_min = et[0]
    et_max = et[-1]
#    if not silent: print("et_mean=%0.1f" %et_mean)

    lon = hdf5_file["Geometry/Point0/Lon"][:,0]
    lon_mean = np.mean(lon[lon > -999.0])
    lon_min = lon[lon > -999.0][0]
    lon_max = lon[lon > -999.0][-1]
    if not silent: print("lon=%0.1f to %0.1f (mean %0.1f)" %(lon_min, lon_max, lon_mean))

    lst_mean = etToLSTHours(et_mean, lon_mean)
    lst_min = etToLSTHours(et_min, lon_min)
    lst_max = etToLSTHours(et_max, lon_max)
    if lst_mean < 12.0:
        time_out = "am"
    else:
        time_out = "pm"
    
    if not silent: print("LST=%0.1f to %0.1f (mean %0.1f) => %s terminator" %(lst_min, lst_max, lst_mean, time_out))

    return lst_mean, time_out


def getSlantAirmass(altitude_in):
    COLUMN = 2
    slantPathFile = os.path.join(BASE_DIRECTORY, "reference_files", "slant_path_airmass_GV_2018.txt")
    slantPathData_in = np.loadtxt(slantPathFile)
    slantPathAltitudes = slantPathData_in[:,0]

    lowerIndex = np.abs(altitude_in - slantPathAltitudes).argmin()
    upperIndex = lowerIndex + 1
    
    lowerAltitude = slantPathAltitudes[lowerIndex]
    upperAltitude = slantPathAltitudes[upperIndex]
    lowerSlantPath = slantPathData_in[lowerIndex, COLUMN]
    upperSlantPath = slantPathData_in[upperIndex, COLUMN]
    
    ratioAltitude = (altitude_in - lowerAltitude) / (upperAltitude - lowerAltitude)    
    ratioSlantPath = lowerSlantPath + ratioAltitude * (upperSlantPath - lowerSlantPath)
    
    return ratioSlantPath
   

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
#    elif len(arrayShape) == 2:
#        nElements = array_in.shape[1]
    
    return np.polyval(np.polyfit(range(nElements), array_in, order_in), range(nElements))






def getContinuumFromSpectrum(row_in, continuum_type):
    """remove continuum from spectrum and output continuum and normalised spectrum. Don't use on very lowest SNR spectra"""
    N_PIXELS_ABSORPTION = 1
    POLYNOMIAL_ORDER1 = 3
    POLYNOMIAL_ORDER2 = 4
    plot_figs = False
    
    def findLocalMinima(row_in):
        localMinima = (np.diff(np.sign(np.diff(row_in))) > 0).nonzero()[0] + 1 # local min
        return localMinima
    
    
    
    def findLocalMaxima(row_in):
        localMaxima = (np.diff(np.sign(np.diff(row_in))) < 0).nonzero()[0] + 1 # local min
        return localMaxima
    
    if continuum_type in "TR":
        localMinMaxIndices = findLocalMaxima(row_in)
    elif continuum_type in ["OD","OP"]:
        localMinMaxIndices = findLocalMinima(row_in)

    polyfit_spectrum = polynomialFit(row_in, POLYNOMIAL_ORDER1)
    if plot_figs:
        plt.plot(polyfit_spectrum)

    
    contPixel=[]
    contTransmittance=[]
    for localMinMaxIndex,localMinMaxPixel in enumerate(localMinMaxIndices):
        trueContinuum = True
        if plot_figs:
            plt.scatter(localMinMaxPixel, row_in[localMinMaxPixel], color="g", marker="x", alpha=0.5)
        for adjacentPixel in range(localMinMaxPixel-N_PIXELS_ABSORPTION, localMinMaxPixel+N_PIXELS_ABSORPTION+1, 1):
            if continuum_type in "TR":
                if row_in[adjacentPixel]<polyfit_spectrum[adjacentPixel]:
                    trueContinuum = False
            elif continuum_type in ["OD","OP"]:
                if row_in[adjacentPixel]>polyfit_spectrum[adjacentPixel]:
                    trueContinuum = False
        if trueContinuum:
            contPixel.append(localMinMaxPixel)
            contTransmittance.append(row_in[localMinMaxPixel])
            
    if len(contPixel)>1:
        lastPixel = len(row_in)-1
        contPixel.append(lastPixel)
        contTransmittance.append(row_in[lastPixel])
        if plot_figs:
            plt.scatter(contPixel, contTransmittance, color="b")
        try: 
            fit = np.polyfit(contPixel, contTransmittance, POLYNOMIAL_ORDER2)
        except np.RankWarning:
            return np.zeros_like(row_in)
        
        continuum = np.polyval(fit, range(row_in.shape[0]))
        if plot_figs:
            plt.plot(continuum)
    else:
        print("Continuum fitting failed")
        continuum = row_in
    return continuum#, spectrumOut #return continuum and continuum-removed spectrum






###3RD VERSION: DO FULL TRANSMITTANCE CALCULATION AND PLOT DUST PROFILES FROM THAT#####

def joinTransmittanceFiles(hdf5_file, hdf5_filename, bin_index, silent=False, top_of_atmosphere=100.0):
    """read in lower alt file and join to high alt file if present.
    If background not subtracted, read in dark file and calibrate to zero level (ingress/egress only)
    """
    
    TOP_OF_ATMOSPHERE = top_of_atmosphere #above this altitude is sun only. Plot polynomial sun spectrum over this range
    TEST_ALTITUDE = top_of_atmosphere #km. check extrapolation at this altitude to define error
#    MAX_PERCENTAGE_ERROR = 5.0 #% if transmittance error greater than this then discard file
    MAX_PERCENTAGE_ERROR = 1.0 #% if transmittance error greater than this then discard file
    NBINS = 4
#    NBINS = 12
    POLYFIT_DEGREE = 3
    PIXEL_NUMBER_START = 190 #centre range of detector
    PIXEL_NUMBER_END = 210 #centre range of detector
    PIXEL_NUMBER = 160 #centre of detector
    
    DETECTOR_DATA_FIELD = "Y"
    ALTITUDE_FIELD = "TangentAltAreoid"
    
    CHOSEN_MERGED_FILE = "Ingress"
    
    #get info from filename
    obspath_split = hdf5_filename.split("_")
    date = obspath_split[0]
    time = obspath_split[1]
    level = obspath_split[2]
    channel = obspath_split[3]
    science_number = obspath_split[4]
    obs_type = obspath_split[5]
    diffraction_order = obspath_split[6]
    file_level = "hdf5_level_" + level
    
#    if (science_number == "1" and obs_type == "E") or (science_number == "2" and obs_type == "I"):
#        if not silent: print("File ok")
#    else:
#        print("Warning: potential obs type (%s) / science number (%s) mismatch" %(obs_type, science_number))
#        stop()

    if not silent: print("Reading in IR file %s" %hdf5_filename)
    
    outputDict = {"error":False}
    if level != "0p1a":
        alt_point0_low = np.mean(get_dataset_contents(hdf5_file, ALTITUDE_FIELD, chosen_group="Geometry/Point0")[0], axis=1)
        lat_point0_low = np.mean(get_dataset_contents(hdf5_file, "Lat", chosen_group="Geometry/Point0")[0], axis=1)
        lon_point0_low = np.mean(get_dataset_contents(hdf5_file, "Lon", chosen_group="Geometry/Point0")[0], axis=1)
    
        nSpectraLow = np.histogram(alt_point0_low, bins=[10.,40.])[0][0]
        nSpectraHigh = np.histogram(alt_point0_low, bins=[60.,90.])[0][0]
        if not silent: print("nSpectra 10-40km = %i" %nSpectraLow)
        if not silent: print("nSpectra 60-90km = %i" %nSpectraHigh)
    
        if nSpectraLow < 40:
            print("Error: %s is not a low altitude file" %hdf5_filename)
            outputDict = {"error":True}
            
        elif np.min(alt_point0_low) != -999.0:
            print("Error: %s doesn't go down to the surface" %hdf5_filename)
            outputDict = {"error":True}
    #    elif (alt_point0_low[0] > 250.0) and (alt_point0_low[-1] > 250.0):
    #        
    #        print("%s is a merged or grazing occultation" %hdf5_filename)
    #        outputDict = {"error":True}
        else:
            #check if grazing occultation
            if (alt_point0_low[0] > 250.0) and (alt_point0_low[-1] > 250.0):
                print("Warning: %s is a merged file" %hdf5_filename)
                merged = True
                mergedCutoff = int(len(alt_point0_low)/2) #just split in half
            else:
                merged=False
                mergedCutoff = 0
    else: #if 0p1a data
        merged=False
        mergedCutoff = 0
        nSpectraHigh = 100
        alt_point0_low = np.zeros(4)
        lat_point0_low = np.zeros(4)
        lon_point0_low = np.zeros(4)

    if not outputDict["error"]:
        """check for background subtraction - if not, find file and do it manually"""
    
        sbsf = get_dataset_contents(hdf5_file,"BackgroundSubtraction")[0][0] #find sbsf flag (just take first value)
        if sbsf == 0:
            if diffraction_order == "0":
                print("Error: %s is a dark file" %hdf5_filename)
                stop()
            else:
                if not silent: print("Subtracting background (low alt)")
                hdf5_filename_dark = "%s_%s_%s_%s_%s_%s_%s" %(date, time, level, channel, science_number, obs_type, "0")
                _, hdf5_file_dark = getFile(hdf5_filename_dark, file_level, 0, silent=silent)
                detector_data_dark = get_dataset_contents(hdf5_file_dark, DETECTOR_DATA_FIELD)[0]
                hdf5_file_dark.close()
                
                detector_data_light = get_dataset_contents(hdf5_file, DETECTOR_DATA_FIELD)[0]
                
                if detector_data_light.shape[0] == (detector_data_dark.shape[0] * 5):
                    print("File has 5x light frames than dark")
                    n_pixels = len(detector_data_light[0, :])
                    detector_data_dark = np.reshape(np.repeat(np.reshape(detector_data_dark, [-1, NBINS*n_pixels]), 5, axis=0), [-1, n_pixels])
                    
                
                detector_data_low = detector_data_light - detector_data_dark
                detector_dark_low = detector_data_dark
        elif sbsf == 1:
            if not silent: print("Background already subtracted (low alt)")
            detector_data_low = get_dataset_contents(hdf5_file, DETECTOR_DATA_FIELD)[0]
            detector_dark_low = np.zeros_like(detector_data_low)
    
        """check for high altitude file"""
        if nSpectraHigh < 40 and not merged: #not for 0p1a files
            if science_number == "1":
                science_number_high = "2"
            elif science_number == "2":
                science_number_high = "1"
                
            if not silent: print("Finding high altitude file")
            hdf5_filename_high = "%s_%s_%s_%s_%s_%s_%s" %(date, time, level, channel, science_number_high, obs_type, diffraction_order)
            _, hdf5_file_high = getFile(hdf5_filename_high, file_level, 0, silent=silent)
            sbsf_high = get_dataset_contents(hdf5_file_high,"BackgroundSubtraction")[0][0] #find sbsf flag (just take first value)
    
            if sbsf_high == 0:
                if diffraction_order == "0":
                    print("Error: %s is a dark file" %hdf5_filename)
                    stop()
                else:                
                    if not silent: print("Subtracting background (high alt)")
                    hdf5_filename_high_dark = "%s_%s_%s_%s_%s_%s_%s" %(date, time, level, channel, science_number_high, obs_type, "0")
                    _, hdf5_file_high_dark = getFile(hdf5_filename_high_dark, file_level, 0, silent=silent)
                    detector_data_high_dark = get_dataset_contents(hdf5_file_high_dark, DETECTOR_DATA_FIELD)[0]
                    hdf5_file_high_dark.close()
                    
                    detector_data_high_light = get_dataset_contents(hdf5_file_high, DETECTOR_DATA_FIELD)[0]
                    detector_data_high = detector_data_high_light - detector_data_high_dark
            elif sbsf == 1:
                if not silent: print("Background already subtracted (high alt)")
                detector_data_high = get_dataset_contents(hdf5_file_high, DETECTOR_DATA_FIELD)[0]
                detector_data_high_dark = np.zeros_like(detector_data_high)
    
            alt_point0_high = np.mean(get_dataset_contents(hdf5_file_high, ALTITUDE_FIELD, chosen_group="Geometry/Point0")[0], axis=1)
    
    #        get lat/lons
            lat_point0_high = np.mean(get_dataset_contents(hdf5_file_high, "Lat", chosen_group="Geometry/Point0")[0], axis=1)
            lon_point0_high = np.mean(get_dataset_contents(hdf5_file_high, "Lon", chosen_group="Geometry/Point0")[0], axis=1)
    
    
            detector_data_all = np.concatenate((detector_data_low, detector_data_high))
            detector_dark_all = np.concatenate((detector_data_dark, detector_data_high_dark))
            alt_point0_all = np.concatenate((alt_point0_low, alt_point0_high))
            lat_point0_all = np.concatenate((lat_point0_low, lat_point0_high))
            lon_point0_all = np.concatenate((lon_point0_low, lon_point0_high))
        else:
            if not merged:
                detector_data_all = detector_data_low
                detector_dark_all = detector_dark_low
                alt_point0_all = alt_point0_low
                lat_point0_all = lat_point0_low
                lon_point0_all = lon_point0_low
            elif CHOSEN_MERGED_FILE == "Ingress":
                detector_data_all = detector_data_low[mergedCutoff:,:]
                detector_dark_all = detector_dark_low[mergedCutoff:,:]
                alt_point0_all = alt_point0_low[mergedCutoff:]
                lat_point0_all = lat_point0_low[mergedCutoff:]
                lon_point0_all = lon_point0_low[mergedCutoff:]
            elif CHOSEN_MERGED_FILE == "Egress":
                detector_data_all = detector_data_low[:mergedCutoff,:]
                detector_dark_all = detector_dark_low[:mergedCutoff,:]
                alt_point0_all = alt_point0_low[:mergedCutoff]
                lat_point0_all = lat_point0_low[:mergedCutoff]
                lon_point0_all = lon_point0_low[:mergedCutoff]
              
        """manually correct remaining bad pixels"""
        if bin_index == 0:
            detector_data_all[:,84] = np.mean(detector_data_all[:,[83,85]], axis=1)
            detector_data_all[:,124] = np.mean(detector_data_all[:,[123,125]], axis=1)
            detector_data_all[:,269] = np.mean(detector_data_all[:,[268,270]], axis=1)
        elif bin_index == 2:
            detector_data_all[:,256] = np.mean(detector_data_all[:,[255,257]], axis=1)
    
    
    
        #get data for desired bin only
        detector_data_split = splitIntoBins(detector_data_all, NBINS)[bin_index]
        detector_dark_split = splitIntoBins(detector_dark_all, NBINS)[bin_index]
        alt_point0_split = splitIntoBins(alt_point0_all, NBINS)[bin_index]
        lat_point0_split = splitIntoBins(lat_point0_all, NBINS)[bin_index]
        lon_point0_split = splitIntoBins(lon_point0_all, NBINS)[bin_index]
        
        detector_data = detector_data_split[alt_point0_split > -100.0, :]
        detector_dark = detector_dark_split[alt_point0_split > -100.0, :]
        alt_point0 = alt_point0_split[alt_point0_split > -100.0]
        lat_point0 = lat_point0_split[alt_point0_split > -100.0]
        lon_point0 = lon_point0_split[alt_point0_split > -100.0]
        
        sort_indices = np.argsort(alt_point0)
        
        detector_data_sorted = detector_data[sort_indices, :]
        detector_dark_sorted = detector_dark[sort_indices, :]
        alt_point0_sorted = alt_point0[sort_indices]
        lat_point0_sorted = lat_point0[sort_indices]
        lon_point0_sorted = lon_point0[sort_indices]
    
    
        
    #    find start/end lat/lons for label
        index_50 = np.abs(alt_point0_sorted - 50.0).argmin()
        index_0 = np.abs(alt_point0_sorted - 0.0).argmin()
        lat_range = [lat_point0_sorted[index_50], lat_point0_sorted[index_0]]        
        lon_range = [lon_point0_sorted[index_50], lon_point0_sorted[index_0]]        
        
        #get indices for top of atmosphere
        top_indices = np.where(alt_point0_sorted>TOP_OF_ATMOSPHERE)[0]
        if len(top_indices) < 10:
            print("Error: Insufficient points above %i. n points = %i" %(TOP_OF_ATMOSPHERE, len(top_indices)))
            print(hdf5_filename)
            stop()
            
        detector_data_sun = detector_data_sorted[top_indices,:]
    
        detector_data_sun_mean = np.mean(detector_data_sun, axis=0)
    
        detector_data_polyfit = np.polyfit(alt_point0_sorted[top_indices], np.mean(detector_data_sun[:, PIXEL_NUMBER_START:PIXEL_NUMBER_END], axis=1)/ np.mean(detector_data_sun_mean[PIXEL_NUMBER_START:PIXEL_NUMBER_END]), POLYFIT_DEGREE)
        detector_data_polyval = np.polyval(detector_data_polyfit, alt_point0_sorted)
    
        detector_data_extrap = np.zeros_like(detector_data_sorted)
        for pixel_number in range(320):
            detector_data_extrap[:,pixel_number] = detector_data_polyval*detector_data_sun_mean[pixel_number]
    
    #    index_atmos_top = np.abs(alt_point0_sorted - TOP_OF_ATMOSPHERE).argmin()
        index_atmos_top = np.abs(alt_point0_sorted - TEST_ALTITUDE).argmin()
    #    plt.plot(alt_point0_sorted, detector_data_sorted[:,160])
    #    plt.plot(alt_point0_sorted[top_indices], detector_data_sun[:,160])
    #    plt.plot(alt_point0_sorted, detector_data_polyval*detector_data_sun_mean[160])
    
        detector_data_transmittance = detector_data_sorted / detector_data_extrap
        y_data = detector_data_transmittance
        y_data_raw = detector_data_sorted
        y_dark_raw = detector_dark_sorted
        
        y_error = np.abs((detector_data_extrap[index_atmos_top,PIXEL_NUMBER] - detector_data_sorted[index_atmos_top,PIXEL_NUMBER])/detector_data_sorted[index_atmos_top,PIXEL_NUMBER])
    
        if not silent: print("%ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
        if y_error * 100 > MAX_PERCENTAGE_ERROR:
            print("Warning: error too large. %ikm error = %0.1f" %(TEST_ALTITUDE, y_error * 100))
            print(hdf5_filename)
            outputDict = {"error":True}
    #        stop()
        else:
            """get extra data from first file only"""
            x = get_dataset_contents(hdf5_file, "X")[0]
            ls = get_dataset_contents(hdf5_file, "LSubS")[0][0][0]
            x_data = x[0]
            obsDatetime = datetime.strptime(hdf5_filename[0:15], '%Y%m%d_%H%M%S')
            lst,terminator = getCorrectLST(hdf5_file, silent=silent)
            if not silent: print("lat_mean = %0.1f" %np.mean(lat_point0_sorted))
            label_out = date+"-"+time+"-"+obs_type+" lat=%0.0f to %0.0f, lon=%0.0f to %0.0f" %(lat_range[0],lat_range[1],lon_range[0],lon_range[1])
        
            
            outputDict = {"error":False, "x":x_data, "y":y_data, "y_raw":y_data_raw, "y_dark":y_dark_raw, \
                          "alt":alt_point0_sorted, "order":int(diffraction_order), \
                          "lat":lat_point0_sorted, "lon":lon_point0_sorted, \
                          "ls":ls, "sbsf":sbsf, \
                          "lat_range":lat_range, "lon_range":lon_range, \
                          "obs_datetime":obsDatetime, "lst":lst, "terminator":terminator, \
                          "label":label_out}
    return outputDict




def plotVerticalProfiles3(hdf5_files, hdf5_filenames, fig_in, ax_arr, ax_index, ax_max, bin_index, subtitle, plot_type={"type":"opacity", "colour":"latitude", "linestyle":"terminatorTime"}, start_date=0.0):
    PIXEL_NUMBER = 160
    ANCHOR_TO_ONE = False
    ANCHOR_TO_ZERO = False
    
    continuumOpacityAll = []
    continuumOpticalDepthAll = []
    continuumTransmittanceAll = []
    altitudesAll = []
    datetimesAll = []
    terminatorTypesAll = []
    labelsAll = []
    latsAll = []
    lonsAll = []
    lstAll = []
    lsAll = []
    opticalDepthAltitudeAll = []
    for fileIndex in range(len(hdf5_filenames)):#range(100):
        print("fileIndex=%i (%s)" %(fileIndex, hdf5_filenames[fileIndex]))
    
        obsDict = joinTransmittanceFiles(hdf5_files[fileIndex], hdf5_filenames[fileIndex], bin_index, silent=True)
        
        if not obsDict["error"]:
            #assume all diffraction orders chosen are the same
#            diffractionOrder = obsDict["order"]
            """choose 0-60km region only"""            
            start_index = 0
            end_index = np.max(np.where(obsDict["alt"]<120.0)[0])
            
            continuumOpacity = []
            continuumOpticalDepth = []
            continuumTransmittance = []
            
            """for each spectrum at each altitude, calculate values to plot"""
        #            print("start_index=%i, end_index=%i" %(start_index,end_index))
        #    plt.figure(figsize=(FIG_X, FIG_Y))
            topFound = False
            bottomFound = False
            for index in range(start_index, end_index):
        #            if plot_type["anchor"]:
        #                transmittance = transmittances[index,:]/transmittances[top_index,:]
        #            else:
                transmittance = obsDict["y"][index,:]
                if ANCHOR_TO_ONE:
                    if np.max(transmittance[150:200]) > 0.995:
            #            plt.plot(transmittance[150:200])
                        transmittance = np.ones(320)
                        topFound = True
                    if topFound:
                        transmittance = np.ones(320)
                if ANCHOR_TO_ZERO:
                    if np.mean(transmittance[150:200]) < 0.005:
            #            plt.plot(transmittance[150:200])
                        transmittance = np.zeros(320)
                        bottomFound = True
                    if bottomFound:
                        continuumOpacity = [np.inf] * len(continuumOpacity)
                        bottomFound = False
        
                slantAirmass = getSlantAirmass(obsDict["alt"][index])
                opticalDepths = -1.0 * np.log(np.abs(transmittance))
                opticalOpacities = opticalDepths / slantAirmass
                
                continuumOP = np.ones(320) * np.min(opticalOpacities[150:200])
        #        continuumOP = getContinuumFromSpectrum(opticalOpacities, "OP")
                continuumOD = np.ones(320) * np.min(opticalDepths[150:200])
#                continuumOD = getContinuumFromSpectrum(opticalDepths, "OD")
                continuumTR = np.ones(320) * np.max(transmittance[150:200])


               
                continuumOpacity.append(continuumOP[PIXEL_NUMBER]) #centre of detector
                continuumOpticalDepth.append(continuumOD[PIXEL_NUMBER]) #centre of detector
                continuumTransmittance.append(continuumTR[PIXEL_NUMBER]) #centre of detector
        #        plt.plot(opticalOpacities, label=obsDict["alt"])
        #        plt.plot(continuumOP)
                
        
        
            altitudes = obsDict["alt"][start_index:end_index]
            latitudes = obsDict["lat"][start_index:end_index]
            longitudes = obsDict["lon"][start_index:end_index]
            ls = obsDict["ls"]
            
            #remove negative transmittances and ODs
            continuumOpacity = np.asarray(continuumOpacity)# + 1.0
            continuumOpticalDepth = np.asarray(continuumOpticalDepth)
            continuumTransmittance = np.asarray(continuumTransmittance)

            lstHours,terminatorType = getCorrectLST(hdf5_files[fileIndex], silent=True)

            #find lowest point where OD < 1.0
            indicesOD = np.where(continuumOpticalDepth <1.0)[0]
            #check OD does drop below value
            if len(indicesOD) > 0:
                    opticalDepthIndex = np.min(indicesOD)
                    opticalDepthAltitudeAll.append(altitudes[opticalDepthIndex])
                    print("Optical depth 1.0 at %0.1f km" %altitudes[opticalDepthIndex])
            else:
                print("Error optical depth too low throughout")
                opticalDepthAltitudeAll.append(np.max(altitudes))

        
            continuumOpacityAll.append(continuumOpacity)
            continuumOpticalDepthAll.append(continuumOpticalDepth)
            continuumTransmittanceAll.append(continuumTransmittance)
            altitudesAll.append(altitudes)
            latsAll.append(latitudes)
            lonsAll.append(longitudes)
            datetimesAll.append(obsDict["obs_datetime"])
            terminatorTypesAll.append(obsDict["terminator"])
            labelsAll.append(obsDict["label"])
            lstAll.append(lstHours)
            lsAll.append(ls)
    
    
    #    plt.plot(obsDict["y"][:, 160], obsDict["alt"])
    #    plt.plot(continuumOpacity, altitudes)
    #plt.ylim([0.0, 80.0])
    #plt.xlim([-0.1, 1.1])
    
    timeDeltaAll = []
    if start_date == 0.0:
        startingDatetime = datetimesAll[0]
    else:
        startingDatetime = start_date
    for obsDatetime in datetimesAll:
        timeDeltaAll.append((obsDatetime - startingDatetime).total_seconds() / 3600.0 / 24.0)
    
    
    if "Northern_latitudes" in subtitle:
        lat_range = [60,90]
    elif "Southern_latitudes" in subtitle:
        lat_range = [-50,-70]
    elif "Mid_latitudes" in subtitle:
        lat_range = [-30,30]
    elif "All_Occultations" in subtitle:
        lat_range = [-90,90]
    else:
        print("Error: %s not defined" %subtitle)
   
    
    if plot_type["colour"] == "time":
        #colour by time
        cmap = plt.get_cmap('Spectral')
        colours = [cmap(i) for i in np.arange(len(timeDeltaAll))/len(timeDeltaAll)]
    elif plot_type["colour"] == "observationNumber":
        #colour by observation number
        cmap = plt.get_cmap('jet_r')
        colours = [cmap(i) for i in np.arange(len(continuumOpacityAll))/len(continuumOpacityAll)]
    elif plot_type["colour"] in ["latitude", "localLatitude"]:
        #colour by latitude
        cmap = plt.get_cmap('brg')
        meanLatitudes = [np.mean(latitudes) for latitudes in latsAll]
#        colours = [cmap((meanLatitude + 90.0) / 180.0) for meanLatitude in meanLatitudes]
        colours = [cmap((meanLatitude - lat_range[0]) / (lat_range[1]-lat_range[0])) for meanLatitude in meanLatitudes]
    
    
    if plot_type["linestyle"] in [":", "-.", "--", "-"]:
        linestyles = [plot_type["linestyle"]] * 2000
    elif plot_type["linestyle"] in [":", "-.", "--", "-"]:
        linestyles = [":", "-.", "--", "-"] * 2000
    elif plot_type["linestyle"] == "terminatorTime":
        linestyles = ["-" if terminatorType == "am" else ":" for terminatorType in terminatorTypesAll] #sunrise is solid line, sunset is dotted line
    
    
    if plot_type["type"] == "transmittance":
        for spectrumIndex,continuumTransmittance in enumerate(continuumTransmittanceAll):
            if len(continuumTransmittance) > 0:
                altitude = altitudesAll[spectrumIndex]
                label = "%i : " %spectrumIndex + labelsAll[spectrumIndex]
                ax_arr[ax_index[0], ax_index[1]].plot(continuumTransmittance, altitude, label=label, color=colours[spectrumIndex], linestyle=linestyles[spectrumIndex], alpha=0.5)
        ax_arr[ax_index[0], ax_index[1]].set_xlim([-0.1, 1.1])
    elif plot_type["type"] == "opticalDepth":
        for spectrumIndex,continuumOpticalDepth in enumerate(continuumOpticalDepthAll):
            if len(continuumOpticalDepth) > 0:
                altitude = altitudesAll[spectrumIndex]
                label = "%i : " %spectrumIndex + labelsAll[spectrumIndex]
                ax_arr[ax_index[0], ax_index[1]].plot(continuumOpticalDepth, altitude, label=label, color=colours[spectrumIndex], linestyle=linestyles[spectrumIndex], alpha=0.5)
        ax_arr[ax_index[0], ax_index[1]].set_xlim([-0.1, 5.0])
    elif plot_type["type"] == "opacity":
        for spectrumIndex,continuumOpacity in enumerate(continuumOpacityAll):
            if len(continuumOpacity) > 0:
                altitude = altitudesAll[spectrumIndex]
                label = "%i : " %spectrumIndex + labelsAll[spectrumIndex]
                ax_arr[ax_index[0], ax_index[1]].plot(continuumOpacity, altitude, label=label, color=colours[spectrumIndex], linestyle=linestyles[spectrumIndex], alpha=0.5)

#    if ax_index[0] == ax_max[0]: #if last row
#        ax_arr[ax_index[0], ax_index[1]].set_xlabel(xLabelTitle)
#    if ax_index[1] == 0: #if first column
#        ax_arr[ax_index[0], ax_index[1]].set_ylabel("Tangent altitude above areoid (km)")



    if "Northern_latitudes" in subtitle:
#            ax3.set_xlim([-0.1, 3.0])
        ax_arr[ax_index[0], ax_index[1]].set_ylim([-5.0, 75.0])
    elif "Southern_latitudes" in subtitle:
#            ax3.set_xlim([-0.1, 3.0])
        ax_arr[ax_index[0], ax_index[1]].set_ylim([-5.0, 75.0])
    elif "Mid_latitudes" in subtitle:
#            ax3.set_xlim([-0.1, 3.0])
        ax_arr[ax_index[0], ax_index[1]].set_ylim([-5.0, 75.0])
#        ax3.set_xscale("log")
#        ax3.set_xlim([0.9, 6.0])
        
    
    
#    ax3.set_title("%s: Order %i from %s to %s" %(subtitle.replace("_"," "), diffractionOrder, hdf5_filenames[0][0:15], hdf5_filenames[-1][0:15]))
    ax_arr[ax_index[0], ax_index[1]].set_title("%s" %(subtitle.replace("_"," ")), fontsize="large")
    if plot_type["colour"] == "time":
        if ax_index[1] == ax_max[1]: #if last column
            norm = mpl.colors.Normalize(vmin=0.0, vmax=max(timeDeltaAll))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            cbar3 = fig_in.colorbar(sm, ax=ax_arr[ax_index[0], ax_index[1]], format="%i")
            cbar3.set_label("Day number (from %s)" %startingDatetime.strftime("%Y-%m-%d"), rotation=270, labelpad=20, fontsize="medium")
    if plot_type["colour"] == "latitude":
        if ax_index[1] == ax_max[1]: #if last column
            norm = mpl.colors.Normalize(vmin=-90.0, vmax=90.0)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            cbar3 = fig_in.colorbar(sm, ax=ax_arr[ax_index[0], ax_index[1]], format="%i")
            cbar3.set_label("Latitude in centre of occultation", rotation=270, labelpad=20, fontsize="medium")
    if plot_type["colour"] == "localLatitude":
        if ax_index[1] == ax_max[1]: #if last column
            norm = mpl.colors.Normalize(vmin=lat_range[0], vmax=lat_range[1])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            cbar3 = fig_in.colorbar(sm, ax=ax_arr[ax_index[0], ax_index[1]], format="%i")
            cbar3.set_label("Latitude in centre of occultation", rotation=270, labelpad=20, fontsize="medium")
    
#    if SAVE_FIGS:
#        plt.savefig(BASE_DIRECTORY + os.sep + "SO_%s_%s_%s_order_%i_bin%i_%s_%s.png" %(plot_type["type"], plot_type["colour"], subtitle, diffractionOrder, bin_index, hdf5_filenames[0][0:15], hdf5_filenames[-1][0:15]))
    
    if SAVE_FILES:
#        output = ["%0.0f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" %(sp.utc2et(time.strftime("%Y%m%d")+" UTC"), sp.lspcn("MARS", sp.utc2et(time+" UTC"), "NONE") * sp.dpr(), lon * 180.0 / np.pi, lat * 180.0 / np.pi, colour, lst) for time, lon, lat, colour, lst in zip(obsTimeArray, lonsArray, latsArray, colours2, lstArray)]
#        output.insert(0, "ObservationTime (J2000), Ls (deg), Longitude (deg), Latitude (deg), %s (km), LST (hours)" %colorbarLabel)

        output = ["%0.0f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" %(sp.utc2et(datetimes.strftime("%Y-%m-%d %H:%M:%S UTC")), \
             sp.lspcn("MARS", sp.utc2et(datetimes.strftime("%Y-%m-%d %H:%M:%S UTC")), "NONE") * sp.dpr(), lons[0], lons[-1], lats[0], lats[-1], lsts) \
             for datetimes, lons, lats, lsts in zip(datetimesAll, lonsAll, latsAll, lstAll)]
        output.insert(0, "ObservationStartTime, Ls (deg), LongitudeStart (deg), LongitudeEnd (deg), LatitudeStart (deg), LatitudeEnd (deg), LST (hours)")

        writeOutput(startingDatetime.strftime("%Y%m%d")+"_"+plot_type["type"]+"_"+subtitle+".csv", output)
    return lonsAll, latsAll, lsAll, opticalDepthAltitudeAll


                                        


def plotAllVerticalProfiles(obspaths, plot_type):
    """plot vertical profiles for a given bin and observational constraints"""
    binIndex = 1

    fig1 = plt.figure(figsize=(FIG_X - 2, FIG_Y + 2))
    ax1 = fig1.add_subplot(111, projection="mollweide")
    ax1.grid(True)


    nrows = 3
    ncols = 3
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(FIG_X + 5, FIG_Y + 5), sharex=True, sharey=True)
    mainPlot = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    xLabelTitle = {"transmittance":"Continuum line-of-sight transmittance", "opticalDepth":"Continuum line-of-sight optical depth", "opacity":"Continuum opacity (relative to zenith)"}[plot_type["type"]]
    yLabelTitle = "Tangent altitude above Mars areoid (km)"
    mainPlot.set_xlabel(xLabelTitle, fontsize="x-large")
    mainPlot.set_ylabel(yLabelTitle, fontsize="x-large")
    mainPlot.set_title("Continuum line-of-sight optical depths for SO diffraction order 121: \nVariations by time and tangent point latitude", fontsize="x-large", y=1.03)
    plt.tight_layout()


    plt.subplots_adjust(hspace=0.15)
    for plotIndex in [0,1,2]:
#    for plotIndex in [2]:
        if plotIndex == 0:
            searchDatasetsMinMax = [["Lat","mean","gt",60.0,"Geometry/Point0"]]; subtitleBase = "Northern_latitudes"
            hdf5FilesAll, hdf5FilenamesAll, titles = makeFileList(obspaths, fileLevel, search_datasets_min_max=searchDatasetsMinMax, silent=True)
        elif plotIndex == 1:    
            searchDatasetsMinMax = [["Lat","mean","lt",30.0,"Geometry/Point0"], ["Lat","mean","gt",-30.0,"Geometry/Point0"]]; subtitleBase = "Mid_latitudes"
            hdf5FilesAll, hdf5FilenamesAll, titles = makeFileList(obspaths, fileLevel, search_datasets_min_max=searchDatasetsMinMax, silent=True)
        elif plotIndex == 2:
            searchDatasetsMinMax = [["Lat","mean","lt",-50.0,"Geometry/Point0"], ["Lat","mean","gt",-70.0,"Geometry/Point0"]]; subtitleBase = "Southern_latitudes"
            hdf5FilesAll, hdf5FilenamesAll, titles = makeFileList(obspaths, fileLevel, search_datasets_min_max=searchDatasetsMinMax, silent=True)
        elif plotIndex == 3:
            hdf5FilesAll, hdf5FilenamesAll, titles = makeFileList(obspaths, fileLevel); subtitleBase = "All_Occultations"
    
        hdf5FilesChosen = []
        hdf5FilenamesChosen = []
        for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5FilesAll, hdf5FilenamesAll)):
            if "201804" in hdf5Filename or "201805" in hdf5Filename:
                hdf5FilesChosen.append(hdf5File)
                hdf5FilenamesChosen.append(hdf5Filename)
        startDate = datetime.strptime("20180421", '%Y%m%d')
        subtitle = subtitleBase + " April-May (Ls: 163-185)"
        print("************Plotting %s************" %subtitle)
        lons, lats, ls, minalts = plotVerticalProfiles3(hdf5FilesChosen, hdf5FilenamesChosen, fig, axarr, [plotIndex, 0], [nrows-1, ncols-1], binIndex, subtitle, plot_type=plot_type, start_date=startDate)
        colours = ["lightgreen", "salmon", "lightblue"]
        for lonobs, latobs in zip(lons, lats):
            ax1.scatter(lonobs * np.pi / 180.0, latobs * np.pi / 180.0, c=colours[plotIndex], marker='o', linewidth=0, alpha=0.8)

        hdf5FilesChosen = []
        hdf5FilenamesChosen = []
        for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5FilesAll, hdf5FilenamesAll)):
            if "201806" in hdf5Filename or "201807" in hdf5Filename:
                hdf5FilesChosen.append(hdf5File)
                hdf5FilenamesChosen.append(hdf5Filename)
        startDate = datetime.strptime("20180601", '%Y%m%d')
        subtitle = subtitleBase + " June-July (Ls: 185-221)"
        print("************Plotting %s************" %subtitle)
        lons, lats, ls, minalts = plotVerticalProfiles3(hdf5FilesChosen, hdf5FilenamesChosen, fig, axarr, [plotIndex, 1], [nrows-1, ncols-1], binIndex, subtitle, plot_type=plot_type, start_date=startDate)
        colours = ["mediumspringgreen", "sienna", "blue"]
        for lonobs, latobs in zip(lons, lats):
            ax1.scatter(lonobs * np.pi / 180.0, latobs * np.pi / 180.0, c=colours[plotIndex], marker='o', linewidth=0, alpha=0.8)

        hdf5FilesChosen = []
        hdf5FilenamesChosen = []
        for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5FilesAll, hdf5FilenamesAll)):
            if "201808" in hdf5Filename or "201809" in hdf5Filename:
                hdf5FilesChosen.append(hdf5File)
                hdf5FilenamesChosen.append(hdf5Filename)
        startDate = datetime.strptime("20180801", '%Y%m%d')
        subtitle = subtitleBase + " August-September (Ls: 221-246)"
        print("************Plotting %s************" %subtitle)
        lons, lats, ls, minalts = plotVerticalProfiles3(hdf5FilesChosen, hdf5FilenamesChosen, fig, axarr, [plotIndex, 2], [nrows-1, ncols-1], binIndex, subtitle, plot_type=plot_type, start_date=startDate)
        colours = ["green", "darkred", "mediumblue"]
        for lonobs, latobs in zip(lons, lats):
            ax1.scatter(lonobs * np.pi / 180.0, latobs * np.pi / 180.0, c=colours[plotIndex], marker='o', linewidth=0, alpha=0.8)
            
#        hdf5FilesChosen = []
#        hdf5FilenamesChosen = []
#        for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5FilesAll, hdf5FilenamesAll)):
#            if "201810" in hdf5Filename or "201811" in hdf5Filename:
#                hdf5FilesChosen.append(hdf5File)
#                hdf5FilenamesChosen.append(hdf5Filename)
#        startDate = datetime.strptime("20181001", '%Y%m%d')
#        subtitle = subtitleBase + " October-November"
#        print("************Plotting %s************" %subtitle)
#        lons, lats, ls, minalts = plotVerticalProfiles3(hdf5FilesChosen, hdf5FilenamesChosen, fig, axarr, [plotIndex, 3], [nrows-1, ncols-1], binIndex, subtitle, plot_type=plot_type, start_date=startDate)
#        colours = ["darkgreen", "darkred", "darkblue"]
#        for lonobs, latobs in zip(lons, lats):
#            ax1.scatter(lonobs * np.pi / 180.0, latobs * np.pi / 180.0, c=colours[plotIndex], marker='o', linewidth=0, alpha=0.8)

    textLocation = [0.5, 65]
    axarr[0,0].text(textLocation[0], textLocation[1], "a", fontsize="large")
    axarr[0,1].text(textLocation[0], textLocation[1], "b", fontsize="large")
    axarr[0,2].text(textLocation[0], textLocation[1], "c", fontsize="large")
    axarr[1,0].text(textLocation[0], textLocation[1], "d", fontsize="large")
    axarr[1,1].text(textLocation[0], textLocation[1], "e", fontsize="large")
    axarr[1,2].text(textLocation[0], textLocation[1], "f", fontsize="large")
    axarr[2,0].text(textLocation[0], textLocation[1], "g", fontsize="large")
    axarr[2,1].text(textLocation[0], textLocation[1], "h", fontsize="large")
    axarr[2,2].text(textLocation[0], textLocation[1], "i", fontsize="large")
    if SAVE_FIGS:
        plt.savefig(BASE_DIRECTORY + os.sep + "SO_coverage_%s_%s.pdf" %(plot_type["type"], plot_type["colour"]), dpi=300)







def plotDustVsLs(hdf5_files, hdf5_filenames, subtitle):
    """plot dust profile latitude vs ls for minimum altitudes"""
    binIndex = 1
    startDate = datetime.strptime("20180421", '%Y%m%d')
    
    #dummy plot for function
    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(FIG_X, FIG_Y))
    lons, lats, ls, minalts = plotVerticalProfiles3(hdf5_files, hdf5_filenames, fig, ax, [0, 0], [0 , 0], binIndex, subtitle, plot_type={"type":"opticalDepth", "colour":"latitude", "linestyle":"-", "anchor":True}, start_date=startDate)
    
    fig, ax = plt.subplots(figsize=(FIG_X + 2, FIG_Y + 1))
    lsArray = [ls[index] for index,sublist in enumerate(lons) for item in sublist]
    minAltArray = [minalts[index] for index,sublist in enumerate(lons) for item in sublist]
    latsArray = []
    for latArray in lats: 
        latsArray.extend(latArray)
    latsArray = np.asfarray(latsArray)
    #plot = ax.scatter(lsArray, latsArray, c=minAltArray, cmap=plt.cm.jet, marker='o', linewidth=0)
    plot = ax.scatter(lsArray, latsArray, c=minAltArray, cmap=plt.cm.jet, marker='o', linewidth=0, vmax=60)
    cbar = fig.colorbar(plot)
    cbar.set_clim(0, 60)
    colorbarLabel = "Lowest altitude where optical depth < 1.0"
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
    
    ax.set_ylim([-90, 90])
    ax.set_xlim([min(lsArray)-1, max(lsArray)+1])
    ax.set_xlabel("Ls (degrees)")
    ax.set_ylabel("Latitude (degrees)")
#    ax.set_title("Optical depth vs. latitude")
    ax.set_title("SO diffraction order 121: continuum line-of-sight optical depths versus Ls and observation latitude")
    
    months = np.arange(4, 13, 1)
    lsMonths = [sp.lspcn("MARS", sp.utc2et(datetime(2018, month, 1).strftime("%Y-%m-%d")), "NONE") * sp.dpr() for month in months]
#    for lsMonth in lsMonths:
#        ax.plot([lsMonth, lsMonth], [-90, 90], "k--")

    if SAVE_FIGS:
        plt.savefig(BASE_DIRECTORY + os.sep + "SO_optical_depth_vs_ls_vs_latitude.pdf", dpi=300)



    fig, ax = plt.subplots(figsize=(FIG_X + 2, FIG_Y + 1))
    plot = ax.scatter(lsArray, minAltArray, c=latsArray, cmap=plt.cm.Set1, marker='o', linewidth=0, vmin=-90, vmax=90)
    cbar = fig.colorbar(plot)
    cbar.set_clim(-90, 90)
    colorbarLabel = "Latitude"
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)

    for lsMonth in lsMonths:
        ax.plot([lsMonth, lsMonth], [0, 60], "k--")
    
    ax.set_xlim([min(lsArray)-1, max(lsArray)+1])
    ax.set_xlabel("Ls (degrees)")
    ax.set_ylabel("Lowest altitude where optical depth < 1.0")
    ax.set_title("Optical depth vs. latitude")

    if SAVE_FIGS:
        plt.savefig(BASE_DIRECTORY + os.sep + "SO_optical_depth_vs_latitude_band.pdf", dpi=300)
    
    return lsArray, latsArray, minAltArray







#if True:
def correctNonLinearity1(hdf5Files, hdf5Filenames, title):
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(len(hdf5Filenames))/len(hdf5Filenames)]
    PLOT_INTERMEDIATE_FIGURES = False
    
    if title == "ch4 paper 134 apr":
        #order 134 april 2018 ignore 1st bin
        BIN_INDICES = [1,2,3]
        DESIRED_TRANSMITTANCES_ALL = [[0.0, 0.0], [0.27464, 0.38273], [0.26592, 0.37367], [0.25350, 0.36581]] #order 134. 2x spectra, first bin is not used!
        WAVENUMBER_ABSORPTION_START = 3025.6 #order 134
#        ABSORPTION_PIXELS = 10 #order 134
        CHOSEN_FILENAME = '20180430_162901_0p3a_SO_1_E_134' #order 134
        LINEAR_PIXEL = 18
        DETECTOR_REGION = np.arange(160,250)
        METHANE_PIXELS = np.arange(70,90)
        WAVENUMBER_SHIFT = -0.10
        EXTRA_TEXT = "Ls 167.9\n78.4N, 32.6E"
        ORDER = 134
    
    if title == "ch4 paper 134 aug":
        ##order 134 aug 2018 all bins 3 altitudes
        BIN_INDICES = [0,1,2,3]
#        BIN_INDICES = [1]
        #SO non linearity corrected values
#        DESIRED_TRANSMITTANCES_ALL = [[0.32321033, 0.44776225, 0.525089], [0.22202936, 0.34463188, 0.45793402], \
#                                      [0.24465266, 0.36952582, 0.47473285], [0.2721203, 0.39733413, 0.48762965]] #order 134, 3x spectra for 4 bins
        #non linearity correction off
        DESIRED_TRANSMITTANCES_ALL = [[0.30861, 0.43511, 0.51386], [0.20562, 0.32963, 0.44427], \
                                      [0.23073, 0.35792, 0.46512], [0.25863, 0.38609, 0.47812]] #order 134, 3x spectra for 4 bins
        WAVENUMBER_ABSORPTION_START = 3025.5 #order 134 main h2o band
        WAVENUMBER_ABSORPTION_END = 3026.0 #order 134 main h2o band
#        ABSORPTION_PIXELS = 6 #order 134
        CHOSEN_FILENAME = '20180821_052431_0p3a_SO_1_E_134' #order 134 august. This one will be calibrated using the other files
        LINEAR_PIXEL = 18
        DETECTOR_REGION = np.arange(160,250)
        METHANE_PIXELS = np.arange(70,90)
        WAVENUMBER_METHANE_START = 3027.0
        WAVENUMBER_SHIFT = -0.08
        SNR_SINGLE_PIXEL = 1000
        EXTRA_TEXT = "Ls 234.5\n56.4N, 34.3E"
        ORDER = 134
        SIMULATION_FILE_INDICES = [157,247]

    
    if title == "ch4 paper 135 aug":
        ##order 135 aug 2018 all bins 3 altitudes
        BIN_INDICES = [0,1,2,3]
        #SO non linearity corrected values
#        DESIRED_TRANSMITTANCES_ALL = [[0.21286258, 0.3408766, 0.4623971], [0.23665068, 0.36100715, 0.47102085], \
#                                      [0.25955728, 0.38554832, 0.48366195], [0.287145, 0.41101104, 0.4963896]] #order 134, 3x spectra for 4 bins
        #non linearity correction off
        DESIRED_TRANSMITTANCES_ALL = [[0.19610, 0.32639, 0.44987], [0.22013, 0.34595, 0.45730], \
                                      [0.24567, 0.37404, 0.47406], [0.27360, 0.39978, 0.48687]] #order 134, 3x spectra for 4 bins

        WAVENUMBER_ABSORPTION_START = 3048.43 #order 135
        WAVENUMBER_ABSORPTION_END = 3049.4 #order 134 main h2o band
#        ABSORPTION_PIXELS = 14 #order 135
        CHOSEN_FILENAME = '20180821_052431_0p3a_SO_1_E_135' #order 135
        LINEAR_PIXEL = 18
        DETECTOR_REGION = np.arange(160,250)
        METHANE_PIXELS = np.arange(70,90)
        WAVENUMBER_METHANE_START = 3050.0
        WAVENUMBER_SHIFT = -0.08
        SNR_SINGLE_PIXEL = 1000
        EXTRA_TEXT = "Ls 234.5\n56.4N, 34.3E"
        ORDER = 135
        SIMULATION_FILE_INDICES = [160,250]

    PLOTTED_TRANSMITTANCE_ALL = [i[0] for i in DESIRED_TRANSMITTANCES_ALL] #plot first transmittance of each bin
    
    obsDictOut = []
    outputSpectra = []
    outputWavenumbers = []
    outputAltitudes = []
    yPlottedMeanAllBins = []
    for bin_index in BIN_INDICES: #loop through each bin separately, making 1 figure per bin
        
        DESIRED_TRANSMITTANCES = DESIRED_TRANSMITTANCES_ALL[bin_index]
        PLOTTED_TRANSMITTANCE = PLOTTED_TRANSMITTANCE_ALL[bin_index]
    
        yPlottedMeanAll = []
        yNormalisedFoundMeanAll = []
        if PLOT_INTERMEDIATE_FIGURES:
            fig1, axes = plt.subplots(len(DESIRED_TRANSMITTANCES), 2, figsize=(FIG_X, FIG_Y)) #plot 3 transmittances for each bin on one plot
        for transmittanceIndex, desiredTransmittance in enumerate(DESIRED_TRANSMITTANCES): #loop through each transmittance, searching each file for where a transmittance close to that value is found
           
            fileIndexFoundAll = []
            yPlottedAll = []
            yNormalisedFoundAll = []
            yAbsorptionRemovedAll = []
            wavenumbersAll = []
            absorptionRegionIndicesAll = []
            for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
            
                obsDict1 = joinTransmittanceFiles(hdf5_file, hdf5_filename, bin_index, silent=True) #use shape method, returns dictionary
                y1 = obsDict1["y"]
                x = obsDict1["x"]
                alt = obsDict1["alt"]
                wavenumbers = x[DETECTOR_REGION] #choose limited region of spectrum for analysis
                y = y1[:, DETECTOR_REGION]
                
                wavenumbersAll.append(wavenumbers) #store wavenumbers for each file separately
                
                absorptionIndexStart = np.argmin(wavenumbers < WAVENUMBER_ABSORPTION_START) #find index of h20 absorption start/end
                absorptionIndexEnd = np.argmin(wavenumbers < WAVENUMBER_ABSORPTION_END)
                #loop through doing calculations on all spectra
                yPoly = np.zeros_like(y)
                yNormalised = np.zeros_like(y)
                yFitted = np.zeros_like(y)
                for spectrumIndex, spectrum in enumerate(y): #loop through spectra in file, removing baseline shape
                    #only do polyfit on part of spectrum without absorption
                    yPolyIndices = np.concatenate((np.arange(0,absorptionIndexStart),np.arange(absorptionIndexEnd, len(wavenumbers))))
                    yPolyCoeffs = np.polyfit(wavenumbers[yPolyIndices], spectrum[yPolyIndices], 3)
                    yPoly[spectrumIndex, :] = np.polyval(yPolyCoeffs, wavenumbers)
                    yNormalised[spectrumIndex,:] = spectrum / yPoly[spectrumIndex,:]
                    yFitted[spectrumIndex,:] = yNormalised[spectrumIndex,:] * np.mean(spectrum) #multiply flattened+normalised spectrum by mean to get flat spectrum of approx correct transmittance
#                    if spectrum[80] > 0.2: #test continuum shape removal
#                        plt.figure()
#                        plt.plot(wavenumbers, spectrum)
#                        plt.plot(wavenumbers[yPolyIndices], spectrum[yPolyIndices], "k--")
#                        plt.plot(wavenumbers, yPoly[spectrumIndex, :])
            
                #pixel non-linearity requires many spectra with transmittance close to value of spectrum in desired file
                #check 1 pixel only to see if any transmittances are near the desired transmittance
                spectrumIndexAtmos = np.where(((desiredTransmittance - 0.025) < yFitted[:,LINEAR_PIXEL]) & ((desiredTransmittance + 0.025) > yFitted[:,LINEAR_PIXEL]))[0]
#                print("Searching %s bin %i for transmittances between %0.3f and %0.3f: %i found" %(hdf5_filename, bin_index, (desiredTransmittance - 0.025),(desiredTransmittance + 0.025),len(spectrumIndexAtmos)))
        
                #if file contains a spectrum close to the desired transmittance
                if len(spectrumIndexAtmos) > 0:
                    spectrumIndexAtmos = np.min(spectrumIndexAtmos) #get frame index of desired transmittance spectrum (first if several)
            
            
                    yPlotted = yNormalised[spectrumIndexAtmos, :] * desiredTransmittance #multiple normalised spectrum by desired value to get spectrum at exactly that transmittance
                    yNormalisedFound = yNormalised[spectrumIndexAtmos, :] #store normalised spectrum also
        
                    altitude = alt[spectrumIndexAtmos] #store altitude of frame at desired transmittance value
    
                    #store data for chosen occultation and altitude. Don't use for pixel non-linearity calibration
                    if hdf5_filename == CHOSEN_FILENAME:
                        
                        #save dictionary for all bins of chosen file
                        if desiredTransmittance == DESIRED_TRANSMITTANCES[0]: #don't save when looping for each transmittance
                            obsDictOut.append(obsDict1)
                        outputSpectra.append(yFitted[spectrumIndexAtmos])
                        outputWavenumbers = wavenumbers
                        outputAltitudes.append(altitude)
                        print("Transmittance on pixel 52 of desired spectrum at %0.1fkm (T=%0.3f) bin %i = %0.5f" %(altitude, desiredTransmittance, bin_index, np.mean(yFitted[spectrumIndexAtmos][LINEAR_PIXEL])))
    
                    else: #if desired transmittance found in another file (not the one to be calibrated) then use for non linearity correction
                        pixels = np.arange(len(yPlotted))
                        
                        yStd = np.std(np.concatenate((yPlotted[0:absorptionIndexStart],yPlotted[absorptionIndexEnd:])))
                        yMean = np.mean(np.concatenate((yPlotted[0:absorptionIndexStart],yPlotted[absorptionIndexEnd:])))
                        if PLOT_INTERMEDIATE_FIGURES:
                            #plot in 1st column, x axis=pixels
                            axes[transmittanceIndex, 0].plot(pixels, yPlotted, color=colours[fileIndex])
                            axes[transmittanceIndex, 0].plot(pixels, np.tile(yMean, len(yPlotted)) - np.tile(yStd, len(yPlotted)), "--", color=colours[fileIndex])
                            axes[transmittanceIndex, 0].plot(pixels, np.tile(yMean, len(yPlotted)) + np.tile(yStd, len(yPlotted)), "--", color=colours[fileIndex])
                    
                        snrMethaneRegion = np.mean(yPlotted[METHANE_PIXELS]) / np.std(yPlotted[METHANE_PIXELS])
                        if desiredTransmittance == PLOTTED_TRANSMITTANCE:
                            print("%s: T=%0.3f@%0.1fkm: uncorrected SNR=%0.1f" %(hdf5_filename, np.mean(np.concatenate((yFitted[spectrumIndexAtmos, 0:absorptionIndexStart],yFitted[spectrumIndexAtmos, absorptionIndexEnd:]))), altitude, snrMethaneRegion))
                        
                        #remove water absorption from non-linearity correction spectra
                        yAbsorptionRemoved = np.copy(yPlotted)
                        yAbsorptionRemoved[absorptionIndexStart:absorptionIndexEnd] = desiredTransmittance
                        """warning: never use these pixels to calculate std, otherwise result will be artificially high!"""
                        absorptionRegionIndicesAll.append([absorptionIndexStart,absorptionIndexEnd])
                        
                        fileIndexFoundAll.append(fileIndex)
                        yPlottedAll.append(yPlotted)
                        yNormalisedFoundAll.append(yNormalisedFound)
                        yAbsorptionRemovedAll.append(yAbsorptionRemoved)
                    
                else: #desired transmittances should always match those in the chosen file. If desired transmittance not found in chosen file, print all
                    if hdf5_filename == CHOSEN_FILENAME:
                        print("Error: transmittance on pixel %i doesn't match desired value (T=%0.3f) for bin %i" %(LINEAR_PIXEL, desiredTransmittance, bin_index))
                        print(yFitted[10:20,LINEAR_PIXEL])
            
            yPlottedAll = np.asfarray(yPlottedAll)
            yNormalisedFoundAll = np.asfarray(yNormalisedFoundAll)
            yPlottedMean = np.mean(yAbsorptionRemovedAll, axis=0)
            yNormalisedFoundMean = np.mean(yNormalisedFoundAll, axis=0)
            #save correction for each altitude
            yPlottedMeanAll.append(yPlottedMean)
            yNormalisedFoundMeanAll.append(yNormalisedFoundMean)
            
            absorptionRegionIndicesAll = np.asarray(absorptionRegionIndicesAll)
            
            if PLOT_INTERMEDIATE_FIGURES:
                #plot mean correction
                axes[transmittanceIndex, 0].plot(pixels, yPlottedMean, "--k")
            
            print("Desired transmittance = %0.3f" %desiredTransmittance)
            for fileIndex, spectrum in zip(fileIndexFoundAll, yPlottedAll):
                yCorrected = spectrum / yPlottedMean * desiredTransmittance
                if PLOT_INTERMEDIATE_FIGURES:
                    axes[transmittanceIndex, 1].plot(wavenumbersAll[fileIndex], yCorrected, color=colours[fileIndex])

                methaneIndexStart = np.argmin(wavenumbersAll[fileIndex] < WAVENUMBER_METHANE_START)
                
                if methaneIndexStart < np.max(absorptionRegionIndicesAll):
                    print("Warning!")
                yMeanMethaneRegion = np.mean(yCorrected[methaneIndexStart:])
                yStdMethaneRegion = np.std(yCorrected[methaneIndexStart:])
                snrMethaneRegion = yMeanMethaneRegion / yStdMethaneRegion
                wavenumbersMethaneRegion = wavenumbersAll[fileIndex][methaneIndexStart:]
            
                yMeanMinusError = np.tile(yMeanMethaneRegion, len(wavenumbersMethaneRegion)) - np.tile(yStdMethaneRegion, len(wavenumbersMethaneRegion))
                yMeanPlusError = np.tile(yMeanMethaneRegion, len(wavenumbersMethaneRegion)) + np.tile(yStdMethaneRegion, len(wavenumbersMethaneRegion))
                if PLOT_INTERMEDIATE_FIGURES:
                    axes[transmittanceIndex, 1].plot(wavenumbersMethaneRegion, yMeanMinusError, "--", color=colours[fileIndex])
                    axes[transmittanceIndex, 1].plot(wavenumbersMethaneRegion, yMeanPlusError, "--", color=colours[fileIndex])

                if desiredTransmittance == PLOTTED_TRANSMITTANCE:
                    #ch4 absorption zone
#                    print("%s bin %i: corrected SNR=%0.1f" %(hdf5Filenames[fileIndex], bin_index, snrMethaneRegion))
                    #plot region where methane SNR is calculated
                    yMeanMethaneRegion = np.tile(yMean, len(wavenumbersMethaneRegion))                  
                    if PLOT_INTERMEDIATE_FIGURES:
                        axes[transmittanceIndex, 1].plot(wavenumbersMethaneRegion, yMeanMethaneRegion, "--", color="k")
                    
        yPlottedMeanAll = np.asfarray(yPlottedMeanAll)
        yPlottedMeanAllBins.append(yPlottedMeanAll)
    
        yNormalisedFoundMeanAll = np.asfarray(yNormalisedFoundMeanAll)

    yPlottedMeanAllBins = np.asfarray(yPlottedMeanAllBins).reshape(len(BIN_INDICES)*len(DESIRED_TRANSMITTANCES_ALL[0]), len(pixels))
    
    outputSpectra = np.asfarray(outputSpectra)
    outputAltitudes = np.asfarray(outputAltitudes)
    
    #now correct spectra using derived correction. Scale to match output Spectrum
    correctedSpectra = outputSpectra / yPlottedMeanAllBins * np.tile(np.mean(np.concatenate((outputSpectra[:, 0:absorptionIndexStart],outputSpectra[:, absorptionIndexEnd:]), axis=1), axis=1), (len(pixels),1)).T
   
    print("Calculating final values")
    flattened_transmittances = [t for ts in DESIRED_TRANSMITTANCES_ALL for t in ts]
    for spectrumIndex in range(len(outputSpectra[:,0])):
    
        snrMethaneRegionBefore = np.mean(outputSpectra[spectrumIndex, METHANE_PIXELS]) / np.std(outputSpectra[spectrumIndex, METHANE_PIXELS])
        snrMethaneRegionAfter = np.mean(correctedSpectra[spectrumIndex, METHANE_PIXELS]) / np.std(correctedSpectra[spectrumIndex, METHANE_PIXELS])
        print("%s (T=%0.3f): SNR = %0.1f; corrected SNR=%0.1f" %(CHOSEN_FILENAME, flattened_transmittances[spectrumIndex], snrMethaneRegionBefore, snrMethaneRegionAfter))
    
    binnedSpectrum = np.mean(outputSpectra, axis=0)
    binnedCorrectedSpectrum = np.mean(correctedSpectra, axis=0)
    
    snrMethaneRegionBinnedBefore = np.mean(binnedSpectrum[METHANE_PIXELS]) / np.std(binnedSpectrum[METHANE_PIXELS])
    snrMethaneRegionBinnedAfter = np.mean(binnedCorrectedSpectrum[METHANE_PIXELS]) / np.std(binnedCorrectedSpectrum[METHANE_PIXELS])
    print("%s: Binned SNR = %0.1f; binned corrected SNR=%0.1f" %(CHOSEN_FILENAME, snrMethaneRegionBinnedBefore, snrMethaneRegionBinnedAfter))

    #get mean transmittance for scaling simulations
    scalar = np.mean(binnedCorrectedSpectrum[METHANE_PIXELS])

    #plot approx error bars using value from Loic value (220) * sqrt (no. of spectra)
    #Loic =220
    #1000SNR = 354 (or 547 if square root)
    #2000SNR = 708
    ySNR = np.ones(len(binnedCorrectedSpectrum)) * SNR_SINGLE_PIXEL * np.sqrt(scalar) * np.sqrt(12)
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    yErr = binnedCorrectedSpectrum / ySNR
    plt.errorbar(outputWavenumbers + WAVENUMBER_SHIFT, binnedCorrectedSpectrum, color="k", yerr = yErr, label="SO data")

    
    #overplot methane and water lines

    def plotTransSimulation(file_name, colour, label, scalar):
        
#        directory = r"X:\projects\planetary\iant"
#        data_in = np.loadtxt(os.path.join(directory, file_name))
        data_in = np.loadtxt(os.path.join(BASE_DIRECTORY, file_name+".txt"))
        waven_in = data_in[SIMULATION_FILE_INDICES[0]:SIMULATION_FILE_INDICES[1],0]
        trans_in = data_in[SIMULATION_FILE_INDICES[0]:SIMULATION_FILE_INDICES[1],1]
        plt.plot(waven_in, trans_in * scalar, color=colour, linestyle="--", label=label)
        
        return waven_in, trans_in
    
    #read in and overplot Justin simulations
    if ORDER == 134:
        fileNames = ["trans134_H2O_15000p0_0p22_for_paper", "trans134_CH4_1p0_0p22_for_paper", "trans134_CH4_0p5_0p22_for_paper"]
        colours = ["b","g","r"]
        labels = ["H2O 15ppmv", "CH4 1ppbv", "CH4 0.5ppbv"]
        
    if ORDER == 135:
        fileNames = ["trans135_H2O_15000p0_0p22_for_paper", "trans135_CH4_1p0_0p22_for_paper", "trans135_CH4_0p5_0p22_for_paper"]
        colours = ["b","g","r","c"]
        labels = ["H2O 15ppmv", "CH4 1ppbv", "CH4 0.5ppbv"]
        
    outputFilename = "nomad_order_%i_so_simulation" %ORDER
    if SAVE_FILES:
        write_log(outputFilename,"wavenumber, "+("%s, "*len(labels) %tuple(labels))[:-2], silent=True, delete=True)
    
    
    sim_wavenumbers = []
    sim_trans = []
    
    
    
    for fileName, colour, label in zip(fileNames, colours, labels):
        w, t = plotTransSimulation(fileName, colour, label, scalar)
        sim_wavenumbers.append(w)
        sim_trans.append(t)
    
    if SAVE_FILES:
        for wavenumber, transmittance1, transmittance2, transmittance3 in zip(sim_wavenumbers[0], sim_trans[0], sim_trans[1], sim_trans[2]):
            write_log(outputFilename, "%0.4f, %0.4f, %0.4f, %0.4f" %(wavenumber, transmittance1 * scalar, transmittance2 * scalar, transmittance3 * scalar), silent=True)
        
            
    plt.legend()
    offset = 0.01 #for writing text on figure
    plt.text(outputWavenumbers[0]+4.0, scalar - offset, "Altitude above areoid %0.1f-%0.1fkm\n%s" %(np.min(outputAltitudes), np.max(outputAltitudes), EXTRA_TEXT))
    
    if SAVE_FILES:
        outputFilename = "nomad_order_%i_so_data" %ORDER
        write_log(outputFilename,"wavenumber, transmittance, error", silent=True, delete=True)
        for wavenumber, transmittance, yerr in zip(outputWavenumbers + WAVENUMBER_SHIFT, binnedCorrectedSpectrum, yErr):
            write_log(outputFilename, "%0.4f, %0.4f, %0.7f" %(wavenumber, transmittance, yerr), silent=True)
    
    plt.xlabel("Wavenumber, cm-1")
    plt.ylabel("Transmittance")
    plt.ylim([scalar-0.011, scalar+0.002])
    if SAVE_FIGS:
        plt.savefig(CHOSEN_FILENAME+".png", dpi=300)


#    #write corrections to file
#    write_log(CHOSEN_FILENAME, "*********")
#    write_log(CHOSEN_FILENAME, "%i,"*len(DETECTOR_REGION) %tuple(DETECTOR_REGION), silent=True)
#    for correctionSpectrum in yPlottedMeanAllBins:
#        write_log(CHOSEN_FILENAME, "%0.5f,"*len(correctionSpectrum) %tuple(correctionSpectrum), silent=True)


#    return obsDictOut



#y_raw = obsDictOut[1]["y_raw"]
#alt = obsDictOut[1]["alt"]
#plt.plot(alt, y_raw[:, 200])
#plt.plot(y_raw[:, 200])
#
#y_raw_mean = np.mean(y_raw[:,160:240], axis=1)
#spectrum_indices = range(140,200)
#y_scaled = np.asfarray([y_raw[i, :]/y_raw_mean[i] for i in spectrum_indices])
#std_normal = np.asfarray([np.std(y_raw[spectrum_indices, px]) for px in range(320)])
#std_scaled = np.asfarray([np.std(y_scaled[:, px]) for px in range(320)])
#snr_normal = np.asfarray([y_raw[:,px]/std_normal[px] for px in range(320)]).T
#snr_scaled = np.asfarray([y_scaled[:,px]/std_scaled[px] for px in range(320)]).T
#plt.figure()
#for i in spectrum_indices: plt.plot(y_raw[i, :])
#plt.figure()
#for i in range(len(spectrum_indices)): plt.plot(y_scaled[i, :])


#####BEGIN SCRIPTS########

"""plot vertical profiles for paper"""
if title == "dust vertical profiles":
#    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, silent=True)
#    plotAllVerticalProfiles(obspaths, {"type":"transmittance", "colour":"localLatitude", "linestyle":"-"})
#    plotAllVerticalProfiles({"type":"opticalDepth", "colour":"time", "linestyle":"-"})
    plotAllVerticalProfiles(obspaths, {"type":"opticalDepth", "colour":"localLatitude", "linestyle":"-"})

"""plot dust profile latitude vs ls for minimum altitudes"""
if title == "dust altitude vs ls":
    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel); subtitleBase = "All_Occultations"
    lsArray, latsArray, minAltArray = plotDustVsLs(hdf5Files, hdf5Filenames, subtitleBase)





"""stuff"""
#hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, obstypes=["I","E"], search_datasets_min_max=searchDatasetsMinMax)
#plotOpticalDepths(hdf5Files, hdf5Filenames, subtitle, plot_type={"type":"opticalDepth", "colour":"time", "linestyle":"terminatorTime"}, start_date=datetime.strptime(obspaths[0][1:7], '%Y%m'))
#
#hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, obstypes=["I","E"], search_datasets_min_max=searchDatasetsMinMax)
#plotOpticalDepths(hdf5Files, hdf5Filenames, subtitle, plot_type={"type":"opticalDepth", "colour":"time", "linestyle":"terminatorTime"}, start_date=datetime.strptime(obspaths[0][1:7], '%Y%m'))
#
#plotOpticalDepths(hdf5Files, hdf5Filenames, subtitle, plot_type={"type":"opticalDepth", "colour":"time", "linestyle":"terminatorTime"}, start_date=datetime.strptime("201804", '%Y%m'))



#plotOpticalDepths(hdf5Files, hdf5Filenames, subtitle, plot_type={"type":"transmittance", "colour":"time", "linestyle":"terminatorTime"})
#plotOpticalDepths(hdf5Files, hdf5Filenames, subtitle, plot_type={"type":"opticalDepth", "colour":"latitude", "linestyle":"terminatorTime"})


#printFileNames(obspaths, titles, write_to_file=True)
#plotVerticalProfiles(obspaths, hdf5Files, hdf5Filenames, [0,1,2,3])



"""plot CH4 SNR"""
if title in ["ch4 paper 134 apr", "ch4 paper 134 aug", "ch4 paper 135 aug", "detector correction"]:
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
#    for i in hdf5Filenames: print("\""+i[0:15]+"\",")
#    stop()
    obsDicts = correctNonLinearity1(hdf5Files, hdf5Filenames, title)
    
    
    #plot CH4 spectra without correcting non-linearity
#    DETECTOR_REGION = np.arange(160,250)
#    WATER_ABSORPTION = np.arange(33,65)
#    plt.figure()
#    altitudes = []
#    spectra = []
#    for obsDict in obsDicts:
#        x = obsDict["x"][DETECTOR_REGION]
#        plotIndices = range(14,17)
#        for plotIndex in plotIndices:
#            spectrum = obsDict["y"][plotIndex,DETECTOR_REGION]
#            altitudes.append(obsDict["alt"][plotIndex])
#            polyfit = np.polyfit(list(range(WATER_ABSORPTION[0]))+list(range(WATER_ABSORPTION[-1],len(x))), spectrum[list(range(WATER_ABSORPTION[0]))+list(range(WATER_ABSORPTION[-1],len(x)))], 4)
#            polyval = np.polyval(polyfit, range(len(x)))
#            spectrumCorrected = spectrum / polyval
#            spectra.append(spectrumCorrected)
##            plt.plot(spectrumCorrected)
##
#    spectraMean = np.mean(np.asfarray(spectra), axis=0)
#    ySNR = np.ones(len(spectraMean)) * 220 * np.sqrt(12)
#    yErr = (spectraMean * 0.38) / ySNR
#    plt.errorbar(x, spectraMean * 0.38, color="k", yerr = yErr)
#    plt.ylim([0.37, 0.381])




"""plot CH4 SNR"""
if title == "occultation map":
    
    colours = {134:"b", 136:"r"}
    for order in [134,136]:
        obspaths = ["*20180*_0p3a_SO*_%s" %order]
        hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
        centreLats = []
        centreLons = []
        filesPlotted = []
        hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
        for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
            hdf5Timestamp = hdf5Filename[0:15]
            if hdf5Timestamp not in filesPlotted:
                lat = hdf5File["Geometry/Point0/Lat"][:,0]
                lon = hdf5File["Geometry/Point0/Lon"][:,0]
                centreLat = np.mean(lat[lat > -900.0])
                centreLon = np.mean(lon[lon > -900.0])
                centreLats.append(centreLat)
                centreLons.append(centreLon)
                filesPlotted.append(hdf5Timestamp)
            
        plt.scatter(centreLons, centreLats, color=colours[order], label="Order %i" %order)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("NOMAD Occultation Coverage April-September 2018")
    plt.legend(loc=4)

""""part to make simulations!"""
if title == "make simulations""":

    import sys
    
    if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
        print("Running on linux")
#        BASE_DIRECTORY = "/bira-iasb/projects/planetary/iant"
        NOMAD_DIRECTORY = "/bira-iasb/projects/NOMAD"
    
    else:
        print("Running on windows")
#        BASE_DIRECTORY = r"X:\projects\planetary\iant"
        NOMAD_DIRECTORY = r"X:\projects\NOMAD"
        
    sys.path.append(os.path.join(BASE_DIRECTORY, "nomad_tools"))
    sys.path.append(os.path.join(BASE_DIRECTORY, "pytran"))
    #sys.path.append(os.path.join(NOMAD_DIRECTORY, "NOMAD", "Radiative_Transfer", "Tools"))
    
    
    
    import pytran
    
    import nomadtools
    import nomadtools.paths   #need to fix this
    from nomadtools import gem_tools
    
    from NOMAD_instrument import freq_mp, F_blaze, F_aotf_3sinc
    
    
    
    
    def get_solar_hr(nu_hr, ):
        '''  '''
        from scipy import interpolate
    
        solar_file = os.path.join(NOMAD_DIRECTORY, "Science", "Radiative_Transfer", "Auxiliary_files", "Solar", "Solar_irradiance_ACESOLSPEC_2015.dat")
    
        print('Reading in solar file %s'%solar_file)
    
        nu_solar = []
        I0_solar = []
        nu_min = nu_hr[0] - 1.
        nu_max = nu_hr[-1] + 1.
        with open(solar_file) as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            for line in f:
    
                nu, I0 = [float(val) for val in line.split()]
                if nu < nu_min:
                    continue
                if nu > nu_max:
                    break
                nu_solar.append(nu)
                I0_solar.append(I0)
        
        f_solar = interpolate.interp1d(nu_solar, I0_solar)
        I0_solar_hr = f_solar(nu_hr)
    
        return I0_solar_hr
    
    
    class NOMAD_sim(object):
    
        def __init__(self, order=121, adj_orders=2, molecule='H2O', str_min=1.0e-25, iso_num=None, vmr=None,
                    apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime',
                    TangentAlt=None, spec_res=0.2, pixel_shift=0.0):
    
            import numpy as np
            from scipy import interpolate
    
            if TangentAlt is None:
                TangentAlt = np.arange(0., 150., 10.)
            NbZ = len(TangentAlt)
            self.NbZ = NbZ
    
            atmofile = gem_tools.get_apriori_files(name='atmo', apriori_version=apriori_version, apriori_zone=apriori_zone) 
            print('Reading in atmo from ', atmofile)
            atmo_in = {}
            atmo_in['Z'], atmo_in['T'], atmo_in['P'], atmo_in['NT'] = np.loadtxt(os.path.join(nomadtools.paths.GLOBAL_ATMOSPHERE_DIR, atmofile), comments='%', usecols=(0,1,2,3,), unpack=True)
    
            atmo = {}
            atmo['Z'] = TangentAlt[:]
            fun_T = interpolate.interp1d(atmo_in['Z'][::-1], atmo_in['T'][::-1])
            fun_P = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['P'][::-1]))
            fun_NT = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['NT'][::-1]))
            atmo['T'] = [fun_T(z) for z in TangentAlt]
            atmo['P'] = np.exp([fun_P(z) for z in TangentAlt])
            atmo['NT'] = np.exp([fun_NT(z) for z in TangentAlt])
    
            self.atmo = atmo
    
            if vmr is None:
                xa_file, sa_file = gem_tools.get_apriori_files(name=molecule, apriori_version=apriori_version, apriori_zone=apriori_zone)
                print('Reading in apriori vmr from ', xa_file)
                za_in, xa_in = np.loadtxt(os.path.join(nomadtools.paths.GLOBAL_ATMOSPHERE_DIR, xa_file), comments='%', usecols=(0,1,), unpack=True)
                sa_in = np.loadtxt(os.path.join(nomadtools.paths.GLOBAL_ATMOSPHERE_DIR, sa_file), comments='%', usecols=(1,))
                xa_fun = interpolate.interp1d(za_in[::-1], xa_in[::-1])
                sa_fun = interpolate.interp1d(za_in[::-1], sa_in[::-1])
                self.xa = xa_fun(atmo['Z'])*1e-6
                self.sa = sa_fun(atmo['Z'])
            else:
                print('Setting vmr to constant %f ppm'%vmr)
                self.xa = np.ones_like(atmo['Z'])*vmr*1e-6
                self.sa = np.ones_like(atmo['Z'])
    
            self.order = order
            self.adj_orders = adj_orders
    
            nu_hr_min = freq_mp(order-adj_orders, 0) - 5.
            nu_hr_max = freq_mp(order+adj_orders, 320.) + 5.
            dnu = 0.001
            Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
            nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
            print('hih resolution range %.1f to %.1f (with %d points)' % (nu_hr_min, nu_hr_max, Nbnu_hr))
            dnu = nu_hr[1]-nu_hr[0]
            I0_solar_hr = get_solar_hr(nu_hr)
    
            self.Nbnu_hr = Nbnu_hr
            self.nu_hr = nu_hr
            self.dnu = dnu
            self.I0_hr = I0_solar_hr
    
            # 
            HITRANDIR = os.path.join(NOMAD_DIRECTORY, "Science", "Radiative_Transfer", "Auxiliary_files", "Spectroscopy")
            M = pytran.get_molecule_id(molecule)
            filename = os.path.join(HITRANDIR, '%02i_hit16_2000-5000_CO2broadened.par' % M)
            if not os.path.exists(filename):
                filename = os.path.join(HITRANDIR, '%02i_hit16_2000-5000.par' % M)
            LineList = pytran.read_hitran2012_parfile(filename, nu_hr_min, nu_hr_max, Smin=str_min)
            nlines = len(LineList['S'])
            print('Found %i lines' % nlines)
            self.LineList = LineList
            
    
            self.sigma_hr = np.zeros((NbZ,Nbnu_hr))
            self.tau_hr = np.zeros((NbZ,Nbnu_hr))
            for i in range(NbZ):
                print("%d of %d" % (i, NbZ))
                self.sigma_hr[i,:] =  pytran.calculate_hitran_xsec(LineList, M, nu_hr, T=atmo['T'][i], p=atmo['P'][i]*1e3)
    
            self.Trans_hr = np.ones((NbZ,Nbnu_hr))
    
            #
            pixels = np.arange(320)
            NbP = len(pixels)
            self.pixels = pixels
            self.NbP = NbP
            self.nu_p = freq_mp(order, pixels, p0=pixel_shift)
    
            self.Trans_p = np.ones((NbZ,NbP))
    
            print("Computing convolution matrix")
            W_conv = np.zeros((NbP,Nbnu_hr))
            sconv = spec_res/2.355
            for iord in range(order-adj_orders, order+adj_orders+1):
                nu_pm = freq_mp(iord, pixels, p0=pixel_shift)
                W_blaze = F_blaze(iord, pixels)
                for ip in pixels:
                    W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
            self.W_conv = W_conv
    
    
        def forward_model(self, xa_fact=None):
    
            import numpy as np
    
            print("Forward model")
    
            if xa_fact==None:
                xa_fact = np.ones(self.NbZ)
            #print(xa_fact)
    
            #
            Rp = 3396.
            s = np.zeros(self.NbZ)
            dl = np.zeros((self.NbZ,self.NbZ))
            for i in range(self.NbZ):
                s[i:] = np.sqrt((Rp+self.atmo['Z'][i:])**2-(Rp+self.atmo['Z'][i])**2)
                #print(i, s[i:])
                if i < self.NbZ-1:
                    dl[i,i] = s[i+1] - s[i]
                if i < self.NbZ-2:
                    dl[i,(i+1):-1] = s[(i+2):] - s[i:-2]
                dl[i,-1] = s[-1] - s[-2] + 2*10. /np.sqrt(1.-((Rp+self.atmo['Z'][i])/(Rp+self.atmo['Z'][-1]+1.))**2) 
                #print(dl[i,i:])
            dl *= 1e5
            self.dl = dl
            
            #
            self.tau_hr[:,:] = 0.0
            for i in range(self.NbZ):
                for j in range(i,self.NbZ):
                    self.tau_hr[i,:] += (xa_fact[j]*self.xa[j]*self.atmo['NT'][j]*dl[i,j])*self.sigma_hr[j,:]
                self.Trans_hr[i,:] = np.exp(-self.tau_hr[i,:])
    
            # 
            W_aotf = F_aotf_3sinc(self.order, self.nu_hr)
            I0_hr = W_aotf * self.I0_hr             # nhr
            I0_p = np.matmul(self.W_conv, I0_hr)    # np x 1
            I_hr = I0_hr[None,:] * self.Trans_hr    # nz x nhr
            I_p = np.matmul(self.W_conv, I_hr.T).T  # nz x np
            self.Trans_p = I_p / I0_p[None,:]       # nz x np
    
    
    def writeCsvFile(file_name, lines_to_write):
        """function to write to csv file"""
        logFile = open(BASE_DIRECTORY+os.sep+file_name, "w")
        for line_to_write in lines_to_write:
            logFile.write(line_to_write+'\n')
        logFile.close()
    
    
    
    
    TangentAlt = np.arange(0.8, 100., 5.)
    index = 1
    diffractionOrder = 134
    #diffractionOrder = 135
    
    #vmr = 0.250 / 1.0e3
    #vmr = 0.5 / 1.0e3 #ppbv
    #vmr = 1.0 / 1.0e3 #ppbv
    vmr = 15.0#ppmv for water only
    
    #molecule = "CH4"
    molecule = "H2O"
    #spec_res=0.15
    #spec_res=0.2
    spec_res=0.22
    
    if diffractionOrder == 134:
        pixel_shift = -6.0279 #134
    elif diffractionOrder == 135:
        pixel_shift = -9.0207 #135
    
    if molecule == "CH4":
        str_min = 1.0e-20
    if molecule == "H2O":
        str_min = 1.0e-25
    
    sim = NOMAD_sim(diffractionOrder, adj_orders=1, molecule=molecule, vmr=vmr, str_min=str_min, TangentAlt=TangentAlt, pixel_shift=pixel_shift, spec_res=spec_res)
    sim.forward_model()
    
    if diffractionOrder == 134:
        print(sim.nu_p[157])
        print(sim.nu_p[158])
        desired_wavenumber = 3022.7391 #first value in data
        sim_wavenumber = sim.nu_p[157]
        sim_wavenumber_delta = sim.nu_p[157] - sim.nu_p[158]
    elif diffractionOrder == 135:
        print(sim.nu_p[160])
        print(sim.nu_p[161])
        desired_wavenumber = 3045.2974
        sim_wavenumber = sim.nu_p[160]
        sim_wavenumber_delta = sim.nu_p[160] - sim.nu_p[161]
    
    
    desired_pixel_shift = (desired_wavenumber - sim_wavenumber) / sim_wavenumber_delta
    print(pixel_shift - desired_pixel_shift)
    
    outputLines = []
    for wavenumber, transmittance in zip(sim.nu_p, sim.Trans_p[index,:]):
        outputLines.append("%0.4f %0.5f" %(wavenumber, transmittance))
    
    print("%0.2fkm" %TangentAlt[index])
    writeCsvFile("trans%i_%s_%s_%s_for_paper.txt" %(diffractionOrder, molecule, str(vmr * 1000).replace(".","p"), str(spec_res).replace(".","p")), outputLines)
