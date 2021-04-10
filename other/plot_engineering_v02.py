# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:18:16 2018

@author: iant

PLOT TEMPERATURES AND OTHER ENGINEERING DATA

"""


import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime
# import re
#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap


#from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

from tools.file.hdf5_functions import make_filelist, get_file
from tools.file.paths import FIG_X, FIG_Y, paths
#from analysis_functions_v01b import spectralCalibration,write_log,get_filename_list,stop
#from filename_lists_v01 import getFilenameList

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    print("Running on windows")
    import spiceypy as sp
    
    from tools.spice.load_spice_kernels import load_spice_kernels
    
    load_spice_kernels()


#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True




####CHOOSE FILENAMES######

"""plot engineering temperatures"""
#fileLevel = "hdf5_level_0p1a"
#obspaths = ["*2018*_0p1a_"]
#obspaths = ["*201804*_0p1a_"]
#obspaths = ["*201805*_0p1a_"]
#obspaths = ["*201810*_0p1a_"]


"""plot so/lno detector cooldowns"""
#fileLevel = "hdf5_level_0p3a"
#obspaths = ["*2018*_0p3a_SO_1*_190"]; channel="so"
#obspaths = ["*2018*_0p3a_LNO_1*_168"]; channel="lno"


"""check n spectra"""
#fileLevel = "hdf5_level_0p1d"
#obspaths = ["*2018052*_0p1d_LNO"]


"""make list of ls vs time"""
#None


"""get coefficients or simple dataset from a range of files"""
#fileLevel = "hdf5_level_0p3a"
#obspaths = ["*2018*_0p3a_LNO_*_L_164"]


"""plot relative signal strengths of SO channel bins to check alignment"""
fileLevel = "hdf5_level_0p1a"
obspaths = ["*201812*_0p1a_SO_1"]


"""plot dark current for 5 x darks"""
#fileLevel = "hdf5_level_0p1a"
#regex = re.compile("201901[0-9][0-9]_.*_0p1a_SO_1")
#regex = re.compile("20181223_045314_0p1a_SO_1") #5x order 135
#regex = re.compile("20181223_084859_0p1a_SO_1") #119, #130, #145, #171, #191, #dark
#    "20181229_165026_0p1a_SO_1", #5x dark egress
#    "20181229_204608_0p1a_SO_1", #5x dark egress
#    "20181223_045314_0p1a_SO_1", #1x dark ingress
#    "20190116_182249_0p1a_SO_1", #5x dark ingress
#title = "plot dark residual"

"""plot diffraction order statistics"""
# fileLevel = "hdf5_level_1p0a"
#regex = re.compile("201[89][01][0-9][0-9][0-9]_.*_SO_.*")
#regex = re.compile("(20191[1-2]|202001)[0-9][0-9]_.*_SO_.*")
#regex = re.compile("(20191[1-2]|202001)[0-9][0-9]_.*_SO_.*")
#regex = re.compile("20200[5-6][0-9][0-9]_.*_SO_.*")
#title = "so order statistics"
#regex = re.compile("201[89][01][0-9][0-9][0-9]_.*_LNO_.*")
#regex = re.compile("(20191[1-2]|202001)[0-9][0-9]_.*_LNO_.*")
# regex = re.compile("20200[5-6][0-9][0-9]_.*_LNO_.*")
# title = "lno order statistics"
#regex = re.compile("201[89][01][0-9][0-9][0-9]_.*_UVIS_.*")
#title = "uvis obs type statistics"



"""check uvis pipeline progression"""
#fileLevel = "hdf5_level_1p0a"
#regex = re.compile("201[89][01][0-9][0-9][0-9]_.*UVIS.*")
#title = "make pipeline level lists" #run this first
#title = "check file progression"

"""plot uvis statistics"""




############FUNCTIONS#############
def getHdf5Temperature(hdf5File, obspath, channel):
    print("Reading in file %s" %(obspath))

    if channel == "so":
        temperature_field = "SENSOR_1_TEMPERATURE_SO"
        hsk_time_strings = hdf5File["Housekeeping/DateTime"][...]
        end_index = 600
        indices = np.arange(1, np.min([end_index, len(hsk_time_strings)]), 10)
    elif channel == "lno":
        temperature_field = "SENSOR_1_TEMPERATURE_LNO"
        hsk_time_strings = hdf5File["Housekeeping/DateTime"][...]
        end_index = 40
        indices = np.arange(1, np.min([end_index, len(hsk_time_strings)]), 5)
    elif channel == "uvis":
        temperature_field = "TEMP_2_CCD"
        hsk_time_strings = hdf5File["DateTime"][...]
        indices = np.arange(1, len(hsk_time_strings), 10)

    hsk_time_strings = hsk_time_strings[indices]
    variable = hdf5File["Housekeeping/%s" %temperature_field][indices]
    hsk_time = [sp.utc2et(hsk_time_string) for hsk_time_string in hsk_time_strings]
    

    return hsk_time, variable 
    



def getHdf5DetectorTemperature(hdf5File, obspath, channel):
    print("Reading in file %s" %(obspath))

    if channel == "so":
        temperature_field = "FPA1_FULL_SCALE_TEMP_SO"
        start_index = 1
        end_index = 590
    elif channel == "lno":
        temperature_field = "FPA1_FULL_SCALE_TEMP_LNO"
        start_index = 1
        end_index = 40

    hsk_time_strings = hdf5File["Housekeeping/DateTime"][start_index:end_index]
    variable = hdf5File["Housekeeping/%s" %temperature_field][start_index:end_index]
    hsk_time = [sp.utc2et(hsk_time_string) for hsk_time_string in hsk_time_strings]

    return hsk_time, variable 



def writeOutput(filename, lines_to_write):
    """function to write output to a file"""
    outFile = open("%s.txt" %filename, 'w')
    for line_to_write in lines_to_write:
        outFile.write(line_to_write+'\n')
    outFile.close()


def writeHdf5Temperatures(hdf5Filenames, obspaths, plot_from_file=False):
    
    """search through all files, write temperatures to file, and plot"""
#    plot_from_file = False
    """read temperatures from previous made file, and plot"""
    plot_from_file = True
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(FIG_X, FIG_Y))
    
    if not plot_from_file:
        so_times = []
        lno_times = []
        uvis_times = []
        so_temperatures = []
        lno_temperatures = []
        uvis_temperatures = []
        for file_index, hdf5_filename in enumerate(hdf5Filenames):
            name, hdf5_file = get_file(hdf5_filename, fileLevel, file_index)
            
            if "SO" in hdf5_filename:
                channel = "so"
            elif "LNO" in hdf5_filename:
                channel = "lno"
            elif "UVIS" in hdf5_filename:
                channel = "uvis"
            time, temperature = getHdf5Temperature(hdf5_file, hdf5_filename, channel)
            hdf5_file.close()
        
            if channel == "so":
                so_times.append(time)
                so_times.append([np.nan])
                so_temperatures.append(temperature)
                so_temperatures.append([np.nan])
            elif channel == "lno":
                lno_times.append(time)
                lno_times.append([np.nan])
                lno_temperatures.append(temperature)
                lno_temperatures.append([np.nan])
            elif channel == "uvis":
                uvis_times.append(time)
                uvis_times.append([np.nan])
                uvis_temperatures.append(temperature)
                uvis_temperatures.append([np.nan])
        
        
        so_times = [item for sublist in so_times for item in sublist]
        lno_times = [item for sublist in lno_times for item in sublist]
        uvis_times = [item for sublist in uvis_times for item in sublist]
        so_temperatures = [item for sublist in so_temperatures for item in sublist]
        lno_temperatures = [item for sublist in lno_temperatures for item in sublist]
        uvis_temperatures = [item for sublist in uvis_temperatures for item in sublist]
        
    else:
        
#        startingIndex = 1
#        endingIndex = 1000
        with open("NOMAD_Temperatures_2018_0p1a_.txt") as temperatureFile:
            temperatureLines = temperatureFile.readlines()
        
        so_times = []
        lno_times = []
        uvis_times = []
        so_temperatures = []
        lno_temperatures = []
        uvis_temperatures = []
        for lineIndex, line in enumerate(temperatureLines):
#            if startingIndex < lineIndex < endingIndex:
            if lineIndex > 0:
                content = line.split(",")
                if content[0] == "SO":
                    so_times.append(np.float(content[1]))
                    so_temperatures.append(np.float(content[2]))
                elif content[0] == "LNO":
                    lno_times.append(np.float(content[1]))
                    lno_temperatures.append(np.float(content[2]))
                elif content[0] == "UVIS":
                    uvis_times.append(np.float(content[1]))
                    uvis_temperatures.append(np.float(content[2]))
    startTime = lno_times[0]
    endTime = lno_times[-2]
    timeRange = np.arange(startTime, endTime, (3600 * 24 * 30))
    
    def getDate(et):
        return sp.et2utc(et, "C", 0)[0:11]
    
    ax1.plot(uvis_times, uvis_temperatures, "r", label="UVIS", alpha=0.5)
    ax1.plot(so_times, so_temperatures, "g", label="SO", alpha=1)
    ax1.plot(lno_times, lno_temperatures, "b", label="LNO", alpha=1)
#    ax1.scatter(uvis_times, uvis_temperatures, c="r", label="UVIS", alpha=0.5)
#    ax1.scatter(so_times, so_temperatures, c="g", label="SO", alpha=0.5)
#    ax1.scatter(lno_times, lno_temperatures, c="b", label="LNO", alpha=0.5)
    ax1.legend()
    ax1.set_ylabel("Temperature (C)")
    if plot_from_file:
        ax1.set_title("NOMAD Temperatures %s - %s" %(getDate(startTime),getDate(endTime)))
    else:
        ax1.set_title("NOMAD Temperatures %s - %s" %(hdf5Filenames[0].split("_")[0],hdf5Filenames[-1].split("_")[0]))
    plt.gca().xaxis.set_major_locator(mtick.FixedLocator(timeRange)) # Set tick locations
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,p:getDate(x)))
    
    if not plot_from_file:
        output = ["Channel,Time,Temperature"]
        soOutput = ["SO,%s,%s" %(so_time, so_temperature) for so_time, so_temperature in zip(so_times, so_temperatures)]
        lnoOutput = ["LNO,%s,%s" %(lno_time, lno_temperature) for lno_time, lno_temperature in zip(lno_times, lno_temperatures)]
        uvisOutput = ["UVIS,%s,%s" %(uvis_time, uvis_temperature) for uvis_time, uvis_temperature in zip(uvis_times, uvis_temperatures)]
        writeOutput(BASE_DIRECTORY+os.sep+"NOMAD_Temperatures_"+obspaths[0].replace("*",""), output + soOutput + lnoOutput + uvisOutput)


def getNumberOfSpectra(obspaths, fileLevel):
    """print number of spectra in each file"""
    hdf5Files, hdf5Filenames, titles = make_filelist(obspaths, fileLevel)
    for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames):
        detectorData = hdf5File["Science/Y"][...]
        nSpectra = detectorData.shape[0]
#        if nSpectra == 920:
        print("nSpectra=%i, hdf5Filename=%s" %(nSpectra, hdf5Filename))


def writeTimeLsToFile():
    """make list of time vs ls"""
    SPICE_TARGET = "MARS"
    SPICE_ABERRATION_CORRECTION = "None"
    
    DATETIME_FORMAT = "%d/%m/%Y %H:%M"
    
    
    from datetime import datetime, timedelta
    
    linesToWrite = []
    datetimeStart = datetime(2018, 3, 1, 0, 0, 0, 0)
    for hoursToAdd in range(0, 24*31*12*3, 6): #3 years
        newDatetime = (datetimeStart + timedelta(hours=hoursToAdd)).strftime(DATETIME_FORMAT)
        ls = sp.lspcn(SPICE_TARGET, sp.utc2et(str(datetimeStart + timedelta(hours=hoursToAdd))), SPICE_ABERRATION_CORRECTION) * sp.dpr()
        linesToWrite.append("%s\t%0.1f" %(newDatetime, ls))
    
    writeOutput("Time_vs_Ls.txt", linesToWrite)

def getDatasetFromFiles(hdf5_files, hdf5_filenames, dataset_path, first_value_only=False):
    dataset_out = []
    for hdf5_file, hdf5_filename in zip(hdf5_files, hdf5_filenames):
        dataset = hdf5_file[dataset_path][0]
        if first_value_only:
            dataset_out.append(dataset[0])
        else:
            dataset_out.append(dataset)
    return dataset_out
    

def plotDetectorTemperatures(hdf5_files, hdf5_filenames, channel):

    cmap = plt.get_cmap('jet')
    n_files = len(hdf5_filenames)
    
    #colour denotes file index
    colours = [cmap(i) for i in np.arange(n_files)/n_files]

    plt.subplots(1, 1, figsize=(FIG_X, FIG_Y))
    plt.plot([0,600],[85,85], "--k")
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        timeList, temperature = getHdf5DetectorTemperature(hdf5_file, hdf5_filename, channel)
        timeDelta = np.asfarray(timeList) - timeList[0]

        if temperature[0]>250.0:
            plt.plot(timeDelta, temperature, alpha=0.1, color=colours[file_index])

    plt.title("%s detector cooldown times %s - %s" %(channel.upper(), hdf5_filenames[0][0:8], hdf5_filenames[-1][0:8]))
    plt.ylabel("Detector Temperature (K)")
    plt.xlabel("Time (seconds)")
    plt.xlim([0,600])
#    plt.yscale("log")
    print("%i files found" %n_files)
    


def getOccultationReferenceCounts(hdf5File, chosen_aotf_frequency=-999.0):

#    obs_start_time = hdf5File["Timestamp"][0]
    obs_start_string = hdf5File["Geometry/ObservationDateTime"][0][0].decode("utf-8")
    
    aotf_freq = hdf5File["Channel/AOTFFrequency"][...]
    
    if chosen_aotf_frequency == -999.0:
        chosen_aotf_frequency = aotf_freq[0]
    matching_indices_boolean = aotf_freq == chosen_aotf_frequency
    matching_indices = np.where(matching_indices_boolean == True)[0]
    
    if len(matching_indices) == 0:
        print("AOTF frequency %0.0f not found in file; skipping" %chosen_aotf_frequency)
        return 0, "", 0, np.zeros(4)
    else:
#        print("AOTF frequency %0.0f found; getting info" %chosen_aotf_frequency)
        first_match = matching_indices[0]
        last_match = matching_indices[-1]
        binned_counts_start = np.mean(hdf5File["Science/Y"][first_match,:,160:240], axis=1)
        binned_counts_end = np.mean(hdf5File["Science/Y"][last_match,:,160:240], axis=1)
        
#        binned_counts_start = np.mean(hdf5File["Science/Y"][matching_indices[0:5],:,160:240], axis=(0,2))
#        binned_counts_end = np.mean(hdf5File["Science/Y"][matching_indices[-6:-1],:,160:240], axis=(0,2))
        
        if np.mean(binned_counts_start) > np.mean(binned_counts_end): #if ingress
            max_counts = np.max(binned_counts_start)
            counts_out = binned_counts_start / max_counts
        else:
            max_counts = np.max(binned_counts_end)
            counts_out = binned_counts_end / max_counts
    
        return obs_start_string, max_counts, counts_out


def plotBinStrengths(hdf5Files, hdf5Filenames, obspaths):
    et_string_all = []
    max_counts_all = []
    relative_counts_all = []
    for fileIndex, (hdf5File, hdf5Filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        print("%i/%i: Reading in file %s" %(fileIndex, len(hdf5Filenames), hdf5Filename))

    #    et, et_string, max_counts, relative_counts = getOccultationReferenceCounts(hdf5File, hdf5Filename, 17859.0)
        et_string, max_counts, relative_counts = getOccultationReferenceCounts(hdf5File)
        
        if len(relative_counts) == 4: #just take nominal 6 order data
            et_string_all.append(et_string)
            max_counts_all.append(max_counts)
            relative_counts_all.append(relative_counts)
            
            if np.min(relative_counts) < 0.9:
                print("File %s has minimum relative counts of %0.2f (max counts = %0.0f)" %(hdf5Filename, np.min(relative_counts), max_counts))
    
    relative_counts_array = np.asfarray(relative_counts_all)
    et_array = np.asfarray([sp.utc2et(string) for string in et_string_all])
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title("SO channel relative counts for each bin\nSearch string: %s" %obspaths[0])
    for bin_index in range(4):
        plt.scatter(et_array, relative_counts_array[:,bin_index], label="Bin %i" %bin_index, marker=".")
    plt.xlabel("Ephemeris Time (s)")
    plt.ylabel("Relative counts for each bin")
    plt.legend()
    plt.grid(True)
    
    months = np.arange(4, 13, 1)
    monthStarts = [sp.utc2et(datetime(2018, month, 1).strftime("%Y-%m-%d")) for month in months]
    monthNames = [datetime(2018, month, 1).strftime("%B") for month in months]
    for monthStart, monthName in zip(monthStarts, monthNames):
        plt.axvline(x=monthStart, color='k', linestyle='--')
        plt.text(monthStart+100000, 0.7, monthName)
    


def plotDarkCurrentResidual(regex, fileLevel, diffractionOrder):
#if True:
    diffractionOrder = 171  
    chosenAotfFrequency = {119:15657, 132:17566, 133:17712, 134:17859, 135:18005, 136:18152, 149:20052, \
                           167:22674, 168:22820, 169:22965, 170:23110, 171:23255, 189:25864, 190:26008, 191:26153}[diffractionOrder]

    hdf5Files, hdf5Filenames, titles = make_filelist(regex, fileLevel)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        print("%i/%i: Reading in file %s" %(fileIndex, len(hdf5Filenames), hdf5_filename))
        aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
        
        #plot 1x dark frames only
    #    if aotfFrequency[0] in singleAotfFrequencies and aotfFrequency[0] == aotfFrequency[1] == aotfFrequency[2] == aotfFrequency[3]:
        #plot 5x dark frames only
    #        if aotfFrequency[0] in singleAotfFrequencies and aotfFrequency[1] == aotfFrequency[2] == aotfFrequency[3] == aotfFrequency[4] == 0.0:
        if True:
            print("%s, #%i" %(hdf5_filename, diffractionOrder))
    
            detectorData = hdf5_file["Science/Y"][...]
            lightIndices = np.where(aotfFrequency == chosenAotfFrequency)[0]
            darkIndices = np.where(aotfFrequency == 0.0)[0]
            detectorDataLight = np.asfarray([detectorData[index,:,:] for index in lightIndices])
            detectorDataDark = np.asfarray([detectorData[index,:,:] for index in darkIndices])
            
            chosenPixel = 200
            
            for chosenBin in [0, 1, 2, 3]:
                ax1.scatter(darkIndices, detectorDataDark[:, chosenBin, chosenPixel], s=5, marker="*", label="%s dark bin %i" %(hdf5_filename, chosenBin))
                ax2.plot(lightIndices, detectorDataLight[:, chosenBin, chosenPixel], label="%s light bin %i" %(hdf5_filename, chosenBin))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax2.set_xlabel("Frame index")
    ax1.set_ylabel("Dark Frame Counts, Pixel %i" %(chosenPixel))
    ax2.set_ylabel("Light Frame Counts, Pixel %i" %(chosenPixel))
    
    if hdf5_filename == "20181223_045314_0p1a_SO_1":
        sun_light_index = 1000 #light frame above atmosphere
        sun_dark_index = 200 #dark frame above atmosphere
        atmos_light_index = 1110 #light frame in atmosphere
        atmos_dark_index = 222 #dark frame in atmosphere
        mars_dark_index = 240 #real dark frame viewing mars
    elif hdf5_filename == "20190116_182249_0p1a_SO_1":
        sun_dark_index = 2338
        mars_dark_index = 3666
    elif hdf5_filename == "20181223_084859_0p1a_SO_1":
        sun_light_index = 175 #light frame above atmosphere
        sun_dark_index = 175 #dark frame above atmosphere
        if diffractionOrder == 171:
            atmos_light_index = 218 #light frame in atmosphere
            atmos_dark_index = 218 #dark frame in atmosphere
        if diffractionOrder == 191:
            atmos_light_index = 216 #light frame in atmosphere
            atmos_dark_index = 216 #dark frame in atmosphere
        mars_dark_index = 225 #real dark frame viewing mars
    else:
        stop()
    
    #generate pixel bin map
    subtractedResidualMap = np.asfarray([detectorDataDark[sun_dark_index, index, :] - detectorDataDark[mars_dark_index, index, :] for index in range(4)])
    plt.figure()
    plt.imshow(subtractedResidualMap, aspect=20) #where 300000 is the light frame counts
    plt.colorbar()
    plt.xlabel("Pixel Number")
    plt.ylabel("Bin Number")
    plt.title("Signal difference between solar dark frame and mars dark frame")

    plt.figure()
    for chosenBin in [0, 1, 2, 3]:
        plt.plot(detectorDataDark[sun_dark_index, chosenBin, :] - detectorDataDark[mars_dark_index, chosenBin, :], label="Bin %i" %chosenBin)
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Signal difference")
    plt.title("Signal difference between solar dark frame and mars dark frame")
    
    
    #see effect on transmittance using contaminated dark and mars dark
    chosenBin = 3
    sun_spectrum_straylight = detectorDataLight[sun_light_index, chosenBin, :] - detectorDataDark[sun_dark_index, chosenBin, :]
    sun_spectrum = detectorDataLight[sun_light_index, chosenBin, :] - detectorDataDark[mars_dark_index, chosenBin, :]
    
    atmos_spectrum_straylight = detectorDataLight[atmos_light_index, chosenBin, :] - detectorDataDark[atmos_dark_index, chosenBin, :]
    atmos_spectrum = detectorDataLight[atmos_light_index, chosenBin, :] - detectorDataDark[mars_dark_index, chosenBin, :]
    
    transmittance_straylight = atmos_spectrum_straylight / sun_spectrum_straylight
    transmittance = atmos_spectrum / sun_spectrum
    
    plt.figure()
    plt.plot(transmittance_straylight, label="Normal transmittance calculation")
    plt.plot(transmittance, label="Transmittance using true darks")
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Transmittance")
    plt.title("Transmittance calculation comparison, \n%s order %i" %(hdf5_filename, diffractionOrder))



"""plot statistics for each diffraction order"""
def plotDiffractionOrderBarChart(regex, fileLevel, channel):
    hdf5Files, hdf5Filenames, titles = make_filelist(regex, fileLevel, open_files=False, silent=True)
    
    if channel in ["so", "lno"]:
        orders = range(200)
        diffractionOrderFilenames = [[] for _ in orders]
        for hdf5_filename in hdf5Filenames:
            hdf5_filename_split = hdf5_filename.split("_")
            if len(hdf5_filename_split) == 7:
                diffractionOrder = hdf5_filename.split("_")[6]
                diffractionOrderFilenames[int(diffractionOrder)].append(hdf5_filename)
    #        else:
    #            if hdf5_filename_split[-1] == "S":
    #                name, hdf5_file = get_file(hdf5_filename, fileLevel, 0, silent=True)
    #                detectorData = hdf5_file["Science/Y"][...]
    #                nSpectra = detectorData.shape[0]
    #                print("nSpectra=%i, hdf5_filename=%s" %(nSpectra, hdf5_filename))
    #                print(list(set(list(hdf5_file["Channel/DiffractionOrder"][...]))))
                
        nDiffractionOrders = [len(values) for values in diffractionOrderFilenames]
        
        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.bar(orders[118:198], nDiffractionOrders[118:198])
        plt.title("Number of times each diffraction order was measured for search %s" %regex.pattern)
        plt.ylabel("Number of observations")
        plt.xlabel("Diffraction Order")
        if SAVE_FIGS:
            plt.savefig(os.path.join(paths["BASE_DIRECTORY"], "%s_diffraction_order_statistics.png" %channel.lower()))
    
    elif channel in ["uvis"]:

        uvisObsTypes = {
                "Occultation\nfull spectrum\nfull resolution":0, \
                "Occultation\nfull spectrum\nreduced resolution":1, \
                "Occultation\nreduced spectrum\nfull resolution":2, \
                "Nadir\nfull spectrum\nfull resolution":3, \
                "Nadir\nfull spectrum\nreduced resolution":4, \
                "Nadir\nreduced spectrum\nfull resolution":5, \
                }
        uvisObsTypeFilenames = [[] for _ in list(uvisObsTypes.keys())]
        
        for file_index, hdf5_filename in enumerate(hdf5Filenames):
            if len(hdf5Filenames) > 100:
                if np.mod(file_index, 100) == 0:
                    print("Processing %s files %i/%i" %(fileLevel, file_index, len(hdf5Filenames)))


            hdf5_filename_split = hdf5_filename.split("_")
            if len(hdf5_filename_split) == 5:
                observationType = hdf5_filename.split("_")[4]
                
                if observationType in ["I", "E"]:
                    name, hdf5_file = get_file(hdf5_filename, fileLevel, 0, silent=True)
                    h_end = hdf5_file["Channel/HEnd"][0]
                    if h_end == 1047:
                        binning = hdf5_file["Channel/HorizontalAndCombinedBinningSize"][0]
                        if binning == 0:
                            index = uvisObsTypes["Occultation\nfull spectrum\nfull resolution"]
                        else:
                            index = uvisObsTypes["Occultation\nfull spectrum\nreduced resolution"]

                    elif h_end < 1047:
                        index = uvisObsTypes["Occultation\nreduced spectrum\nfull resolution"]


                elif observationType in ["D"]:
                    name, hdf5_file = get_file(hdf5_filename, fileLevel, 0, silent=True)
                    h_end = hdf5_file["Channel/HEnd"][0]
                    if h_end == 1047:
                        binning = hdf5_file["Channel/HorizontalAndCombinedBinningSize"][0]
                        if binning == 0:
                            index = uvisObsTypes["Nadir\nfull spectrum\nfull resolution"]
                        else:
                            index = uvisObsTypes["Nadir\nfull spectrum\nreduced resolution"]

                    elif h_end < 1047:
                        index = uvisObsTypes["Nadir\nreduced spectrum\nfull resolution"]
                    
                uvisObsTypeFilenames[index].append(hdf5_filename)
        
        
        nObsTypes = [len(values) for values in uvisObsTypeFilenames]
        
        plt.figure(figsize=(FIG_X+1, FIG_Y+1))
        plt.bar(range(len(uvisObsTypes)), nObsTypes)
        plt.title("Number of times each observation type was measured for search %s" %regex.pattern)
        plt.ylabel("Number of observations")
        plt.xlabel("Observation Type")
        
        plt.xticks(range(len(uvisObsTypes)), list(uvisObsTypes.keys()))
        plt.tight_layout()
        
        if SAVE_FIGS:
            plt.savefig(os.path.join(BASE_DIRECTORY, "%s_observation_type_statistics.png" %channel.lower()))

                



"""plot instrument temperatures"""
#hdf5Files, hdf5Filenames, titles = make_filelist(obspaths, fileLevel, open_files=False)
#writeHdf5Temperatures(hdf5Filenames)

#plot from file
#writeHdf5Temperatures(["20180315_","20181020_"], plot_from_file=True)

"""write number of files"""
#getNumberOfSpectra(regex, fileLevel)


"""plot detector temperature variations"""
#hdf5Files, hdf5Filenames, titles = make_filelist(obspaths, fileLevel)
#plotDetectorTemperatures(hdf5Files, hdf5Filenames, channel)


"""write list of Ls to file"""
#writeTimeLsToFile()



"""compare spectral coefficients and temperature dependent shifts"""
#pixelSpectralCoefficients = getDatasetFromFiles(hdf5Files, hdf5Filenames, "Channel/PixelSpectralCoefficients", first_value_only=True)
#pixel1 = getDatasetFromFiles(hdf5Files, hdf5Filenames, "Channel/Pixel1", first_value_only=True)
#temperature = getDatasetFromFiles(hdf5Files, hdf5Filenames, "Housekeeping/SENSOR_2_TEMPERATURE_LNO")
#xAll = getDatasetFromFiles(hdf5Files, hdf5Filenames, "Science/X", first_value_only=True)
#x0 = [value[0] for value in xAll]
#x319 = [value[319] for value in xAll]


"""plot relative signal strengths of SO channel bins to check alignment"""
hdf5Files, hdf5Filenames, titles = make_filelist(obspaths, fileLevel)
plotBinStrengths(hdf5Files, hdf5Filenames, obspaths)



"""plot dark current for 5 x darks and residual signal on detector bins"""
#if title == "plot dark residual":
#    plotDarkCurrentResidual(regex, fileLevel, 149)





if title == "so order statistics":
    channel = "so"
    plotDiffractionOrderBarChart(regex, fileLevel, channel)
if title == "lno order statistics":
    channel = "lno"
    plotDiffractionOrderBarChart(regex, fileLevel, channel)
if title == "uvis obs type statistics":
    channel = "uvis"
    plotDiffractionOrderBarChart(regex, fileLevel, channel)
    






"""check uvis pipeline progression"""
if title == "make pipeline level lists": #run this first
    """make file lists"""

    fileLevels = ["hdf5_level_0p1a", "hdf5_level_0p2a", "hdf5_level_1p0a"]
    
    for fileLevel in fileLevels:
        hdf5Files, hdf5Filenames, _ = make_filelist(regex, fileLevel, open_files=False, silent=True)
        acceptedFilenames = []
        
        for file_index, (hdf5_filepath, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
            if len(hdf5Filenames) > 100:
                if np.mod(file_index, 100) == 0:
                    print("Processing %s files %i/%i" %(fileLevel, file_index, len(hdf5Filenames)))
    
    
            hdf5_file = h5py.File(hdf5_filepath, "r")
            hStarts = hdf5_file["Channel/HStart"][...]
            hStart = hStarts[0]
            hEnd = hdf5_file["Channel/HEnd"][0]
            nFrames = len(hStarts)
            vStart = hdf5_file["Channel/VStart"][0]
            vEnd = hdf5_file["Channel/VEnd"][0]
            
            mode = hdf5_file["Channel/Mode"][0] #1=SO, 2=Nadir. Higher values=Calibration
            acquistionMode = hdf5_file["Channel/AcquisitionMode"][0] #0=unbinned, 1=vertical binning, 2=horizontal /square binning
            integrationTime = hdf5_file["Channel/IntegrationTime"][0]
        
    #        print(hdf5_filename, mode, acquistionMode, vStart, vEnd, hStart, hEnd, integrationTime)
            acceptedFilenames.append("%s\t%i\t%i\t%i\t%i\t%i\t%i\t%i" %(hdf5_filename, mode, acquistionMode, vStart, vEnd, hStart, hEnd, integrationTime))
        writeOutput(os.path.join(BASE_DIRECTORY, "uvis_filelist_%s" %fileLevel), acceptedFilenames)
    

if title == "check file progression": #use previously generated file lists
    """read in existing file lists"""

    fileLevelsDict = {"hdf5_level_0p1a":[], "hdf5_level_0p2a":[], "hdf5_level_1p0a":[]}
    outputLines = ["### mode, acquistionMode, vStart, vEnd, hStart, hEnd, integrationTime ###"]
    for fileLevel in fileLevelsDict.keys():
        
        with open(os.path.join(BASE_DIRECTORY, "uvis_filelist_%s.txt" %fileLevel), "r") as f:
            lines = f.readlines()
            lines_split = [[line.split("\t")[index] if index==0 else int(line.split("\t")[index]) for index in range(len(line.split()))] for line in lines]
            
            filenameDict = {}
            for line_split in lines_split:
                filenameDict[line_split[0]] = line_split[1:]
            fileLevelsDict[fileLevel] = filenameDict
                    
    for filename01a, fileproperties01a in fileLevelsDict["hdf5_level_0p1a"].items():
    #    print("checking", filename01a)
        fileTime01a = filename01a[0:15]
        
        found02a=False
        for filename02a, fileproperties02a in fileLevelsDict["hdf5_level_0p2a"].items():
            fileTime02a = filename02a[0:15]
            if fileTime01a == fileTime02a:
                found02a=True
                foundFilename02a = filename02a
                foundFileproperties02a = fileproperties02a
                
        fileTime02a = ""
        filename02a = ""
        fileproperties02a = ""
        
        if not found02a:
            comment = "***"
            if fileproperties01a[0] > 2:
                comment += " calibration mode!"
            if fileproperties01a[1] == 1:
                comment += " vertically binned!"
            if fileproperties01a[1] == 2:
                comment += " horizontally binned!"
            
            if comment == "***":
                comment += "why?"
            print(filename01a, "not found in 0p2a (", ", ".join([str(i) for i in fileproperties01a]), ")", comment)
            outputLines.append(filename01a + " not found in 0p2a (" + ", ".join([str(i) for i in fileproperties01a]) + ") " + comment)
        else:
    #        print(filename01a, "found in 0p2a", ", ".join([str(i) for i in fileproperties02a]))
            found10a=False
            foundFilename10a = [] #merged can be split
            for filename10a, fileproperties10a in fileLevelsDict["hdf5_level_1p0a"].items():
                fileTime10a = filename10a[0:15]
                if fileTime01a == fileTime10a:
                    found10a=True
                    foundFilename10a.append(filename10a)
            if found10a:
                True
    #            if len(foundFilename10a) == 1:
    #                print(filename01a, "-->", foundFilename02a, "-->", foundFilename10a[0])
    #                outputLines.append(filename01a + " --> " + foundFilename02a + " --> " + foundFilename10a[0] + " ("+ ", ".join([str(i) for i in fileproperties10a]) + ")")
    #            else:
    #                print(filename01a, "-->", foundFilename02a, "-->", " & ".join(foundFilename10a))
    #                outputLines.append(filename01a + " --> " + foundFilename02a + " --> " + " & ".join(foundFilename10a) + " ("+ ", ".join([str(i) for i in fileproperties10a]) + ")")
            else:
                comment = "***"
    #            if foundFileproperties02a[6] > 14000:
    #                comment += " saturation?"
                if foundFileproperties02a[0] > 2:
                    comment += " calibration mode!"
                if foundFileproperties02a[1] == 1:
                    comment += " vertically binned!"
                if foundFileproperties02a[1] == 2:
                    comment += " horizontally binned!"
                if "UVIS_C" in foundFilename02a:
                    comment += " calibration C observation!" 
                if "UVIS_L" in foundFilename02a:
                    comment += " limb L observation!" 
                if comment == "***":
                    comment += "why?"
                print(filename01a, "found in 0p2a but not found in 1p0a (", ", ".join([str(i) for i in fileproperties10a]), ")", comment)
                outputLines.append(filename01a + " found in 0p2a (%s) but not found in 1p0a " %foundFilename02a + " (" + ", ".join([str(i) for i in foundFileproperties02a]) + ") " + comment)
    
    writeOutput(os.path.join(BASE_DIRECTORY, "uvis_errorlist"), outputLines)




