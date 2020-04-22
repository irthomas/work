# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:37:52 2016

@author: ithom

MORE CALIBRATION ANALYSIS
"""

import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime

#import bisect
#from mpl_toolkits.basemap import Basemap
#from scipy import interpolate, signal

from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#from hdf5_functions_v02b import get_hdf5_attributes,get_dataset_contents,write_to_hdf5
#from analysis_functions_v01b import chisquared,get_spectra,savitzky_golay


from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import write_to_hdf5
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, getFile, makeFileList, printFileNames
from hdf5_functions_v03 import getFilesFromDatastore
#from analysis_functions_v01b import spectralCalibration,write_log,get_filename_list,stop
from filename_lists_v01 import getFilenameList



#if os.path.exists(os.path.normcase(r"X:\linux\Data")):
##    DATA_DIRECTORY = os.path.normcase(r"X:\projects\NOMAD\data\db_test\test\iant\hdf5")
#    DATA_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
##    DATA_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
#    BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
#    PFM_AUXILIARY_FILES = os.path.join(BASE_DIRECTORY, "data", "pfm_auxiliary_files_all")
##    PFM_AUXILIARY_FILES = BASE_DIRECTORY
#    DIRECTORY_STRUCTURE = True
#    FIG_X = 10
#    FIG_Y = 7
#    SEARCH_DATASTORE = False
#    DATASTORE_SERVER = []
#    DATASTORE_DIRECTORY = ""



rcParams["axes.formatter.useoffset"] = False
model="PFM"

SAVE_FIGS=False
#SAVE_FIGS=True

#SAVE_FILES=False
SAVE_FILES=True




"""Ground Calibration"""

"""NEC data"""

"""MCO data"""
#title = "LNO MCO Dark" #make corrected frame HDF5 file
#fileLevel="hdf5_level_0p1a"
#obspaths=["20161125_061950_0p1a_LNO_1","20161125_070250_0p1a_LNO_1",\
#        "20161125_082550_0p1a_LNO_1","20161125_090850_0p1a_LNO_1",\
#        "20161125_103150_0p1a_LNO_1","20161125_111450_0p1a_LNO_1",\
#        "20161125_123750_0p1a_LNO_1","20161125_132050_0p1a_LNO_1"]


#title="SO MCO Dark" #make corrected frame HDF5 file
#fileLevel="hdf5_level_0p1a"
#obspaths=["20161125_020250_0p1a_SO_1","20161125_032550_0p1a_SO_1"]



#title="LNO Bad Pixel Map" #derive bad pixel map
#obspaths = "LNO_MCO_Dark_Integration_Time_Stepping_Corrected_Full_Frame_-19C"
#fileLevel=""

#title="SO Bad Pixel Map" #derive bad pixel map
#obspaths = "SO_MCO_Dark_Integration_Time_Stepping_Corrected_Full_Frame_-12C"
#fileLevel=""


#title="LNO Non Linearity Correction"; #build non-linearity file
#obspaths = "LNO_MCO_Dark_Integration_Time_Stepping_Corrected_Full_Frame_-19C"
#fileLevel=""

title="SO Non Linearity Correction"; #build non-linearity file
obspaths = "SO_MCO_Dark_Integration_Time_Stepping_Corrected_Full_Frame_-12C"
fileLevel=""








def chisquared(xvector,yarray,plot_fig=False):
    """calculate mean gradient and chisquared total for given 2d array and xvalues"""
    import matplotlib.pyplot as plt

    chisq_total=0
    mean_gradients=[]
    if plot_fig:
        plt.figure(figsize=(10,8))
    for index in range(yarray.shape[1]):
        yvector=yarray[:,index]
        gradient,offset = np.polyfit(xvector,yvector,1)
        fitted_points = xvector * gradient + offset
    
        #run chisquared total on 
        chisq=(yvector-fitted_points)**2.0
        chisq_total+=np.sum(chisq)
        mean_gradient = np.mean(gradient)
        mean_gradients.append(mean_gradient)
#        mean_offset = np.mean(offset)
        
        if plot_fig:
            plt.scatter(xvector,fitted_points,linewidth=0,alpha=0.3)
            plt.plot(xvector,yvector)    
            plt.xlabel("Integration time (ms)")
            plt.ylabel("Pixel ADC value")
#            plt.title("Gradient=%f, Chisq=%f" %(mean_gradient,chisq_total))
    return np.mean(np.asfarray(mean_gradients)),chisq_total



def savitzky_golay(y, window_size, order, deriv=0, rate=1): #deriv and rate only work on the derivate (set deriv=1)
    """function to calculate Savitzky Golay filter for an input spectrum, a type of moving-average filter"""
    """if given the correct parameters, the general spectral shape will be returned without absorption lines"""
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')






def writeSteppingData(hdf5_files, hdf5_filenames, channel):
    #if option==2:
    """write hdf5 file containing all dark stepping from MCO1, corrected to the first measurement using the regions of overlap to normalise subsequent frames"""
    if channel == "so":
        nIntTimes = 960
    elif channel == "lno":
        nIntTimes = 480
    SCALE_TO_MATCH = True
#    SCALE_TO_MATCH = False

    full_frame = np.zeros((nIntTimes,256,320)) * np.nan
    plt.figure(figsize=(FIG_X, FIG_Y))
#    plt.legend()
#    crossover_lines = [4,4,4,4,4,8,4] #number of lines that overlap the previous 
    
    for file_index,hdf5_file in enumerate(hdf5_files):
        detectorData,_,_ = get_dataset_contents(hdf5_file,"Y")
        intTimeAll = get_dataset_contents(hdf5_file,"IntegrationTime")[0]
        windowTopAll = get_dataset_contents(hdf5_file,"WindowTop")[0]
        frames_to_compare = range(80,100)
        
        if file_index==0:
            if channel=="lno":
                temperature,_,_ = get_dataset_contents(hdf5_file,"AOTF_TEMP_LNO") #get data for LNO
            elif channel=="so":
                temperature,_,_ = get_dataset_contents(hdf5_file,"AOTF_TEMP_SO") #get data for SO
            nomad_temperature = np.mean(temperature[1:10])
        
        frame_to_plot=100
        
        windowtop = windowTopAll[0]
        windowbottom = windowtop+24
#        print(windowtop,windowbottom
    
        old_new_ratio = [] #loop through lines, checking if data overlapping
        for line_index,windowline in enumerate(range(windowtop,windowbottom)):
            if not np.isnan(full_frame[0,windowline,0]): #check first pixel of one frame for nan
                for frame_to_compare in frames_to_compare:
                    #if line overlaps, find mean ratio of old and new lines
                    old_new_ratio.append(np.mean(full_frame[frame_to_compare,windowline,:]/detectorData[frame_to_compare,line_index,:]))
                    plt.plot(full_frame[frame_to_compare,windowline,:]/detectorData[frame_to_compare,line_index,:], label=line_index)
        if old_new_ratio==[] or not SCALE_TO_MATCH: #if no previous frame or if turned off
            average_old_new_ratio = 1.0 #don't scale
        else:
            average_old_new_ratio = np.mean(old_new_ratio)
        corrected_detector_data = detectorData[:,:,:] * average_old_new_ratio
        full_frame[:,windowtop:windowbottom,:]=corrected_detector_data[:,:,:]

    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.title(title+" Log Scale")
    plt.xlabel("Spectral Dimension")
    plt.ylabel("Spatial Dimension")
    plt.imshow(np.log(full_frame[frame_to_plot,:,:]-0.90*np.nanmean(full_frame[frame_to_plot,:,:])),interpolation='none',cmap=plt.cm.gray)
    plt.colorbar()
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.plot(full_frame[frame_to_plot,150,:])    
    plt.plot(full_frame[frame_to_plot,:,155])    

    if SAVE_FILES:
        output_filename = "%s_Integration_Time_Stepping_Corrected_Full_Frame_%iC" %(title.replace(" ","_"),nomad_temperature)
        #write corrected dark frame dataset to file
        hdf5_file = h5py.File(os.path.join(BASE_DIRECTORY,output_filename+".h5"), "w")
        write_to_hdf5(hdf5_file,full_frame,output_filename,np.float,frame=channel)
        write_to_hdf5(hdf5_file,intTimeAll,"IntegrationTime",np.float)
        comments = "Files Used: "
        for hdf5_filename in hdf5_filenames:
            comments = comments + hdf5_filename + "; "
        hdf5_file.attrs["Comments"] = comments
        hdf5_file.attrs["DateCreated"] = str(datetime.now())
        hdf5_file.close()
    
    
    
    
    
def writeBadPixelMap(hdf5_stepping_filename, channel):
    #if option==3:
    """read in dark stepping hdf5 file, make clickable figure. output bad pixel map"""
#    sensitivity = "Normal"
#    sensitivity = "Sensitive"
    sensitivity = "Very_Sensitive"
    
    if channel == "lno":
        temperature=-19 #take from filename in future
        if sensitivity == "Normal":
            max_deviation_from_mean_gradient = 3.0
            max_log_chisq = 18.0
        if sensitivity == "Sensitive":
            max_deviation_from_mean_gradient = 3.0
            max_log_chisq = 15.0
        if sensitivity == "Very_Sensitive":
            max_deviation_from_mean_gradient = 2.0
            max_log_chisq = 14.0
    elif channel == "so":
        temperature=-12 #take from filename in future
        if sensitivity == "Normal":
            max_deviation_from_mean_gradient = 5.0
            max_log_chisq = 20.0
        if sensitivity == "Sensitive":
            max_deviation_from_mean_gradient = 3.0
            max_log_chisq = 19.0
        if sensitivity == "Very_Sensitive":
            max_deviation_from_mean_gradient = 2.0
            max_log_chisq = 18.5



#    plot_type="gradient"
#    plot_type="deviation_from_mean_gradient"
    plot_type="chisq"
#    plot_type="none"


    filename = os.path.normcase(BASE_DIRECTORY + os.sep + hdf5_stepping_filename+".h5") #choose a file
#    nomad_temperature = filename[(filename.find("frame_")+6):(filename.find("c.h5"))]    
    hdf5_file = h5py.File(filename, "r") #open file
    corrected_darks = get_dataset_contents(hdf5_file, hdf5_stepping_filename, return_calibration_file=True)[0]
    int_time_all = get_dataset_contents(hdf5_file, "IntegrationTime", return_calibration_file=True)[0]
    
    frame_gradient=np.zeros((256,320)) * np.nan
    frame_chisq=np.zeros((256,320))
    frame_ranges = [[0,200],[256,456]]    
    
    inttimes=int_time_all[frame_ranges[0][0]:frame_ranges[0][1]]
    for h_index in range(320):
        for v_index in range(256):
            pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
            if not np.isnan(pixelvalues[0,0]): #check if data is there
                frame_gradient[v_index,h_index],frame_chisq[v_index,h_index]=chisquared(inttimes,pixelvalues)

    def on_click(event): #make plot clickable
        global SAVE_FIGS
        v_index=int(np.round(event.ydata))
        h_index=int(np.round(event.xdata))
        print('xdata=%i, ydata=%i' %(h_index,v_index))
        pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
        _,_=chisquared(inttimes,pixelvalues,plot_fig=True)
        plt.title("Integration time stepping for pixel (%i,%i)" %(h_index,v_index))
        plt.tight_layout()
        if SAVE_FIGS: plt.savefig("%s_Integration_time_stepping_for_pixel_%i_%i_%iC.png" %(channel.upper(),h_index,v_index,temperature))
       
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax = fig.add_subplot(111)
    if plot_type=="gradient":
        cax = ax.imshow(frame_gradient,interpolation='none',cmap=plt.cm.gray)
        plt.title("Integration time stepping: gradient of pixel value vs. int time")
    elif plot_type=="deviation_from_mean_gradient":
        cax = ax.imshow(np.abs(frame_gradient-np.nanmean(frame_gradient)),interpolation='none',cmap=plt.cm.gray)
        plt.title("Integration time stepping: gradient of pixel value vs. int time")
    elif plot_type=="chisq":
        cax = ax.imshow(np.log(frame_chisq),interpolation='none',cmap=plt.cm.gray)
        plt.title("Integration time stepping: sum of chi-squared values compared to linear fit")
    plt.colorbar(cax)
    plt.xlabel("Horizontal pixel number (spectral direction)")
    plt.ylabel("Vertical pixel number (spatial direction)")
#    if SAVE_FIGS: plt.savefig("Integration_time_stepping_sum_of_chi-squared_values_compared_to_linear_fit_%iC" %temperature)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    bad_pixel_map = np.zeros((256,320), dtype=bool)
    gradient_deviation_map = np.abs(frame_gradient-np.nanmean(frame_gradient))
    chisq_log_map = np.log(frame_chisq)
    
    for h_index in range(320):
        for v_index in range(256):
            if gradient_deviation_map[v_index,h_index]>max_deviation_from_mean_gradient or chisq_log_map[v_index,h_index]>max_log_chisq:
                bad_pixel_map[v_index,h_index]=1
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.imshow(bad_pixel_map,interpolation='none',cmap=plt.cm.gray)
#    plt.colorbar()
    
    print("%i bad pixels included in map" %np.nansum(bad_pixel_map))

    if SAVE_FILES:
        output_filename = "%s_%s" %(title.replace(" ", "_"), sensitivity)
        #write bad pixel map to file
        hdf5_file = h5py.File(os.path.join(BASE_DIRECTORY, output_filename+".h5"), "w")
        write_to_hdf5(hdf5_file, bad_pixel_map, "Bad_Pixel_Map", np.bool, frame=channel)
        comments = "Files Used: %s; Maximum deviation from mean gradient: %0.1f; Maximum log chi squared value: %0.1f" %(hdf5_stepping_filename+".h5",max_deviation_from_mean_gradient,max_log_chisq)
        hdf5_file.attrs["Comments"] = comments
        hdf5_file.attrs["DateCreated"] = str(datetime.now())
        hdf5_file.close()
        




def writeNonLinearity(hdf5_stepping_filename, channel):
    #if option==4:
    USE_MEAN_NON_LINEARITY = True
    """read in dark stepping hdf5 file, make non-linearity correction lookup table and check it"""

    nonlinear_region=[0,15]
    linear_region=[25,160]
    
    max_steps = 250
    frame_ranges = [[0,max_steps],[256,256+max_steps]]    
    filename = os.path.join(BASE_DIRECTORY, hdf5_stepping_filename+".h5")
    hdf5_file = h5py.File(filename, "r") #open file
    temperature = obspaths.split("_")[-1]

    corrected_darks = get_dataset_contents(hdf5_file, hdf5_stepping_filename, return_calibration_file=True)[0]
    int_time_all = get_dataset_contents(hdf5_file,"IntegrationTime",return_calibration_file=True)[0]
    
#    frame_gradient=np.zeros((256,320)) * np.nan
#    frame_chisq=np.zeros((256,320))
    
    inttimes=int_time_all[frame_ranges[0][0]:frame_ranges[0][1]]

    non_linear_region_counts_real = [1050,2000] #grid for filling in bad pixels
    non_linear_region_counts_ideal = [900,1800]
    adu_ideal=np.arange(non_linear_region_counts_ideal[0],non_linear_region_counts_ideal[1])
#    adu_real=np.arange(non_linear_region_counts_real[0],non_linear_region_counts_real[1])
    lookup_table=np.zeros((non_linear_region_counts_real[1]+1,256,320))
    for index in range(lookup_table.shape[0]):
        lookup_table[index,:,:] = index

    fitted_deviation_all = []
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    for v_index in range(108,144):
        for h_index in range(320):
    
            pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
            if not np.isnan(pixelvalues[0,0]): #check if data is there
            
                linearcoeffs1 = np.polyfit(inttimes[linear_region[0]:linear_region[1]],pixelvalues[linear_region[0]:linear_region[1],0],1)
                linearcoeffs2 = np.polyfit(inttimes[linear_region[0]:linear_region[1]],pixelvalues[linear_region[0]:linear_region[1],1],1)
                linearfit1 = np.polyval(linearcoeffs1, inttimes)
                linearfit2 = np.polyval(linearcoeffs2, inttimes)
                sg_fit1 = savitzky_golay(pixelvalues[nonlinear_region[0]:nonlinear_region[1],0], 15, 2)
                sg_fit2 = savitzky_golay(pixelvalues[nonlinear_region[0]:nonlinear_region[1],1], 15, 2)
                
        
                deviation_coefficients = np.polyfit(np.mean([linearfit1[nonlinear_region[0]:nonlinear_region[1]], linearfit2[nonlinear_region[0]:nonlinear_region[1]]], axis=0), \
                            np.mean([sg_fit1-linearfit1[nonlinear_region[0]:nonlinear_region[1]], sg_fit2-linearfit2[nonlinear_region[0]:nonlinear_region[1]]], axis=0),2)
                fitted_deviation = np.polyval(deviation_coefficients, adu_ideal)
                
                old_values = adu_ideal
                if np.max(fitted_deviation) < 250 and np.min(fitted_deviation) > -10: #check for sensible values, otherwise probably a bad pixel
                    new_values = adu_ideal+fitted_deviation
                    new_values_interpolated = np.interp(range(int(min(new_values)),int(max(new_values))),new_values,old_values)
             
                    lookup_table[range(int(min(new_values)),int(max(new_values))), v_index, h_index] = new_values_interpolated
                
                    fitted_deviation_all.append(fitted_deviation)
                else:
                    print("Warning: bad data, skipping (%i,%i)" %(h_index,v_index))
#                    lookup_table[adu_real, v_index, h_index] = adu_real
                
                if (h_index == 268 and v_index == 128) or (h_index == 268 and v_index == 114) or (h_index == 150 and v_index == 132) or (h_index == 150 and v_index == 110):

                    plt.scatter(inttimes,pixelvalues[:, 0], alpha=0.3)
                    plt.scatter(inttimes,pixelvalues[:, 1], alpha=0.3)
                    plt.plot(inttimes, linearfit1)
                    plt.plot(inttimes, linearfit2)
                    plt.plot(inttimes[nonlinear_region[0]:nonlinear_region[1]], sg_fit1)
                    plt.plot(inttimes[nonlinear_region[0]:nonlinear_region[1]], sg_fit2)
                    plt.title("Integration time stepping: non-linear vs fitted linear pixel values")
                    plt.xlabel("Integration time (ms)")
                    plt.ylabel("Signal ADU")
                    if SAVE_FIGS: 
                        plt.savefig("Integration_time_stepping_nonlinear_vs_fitted_values_%i_%i_%iC" %(h_index,v_index,temperature))
                    print("Detector reads counts of 1200 to 1210. Lookup table replaces with values of:")
                    print(lookup_table[1200:1210,v_index,h_index])
#                    stop()

    #do mean
    fitted_deviation_mean = np.mean(np.asfarray(fitted_deviation_all), axis=0)
    new_values_mean = adu_ideal+fitted_deviation_mean
    new_values_interpolated_mean = np.interp(range(int(min(new_values_mean)),int(max(new_values_mean))),new_values_mean,adu_ideal)
    lookup_line = np.asfarray(np.arange((non_linear_region_counts_real[1]+1)))
    lookup_line[range(int(min(new_values_mean)),int(max(new_values_mean)))] = new_values_interpolated_mean
    print("Detector reads counts of 1200 to 1210. Mean lookup table replaces with values of:")
    print(lookup_line[1200:1210])
    
    #use mean non-linearity for whole lookup table!
    if USE_MEAN_NON_LINEARITY:
        for v_index in range(108,144):
            for h_index in range(320):
                lookup_table[range(int(min(new_values_mean)),int(max(new_values_mean))), v_index, h_index] = new_values_interpolated_mean
    

    if SAVE_FILES:
        #write lookup table to file
        hdf5_file = h5py.File(BASE_DIRECTORY+os.sep+r"Non_linearity_look_up_table_%s_channel_%iC_temperature.h5" %(channel,temperature), "w")
        write_to_hdf5(hdf5_file,lookup_table,"Non_Linearity_Lookup_Table",np.float32,frame=channel)
        hdf5_file.close()


def writeNonLinearity2(hdf5_stepping_filename, channel):
    """this time, write a conversion factor expressing deviation from linear. one formula for all pixels"""

#    nonlinear_region=[0,15]
#    linear_region=[25,160]
    max_steps = 160
#    frame_ranges = [[0,max_steps],[256,256+max_steps],[512,512+max_steps]]    
    frame_ranges = [[256,256+max_steps],[512,512+max_steps]]    

    filename = os.path.join(BASE_DIRECTORY, hdf5_stepping_filename+".h5")
    hdf5_file = h5py.File(filename, "r") #open file
    temperature = obspaths.split("_")[-1]

    dark_stepping_data = get_dataset_contents(hdf5_file, hdf5_stepping_filename, return_calibration_file=True)[0]
    int_time_all = get_dataset_contents(hdf5_file,"IntegrationTime",return_calibration_file=True)[0]
    
    inttimes=int_time_all[frame_ranges[0][0]:frame_ranges[0][1]]
    print(inttimes)

    deviation_all = []
    
    n_spectra = 160
    cmap = plt.get_cmap('jet')
    colours = [cmap(i) for i in np.arange(n_spectra)/n_spectra]

    colour_loop = -1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_X, FIG_Y), sharex=True)
    for v_index in range(108,144):
        for h_index in range(320):
            for frame_range_index, frame_range in enumerate(frame_ranges):
                selected_dark = dark_stepping_data[frame_range[0]:frame_range[1], v_index, h_index]
                mean_dark_high = np.mean(selected_dark[140:150])

            for frame_range_index, frame_range in enumerate(frame_ranges):
#                selected_dark = dark_stepping_data[frame_range[0]:frame_range[1], v_index, h_index]
#                mean_dark_high = np.mean(selected_dark[140:150])
#                normalised_dark = selected_dark / mean_dark_high
                
                normalised_dark = selected_dark
#                    mean_dark_low = np.mean(selected_dark[60:70])
#                    mean_dark_high = np.mean(selected_dark[140:150] - mean_dark_low)
#                    normalised_dark = (selected_dark - mean_dark_low) / mean_dark_high

                polyfit = np.polyfit(np.arange(60,150), normalised_dark[60:150], 1)
                linear_dark = np.polyval(polyfit, np.arange(160))
                deviation = (normalised_dark - linear_dark) / linear_dark + 1.0

                chisq=np.sum((normalised_dark[60:150] - linear_dark[60:150])**2.0)
#                if chisq > 10000: #0.00005: #filter out dodgy pixels
#                    print(chisq)
                if chisq < 10000: #0.00005: #filter out dodgy pixels
                    deviation_all.append(deviation)

                    if h_index in [160, 180, 200, 220]:
                        if frame_range_index == 0:
                            colour_loop += 1
                        ax1.plot(linear_dark, normalised_dark, color=colours[colour_loop])
                        ax1.plot(linear_dark, linear_dark, "--", color=colours[colour_loop])
                        if v_index in [110, 120, 130, 140] and frame_range_index==0:
                            ax2.plot(linear_dark, deviation, color=colours[colour_loop], label="%i:%i %i" %(h_index, v_index, frame_range_index))
                        else:
                            ax2.plot(linear_dark, deviation, color=colours[colour_loop])

    plt.legend()
    deviation_all = np.asfarray(deviation_all)
    deviation_mean = np.mean(deviation_all, axis=0)
    
    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * np.exp(-b * x) + 1.0
    popt, pcov = curve_fit(func, linear_dark, deviation_mean, p0=[1.2, 0.0032])
    
    ax2.plot(linear_dark, deviation_mean, c="k")
    ax2.plot(linear_dark, func(linear_dark, *popt), "k--")
    
    ax1.set_ylabel("Detector counts (per pixel per accumulation)")
    ax2.set_ylabel("Deviation from linear fit")
    ax2.set_xlabel("Detector counts linear fit")
    
    fig.suptitle("SO Non Linearity")
    
    print("fit: a=%0.6f, b=%0.6f" % tuple(popt))
    
    return linear_dark, deviation_mean
   


if title == "SO MCO Dark":
    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
    writeSteppingData(hdf5Files, hdf5Filenames, "so")
if title == "LNO MCO Dark":
    hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
    writeSteppingData(hdf5Files, hdf5Filenames, "lno")


if title == "SO Bad Pixel Map":
    writeBadPixelMap(obspaths, "so")
if title == "LNO Bad Pixel Map":
    writeBadPixelMap(obspaths, "lno")
    
    
if title == "SO Non Linearity Correction":
    linear_dark, deviation_mean = writeNonLinearity2(obspaths, "so")

