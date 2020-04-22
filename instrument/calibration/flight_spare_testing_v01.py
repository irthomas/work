# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:37:52 2016

@author: ithom

MORE CALIBRATION ANALYSIS
"""

import os
import h5py
import numpy as np
import numpy.linalg as la
import gc
from datetime import datetime

import bisect
from mpl_toolkits.basemap import Basemap
from scipy import interpolate, signal

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
#import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v02b import get_hdf5_attributes,get_dataset_contents,write_to_hdf5
#from spice_functions_v01 import convert_hdf5_time_to_spice_utc,find_boresight,find_rad_lon_lat,py_ang
from analysis_functions_v01b import interpolate_bad_pixel,savitzky_golay,sg_filter,chisquared,find_order,fft_filter,fft_filter2,spectral_calibration,spectral_calibration_simple,write_log,get_filename_list,stop,get_spectra
from pipeline_config_v04 import figx,figy,DATA_ROOT_DIRECTORY,BASE_DIRECTORY,KERNEL_DIRECTORY,AUXILIARY_DIRECTORY

rcParams["axes.formatter.useoffset"] = False
file_level=""
model="PFM"

save_figs=False
#save_figs=True

#save_files=True
save_files=False

"""Ground Calibration"""
#title="LNO UVIS Full Frame Testing"; option=16; file_level="hdf5_level_0p1c"; DATA_ROOT_DIRECTORY=r"X:\projects\NOMAD\Data\flight_spare\db"
title="LNO Testing Dec2017"; option=1; file_level="hdf5_level_0p1c"


"""NEC data"""

"""MCO data"""

"""All Data"""




channel={"SO ":"so", "SO-":"so", "LNO":"lno", "UVI":"uvis"}[title[0:3]]
detector_centre={"so":128, "lno":152, "uvis":0}[channel]

obspaths={"LNO UVIS Full Frame Testing":[#"20170811_074929_LNO","20170811_090500_LNO","20170811_101936_LNO","20170811_113506_LNO","20170811_125009_LNO","20170811_140451_LNO",\
                                       #"20170814_065955_LNO","20170814_081526_LNO","20170814_093005_LNO","20170814_121539_LNO",
                                       "20170829_071814_LNO","20170829_083344_LNO","20170829_094848_LNO","20170829_110336_LNO","20170829_121840_LNO",\
                                       "20170830_080553_LNO","20170830_092112_LNO","20170830_103627_LNO","20170830_115130_LNO","20170830_130616_LNO",\
                                       "20170831_073257_LNO","20170831_084800_LNO","20170831_100305_LNO","20170831_141051_LNO",\
                                       "20170904_075859_LNO","20170904_091422_LNO","20170904_102932_LNO","20170904_114434_LNO","20170904_125911_LNO"],
        "LNO Testing Dec2017":["20171205_112324_LNO","20171205_120855_LNO","20171205_125334_LNO","20171205_142353_LNO", \
                               "20171207_085523_LNO","20171207_094026_LNO","20171207_102531_LNO","20171207_111047_LNO", \
                               "20171207_124324_SO","20171207_132338_SO","20171207_140352_SO","20171207_144357_SO","20171207_153808_SO", \
                               "20171208_080537_SO","20171208_082736_SO","20171208_084936_SO","20171208_091404_LNO"], \

        }[title]



bad_files=["20150427_023529_LNO_1","20150427_023529_LNO_2"]

if obspaths in bad_files:
    print("Warning: Bad file loading")
    
os.chdir(BASE_DIRECTORY)

hdf5_files=[]
for obspath in obspaths:
    if obspath != "":
        year = obspath[0:4]
        month = obspath[4:6]
        day = obspath[6:8]
        if year=="2015" and month=="09": #fudge to make PFM and FS work together
            filename=os.path.normcase(BASE_DIRECTORY+os.sep+"Data"+os.sep+"flight_spare"+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
        else:
            filename=os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
        hdf5_files.append(h5py.File(filename, "r")) #open file, add to list
    
#        print("File %s has the following attributes:" %(filename)) #print(attributes from file
#        attributes,attr_values=get_hdf5_attributes(hdf5_files[-1])
#        for index in range(len(attributes)):
#            print("%s: %s" %(list(attributes)[index],list(attr_values)[index]))


if option==0:
    """simple routines to plot frames"""
    
#    if title=="LNO Solar Miniscan Testing":
#        frame_range=[78,156,234]
#    elif title=="LNO T1 Blackbody=150C":
#        frame_range=range(10,13)
#    
##    get_spectra(hdf5_files[0],"raw_frames",range(10,13),range(7,19),extras=["binned","spectral_calibration","smoothed"],plot_figs=True)
#    get_spectra(hdf5_files[0],"raw_frames",frame_range,range(7,19),extras=["binned","normalised","smoothed"],plot_figs=True,aotf=16749)
#    get_spectra(hdf5_files[1],"raw_frames",range(10,13),range(7,19),extras=["binned","spectral_calibration","smoothed"],plot_figs=True,overplot=True)

    linecolour_all = ["r","orange","y","yellow","lime","g","c","b","purple","m","grey","k"] * 2
    frame_range=range(200)
    row_range=range(8,16)
    extras=[""]#["normalised"]
    for file_index,hdf5_file in enumerate(hdf5_files[4:8]):
        temp = np.mean(get_dataset_contents(hdf5_file,"AOTF_TEMP_%s" %channel.upper())[0][2:10])
        for row_index,row in enumerate(row_range):
            if row_index==0:
                legend = "%s, %0.1fC" %(obspaths[file_index],temp)
            else:
                legend=""
            if file_index == 0 and row_index == 0:
                get_spectra(hdf5_file,"raw_frames",frame_range,[row],extras=extras,plot_figs=True,aotf=16749,colour=linecolour_all[file_index],legend=legend)
            else:
                get_spectra(hdf5_file,"raw_frames",frame_range,[row],extras=extras,plot_figs=True,aotf=16749,overplot=True,colour=linecolour_all[file_index],legend=legend)
            if row_index==0:
                get_spectra(hdf5_file,"raw_frames",frame_range,row_range,extras=["binned"],plot_figs=True,aotf=16749,overplot=True,alpha=1.0,colour=linecolour_all[file_index],legend=legend)

if option==1:
    
    for fileIndex,hdf5_file in enumerate(hdf5_files):

    #    time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
    #    date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
        detectorDataAll = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        binning = get_dataset_contents(hdf5_file,"Binning")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotfFrequency = get_dataset_contents(hdf5_file,"AOTFFrequency")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        
        nFrames = detectorDataAll.shape[0]
        frameNumber = 1
        columnNumber = 180
#        vPixels = np.arange(detectorDataAll.shape[1])
        soStartPixel = 2
        lnoStartPixel = 0
        
        plt.figure(figsize = (figx-5,figy-3))
        plt.imshow(detectorDataAll[frameNumber,:,:], aspect=[binning[frameNumber] if binning[frameNumber]>0 else 1][0])
        print(binning[frameNumber])
        plt.title(obspaths[fileIndex] + " (%i frames, %ikHz, binning=%i)" %(nFrames,aotfFrequency[frameNumber],binning[frameNumber]))
        
    
        detectorLine = detectorDataAll[frameNumber,:,columnNumber]
        


"""plot LNO spectra from UVIS FS Full Frame testing 2017"""
if option==16:
    file_choices=range(len(obspaths))
    
    for file_choice in file_choices:
    
        """read in data from file"""
        detector_data,_,_ = get_dataset_contents(hdf5_files[file_choice],"Y")
        aotf_freq_all = get_dataset_contents(hdf5_files[file_choice],"AOTFFrequency")[0]
    
        """calculate the mean"""
        aotfs_to_plot=[16593,24640]
        
        detector_rows_to_bin = [4,5,6,7,8]
        
        fig1 = plt.figure(figsize=(figx,figy))
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        for aotf_index,aotf_to_plot in enumerate(aotfs_to_plot):
            chosen_frame_indices = list(np.where(aotf_freq_all==aotf_to_plot)[0])
            
            for frame_index in chosen_frame_indices:#range(10,81,5):
                detector_data_binned = np.mean(detector_data[frame_index,detector_rows_to_bin,:], axis=0)
                if aotf_index==0:
                    ax1.plot(detector_data_binned,label="Frame %i" %frame_index, alpha=0.3)
                    ax1.set_xlabel("Pixel Number")
                    ax1.set_ylabel("Raw Signal")
                elif aotf_index==1:
                    ax2.plot(detector_data_binned,label="Frame %i" %frame_index, alpha=0.3)
                    ax2.set_xlabel("Pixel Number")
                    ax2.set_ylabel("Raw Signal")
        fig1.suptitle("%s - File %s" %(title,obspaths[file_choice]))
        if save_figs:
            plt.savefig("%s - File %s" %(title.replace(" ","_"),obspaths[file_choice]))
                
            








