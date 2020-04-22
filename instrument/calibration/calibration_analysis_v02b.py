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
#from mpl_toolkits.basemap import Basemap
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
#from analysis_functions_v01b import interpolate_bad_pixel,savitzky_golay,chisquared,findOrder,fft_filter,fft_filter2,spectralCalibration,spectral_calibration_simple,write_log,get_filename_list,stop,get_spectra
from analysis_functions_v01b import sg_filter
#from pipeline_config_v04 import figx,figy,DATA_ROOT_DIRECTORY,BASE_DIRECTORY,KERNEL_DIRECTORY,AUXILIARY_DIRECTORY

BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
figx = 10
figy = 7

rcParams["axes.formatter.useoffset"] = False
#rcParams.update({'font.size': 18}) #use larger fonts on plots
rcParams.update({'font.size': 12}) #use larger fonts on plots
file_level=""
model="PFM"

save_figs=False
#save_figs=True

save_files=True
#save_files=False

"""Ground Calibration"""
#title="LNO T1 Blackbody=150C"; option=0; file_level="hdf5_level_0p1c" #plot frames only
#title="LNO FS PFM Blackbody All"; option=0; file_level="hdf5_level_0p1c" #plot frames only


"""see option 12 for newer version"""
#title="LNO T1 Blackbody=150C"; option=1; file_level="hdf5_level_0p1c"
#title="LNO FS Blackbody"; option=1.1; file_level="hdf5_level_0p1c"; model="FS"
#title="LNO FS Blackbody All"; option=1.1; file_level="hdf5_level_0p1c"; model="FS"
#title="LNO FS PFM Blackbody All"; option=1.1; file_level="hdf5_level_0p1c"
#title="LNO T1 FS Blackbody=150C"; option=1.1; file_level="hdf5_level_0p1c"; model="FS"

#title="LNO T1 Blackbody=150C"; option=1.1; file_level="hdf5_level_0p1c"
#title="LNO T2 Blackbody=150C"; option=1.1; file_level="hdf5_level_0p1c"
#title="LNO T2 Globar"; option=1.1; file_level="hdf5_level_0p1c"

#title="LNO UVIS Full Frame Testing"; option=16; file_level="hdf5_level_0p1c"; DATA_ROOT_DIRECTORY=r"X:\projects\NOMAD\Data\flight_spare\db"

"""NEC data"""


"""MCO data"""
#title="LNO MCO Dark"; option=2; file_level="hdf5_level_0p1c"  #make corrected frame HDF5 file
#title="SO MCO Dark"; option=2; file_level="hdf5_level_0p1c"  #make corrected frame HDF5 file
#title="LNO Bad Pixel Map"; option=3 #derive bad pixel map
#title="SO Bad Pixel Map"; option=3 #derive bad pixel map
#title="LNO Non Linearity Correction"; option=4 #build non-linearity file



#title="SO Miniscan Data Construction"; option=5 #write temperature-dependent coefficients to file
#title="SO Miniscan Analysis"; option=6 #write temperature-dependent coefficients to file
#title="SO Fullscan Data Construction"; option=7 #write temperature-dependent coefficients to file
#title="SO Fullscan Analysis"; option=8 #write temperature-dependent coefficients to file

#title="SO Spectral Calibration Table"; option=9; file_level="hdf5_level_0p1c" #build non temperature-dependent coefficient file
#title="LNO Spectral Calibration Table"; option=10; file_level="hdf5_level_0p1c" #build temperature-dependent coefficient file
#title="LNO Compare Spectral Calibration"; option=11; 

#title="LNO Radiometric Calibration Table"; option=12; file_level="hdf5_level_0p1c"

#title="LNO Solar Miniscan Analysis"; option=14 #uses premade file (see above)
title="LNO Solar Spectrum"; option=15

#title="LNO Solar Miniscan Testing"; option=0; file_level="hdf5_level_0p1c"


"""All Data"""
#title="LNO All Files Comparison"; option=0; file_level="hdf5_level_0p1c" #plot frames only



channel={"SO ":"so", "SO-":"so", "LNO":"lno", "UVI":"uvis"}[title[0:3]]
detector_centre={"so":128, "lno":152, "uvis":0}[channel]

obspaths={"LNO T1 Blackbody=150C":["20150426_054602_LNO","20150426_030851_LNO"],
        "LNO T2 Blackbody=150C":["20150427_010422_LNO"], 
        "LNO T2 Globar":["20150427_060747_LNO"],
        "LNO MCO Dark":["20161125_061950_LNO","20161125_070250_LNO","20161125_082550_LNO","20161125_090850_LNO","20161125_103150_LNO","20161125_111450_LNO","20161125_123750_LNO","20161125_132050_LNO"], 
        "SO MCO Dark":["20161125_020250_SO","20161125_032550_SO"], 
        "LNO Bad Pixel Map":[""], 
        "SO Bad Pixel Map":[""], 
        "LNO Non Linearity Correction":[""],
        "SO Spectral Calibration Table":[""],
        "LNO Spectral Calibration Table":[""],
        "LNO Compare Spectral Calibration":[""],
        "SO Radiometric Calibration Table":["20160414_230000_SO","20160615_123000_SO","20161121_224950_SO"],
#        "LNO Radiometric Calibration Table":["20160414_233000_LNO","20160615_130000_LNO","20161121_233000_LNO"],
        "LNO Solar Miniscan Analysis":[""],
        "LNO Solar Spectrum":[""],

        "LNO FS Blackbody":["20150904_231759_LNO","20150905_040057_LNO","20150905_052528_LNO",\
                               "20150907_190444_LNO","20150907_195100_LNO","20150907_212248_LNO",\
                               "20150909_232514_LNO","20150910_015625_LNO","20150910_043039_LNO"], 
        "LNO FS Blackbody All":["20150904_231759_LNO","20150905_005006_LNO","20150905_013134_LNO","20150905_031732_LNO","20150905_040057_LNO","20150905_052528_LNO","20150906_034503_LNO",\
                               "20150907_190444_LNO","20150907_195100_LNO","20150907_212248_LNO","20150907_221000_LNO","20150907_234545_LNO","20150908_003506_LNO",\
                               "20150909_223132_LNO","20150909_232514_LNO","20150910_010414_LNO","20150910_015625_LNO","20150910_033745_LNO","20150910_043039_LNO"], 
        "LNO FS PFM Blackbody All":["20150904_231759_LNO","20150905_005006_LNO","20150905_013134_LNO","20150905_031732_LNO","20150905_040057_LNO","20150905_052528_LNO","20150906_034503_LNO",\
                               "20150907_190444_LNO","20150907_195100_LNO","20150907_212248_LNO","20150907_221000_LNO","20150907_234545_LNO","20150908_003506_LNO",\
                               "20150909_223132_LNO","20150909_232514_LNO","20150910_010414_LNO","20150910_015625_LNO","20150910_033745_LNO","20150910_043039_LNO",\
                               "20150426_054602_LNO","20150426_030851_LNO","20150427_010422_LNO"], 

        "LNO T1 FS Blackbody=150C":["20150904_231759_LNO","20150905_005006_LNO","20150905_013134_LNO","20150905_031732_LNO"], #150C BB integration time steps 

        "LNO UVIS Full Frame Testing":[#"20170811_074929_LNO","20170811_090500_LNO","20170811_101936_LNO","20170811_113506_LNO","20170811_125009_LNO","20170811_140451_LNO",\
                                       #"20170814_065955_LNO","20170814_081526_LNO","20170814_093005_LNO","20170814_121539_LNO",
                                       "20170829_071814_LNO","20170829_083344_LNO","20170829_094848_LNO","20170829_110336_LNO","20170829_121840_LNO",\
                                       "20170830_080553_LNO","20170830_092112_LNO","20170830_103627_LNO","20170830_115130_LNO","20170830_130616_LNO",\
                                       "20170831_073257_LNO","20170831_084800_LNO","20170831_100305_LNO","20170831_141051_LNO",\
                                       "20170904_075859_LNO","20170904_091422_LNO","20170904_102932_LNO","20170904_114434_LNO","20170904_125911_LNO"],

        "LNO Solar Miniscan Testing":["20161123_025550_LNO"],
        
        "LNO All Files Comparison":["20150426_054602_LNO","20150426_030851_LNO","20150427_010422_LNO","20160414_233000_LNO","20160615_130000_LNO","20161121_233000_LNO","20161123_025550_LNO"],

        "LNO Radiometric Calibration Table":["20150904_231759_LNO","20150905_005006_LNO","20150905_013134_LNO","20150905_031732_LNO","20150905_040057_LNO","20150905_052528_LNO","20150906_034503_LNO",\
                               "20150907_190444_LNO","20150907_195100_LNO","20150907_212248_LNO","20150907_221000_LNO","20150907_234545_LNO","20150908_003506_LNO",\
                               "20150909_223132_LNO","20150909_232514_LNO","20150910_010414_LNO","20150910_015625_LNO","20150910_033745_LNO","20150910_043039_LNO",\
                               "20150426_054602_LNO","20150426_030851_LNO","20150427_010422_LNO"] 

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
    """analyse ground blackbody data to derive SNR and NESR"""
#    compare_radiance = True #compare to simulated radiances to derive new SNRs?
    compare_radiance = False
    
    sinc2_fwhm = 23.0 #this is the average value across all bins from Arnaud analysis 2016
    
    def planck(xscale,temp,units): #planck function W/m2/sr/spectral unit
        if units=="microns" or units=="m" or units=="wavel":
            c1=1.191042e8
            c2=1.4387752e4
            return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0)
        elif units=="wavenumbers" or units=="w" or units=="waven":
            c1=1.191042e-5
            c2=1.4387752
            return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) /1000.0
        else:
            print("Error: Unknown units given")
#    print(planck(11.0,290.0,"microns")
#    print(planck(908.8,290.0,"wavenumbers")

    def func_sinc2(x, b, c): #inverted sinc2
        b = b+0.0001 #fudge to stop infinity at peak
        return ((np.sin((x-b)/c*2.783)**2.0) / (((x-b)/c*2.783)**2.0))
    
    
    sinc2_xoffset = 3000.0
    sinc2_width = 60.0#
    sinc2_x = np.arange(2000.0001,4000.0001,0.1)
    sinc2 = func_sinc2(sinc2_x,sinc2_xoffset,sinc2_width)

    wavenumbers = np.arange(2900.0,3100,0.1)
    planck_waven = planck(wavenumbers,(273.0+150.0), "wavenumbers")
    sinc2_waven = func_sinc2(wavenumbers,3000.0,sinc2_fwhm)
    sinc2_planck_waven = planck_waven * sinc2_waven
    
    plt.figure()
    plt.plot(wavenumbers,planck_waven/max(planck_waven))
    plt.plot(wavenumbers,sinc2_waven/max(sinc2_waven))
    plt.plot(wavenumbers,sinc2_planck_waven/max(sinc2_planck_waven))
    
    
    
    """code to calculate scaling factor of width of sinc squared function. Value=2.783"""
#    def calc_sinc2_width(sinc2_x,sinc2):
#        sinc2_max_index=np.where(sinc2==max(sinc2))[0][0]
#        sinc2_half_index_min = np.min(np.where(sinc2[0:sinc2_max_index]>0.5)[0])
#        sinc2_half_index_max = np.min(np.where(sinc2[sinc2_max_index:len(sinc2)]<0.5)[0]) + sinc2_max_index
#        sinc2_calc_width = sinc2_x[sinc2_half_index_max]-sinc2_x[sinc2_half_index_min]
#    
#        plt.plot(sinc2_x, sinc2)
#        plt.scatter(sinc2_x[sinc2_half_index_min], sinc2[sinc2_half_index_min])
#        plt.scatter(sinc2_x[sinc2_half_index_max], sinc2[sinc2_half_index_max])
#        
#        return sinc2_calc_width
#    
#    sinc2_widths = np.arange(0.5,100,1)
#    sinc2_calc_widths = []
#    for sinc2_width in sinc2_widths:
#        sinc2_calc_widths.append(calc_sinc2_width(sinc2_x,func_sinc2(sinc2_x,sinc2_xoffset,sinc2_width)))
#    plt.figure()
#    plt.plot(sinc2_widths,sinc2_calc_widths)
#    print(np.polyfit(sinc2_widths,sinc2_calc_widths,2))
    

    """read in data from file"""
    detector_data,_,_ = get_dataset_contents(hdf5_files[0],"Y")
    nacc_all = get_dataset_contents(hdf5_files[0],"NumberOfAccumulations")[0]
    inttime_all = get_dataset_contents(hdf5_files[0],"IntegrationTime")[0]
    aotf_freq_all = get_dataset_contents(hdf5_files[0],"AOTFFrequency")[0]
    backsub_all = get_dataset_contents(hdf5_files[0],"BackgroundSubtraction")[0]

    binned_data = np.asfarray([np.nansum(frame, axis=0) for index,frame in enumerate(list(detector_data))])
    backsub = backsub_all[0]
    nacc = nacc_all[0]
    inttime = inttime_all[0]
    frames=range(2,110)
    frames_to_plot = [60,90,100,105,110]

    """calculate true observation time"""
    if backsub==1:
        true_obs_time = (nacc/2) * inttime
        print("Actual Obs Time = %ims" %true_obs_time)
        ratio_15s = 15000/true_obs_time
    else:
        print("Error: sbsf must be on")

    attributes,attr_values=get_hdf5_attributes(hdf5_files[0])
    for attr_value in list(attr_values):
#        if isinstance(attr_value, str): 
        if "lackbody" in str(attr_value):
            print(attr_value)
            bb_temp_string = str(attr_value).split("lackbody at ")[1].split("C")[0].replace("\\r","")
    try:
        bb_temp = float(bb_temp_string) + 273.0
    except ValueError:
        print("Warning: BB Temp is not a float")
        
    """set up figures"""
    plt.figure(1, figsize=(figx,figy))
    plt.title("SNR for %iC Blackbody" %(bb_temp-273.0))
    plt.xlabel("Wavenumbers cm-1")
    plt.ylabel("SNR")

    plt.figure(2, figsize=(figx,figy))
    plt.title("Blackbody Radiance W/m2/sr/cm-1")
    plt.xlabel("Wavenumbers cm-1")
    plt.ylabel("W/m2/sr/cm-1")

    plt.figure(3, figsize=(figx,figy))
    plt.title("Noise-Equivalent Spectral Radiance W/m2/sr/cm-1")
#    plt.legend()
    plt.yscale("log")
    plt.ylabel("W/m2/sr/cm-1")
    plt.xlabel("Wavenumbers cm-1")
    
    plt.figure(4, figsize=(figx,figy))
    plt.title("NESR Scaled to 15s Observation W/m2/sr/cm-1")
#    plt.legend()
    plt.yscale("log")
    plt.ylabel("W/m2/sr/cm-1")
    plt.xlabel("Wavenumbers cm-1")
    
    plt.figure(5, figsize=(figx,figy))
    plt.title("Standard Deviation Noise vs Order")
    plt.ylabel("Noise Standard Deviation")
    plt.xlabel("Diffraction Order")

    if compare_radiance:

        snr_path = os.path.normcase(r"C:\Users\iant\Documents\Python\snr\dataRad_IR_from_Ann_Carine\Results_SNR_Model_v02\LNO_nadir_LS251_DustVar_MedAlb\LNO_nadir_LS251_DustVar_MedAlb_column3_273K_variable_int_time.txt")
        radiance_path = os.path.normcase(r"C:\Users\iant\Documents\Python\snr\dataRad_IR_from_Ann_Carine\input_files_ACV\LNO_nadir_LS251_DustVar_MedAlb.dat")
        # Effects of Variations in Albedo: Expected SNR for LNO nadir observations for one solar zenith angle (45°), one dust case (OD = 0.4)
        # one surface albedo (0.242), one Sun-Mars distance (1.38AU @ Ls = 251°) and one instrument temperature (273K)
        snr_values = np.loadtxt(snr_path, skiprows=2, usecols=(7,))
        snr_orders = np.loadtxt(snr_path, skiprows=2, usecols=(0,))
        radiance_values = np.loadtxt(radiance_path, usecols=(2,))
        radiance_wavenumbers = np.loadtxt(radiance_path, usecols=(0,))
        
        plt.figure(6, figsize=(figx,figy))
        plt.title("Measured vs Theoretical SNRs for an Typical Mars Radiance")
        plt.ylabel("SNR")
        plt.xlabel("Diffraction Order")
        
    """work backwards from a pixel:
        reverse temperature dependent spectral calibration to determine wavenumber of central order + adjacent orders
        use AOTF function to determine relative contribution of each order
        
    use spectral calibration to determine radiance entering aperture
    use AOTF function to add up contributions from various orders """
       
    for frame in frames:
    
    #    fig=plt.figure(figsize=(figx,figy))
    #    ax1 = plt.subplot2grid((4,4),(0,1), rowspan=3, colspan=3)
    #    plt.imshow(detector_data[frame,:,:])
    #    plt.colorbar()
    #
    #    ax2 = plt.subplot2grid((4,4),(0,0), rowspan=3)
    #    plt.plot(np.transpose(detector_data[frame,:,[60,120,180,240]]))
    #    
    #    ax3 = plt.subplot2grid((4,4),(3,1), colspan=3)
    #    plt.plot(np.transpose(detector_data[frame,[90,120,150,180,210],:]))
    #
    #    plt.figure(figsize=(figx,figy))
    #    plt.plot(np.nansum(detector_data[:,:,160], axis=1))
    
        """copy final value 50 times and join to end of array (stops bad fit at end)"""    
        binned_frame = np.asfarray(list(binned_data[frame,:])+list(binned_data[frame,-1::])*50)
        x_range = range(len(binned_frame))
        polyfit1 = np.polyfit(x_range,binned_frame,9)
        fitted1_binned_frame=np.polyval(polyfit1, x_range)
        fitted1_binned_frame_sg=sg_filter(binned_frame, window_size=99, order=2)
        residual_frame1 = binned_frame - fitted1_binned_frame
        residual_frame1_sg = binned_frame - fitted1_binned_frame_sg
        polyfit2 = np.polyfit(x_range,residual_frame1,9)
        fitted2_binned_frame=np.polyval(polyfit2, x_range)
        fitted2_binned_frame_sg=sg_filter(residual_frame1_sg, window_size=99, order=2)
        residual_frame2 = residual_frame1 - fitted2_binned_frame
        residual_frame2_sg = residual_frame1_sg - fitted2_binned_frame_sg
        sum_fitted_frame = fitted1_binned_frame + fitted2_binned_frame
        sum_fitted_frame_sg = fitted1_binned_frame_sg + fitted2_binned_frame_sg
        
        """remove 50 added values before continuing calculation"""
        binned_frame = binned_frame[:-50:]
        fitted1_binned_frame_sg = fitted1_binned_frame_sg[:-50:]
        sum_fitted_frame_sg = sum_fitted_frame_sg[:-50:]
        residual_frame2_sg = residual_frame2_sg[:-50:]
        
        """calculate planck function"""
        aotf_freq=aotf_freq_all[frame]
        order = findOrder(channel,aotf_freq)
        wavenumbers = spectral_calibration_simple(channel,order)
        frame_planck = planck(wavenumbers,bb_temp,"wavenumbers")

        if frame in frames_to_plot:
            plt.figure(figsize=(figx,figy))
            plt.title("Raw and fitted binning spectra order %i" %order)
            plt.plot(binned_frame, label="Raw Signal")
        #    plt.plot(fitted1_binned_frame, label="polyfit1")
            plt.plot(fitted1_binned_frame_sg, label="S-G Fit to Raw Signal")
        #    plt.plot(fitted1_binned_frame + fitted2_binned_frame, label="polyfit2")
        #    plt.plot(fitted1_binned_frame_sg + fitted2_binned_frame_sg, label="sg fit2")
        #    plt.plot(sum_fitted_frame, label="sum polyfit")
            plt.plot(sum_fitted_frame_sg, label="S-G Fit 2")
            plt.legend()
            plt.xlabel("Pixel Number")
            plt.ylabel("Raw Signal (ADUs)")
            if save_figs:
                plt.savefig("raw_signal_order_%i_%iC_blackbody.png" %(order,bb_temp-273.0))
    
        if frame in frames_to_plot:
            plt.figure(figsize=(figx,figy))
            plt.title("Residuals after fitting order %i" %order)
        #    plt.plot(residual_frame1)
        #    plt.plot(residual_frame1_sg)
        #    plt.plot(residual_frame2)
            plt.plot(residual_frame2_sg)
            plt.xlabel("Pixel Number")
            plt.ylabel("Raw Residual Signal (ADUs)")
            if save_figs:
                plt.savefig("residual_signal_order_%i_%iC_blackbody.png" %(order,bb_temp-273.0))
    
        fitted_frame = sum_fitted_frame_sg
        residual_frame = binned_frame - fitted_frame
        std_frame = np.std(residual_frame)
    
        snr_frame = fitted_frame / std_frame
        plt.figure(1)
        plt.plot(wavenumbers,snr_frame)

        plt.figure(2)
        plt.plot(wavenumbers,frame_planck)
        
        nesr_frame = frame_planck / snr_frame
        plt.figure(3)
        plt.plot(wavenumbers,nesr_frame, label="Order %s" %order)

        nesr_frame_15s = nesr_frame / ratio_15s
        plt.figure(4)
        plt.plot(wavenumbers,nesr_frame_15s, label="Order %s" %order)

        plt.figure(5)
        plt.plot(order,std_frame,"o")
        
        if compare_radiance:
            np.interp()

    if save_figs:
        plt.figure(1)
        plt.savefig("SNR_from_fitted_spectra_all_orders_%iC_blackbody.png" %(bb_temp-273.0))
    if save_figs:
        plt.figure(2)
        plt.savefig("%iC_blackbody_radiance_per_order.png" %(bb_temp-273.0))
    if save_figs:
        plt.figure(3)
        plt.savefig("NESR_all_orders_%iC_blackbody.png" %(bb_temp-273.0))
    if save_figs:
        plt.figure(4)
        plt.savefig("NESR_15s_observation_all_orders_%iC_blackbody.png" %(bb_temp-273.0))
    if save_figs:
        plt.figure(5)
        plt.savefig("standard_deviation_noise_per_order_%iC_blackbody.png" %(bb_temp-273.0))
    if save_figs:
        plt.figure(6)
        plt.savefig("measured_theoretical_SNRs_for_typical_mars_radiance_per_order_%iC_blackbody.png" %(bb_temp-273.0))







if option==1.1:
        
    """analyse ground blackbody data to derive SNR and NESR"""
    """need to account for CSL window transmission!!!"""

    sinc2_fwhm = 23.0 #this is the average value across all bins from Arnaud analysis 2016
    
#    linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"] *10
    linecolours = ["r","r","orange","orange","y","y","yellow","lime","lime","g","g","c","c","b","b","k","k","m","m","grey","grey","k"] *10
    
#    if model=="PFM":
#        detector_rows_to_bin = range(1,24)
#    elif model=="FS":
    detector_rows_to_bin = range(1,22)
    
    def planck(xscale,temp,units): #planck function W/m2/sr/spectral unit
        if units=="microns" or units=="m" or units=="wavel":
            c1=1.191042e8
            c2=1.4387752e4
            return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0)
        elif units=="wavenumbers" or units=="w" or units=="waven":
            c1=1.191042e-5
            c2=1.4387752
            return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) /1000.0
        else:
            print("Error: Unknown units given")
#    print(planck(11.0,290.0,"microns")
#    print(planck(908.8,290.0,"wavenumbers")

    def func_sinc2(x, b, c): #inverted sinc2
        b = b+0.0001 #fudge to stop infinity at peak
        return ((np.sin((x-b)/c*2.783)**2.0) / (((x-b)/c*2.783)**2.0))

#    def func_sinc2(x, b, fwhm): #gaussian
#        c = fwhm / 2.355
#        return np.exp(-(((x-b))**2.0) / (2.0 * c**2.0))
    
    def normalise(dset, **keyword_parameters):
        if ('dset_all' not in keyword_parameters):
            return dset/max(dset)
        else:
            max_value = np.max(keyword_parameters["dset_all"])
            return (dset/max_value)
        
#    def calc_blaze_function(wavenumbers_in,order,temperature):
#        """Arnaud version"""
#        
#        a=-4.1272000000000E+02
#        b=-3.0640000000000E-01
#        c=8.7960000000000E-03
#        d=-1.0055000000000E-05
#                
#        t1=24.5 + 273.0 #design temp
#        lno_t2 = temperature + 273.0
#        lno_exp=(1.0+(a+b*lno_t2+c*lno_t2**2+d*lno_t2**3.0)*0.00001)/(1.0+(a+b*t1+c*t1**2+d*t1**3.0)*0.00001)
#
#        wavelengths_in = 10000.0 / wavenumbers_in
#        wavelengths = wavelengths_in * 1.0e-6
#        
#        theta_b = 63.43 * np.pi / 180.0
#        sigma_old = 248.06e-6
#        sigma = sigma_old*lno_exp
#        gamma = 2.6 * np.pi / 180.0
#        alpha_b = -0.02 * np.pi / 180.0
#        alpha = theta_b + alpha_b
#        beta = np.arcsin(((order * wavelengths)/(sigma * np.cos(gamma)))-np.sin(alpha))
#        
#        s_all = (1.0/wavelengths)*(sigma * np.cos(gamma) * np.cos(alpha) / np.cos(alpha_b))*(np.sin(alpha_b)+np.sin(beta - theta_b))
#
#        blaze = np.zeros_like(wavelengths)
#        for index,s in enumerate(list(s_all)):
#            if alpha >= beta[index]:
#                blaze[index] = (np.sin(s)/s)**2.0
#            if alpha < beta[index]:
#                blaze[index] = (np.cos(beta[index])/np.cos(alpha))**2.0 * (np.sin(s)/s)**2.0
#        return blaze,sigma
#
#
#    def calc_blaze_function2(order,temperature):
#        a=-4.1272000000000E+02
#        b=-3.0640000000000E-01
#        c=8.7960000000000E-03
#        d=-1.0055000000000E-05
#                
#        t1=24.5 + 273.0 #design temp
#        lno_t2 = temperature + 273.0
#        lno_exp=(1.0+(a+b*lno_t2+c*lno_t2**2+d*lno_t2**3.0)*0.00001)/(1.0+(a+b*t1+c*t1**2+d*t1**3.0)*0.00001)
#
#        """OIP style calculation. Variables taken from SNR model"""
#        related_grating_order=np.float(order)
#        px_size=30.0 #um                     #F2
#        px_arr_length=320.0               #F3
#        px_area_length_um=px_size*px_arr_length #um           F4
#        pitch=248.06#248.06 #was 248.0 #µm                           B19
#        blaze_angle=63.43 #°                    B18
#        angle_alpha=2.75 #was 2.74986 #°                  B22
#        angle_i=0.0 #°                            B23
#        f_imager=302.0 #was 300.0 #mm                        B13
#        t_minus_i=blaze_angle-angle_i #°                            B24
#        t_plus_i=blaze_angle+angle_i #°                             B25
#        ldet=px_area_length_um/1000.0
#        spacing_old=pitch
#        lno_spacing=spacing_old*lno_exp
#        lno_density=1000.0/lno_spacing
#        lno_oopangle=angle_alpha #was 2.74986 
#        lno_groovedensity=lno_density/np.cos(lno_oopangle*np.pi/180.0)
#        app_groove_density=lno_groovedensity
#        px_size=30.0 #um                     #F2
#        px_number_321=np.arange(1,321.001,1)
#        px_pos_321=ldet/2.0-(px_number_321-1.0)*px_size*0.001
#        px_number=np.delete(px_number_321, -1) #K
#        px_pos=np.delete(px_pos_321, 0) #M
#        
#        dispersion_angle=np.arctan(px_pos[:]/f_imager)*180.0/np.pi #N
#        wavel_pixel_centre=(np.sin(t_plus_i*np.pi/180.0)+np.sin((t_minus_i-dispersion_angle)*np.pi/180.0))/app_groove_density/related_grating_order*1000.0 #O
#        beta_angle_rad=np.arcsin(related_grating_order*wavel_pixel_centre[:]/pitch/np.cos(angle_alpha*np.pi/180.0)-np.sin(t_plus_i*np.pi/180.0)) #P
#        beta_angle_deg=beta_angle_rad[:]*180.0/np.pi #Q
#        calculation_step=(np.pi/wavel_pixel_centre[:]*(pitch*np.cos(angle_alpha*np.pi/180.0)*np.cos(t_plus_i*np.pi/180.0)/np.cos(angle_i*np.pi/180.0))*(np.sin(angle_i*np.pi/180.0)+np.sin(beta_angle_rad[:]-blaze_angle*np.pi/180.0))) #R
#        
#        blaze_function=np.zeros(px_number.size) #S
#        for loop in np.arange(blaze_function.size):
#            if beta_angle_deg[loop]<t_plus_i:
#                blaze_function[loop]=(np.sin(calculation_step[loop])/calculation_step[loop])**2
#            else:
#                blaze_function[loop]=(np.cos(beta_angle_rad[loop])/np.cos(t_plus_i*np.pi/180.0))**2 * (np.sin(calculation_step[loop])/calculation_step[loop])**2
#            if np.isnan(blaze_function[loop]):
#                blaze_function[loop]=1
#        return blaze_function,lno_spacing
#            
  
    """code to plot blaze functions"""
#    plt.figure(figsize=(figx,figy))
#    aotf_freqs = [15658,15814,15970,16126,16281,16437,16593]
#    temperatures = [-15.,0.,15.]
#    colour_loop = -1
#    for temperature in temperatures:
#        for aotf_freq in aotf_freqs:
#            colour_loop += 1
#            order = spectralCalibration("aotf2order",channel,aotf_freq,0) #find nearest order number
#            wavenumbers = spectralCalibration("pixel2waven",channel,order,0.0) #find wavenumbers for order
##            wavenumbers = spectralCalibration("pixel2waven",channel,order,temperature) #find wavenumbers for order
#            blaze,sigma = calc_blaze_function(wavenumbers,order,temperature)
#            blaze2,sigma2 = calc_blaze_function2(order,temperature)
#            plt.plot(blaze,label="AM: temperature=%iC, sigma=%0.6gm, centre_order=%i" %(temperature,sigma,order),color=linecolours[colour_loop],alpha=0.7)
#            plt.plot(blaze2,label="OIP: temperature=%iC, sigma=%0.6gm, centre_order=%i" %(temperature,sigma2,order),color=linecolours[colour_loop],linestyle="--",alpha=0.7)
#    plt.xlabel("Pixel Number")
#    plt.ylabel("Normalised Blaze Function")
#    plt.title("Blaze Angle Variations with Temperature")
##    plt.legend()
#    halt

    
    def optical_transmission(csl_window=False):
        #0:Wavelength, 1:Lens ZnSe, 2:Lens Si, 3: Lens Ge, 4:AOTF, 5:Par mirror, 6:Planar miror, 7:Detector, 8:Cold filter, 9:Window transmission function
        #10:CSL sapphire window
        optics_all = np.loadtxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_optics_transmission.csv", skiprows=1, delimiter=",")
        if not csl_window:
            optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9])
        else:
            optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9]) * (optics_all[:,10])
        optics_wavenumbers =  10000. / optics_all[:,0]
        return optics_wavenumbers, optics_transmission_total
    """code to plot optics"""
#    plt.figure(figsize=(figx,figy))
#    optics_wavenumbers, optics_transmission=optical_transmission(csl_window=False)
#    plt.plot(optics_wavenumbers, optics_transmission)
#    optics_wavenumbers, optics_transmission=optical_transmission(csl_window=True)
#    plt.plot(optics_wavenumbers, optics_transmission)
#    plt.xlabel("Wavenumber")
#    plt.ylabel("Optical Transmission")
#    plt.title("Total LNO NOMAD Optical Transmission Before/After CSL Window Included")
    


    """use function instead"""
#    get_spectra(hdf5_file,"raw_frames",range(10,13),range(7,20),extras=["binned","smoothed"],plot_figs=True)
#    get_spectra(hdf5_files[0],"raw_frames",range(10,13),range(7,19),extras=["binned","spectral_calibration","smoothed"],plot_figs=True)
#    get_spectra(hdf5_files[1],"raw_frames",range(10,13),range(7,19),extras=["binned","spectral_calibration","smoothed"],plot_figs=True,overplot=True)
##    get_spectra(hdf5_file,"raw_frames",range(8,15),[11,12],extras=["binned","spectral_calibration","smoothed"],plot_figs=True)
#
#    halt
    
#    file_choices=range(2)#len(obspaths))#[0,1,2,3,4,5,6,7,8]
    file_choices=range(len(obspaths))
    pfm_indices = [19,20,21]
    temperature_dependencies = ["real"]#"spectral_calibration"]#,"grating"]
    temperature_to_use = 25.0 #if not real

    aotf_frames_to_plot = []#[16749]#16593,16749,16904] #check aotf matches with aotfs_to_plot, otherwise only radiance/counts conversion will be plotted
    pixels_to_plot = [160] #just for crosses on aotf plot
    aotfs_to_plot = [16593,16749,16904] #which radiance/counts conversions should be plotted?
    aotf_to_average = 16904
    filtered_signal_ratios=[]
    measurement_temperatures=[]
    plotted=False

    for aotf_to_plot in aotfs_to_plot:
        fig1 = plt.figure(figsize=(figx,figy))
        ax1 = fig1.add_subplot(111)
        colour_loop=-1
        for file_choice in file_choices:
    
            """read in data from file"""
            detector_data,_,_ = get_dataset_contents(hdf5_files[file_choice],"Y")
            nacc_all = get_dataset_contents(hdf5_files[file_choice],"NumberOfAccumulations")[0]
            inttime_all = get_dataset_contents(hdf5_files[file_choice],"IntegrationTime")[0]
            aotf_freq_all = get_dataset_contents(hdf5_files[file_choice],"AOTFFrequency")[0]
            binning_all = get_dataset_contents(hdf5_files[file_choice],"Binning")[0]
            backsub_all = get_dataset_contents(hdf5_files[file_choice],"BackgroundSubtraction")[0]
            measurement_temperature = np.mean(get_dataset_contents(hdf5_files[file_choice],"AOTF_TEMP_%s" %channel.upper())[0][2:10])
#            measurement_temperature = 10.0
            print("measurement_temperature %iC" %measurement_temperature)
    
            chosen_frame_indices = list(np.where(aotf_freq_all==aotf_to_plot)[0])
            print("found %i frames with aotf frequency %i" %(len(chosen_frame_indices),aotf_to_plot))
            backsub = backsub_all[0]
            nacc = nacc_all[0]
            inttime = inttime_all[0]
            binning = binning_all[0]+1
            
            print("nacc %i" %nacc)
            print("inttime %i" %inttime)
        
            """calculate true observation time"""
            if backsub==1:
                true_obs_time = (nacc/2) * inttime
                print("Actual Obs Time = %ims" %true_obs_time)
                ratio_15s = 15000/true_obs_time
            else:
                print("Error: sbsf must be on")
        
            blackbody_in_fov=False
            globar_in_fov=False
            attributes,attr_values=get_hdf5_attributes(hdf5_files[file_choice])
            for attr_value in list(attr_values):
        #        if isinstance(attr_value, str): 
                if "lackbody" in str(attr_value):
                    print(attr_value)
                    bb_temp_string = str(attr_value).split("lackbody at ")[1].split("C")[0].replace("\\r","")
                    blackbody_in_fov=True
                if "Globar" in str(attr_value):
                    print(attr_value)
                    globar_in_fov=True
            
            if blackbody_in_fov:
                try:
                    bb_temp = float(bb_temp_string) + 273.0
                except ValueError:
                    print("Warning: BB Temp is not a float")
            elif globar_in_fov:
                bb_temp = 800.0
            else:
                print("Error: neither blackbody nor globar in FOV")
        
        
            for frame_index in chosen_frame_indices:#range(10,81,5):
                
                aotf_freq = aotf_freq_all[frame_index]
                
                
                centre_order = spectralCalibration("aotf2order",channel,aotf_freq,0) #find nearest order number
                print(centre_order)
                orders = [centre_order-4,centre_order-3,centre_order-2,centre_order-1,centre_order,centre_order+1,centre_order+2]
                aotf_centre_wavenumber = spectralCalibration("aotf2waven",channel,aotf_freq,0) #find aotf central wavenumber
                print("aotf_centre_wavenumber")
                print(aotf_centre_wavenumber)
                if "real" in temperature_dependencies:
                    central_order_wavenumbers = spectralCalibration("pixel2waven",channel,centre_order,measurement_temperature) #find wavenumbers for order
                else:
                    central_order_wavenumbers = spectralCalibration("pixel2waven",channel,centre_order,temperature_to_use) #find wavenumbers for order
                print("")
            
                aotf_wavenumbers = np.arange(aotf_centre_wavenumber-100.0,aotf_centre_wavenumber+100.0,0.01) #make aotf function x axis
                planck_waven = planck(aotf_wavenumbers,bb_temp,"wavenumbers") #planck function
                sinc2_waven = func_sinc2(aotf_wavenumbers,aotf_centre_wavenumber,sinc2_fwhm) #make aotf function
                sinc2_planck_waven = planck_waven * sinc2_waven
                
                #optics contribution
                optics_wavenumbers_raw, optics_transmission_raw=optical_transmission(csl_window=True)
                optics_transmission = np.interp(aotf_wavenumbers,optics_wavenumbers_raw[::-1], optics_transmission_raw[::-1])
                sinc2_planck_optics = planck_waven * sinc2_waven * optics_transmission
                
                if aotf_freq in aotf_frames_to_plot:
                    fig2 = plt.figure(figsize=(figx,figy))
                    ax2 = fig2.add_subplot(111)
                    ax2.plot(aotf_wavenumbers,normalise(planck_waven),label="Normalised Planck function at %iK" %bb_temp)
                    ax2.plot(aotf_wavenumbers,normalise(sinc2_waven),label="AOTF passband")
                    ax2.plot(aotf_wavenumbers,normalise(sinc2_planck_waven),label="AOTF passband scaled to Planck function")
                    ax2.plot(aotf_wavenumbers,normalise(optics_transmission),label="Normalised Optics Transmission")
                    ax2.plot(aotf_wavenumbers,normalise(sinc2_planck_optics),label="Optics, AOTF and Planck Function")
                    ax2.set_xlabel("Wavenumbers cm-1")
                    ax2.set_ylabel("Normalised radiance/AOTF passband")
                    ax2.set_title("Diffraction orders superimposed on AOTF function and blackbody radiance at %iK" %bb_temp)
                
                pixels = np.arange(320)
                total_radiances=np.zeros(320)
            
                """sum radiance for each pixel"""
                for order_index,order in enumerate(orders): #loop through order
                    if "real" in temperature_dependencies:
                        wavenumbers = spectralCalibration("pixel2waven",channel,order,measurement_temperature) #find wavenumbers for order
                    else:
                        wavenumbers = spectralCalibration("pixel2waven",channel,order,temperature_to_use) #find wavenumbers for order
                    if "grating" in temperature_dependencies:
                        blaze,_ = calc_blaze_function2(order,measurement_temperature) #blaze function
                    else:
                        blaze,_ = calc_blaze_function2(order,temperature_to_use) #blaze function
                    y_axis = np.zeros(len(wavenumbers))+order_index/10.0
                    

                    if aotf_freq in aotf_frames_to_plot:
                        ax2.scatter(wavenumbers,y_axis,marker=".",color=linecolours[order_index], label="Diffraction Order %i" %order)
                        ax2.plot(wavenumbers,(y_axis+normalise(blaze)*0.1))
                    
                    for pixel_number in pixels: #loop through pixel
                    
                        """for each pixel, find corresponding aotf function heights"""
                        index = np.abs(wavenumbers[pixel_number] - aotf_wavenumbers).argmin()
#                        sinc2_planck_optics_blaze = 
                            
                        if aotf_freq in aotf_frames_to_plot and pixel_number in pixels_to_plot:
                            ax2.scatter(aotf_wavenumbers[index],y_axis[pixel_number],marker="x",color=linecolours[order_index])
                            ax2.scatter(aotf_wavenumbers[index],(y_axis[pixel_number]+normalise(blaze)[pixel_number]*0.1),marker="x",color=linecolours[order_index])
                            ax2.scatter(aotf_wavenumbers[index],normalise(optics_transmission)[index],marker="x",color=linecolours[order_index])
                            ax2.scatter(aotf_wavenumbers[index],normalise(sinc2_planck_waven[index], dset_all=sinc2_planck_waven),marker="x",color=linecolours[order_index])
                            ax2.scatter(aotf_wavenumbers[index],normalise(planck_waven[index], dset_all=planck_waven),marker="x",color=linecolours[order_index])
                            ax2.legend()
                        

                        total_radiances[pixel_number] += sinc2_planck_optics[index] * blaze[pixel_number]
                
            #    print("ratios of sinc2_planck_waven")
            #    print(normalise(sinc2_planck_waven[indices]))
            #    print("total_radiance")
            #    print(total_radiances)
            
                detector_data_binned = np.mean(detector_data[frame_index,detector_rows_to_bin,:], axis=0)
                if aotf_freq in aotf_frames_to_plot:
                    fig3 = plt.figure(figsize=(figx,figy))
                    ax3 = fig3.add_subplot(111)
#                    ax3.plot((normalise(total_radiances)*np.max(detector_data[frame_index,12,:])),label="Sum radiance for order %i" %centre_order)
#                    ax3.plot((normalise(detector_data_binned)*np.max(detector_data[frame_index,12,:])),label="Binned signal")
                    ax3.plot(normalise(total_radiances)*np.max(detector_data_binned),label="Sum radiance for order %i" %centre_order)
                    ax3.plot(detector_data_binned,label="Binned signal")
                    ax3.set_xlabel("Pixel number")
                    ax3.set_ylabel("Blaze function per horizontal detector bin")
                    ax3.set_title(obspaths[file_choice])
                    ax3.set_ylim(ymin=0)
                
                    for detector_row in detector_rows_to_bin:
        #                ax3.plot(normalise(detector_data[frame_index,detector_row,:], dset_all=detector_data[frame_index,:,:]),alpha=0.3)
                        ax3.plot(detector_data[frame_index,detector_row,:],alpha=0.3)
                        ax3.legend()
                    
                binned_signal_1ms_radiance_ratio = detector_data_binned/ true_obs_time / np.float(binning) / total_radiances
                filtered_signal_ratio = fft_filter(binned_signal_1ms_radiance_ratio)
                if aotf_freq==aotf_to_average:
                    filtered_signal_ratios.append(filtered_signal_ratio)
                    measurement_temperatures.append(measurement_temperature)
                
                colour_loop += 1
                """normalise"""
#                ax1.plot(normalise(binned_signal_1ms_radiance_ratio, dset_all=filtered_signal_ratio), alpha=0.3, color=linecolours[colour_loop])
#                ax1.plot(normalise(filtered_signal_ratio),label="Order=%i,IT=%i,BB temp=%iC,NOMAD temp=%iC filtered" %(centre_order,inttime,bb_temp,measurement_temperature), color=linecolours[colour_loop])
                """plot vs wavenumber"""
#                ax1.plot(central_order_wavenumbers,binned_signal_1ms_radiance_ratio,alpha=0.3, color=linecolours[colour_loop])
#                ax1.plot(central_order_wavenumbers,filtered_signal_ratio,label="Order=%i at %iC,IT=%i,BB at %iC," %(centre_order,measurement_temperature,inttime,bb_temp), color=linecolours[colour_loop])
#                ax1.set_xlabel("Wavenumber cm-1")
                """"plot vs pixel number"""
                ax1.plot(binned_signal_1ms_radiance_ratio,alpha=0.3, color=linecolours[colour_loop])
                ax1.plot(filtered_signal_ratio,label="Order=%i at %iC,IT=%i,BB at %iC," %(centre_order,measurement_temperature,inttime,bb_temp), color=linecolours[colour_loop])
                ax1.set_xlabel("Pixel number")
                ax1.set_title("Blaze angles calculated for a range of blackbody and instrument temperatures")
                ax1.set_ylabel("Normalised and binned blaze angle")
                ax1.legend()
    
#        ax1.set_ylim(ymin=0, ymax=0.22)
    indices=[]
    for index,measurement_temperature in enumerate(measurement_temperatures):
#        if index in pfm_indices:
#            if -15.0 < np.asfarray(measurement_temperature) < -5.0:
        if -25.0 < np.asfarray(measurement_temperature) < 25.0:
            indices.append(index)
    mean_measurement_temperature = np.mean(np.asfarray(measurement_temperatures)[indices])
    mean_filtered_signal_ratio = np.mean(np.asfarray(filtered_signal_ratios)[indices], axis=0)
    std_filtered_signal_ratio = np.std(np.asfarray(filtered_signal_ratios)[indices], axis=0)
    
    fig4 = plt.figure(figsize=(figx,figy))
    ax4 = fig4.add_subplot(111)
    ax4.fill_between(range(len(mean_filtered_signal_ratio)), mean_filtered_signal_ratio+std_filtered_signal_ratio, mean_filtered_signal_ratio-std_filtered_signal_ratio, facecolor='red', alpha=0.5, interpolate=True)
    ax4.plot(mean_filtered_signal_ratio)
    ax4.set_xlabel("Pixel number")
    ax4.set_title("Mean radiance conversion")
    ax4.set_ylabel("Pixel counts to radiance ratio")



if option==2:
    """write hdf5 file containing all dark stepping from MCO1, corrected to the first measurement using the regions of overlap to normalise subsequent frames"""

    if channel=="lno":
        full_frame = np.zeros((480,256,320)) * np.nan
    elif channel=="so":
        full_frame = np.zeros((960,256,320)) * np.nan
    plt.figure(figsize=(figx,figy))
#    crossover_lines = [4,4,4,4,4,8,4] #number of lines that overlap the previous 
    
    for file_index,hdf5_file in enumerate(hdf5_files):
        detector_data,_,_ = get_dataset_contents(hdf5_file,"Y")
        inttime_all = get_dataset_contents(hdf5_file,"IntegrationTime")[0]
        windowtop_all = get_dataset_contents(hdf5_file,"WindowTop")[0]
        frames_to_plot = [100]
        frames_to_compare = range(80,100)
        
        if file_index==0:
            if channel=="lno":
                temperature,_,_ = get_dataset_contents(hdf5_file,"AOTF_TEMP_LNO") #get data for LNO
            elif channel=="so":
                temperature,_,_ = get_dataset_contents(hdf5_file,"AOTF_TEMP_SO") #get data for SO
            nomad_temperature = np.mean(temperature[1:10])
        
        frame_to_plot=frames_to_plot[0]
        
        windowtop = windowtop_all[0]
        windowbottom = windowtop+24
#        print(windowtop,windowbottom
    
        old_new_ratio = [] #loop through lines, checking if data overlapping
        for line_index,windowline in enumerate(range(windowtop,windowbottom)):
            if not np.isnan(full_frame[0,windowline,0]): #check first pixel of one frame for nan
                for frame_to_compare in frames_to_compare:
                    #if line overlaps, find mean ratio of old and new lines
                    old_new_ratio.append(np.mean(full_frame[frame_to_compare,windowline,:]/detector_data[frame_to_compare,line_index,:]))
                    plt.plot(full_frame[frame_to_compare,windowline,:]/detector_data[frame_to_compare,line_index,:], label=line_index)
        if old_new_ratio==[]: #if no previous frame
            average_old_new_ratio = 1.0 #use 
        else:
            average_old_new_ratio = np.mean(old_new_ratio)
        corrected_detector_data = detector_data[:,:,:] * average_old_new_ratio
        full_frame[:,windowtop:windowbottom,:]=corrected_detector_data[:,:,:]
    plt.legend()

    plt.figure(figsize=(figx,figy))
    plt.title(title+" Log Scale")
    plt.xlabel("Spectral Dimension")
    plt.ylabel("Spatial Dimension")
    plt.imshow(np.log(full_frame[frame_to_plot,:,:]),interpolation='none',cmap=plt.cm.gray)
    plt.colorbar()
    
    plt.figure(figsize=(figx,figy))
    plt.plot(full_frame[frame_to_plot,150,:])    
    plt.plot(full_frame[frame_to_plot,:,155])    

    if save_files:
        output_filename = "%s_IntegrationTime_Stepping_Corrected_Full_Frame_%iC" %(title.replace(" ","_"),nomad_temperature)
        #write corrected dark frame dataset to file
        hdf5_file = h5py.File(output_filename+".h5", "w")
        write_to_hdf5(hdf5_file,full_frame,output_filename,np.float,frame=channel)
        write_to_hdf5(hdf5_file,inttime_all,"IntegrationTime",np.float)
        comments = "Files Used: "
        for obspath in obspaths:
            comments = comments + obspath + "; "
        hdf5_file.attrs["Comments"] = comments
        hdf5_file.attrs["Date_Created"] = str(datetime.now())
        hdf5_file.close()
    
    
    
    
    
    
    
    
if option==3:
    """read in dark stepping hdf5 file, make clickable figure. output bad pixel map"""
    if channel=="lno":
        max_deviation_from_mean_gradient=3.0
        max_log_chisq=18.0
    elif channel=="so":
        max_deviation_from_mean_gradient=3.0
        max_log_chisq=18.5

#    plot_type="gradient"
#    plot_type="deviation_from_mean_gradient"
    plot_type="chisq"
#    plot_type="none"
    
    """load file manually"""
    if channel=="lno":
        measurementTemperature = -19
    elif channel=="so":
        measurementTemperature = -12
    obspath = "%s_MCO_Dark_IntegrationTime_Stepping_Corrected_Full_Frame_%iC" %(channel.upper(),measurementTemperature); temperature=-19
    filename = os.path.normcase(AUXILIARY_DIRECTORY+os.sep+"non_linearity"+os.sep+obspath+".h5") #choose a file
    nomad_temperature = filename[(filename.find("frame_")+6):(filename.find("c.h5"))]    
    hdf5_file = h5py.File(filename, "r") #open file
    corrected_darks = get_dataset_contents(hdf5_file,"%s_MCO_Dark_IntegrationTime_Stepping_Corrected_Full_Frame_%iC" %(channel.upper(),measurementTemperature),return_calibration_file=True)[0]
    int_time_all = get_dataset_contents(hdf5_file,"IntegrationTime",return_calibration_file=True)[0]
    
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
        global save_figs
        v_index=int(np.round(event.ydata))
        h_index=int(np.round(event.xdata))
        print('xdata=%i, ydata=%i' %(h_index,v_index))
        pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
        _,_=chisquared(inttimes,pixelvalues,plot_fig=True)
        plt.title("Integration time stepping for pixel (%i,%i)" %(h_index,v_index))
        plt.tight_layout()
        if save_figs: plt.savefig("%s_IntegrationTime_stepping_for_pixel_%i_%i_%iC.png" %(channel.upper(),h_index,v_index,temperature))
       
    fig = plt.figure(figsize=(figx,figy))
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
#    if save_figs: plt.savefig("IntegrationTime_stepping_sum_of_chi-squared_values_compared_to_linear_fit_%iC" %temperature)
    
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    
    bad_pixel_map = np.zeros((256,320), dtype=bool)
    gradient_deviation_map = np.abs(frame_gradient-np.nanmean(frame_gradient))
    chisq_log_map = np.log(frame_chisq)
    
    for h_index in range(320):
        for v_index in range(256):
            if gradient_deviation_map[v_index,h_index]>max_deviation_from_mean_gradient or chisq_log_map[v_index,h_index]>max_log_chisq:
                bad_pixel_map[v_index,h_index]=1
    plt.figure(figsize=(figx,figy))
    plt.imshow(bad_pixel_map,interpolation='none',cmap=plt.cm.gray)
    plt.colorbar()

    if save_files:
        output_filename = "%s" %(title.replace(" ","_"))
        #write bad pixel map to file
        hdf5_file = h5py.File(output_filename+".h5", "w")
        write_to_hdf5(hdf5_file,bad_pixel_map,output_filename,np.bool,frame=channel)
        comments = "Files Used: %s; Maximum deviation from mean gradient: %0.1f; Maximum log chi squared value: %0.1f" %(obspath+".h5",max_deviation_from_mean_gradient,max_log_chisq)
        hdf5_file.attrs["Comments"] = comments
        comments2 = "nomad_temperature=%s" %nomad_temperature
        hdf5_file.attrs["Temperature"] = comments2
        hdf5_file.attrs["Date_Created"] = str(datetime.now())
        hdf5_file.close()
        
       
        
        
        
        
        
        
if option==4:
    """read in dark stepping hdf5 file, make non-linearity correction lookup table and check it"""
#    plot_type="gradient"
#    plot_type="deviation_from_mean_gradient"
    plot_type="chisq"
#    plot_type="none"
    linear_region=[25,100]
    
    obspath = "LNO_MCO_Dark_IntegrationTime_Stepping_Corrected_Full_Frame_-19C"
    filename = os.path.normcase(AUXILIARY_DIRECTORY+os.sep+"non_linearity"+os.sep+obspath+".h5") #choose a file
    nomad_temperature = filename[(filename.find("frame_")+6):(filename.find("c.h5"))]    
    hdf5_file = h5py.File(filename, "r") #open file
    corrected_darks = get_dataset_contents(hdf5_file,"LNO_MCO_Dark_IntegrationTime_Stepping_Corrected_Full_Frame_-19C",return_calibration_file=True)[0]
    int_time_all = get_dataset_contents(hdf5_file,"IntegrationTime",return_calibration_file=True)[0]
    
    frame_gradient=np.zeros((256,320)) * np.nan
    frame_chisq=np.zeros((256,320))
    frame_ranges = [[0,224],[256,480]]    
    
    inttimes=int_time_all[frame_ranges[0][0]:frame_ranges[0][1]]
    
    v_index=100
    h_index=100
    pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
    if not np.isnan(pixelvalues[0,0]): #check if data is there
        frame_gradient[v_index,h_index],frame_chisq[v_index,h_index]=chisquared(inttimes,pixelvalues)
        pixelvalues=np.transpose(np.asarray([corrected_darks[:,v_index,h_index][frame_ranges[0][0]:frame_ranges[0][1]],corrected_darks[:,v_index,h_index][frame_ranges[1][0]:frame_ranges[1][1]]]))
        _,_=chisquared(inttimes,pixelvalues,plot_fig=True)
        plt.title("Integration time stepping for pixel (%i,%i)" %(h_index,v_index))
    
#        sg_fit = sg_filter(pixelvalues, window_size=29, order=2)
        sg_fit1 = savitzky_golay(pixelvalues[:,0], 25, 2)
        sg_fit2 = savitzky_golay(pixelvalues[:,1], 25, 2)
        polycoeffs1 = np.polyfit(inttimes[linear_region[0]:linear_region[1]],pixelvalues[linear_region[0]:linear_region[1],0],1)
        polycoeffs2 = np.polyfit(inttimes[linear_region[0]:linear_region[1]],pixelvalues[linear_region[0]:linear_region[1],1],1)
        polyfit1 = np.polyval(polycoeffs1, inttimes)
        polyfit2 = np.polyval(polycoeffs2, inttimes)

        plt.plot(inttimes,sg_fit1)
        plt.plot(inttimes,sg_fit2)
        
        plt.figure(figsize=(figx,figy))
        plt.plot(inttimes,sg_fit1-pixelvalues[:,0])
        plt.plot(inttimes,sg_fit2-pixelvalues[:,1])
        
        plt.figure(figsize=(figx,figy))
        plt.plot(inttimes,polyfit1-pixelvalues[:,0])
        plt.plot(inttimes,polyfit2-pixelvalues[:,1])
        



"""Build spectral calibration table: AOTF-to-order and AOTF-to-pixel coefficients for a range of instrument temperatures over a specified time range.
NOMAD PFM contains AOTF to order calibration that is probably not correct, so AOTF peak lies near the edge of the detector in each order.
Hence AOTFWnCoeffs give a wavenumber near low end of each order. See EXM-NO-TNO-AER-00083-iss0rev0-Spectral_calibration_coefficients_analysis_170322
"""
if option in [9,10]:
    
    """extra failsafe to avoid accidental overwriting"""
#    save_files=False
    save_files=True
    
    if channel=="lno":
##        V01: old AOTF calibration
#        aotfwn = np.asfarray([
#        1.15534E-07,0.1410044,312.7094
#        ])
        
        #V02:
        #Tuning.dat Arnaud analysis 2016
        #% Coefficients of the polynomial fit of the AOTF frequency wavenumber relation for all bins.
#        aotfwn = np.asfarray([
#        -4.7146920982e-07,1.6810158899e-01,1.3571945401e-03
#        ])
    
        #V03:
        #Tuning.dat Goddard analysis 2017
        #% Coefficients of the polynomial fit of the AOTF frequency wavenumber relation for all bins.
        aotfwn = np.asfarray([
        0.00000009409476,0.1422382,300.67657
        ])


        
        
        #TemperatureDependencePixWn.dat lines 11-18
        #% Coefficients of the polynomial fit of the temperature dependence of the degree zero term of the pixels wavenumber relation.
        #% One line per bin. 
        
        ##warning: pixel number 0.5 to 319.5
        temperature_dependence_pixwin = np.asfarray([
        [7.3774311089e-06,-4.5964191563e-04,2.2458864042e+01], \
        [7.8774332495e-06,-4.0940844596e-04,2.2459039705e+01], \
        [6.9050760384e-06,-4.2364224952e-04,2.2459370225e+01], \
        [4.3285323205e-06,-4.6866196180e-04,2.2459699729e+01], \
        [6.2065975179e-06,-4.3390608328e-04,2.2459436253e+01], \
        [5.5427643502e-06,-4.5261326470e-04,2.2459785353e+01], \
        [4.9446630483e-06,-4.2533566503e-04,2.2459781154e+01], \
        [4.9504316845e-06,-4.4540528268e-04,2.2459754976e+01]
        ])
        
        #PixWn.dat lines 12-19
        #% Coefficients of the second order polynomial fit of the pixels wavenumber relation at an AOTF temperature of 0 degrees C.
        #% The value of the constant term for other temperatures can be calculated using the temperature dependence found in TemperatureDependencePixWn.dat.
        #% One line per bin for the 24 bins.
        
        ##warning: pixel number 0.5 to 319.5
        pix_wn = np.asfarray([
        [3.9217986203e-08,5.5128633467e-04,2.2458864042e+01], \
        [4.0112808297e-08,5.5039942750e-04,2.2459039705e+01], \
        [4.4162950887e-08,5.4907080187e-04,2.2459370225e+01], \
        [4.0953092843e-08,5.4962469763e-04,2.2459699729e+01], \
        [4.3227078593e-08,5.4915173633e-04,2.2459436253e+01], \
        [4.1476889820e-08,5.4875930403e-04,2.2459785353e+01], \
        [3.4849882810e-08,5.5095483243e-04,2.2459781154e+01], \
        [3.2311730517e-08,5.5151319870e-04,2.2459754976e+01]
        ])
        
    elif channel=="so":
#        """Geronimo NEC analysis 31/8/16. No temperature dependency"""
#        #Tuning function not determined by Geronimo - using Arnaud ground cal instead
#        tuning = np.asfarray([
#        1.063549979e-07,1.503354583e-01,2.976862348e+02
#        ])
#    
#        #no temeperature dependence
#        
#        ##warning: pixel number 0 to 319
#        temperature_dependence_pixwin = np.asfarray([
#        [0.0,0.0,2.247514316e+01], \
#        [0.0,0.0,2.247514316e+01], \
#        ])
#        
#        ##warning: pixel number 0 to 319
#        pix_wn = np.asfarray([
#        [6.647320856e-08,5.404944045e-04,2.247514316e+01], \
#        [6.647320856e-08,5.404944045e-04,2.247514316e+01], \
#        ])
    
    
        """Geronimo re-analysis August 2017"""
        aotfwn = np.asfarray([
        1.340818e-7,0.1494441,313.91768
        ])
    
        #temperature dependence shown to be same as for LNO: use quadratic coefficients 0 and 1 from LNO and p0 from SO
        temperature_dependence_pixwin = np.asfarray([
        [6.01661616e-06,-4.39826859e-04,22.473422], \
        [6.01661616e-06,-4.39826859e-04,22.473422], \
        ])

        ##warning: pixel number 0 to 319
        pix_wn = np.asfarray([
        [1.751279e-8,5.559526e-4,22.473422], \
        [1.751279e-8,5.559526e-4,22.473422], \
        ])
                
                  

    print("aotfwn")
    print(aotfwn)
    
    pixelorder_to_waven_coeffs = np.mean(pix_wn, axis=0)
    temperature_to_pixel0_coeffs = np.mean(temperature_dependence_pixwin, axis=0)
    print("pixelorder_to_waven_coeffs")
    print(pixelorder_to_waven_coeffs)
    print("temperature_to_pixel0_coeffs")
    print(temperature_to_pixel0_coeffs)
    
    #calculate pixel-wavenumber (0.5 to 319.5) for pixels 0 to 319
    #only for LNO!!#
    if channel=="lno":
        pixels0p5 = np.arange(0.5,319.6,1.0)
    elif channel=="so":
        pixels0p5 = np.arange(0.0,319.1,1.0)
        
    pixels0 = np.arange(0.0,320.0,1.0)
    frame_wavenumbers0p5 = np.polyval(pixelorder_to_waven_coeffs, pixels0p5)
    frame_wavenumbers0 = np.interp(pixels0,pixels0p5,frame_wavenumbers0p5)
    pixelorder_to_waven_coeffs_p0 = np.polyfit(pixels0,frame_wavenumbers0,2)
    frame_wavenumbers0_fit = np.polyval(pixelorder_to_waven_coeffs_p0,pixels0)
    
    print("pixelorder_to_waven_coeffs_p0")
    print(pixelorder_to_waven_coeffs_p0)

#    pixel0_x = np.arange(22.4,22.5,0.001)
    pixel0_x = np.arange(-30,41,1)
    pixel0_y = np.polyval(list(temperature_to_pixel0_coeffs),pixel0_x)
    
    
    plt.figure(figsize=(figx,figy))
    plt.plot(pixels0p5,frame_wavenumbers0p5)
    plt.plot(pixels0,frame_wavenumbers0)
    plt.plot(pixels0,frame_wavenumbers0_fit)

    plt.figure(figsize=(figx,figy))
    plt.plot(pixel0_x,pixel0_y)
    
    calibration_times = [
                b"2015 JAN 01 00:00:00.000", \
                b"2015 JUL 01 00:00:00.000", \
                b"2016 JAN 01 00:00:00.000", \
                b"2016 JUL 01 00:00:00.000", \
                b"2017 JAN 01 00:00:00.000", \
                b"2017 JUN 01 00:00:00.000", \
                ]
    instrument_temperatures = np.arange(-30,40,0.5)

    number_of_coefficients = 3
    number_of_tuning_coefficients = 3
    
    #calculate the reverse tuning function coefficients
    aotf_freqs = np.arange(12000.0,32000.0,0.1)
    wavenumbers = np.polyval(aotfwn,aotf_freqs)
    reverse_tuning = np.polyfit(wavenumbers,aotf_freqs,2)
    
    print("reverse_tuning")
    print(reverse_tuning)
    
    plt.figure(figsize=(figx,figy))
    plt.plot(wavenumbers,aotf_freqs)
    plt.plot(wavenumbers[::100],np.polyval(reverse_tuning,wavenumbers[::100]),".")
    plt.xlabel("Wavenumbers cm-1")
    plt.ylabel("AOTF frequency kHz")
    
    
    #calculate 
    
    
    coefficient_table = np.zeros((len(calibration_times),len(instrument_temperatures),number_of_coefficients))
    tuning_table = np.zeros((len(calibration_times),len(instrument_temperatures),number_of_tuning_coefficients))
    reverse_tuning_table = np.zeros((len(calibration_times),len(instrument_temperatures),number_of_tuning_coefficients))

    for time_index,calibration_time in enumerate(calibration_times):
        for temp_index,instrument_temperature in enumerate(instrument_temperatures):
            temp_dependent_zero_term = np.polyval(list(temperature_to_pixel0_coeffs),instrument_temperature)
            coefficient_table[time_index,temp_index,:] = [pixelorder_to_waven_coeffs[0],pixelorder_to_waven_coeffs[1],temp_dependent_zero_term]
            tuning_table[time_index,temp_index,:] = list(aotfwn)
            reverse_tuning_table[time_index,temp_index,:] = list(reverse_tuning)
            
            
    if save_files:
        output_filename = "%s" %(title.replace(" ","_"))
        #write bad pixel map to file
        hdf5_file = h5py.File(os.path.normcase(BASE_DIRECTORY+os.sep+output_filename+".h5"), "w")
        write_to_hdf5(hdf5_file,coefficient_table,"PixelSpectralCoefficients",np.float,frame="None")
        write_to_hdf5(hdf5_file,tuning_table,"AOTFWnCoefficients",np.float,frame="None")
        write_to_hdf5(hdf5_file,reverse_tuning_table,"WnAOTFCoefficients",np.float,frame="None")
        write_to_hdf5(hdf5_file,instrument_temperatures,"Temperature",np.float,frame="None")
        write_to_hdf5(hdf5_file,calibration_times,"CalibrationDatetime","S25",frame="None")
    
        if channel=="lno":
    #        V02:
            comments = "Analysis by G. Villaneuva (nomad_calib_gsfc.pdf), 2017-08-28. Temperature dependency by A. Mahieux (Spectral_calibration_2016, NOMAD_LNO_calib_report.docx), 2017-02-03"
#            comments = "Analysis by A. Mahieux (Spectral_calibration_2016, NOMAD_LNO_calib_report.docx), 2017-02-03"
        elif channel=="so":
            comments = "Analysis by G. Villaneuva (nomad_calib_gsfc.pdf), 2017-08-28"
        hdf5_file.attrs["Comments"] = comments
        hdf5_file.attrs["Date_Created"] = str(datetime.now())
        hdf5_file.close()

            
            




if option==11:
    """compare 2015 Arnaud and Geronimo spectral calibration"""
    
#    channel="so"
    channel="lno"
    if channel=="so":
        orders=range(96,225)
        temperature_diff_30c_orders=range(97,211)
        temperature_diff_30c = [1.48,1.49,1.51,1.53,1.54,1.55,1.57,1.59,1.61,1.61,\
        1.63,1.65,1.66,1.67,1.69,1.71,1.72,1.74,1.75,1.77,\
        1.78,1.80,1.82,1.82,1.84,1.86,1.88,1.89,1.90,1.92,\
        1.94,1.95,1.97,1.98,2.00,2.01,2.03,2.05,2.05,2.07,\
        2.09,2.11,2.12,2.13,2.15,2.17,2.18,2.20,2.21,2.22,\
        2.24,2.26,2.28,2.28,2.30,2.32,2.34,2.34,2.36,2.38,\
        2.39,2.41,2.42,2.44,2.45,2.47,2.49,2.50,2.51,2.53,\
        2.55,2.56,2.57,2.59,2.61,2.62,2.64,2.65,2.67,2.68,\
        2.70,2.72,2.73,2.74,2.76,2.78,2.79,2.80,2.82,2.84,\
        2.85,2.87,2.88,2.90,2.91,2.93,2.95,2.95,2.97,2.99,\
        3.01,3.01,3.03,3.05,3.07,3.08,3.09,3.11,3.12,3.14,\
        3.16,3.17,3.18,3.20]


    elif channel=="lno":
        orders=range(108,220)
        temperature_diff_30c_orders=range(107,203)
        temperature_diff_30c = [1.64,1.65,1.67,1.68,1.70,1.71,1.73,1.74,1.76,1.77,\
        1.79,1.80,1.82,1.83,1.85,1.86,1.88,1.89,1.91,1.92,\
        1.94,1.95,1.97,1.99,2.00,2.02,2.03,2.05,2.06,2.08,\
        2.09,2.11,2.12,2.14,2.15,2.17,2.18,2.20,2.21,2.23,\
        2.24,2.26,2.27,2.29,2.30,2.32,2.33,2.35,2.37,2.38,\
        2.40,2.41,2.43,2.44,2.46,2.47,2.49,2.50,2.52,2.53,\
        2.55,2.56,2.58,2.59,2.61,2.62,2.64,2.65,2.67,2.68,\
        2.70,2.71,2.73,2.75,2.76,2.78,2.79,2.81,2.82,2.84,\
        2.85,2.87,2.88,2.90,2.91,2.93,2.94,2.96,2.97,2.99,\
        3.00,3.02,3.03,3.05,3.06,3.08]

    spectra = np.zeros((320,len(orders)))
    difference = np.zeros((320,len(orders)))
    for index,order in enumerate(orders):
        spectra[:,index] = spectral_calibration_simple(channel,order,geronimo_flag=False)
        difference[:,index] = np.abs(spectral_calibration_simple(channel,order,geronimo_flag=False)-spectral_calibration_simple(channel,order,geronimo_flag=True))
        
    temperature_diff_30c_centre_wavenumber = np.zeros(len(temperature_diff_30c_orders))
    for index,temperature_diff_30c_order in enumerate(temperature_diff_30c_orders):
        temperature_diff_30c_centre_wavenumber[index] = spectral_calibration_simple(channel,temperature_diff_30c_order,geronimo_flag=False)[160]
        
    plt.figure(figsize=(10,8))
    plt.scatter(temperature_diff_30c_centre_wavenumber,temperature_diff_30c,marker="x",color="red",label="Theoretical difference due to 30C grating temperature change")
    plt.legend(loc="lower right")
    plt.plot(spectra,difference)
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Difference in wavenumbers cm-1")
    plt.title(channel.upper()+": Difference between CSL gas cell and solar line spectral calibrations")
    if save_figs: plt.savefig(channel.upper()+"_difference_between_CSL_gas_cell_and_solar_line_spectral_calibrations.png")



"""make hdf5 file containing radiometric calibration coefficients, used to convert DNs into Radiances to be input into ASIMUT"""
"""data is derived from inflight AOTF calibration on CSL FS and PFM blackbody datasets"""
"""analyse data for lower orders where signal is good, then extrapolate for higher orders using solar fullscan data and broadband solar spectrum"""

"""output h5 file contains one dataset per order. Each dataset has size NTemperatures x NTimes x NPixels"""
"""at present, no variation with temperature or time"""

if option==12:
    
    
#    linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"] *10
    linecolours = ["r","r","orange","orange","y","y","yellow","lime","lime","g","g","c","c","b","b","k","k","m","m","grey","grey","k"] *10
    
#    if model=="PFM":
#        detector_rows_to_bin = range(1,24)
#    elif model=="FS":
    detector_rows_to_bin = range(1,22) #ignore lines where signal low
    aotf_freq = 16749
    
    def planck(xscale,temp,units): #planck function W/m2/sr/spectral unit
        if units=="microns" or units=="m" or units=="wavel":
            c1=1.191042e8
            c2=1.4387752e4
            return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0)
        elif units=="wavenumbers" or units=="w" or units=="waven":
            c1=1.191042e-5
            c2=1.4387752
            return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) /1000.0
        else:
            print("Error: Unknown units given")


    def func_aotf(x, x0): #Goddard model 2017
        if channel=="so":
            iGi0 = -0.472221
            w = 17.358663
            sigmaG = 8.881119
        elif channel=="lno":
            iGi0 = 0.589821
            w = 18.188122
            sigmaG = 12.181137

        x0 = x0+0.0001 #fudge to stop infinity at peak
        i0 = 1.0 #set to 1
        iG = iGi0 * i0

        fsinc = (i0 * w**2.0 * (np.sin(np.pi * (x - x0) / w))**2.0) / (np.pi**2.0 * (x - x0)**2.0)
        fgauss = iG * np.exp(-1.0 * (x - x0)**2.0 / sigmaG**2.0)
        
        
        return fsinc + fgauss #slant not included
    
    
    
    def normalise(dset, **keyword_parameters):
        if ('dset_all' not in keyword_parameters):
            return dset/max(dset)
        else:
            max_value = np.max(keyword_parameters["dset_all"])
            return (dset/max_value)
        
    
    def optical_transmission(csl_window=False):
        #0:Wavelength, 1:Lens ZnSe, 2:Lens Si, 3: Lens Ge, 4:AOTF, 5:Par mirror, 6:Planar miror, 7:Detector, 8:Cold filter, 9:Window transmission function
        #10:CSL sapphire window
        optics_all = np.loadtxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_optics_transmission.csv", skiprows=1, delimiter=",")
        if not csl_window:
            optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9])
        else:
            optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9]) * (optics_all[:,10])
        optics_wavenumbers =  10000. / optics_all[:,0]
        return optics_wavenumbers, optics_transmission_total


    fig3 = plt.figure(figsize=(figx,figy))
    ax3 = fig3.add_subplot(111)
    for file_choice in [0,10,13]:#range(len(obspaths))[0,10,20]:
    
        """read in data from file"""
        detector_data,_,_ = get_dataset_contents(hdf5_files[file_choice],"Y")
        nacc_all = get_dataset_contents(hdf5_files[file_choice],"NumberOfAccumulations")[0]
        inttime_all = get_dataset_contents(hdf5_files[file_choice],"IntegrationTime")[0]
        aotf_freq_all = get_dataset_contents(hdf5_files[file_choice],"AOTFFrequency")[0]
        binning_all = get_dataset_contents(hdf5_files[file_choice],"Binning")[0]
        backsub_all = get_dataset_contents(hdf5_files[file_choice],"BackgroundSubtraction")[0]
        measurement_temperature = np.mean(get_dataset_contents(hdf5_files[file_choice],"AOTF_TEMP_%s" %channel.upper())[0][2:10])
        print("*******************************")
        print("measurement_temperature %iC" %measurement_temperature)

        chosen_frame_indices = list(np.where(aotf_freq_all==aotf_freq)[0])
        print("found %i frames with aotf frequency %i" %(len(chosen_frame_indices),aotf_freq))
        backsub = backsub_all[0]
        nacc = nacc_all[0]
        inttime = inttime_all[0]
        binning = binning_all[0]+1

        print("nacc %i" %nacc)
        print("inttime %i" %inttime)
    
        """calculate true observation time"""
        if backsub==1:
            true_obs_time = (nacc/2) * inttime
            print("Actual Obs Time = %ims" %true_obs_time)
            ratio_15s = 15000/true_obs_time
        else:
            print("Error: sbsf must be on")
    
        blackbody_in_fov=False
        globar_in_fov=False
        attributes,attr_values=get_hdf5_attributes(hdf5_files[file_choice])
        for attr_value in list(attr_values):
    #        if isinstance(attr_value, str): 
            if "lackbody" in str(attr_value):
                print(attr_value)
                bb_temp_string = str(attr_value).split("lackbody at ")[1].split("C")[0].replace("\\r","")
                blackbody_in_fov=True
            if "Globar" in str(attr_value):
                print(attr_value)
                globar_in_fov=True
        
        if blackbody_in_fov:
            try:
                bb_temp = float(bb_temp_string) + 273.0
            except ValueError:
                print("Warning: BB Temp is not a float")
        elif globar_in_fov:
            bb_temp = 800.0
        else:
            print("Error: neither blackbody nor globar in FOV")
            
        frame_index = chosen_frame_indices[0]

        centre_order = spectralCalibration("aotf2order",channel,aotf_freq,0) #find nearest order number
        print("centre_order %0.1f" %centre_order)
    
        aotf_centre_wavenumber = spectralCalibration("aotf2waven",channel,aotf_freq,0) #find aotf central wavenumber
        aotf_wavenumbers = np.arange(aotf_centre_wavenumber-40.0,aotf_centre_wavenumber+40.0,0.01) #make aotf function x axis
        aotf_function = func_aotf(aotf_wavenumbers,aotf_centre_wavenumber) #make aotf function
        source_planck = planck(aotf_wavenumbers,bb_temp,"wavenumbers") #planck function
        aotf_and_source = aotf_function * source_planck
        print("aotf_centre_wavenumber %0.1f" %aotf_centre_wavenumber)
    
        central_order_wavenumbers = spectralCalibration("pixel2waven",channel,centre_order,measurement_temperature) #find wavenumbers for order
        print("central_order_wavenumbers first %0.1f last %0.1f" %(central_order_wavenumbers[0],central_order_wavenumbers[-1]))

        
        #calculate centre order
        #calculate wavenumber at aotf peak and full aotf function
        #multiply by BB source to get input radiance

        #now match pixel to aotf radiance for each pixel
        pixels = np.arange(320)
        total_radiances=np.zeros(320)
        orders = [centre_order-3,centre_order-2,centre_order-1,centre_order,centre_order+1,centre_order+2,centre_order+3]

        for order in orders:
            order_wavenumbers = spectralCalibration("pixel2waven",channel,order,measurement_temperature) #find wavenumbers for order

            for pixel_index,pixel_number in enumerate(pixels): #loop through pixel
                matching_index = np.abs(aotf_wavenumbers - order_wavenumbers[pixel_index]).argmin()
                total_radiances[pixel_index] = aotf_and_source[matching_index]
        

        """sum radiance for each pixel"""
            





        detector_data_binned = np.mean(detector_data[frame_index,detector_rows_to_bin,:], axis=0)
        ax3.plot(central_order_wavenumbers,detector_data_binned,label="Binned signal")
#        for detector_row in detector_rows_to_bin:
#            ax3.plot(central_order_wavenumbers,detector_data[frame_index,detector_row,:],alpha=0.3)
        ax3.legend()
        ax3.set_xlabel("Pixel number")
        ax3.set_ylabel("Blaze function per horizontal detector bin")
        ax3.set_title(obspaths[file_choice])
        ax3.set_ylim(ymin=0)

        ax3.plot(aotf_wavenumbers,normalise(aotf_function)*np.max(detector_data_binned),label="AOTF function")
        ax3.plot(aotf_wavenumbers,normalise(aotf_and_source)*np.max(detector_data_binned),label="AOTF and source function")
        ax3.plot(central_order_wavenumbers,normalise(total_radiances)*np.max(detector_data_binned),label="AOTF and source function summed")
    

    
    

#    file_choices=[0,1,2]#range(len(obspaths))
#    pfm_indices = [19,20,21]
#    temperature_dependencies = ["fake"]#["real"]
#    temperature_to_use = -25.0 #if not using real temperatures
#
#    aotf_frames_to_plot = [16749]#[16749]#16593,16749,16904] #check aotf matches with aotfs_to_plot, otherwise only radiance/counts conversion will be plotted
#    pixels_to_plot = [160] #just for crosses on aotf plot
#    aotfs_to_plot = [16749]#[16593,16749,16904] #which radiance/counts conversions should be plotted?
#    aotf_to_average = 16749#16904
#    filtered_signal_ratios=[]
#    measurement_temperatures=[]
#    plotted=False
#
#    for aotf_to_plot in aotfs_to_plot:
#        fig1 = plt.figure(figsize=(figx,figy))
#        ax1 = fig1.add_subplot(111)
#        colour_loop=-1
#        for file_choice in file_choices:
#    
#            """read in data from file"""
#            detector_data,_,_ = get_dataset_contents(hdf5_files[file_choice],"Y")
#            nacc_all = get_dataset_contents(hdf5_files[file_choice],"NumberOfAccumulations")[0]
#            inttime_all = get_dataset_contents(hdf5_files[file_choice],"IntegrationTime")[0]
#            aotf_freq_all = get_dataset_contents(hdf5_files[file_choice],"AOTFFrequency")[0]
#            binning_all = get_dataset_contents(hdf5_files[file_choice],"Binning")[0]
#            backsub_all = get_dataset_contents(hdf5_files[file_choice],"BackgroundSubtraction")[0]
#            measurement_temperature = np.mean(get_dataset_contents(hdf5_files[file_choice],"AOTF_TEMP_%s" %channel.upper())[0][2:10])
##            measurement_temperature = 10.0
#            print("measurement_temperature %iC" %measurement_temperature)
#    
#            chosen_frame_indices = list(np.where(aotf_freq_all==aotf_to_plot)[0])
#            print("found %i frames with aotf frequency %i" %(len(chosen_frame_indices),aotf_to_plot))
#            backsub = backsub_all[0]
#            nacc = nacc_all[0]
#            inttime = inttime_all[0]
#            binning = binning_all[0]+1
#            
#            print("nacc %i" %nacc)
#            print("inttime %i" %inttime)
#        
#            """calculate true observation time"""
#            if backsub==1:
#                true_obs_time = (nacc/2) * inttime
#                print("Actual Obs Time = %ims" %true_obs_time)
#                ratio_15s = 15000/true_obs_time
#            else:
#                print("Error: sbsf must be on")
#        
#            blackbody_in_fov=False
#            globar_in_fov=False
#            attributes,attr_values=get_hdf5_attributes(hdf5_files[file_choice])
#            for attr_value in list(attr_values):
#        #        if isinstance(attr_value, str): 
#                if "lackbody" in str(attr_value):
#                    print(attr_value)
#                    bb_temp_string = str(attr_value).split("lackbody at ")[1].split("C")[0].replace("\\r","")
#                    blackbody_in_fov=True
#                if "Globar" in str(attr_value):
#                    print(attr_value)
#                    globar_in_fov=True
#            
#            if blackbody_in_fov:
#                try:
#                    bb_temp = float(bb_temp_string) + 273.0
#                except ValueError:
#                    print("Warning: BB Temp is not a float")
#            elif globar_in_fov:
#                bb_temp = 800.0
#            else:
#                print("Error: neither blackbody nor globar in FOV")
#        
#        
#            for frame_index in chosen_frame_indices:#range(10,81,5):
#                
#                aotf_freq = aotf_freq_all[frame_index]
#                
#                
#                centre_order = spectralCalibration("aotf2order",channel,aotf_freq,0) #find nearest order number
#                print(centre_order)
#                orders = [centre_order-3,centre_order-2,centre_order-1,centre_order,centre_order+1,centre_order+2,centre_order+3]
#                aotf_centre_wavenumber = spectralCalibration("aotf2waven",channel,aotf_freq,0) #find aotf central wavenumber
#                print("aotf_centre_wavenumber")
#                print(aotf_centre_wavenumber)
#                if "real" in temperature_dependencies:
#                    central_order_wavenumbers = spectralCalibration("pixel2waven",channel,centre_order,measurement_temperature) #find wavenumbers for order
#                else:
#                    central_order_wavenumbers = spectralCalibration("pixel2waven",channel,centre_order,temperature_to_use) #find wavenumbers for order
#                print("")
#            
#                aotf_wavenumbers = np.arange(aotf_centre_wavenumber-100.0,aotf_centre_wavenumber+100.0,0.01) #make aotf function x axis
#                planck_waven = planck(aotf_wavenumbers,bb_temp,"wavenumbers") #planck function
#                sinc2_waven = func_aotf(aotf_wavenumbers,aotf_centre_wavenumber) #make aotf function
#                sinc2_planck_waven = planck_waven * sinc2_waven
#                
#                #optics contribution
#                optics_wavenumbers_raw, optics_transmission_raw=optical_transmission(csl_window=True)
#                optics_transmission = np.interp(aotf_wavenumbers,optics_wavenumbers_raw[::-1], optics_transmission_raw[::-1])
#                sinc2_planck_optics = planck_waven * sinc2_waven * optics_transmission
#                
#                if aotf_freq in aotf_frames_to_plot:
#                    if not plotted:
#                        fig2 = plt.figure(figsize=(figx,figy))
#                        ax2 = fig2.add_subplot(111)
#                        ax2.plot(aotf_wavenumbers,normalise(planck_waven),label="Normalised Planck function at %iK" %bb_temp)
#                        ax2.plot(aotf_wavenumbers,normalise(sinc2_waven),label="AOTF passband")
#                        ax2.plot(aotf_wavenumbers,normalise(sinc2_planck_waven),label="AOTF passband scaled to Planck function")
#                        ax2.plot(aotf_wavenumbers,normalise(optics_transmission),label="Normalised Optics Transmission")
#                        ax2.plot(aotf_wavenumbers,normalise(sinc2_planck_optics),label="Optics, AOTF and Planck Function")
#                        ax2.set_xlabel("Wavenumbers cm-1")
#                        ax2.set_ylabel("Normalised radiance/AOTF passband")
#                        ax2.set_title("Diffraction orders superimposed on AOTF function and blackbody radiance at %iK" %bb_temp)
#                        plotted=True
#                
#                pixels = np.arange(320)
#                total_radiances=np.zeros(320)
#            
#                """sum radiance for each pixel"""
#                for order_index,order in enumerate(orders): #loop through order
#                    if "real" in temperature_dependencies:
#                        wavenumbers = spectralCalibration("pixel2waven",channel,order,measurement_temperature) #find wavenumbers for order
#                    else:
#                        wavenumbers = spectralCalibration("pixel2waven",channel,order,temperature_to_use) #find wavenumbers for order
#
#                    y_axis = np.zeros(len(wavenumbers))+order_index/10.0
#                    
#
#                    if aotf_freq in aotf_frames_to_plot:
#                        ax2.scatter(wavenumbers,y_axis,marker=".",color=linecolours[order_index], label="Diffraction Order %i" %order)
##                        ax2.plot(wavenumbers,(y_axis+normalise(blaze)*0.1))
#                        ax2.plot(wavenumbers,y_axis)
#                    
#                    for pixel_number in pixels: #loop through pixel
#                    
#                        """for each pixel, find corresponding aotf function heights"""
#                        index = np.abs(wavenumbers[pixel_number] - aotf_wavenumbers).argmin()
##                        sinc2_planck_optics_blaze = 
#                            
#                        if aotf_freq in aotf_frames_to_plot and pixel_number in pixels_to_plot:
#                            if not plotted:
#                                ax2.scatter(aotf_wavenumbers[index],y_axis[pixel_number],marker="x",color=linecolours[order_index])
#    #                            ax2.scatter(aotf_wavenumbers[index],(y_axis[pixel_number]+normalise(blaze)[pixel_number]*0.1),marker="x",color=linecolours[order_index])
#                                ax2.scatter(aotf_wavenumbers[index],(y_axis[pixel_number]),marker="x",color=linecolours[order_index])
#                                ax2.scatter(aotf_wavenumbers[index],normalise(optics_transmission)[index],marker="x",color=linecolours[order_index])
#                                ax2.scatter(aotf_wavenumbers[index],normalise(sinc2_planck_waven[index], dset_all=sinc2_planck_waven),marker="x",color=linecolours[order_index])
#                                ax2.scatter(aotf_wavenumbers[index],normalise(planck_waven[index], dset_all=planck_waven),marker="x",color=linecolours[order_index])
#                                ax2.legend()
#                        
#
##                        total_radiances[pixel_number] += sinc2_planck_optics[index] * blaze[pixel_number]
#                        total_radiances[pixel_number] += sinc2_planck_optics[index]
#                
#            #    print("ratios of sinc2_planck_waven")
#            #    print(normalise(sinc2_planck_waven[indices]))
#            #    print("total_radiance")
#            #    print(total_radiances)
#            
#                detector_data_binned = np.mean(detector_data[frame_index,detector_rows_to_bin,:], axis=0)
#                if aotf_freq in aotf_frames_to_plot:
#                    fig3 = plt.figure(figsize=(figx,figy))
#                    ax3 = fig3.add_subplot(111)
##                    ax3.plot((normalise(total_radiances)*np.max(detector_data[frame_index,12,:])),label="Sum radiance for order %i" %centre_order)
##                    ax3.plot((normalise(detector_data_binned)*np.max(detector_data[frame_index,12,:])),label="Binned signal")
#                    ax3.plot(normalise(total_radiances)*np.max(detector_data_binned),label="Sum radiance for order %i" %centre_order)
#                    ax3.plot(detector_data_binned,label="Binned signal")
#                    ax3.set_xlabel("Pixel number")
#                    ax3.set_ylabel("Blaze function per horizontal detector bin")
#                    ax3.set_title(obspaths[file_choice])
#                    ax3.set_ylim(ymin=0)
#                
#                    for detector_row in detector_rows_to_bin:
#        #                ax3.plot(normalise(detector_data[frame_index,detector_row,:], dset_all=detector_data[frame_index,:,:]),alpha=0.3)
#                        ax3.plot(detector_data[frame_index,detector_row,:],alpha=0.3)
#                        ax3.legend()
#                    
#                binned_signal_1ms_radiance_ratio = detector_data_binned/ true_obs_time / np.float(binning) / total_radiances
#                filtered_signal_ratio = fft_filter(binned_signal_1ms_radiance_ratio)
#                if aotf_freq==aotf_to_average:
#                    filtered_signal_ratios.append(filtered_signal_ratio)
#                    measurement_temperatures.append(measurement_temperature)
#                
#                colour_loop += 1
#                """normalise"""
##                ax1.plot(normalise(binned_signal_1ms_radiance_ratio, dset_all=filtered_signal_ratio), alpha=0.3, color=linecolours[colour_loop])
##                ax1.plot(normalise(filtered_signal_ratio),label="Order=%i,IT=%i,BB temp=%iC,NOMAD temp=%iC filtered" %(centre_order,inttime,bb_temp,measurement_temperature), color=linecolours[colour_loop])
#                """plot vs wavenumber"""
##                ax1.plot(central_order_wavenumbers,binned_signal_1ms_radiance_ratio,alpha=0.3, color=linecolours[colour_loop])
##                ax1.plot(central_order_wavenumbers,filtered_signal_ratio,label="Order=%i at %iC,IT=%i,BB at %iC," %(centre_order,measurement_temperature,inttime,bb_temp), color=linecolours[colour_loop])
##                ax1.set_xlabel("Wavenumber cm-1")
#                """"plot vs pixel number"""
#                ax1.plot(binned_signal_1ms_radiance_ratio,alpha=0.3, color=linecolours[colour_loop])
#                ax1.plot(filtered_signal_ratio,label="Order=%i at %iC,IT=%i,BB at %iC," %(centre_order,measurement_temperature,inttime,bb_temp), color=linecolours[colour_loop])
#                ax1.set_xlabel("Pixel number")
#                ax1.set_title("Blaze angles calculated for a range of blackbody and instrument temperatures")
#                ax1.set_ylabel("Normalised and binned blaze angle")
#                ax1.legend()
#    
##        ax1.set_ylim(ymin=0, ymax=0.22)
#    indices=[]
#    for index,measurement_temperature in enumerate(measurement_temperatures):
##        if index in pfm_indices:
##            if -15.0 < np.asfarray(measurement_temperature) < -5.0:
#        if -25.0 < np.asfarray(measurement_temperature) < 25.0:
#            indices.append(index)
#    mean_measurement_temperature = np.mean(np.asfarray(measurement_temperatures)[indices])
#    mean_filtered_signal_ratio = np.mean(np.asfarray(filtered_signal_ratios)[indices], axis=0)
#    std_filtered_signal_ratio = np.std(np.asfarray(filtered_signal_ratios)[indices], axis=0)
#    
#    fig4 = plt.figure(figsize=(figx,figy))
#    ax4 = fig4.add_subplot(111)
#    ax4.fill_between(range(len(mean_filtered_signal_ratio)), mean_filtered_signal_ratio+std_filtered_signal_ratio, mean_filtered_signal_ratio-std_filtered_signal_ratio, facecolor='red', alpha=0.5, interpolate=True)
#    ax4.plot(mean_filtered_signal_ratio)
#    ax4.set_xlabel("Pixel number")
#    ax4.set_title("Mean radiance conversion")
#    ax4.set_ylabel("Pixel counts to radiance ratio")
#
#



























if option == 14:
    """analyse hdf5 miniscan interpolated dataset. See inflight analysis for other versions using level 0.1c data"""
    sg_window_size=49 #check this
    filename=os.path.normcase(AUXILIARY_DIRECTORY+os.sep+"spectral_calibration"+os.sep+"LNO_Solar_Miniscans_smoothing=%i.h5" %sg_window_size) #choose a file
    hdf5_file = h5py.File(filename, "r")

    #old tunning function
    tuning = np.asfarray([-4.7146920982e-07,1.6810158899e-01,1.3571945401e-03])

#    miniscan_spectra = get_dataset_contents(hdf5_file,"MiniscanSpectraInterpolated",return_calibration_file=True)[0]
    miniscan_spectra = get_dataset_contents(hdf5_file,"MiniscanSpectraSmoothed",return_calibration_file=True)[0]
    miniscan_aotfs = get_dataset_contents(hdf5_file,"MiniscanAOTFsInterpolated",return_calibration_file=True)[0]
    pixels = np.arange(320)

    
    """part to read in nomad solar spectra from text file"""
    solar_spectrum_filename = AUXILIARY_DIRECTORY+os.sep+"spectral_calibration"+os.sep+"nomad_solar_spectrum.txt"
#    solar_spectrum_filename = AUXILIARY_DIRECTORY+os.sep+"spectral_calibration"+os.sep+"nomad_solar_spectrum_flattened_smoothing=%i.txt" %window_size
    solar_spectrum_data = np.loadtxt(solar_spectrum_filename)
    solar_wavenumber = solar_spectrum_data[:,0]
    solar_radiance_flattened = solar_spectrum_data[:,1]


    plt.figure(figsize = (figx,figy))
    plt.plot(solar_wavenumber,solar_radiance_flattened)


    
    line_centres=[7842,8807]
    line_ranges=[[10,80],[60,60]]
#    continuum_ranges = [[105,111,127,133],[220,222,240,245]]
    continuum_ranges = [[105,111,127,133],[220,222,240,245]]

    for line_range,line_centre,continuum_range in zip(line_ranges,line_centres,continuum_ranges):
        fig = plt.figure(figsize = (figx,figy))
        ax1 = plt.subplot2grid((2,1),(0,0))
        ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)

        frames = range(line_centre-line_range[0],line_centre+line_range[1])
        frame_aotfs = miniscan_aotfs[line_centre-line_range[0]:line_centre+line_range[1]]
        
    
        absorption_depths = []
        for frame in frames:
    #    frame=7831
        #line centred on 119 and 120
        #line range 111 and 127
        
            
            continuum_pixel = pixels[range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]    
            continuum_spectrum = miniscan_spectra[frame,range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]
            
            coefficients = np.polyfit(continuum_pixel,continuum_spectrum,2)
            
            absorption_pixel = pixels[range(continuum_range[1],continuum_range[2])]
            absorption_spectrum = miniscan_spectra[frame,range(continuum_range[1],continuum_range[2])]
            absorption_continuum = np.polyval(coefficients, absorption_pixel)
            
            absorption = absorption_spectrum/absorption_continuum
        
            absorption_depths.append(min(absorption))
            if frame==line_centre:
                ax1.plot(miniscan_spectra[frame,:])
                ax1.scatter(continuum_pixel,continuum_spectrum,c="g")
                ax1.scatter(absorption_pixel,absorption_continuum,c="r")
                ax2.plot(absorption_pixel,absorption)
        
    
        plt.figure(figsize = (figx,figy))
        plt.plot(frame_aotfs,absorption_depths)
        
        frame_coefficients = np.polyfit(frames,absorption_depths,2)
        frame_fit = np.polyval(frame_coefficients, frames)
        plt.plot(frame_aotfs,frame_fit)
        x = -1.0*frame_coefficients[1]/(2.0*frame_coefficients[0])
        y = np.polyval(frame_coefficients, x)
        plt.text(x, y+0.005, "%0.3f" %x)
        print("centre at %0.1f" %x)
    
        
        old_tuning = np.polyval(tuning,frame_aotfs)
    
        plt.figure(figsize = (figx,figy))
        plt.plot(old_tuning,absorption_depths)



"""plot solar spectrum for use with miniscan analysis"""
if option==15:
    
    window_size=9999
    sg_window_size=29
    
#    wavenumber_start=2611.0
#    wavenumber_end=3250.0
    wavenumber_start=2400.0
    wavenumber_end=4500.0
    
    
    """part to make nomad solar spectra"""
#    solar_spectrum_filename = BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"Solar_radiance_ACE.dat" #or solspec
    solar_spectrum_filename = BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"Solar_irradiance_ACESOLSPEC_2015.dat" #or solspec
    solar_spectrum_data = np.loadtxt(solar_spectrum_filename, skiprows=6)
#    nomad_solar_range = [346000,808000] #2230 - 4540
    nomad_solar_range = [np.min(np.where(solar_spectrum_data[:,0] > wavenumber_start)),np.max(np.where(solar_spectrum_data[:,0] < wavenumber_end))] #chosen from parameters
    solar_wavenumber = solar_spectrum_data[nomad_solar_range[0]:nomad_solar_range[1],0]
    solar_radiance = solar_spectrum_data[nomad_solar_range[0]:nomad_solar_range[1],1]
    solar_radiance_smoothed = sg_filter(solar_radiance,window_size=window_size,order=2)
    solar_continuum_coeffs = np.polyfit(solar_wavenumber,solar_radiance, 2)
    solar_continuum = np.polyval(solar_continuum_coeffs,solar_wavenumber)
    solar_radiance_smoothed_flattened = solar_radiance_smoothed / solar_continuum   
    solar_radiance_flattened = solar_radiance / solar_continuum   
    if save_files:
        np.savetxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_solar_spectrum.txt", np.transpose(np.asarray([solar_wavenumber,solar_radiance])), fmt="%11.9g")
        np.savetxt(BASE_DIRECTORY+os.sep+"reference_files"+os.sep+"nomad_solar_spectrum_flattened_smoothing=%i.txt" %window_size, np.transpose(np.asarray([solar_wavenumber,solar_radiance_flattened])), fmt="%11.9g")
    plt.figure(figsize = (figx,figy))
    plt.plot(solar_wavenumber,solar_radiance)
    plt.plot(solar_wavenumber,solar_radiance_smoothed)
    plt.plot(solar_wavenumber,solar_continuum)

    plt.figure(figsize = (figx,figy))
    plt.plot(solar_wavenumber,solar_radiance_flattened)

    plt.figure(figsize = (figx,figy))
    plt.plot(solar_wavenumber,solar_radiance_smoothed_flattened)


    """need to convolve spectra with gaussian/passband 0.4cm-1 wide!"""

    win = signal.hann(50)
    solar_radiance_filtered = signal.convolve(solar_radiance_flattened, win, mode='same') / sum(win)
    
    plt.figure(figsize = (figx,figy))
    plt.plot(solar_wavenumber,solar_radiance_filtered)
    

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
                
            








