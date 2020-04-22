# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 17:54:54 2017

@author: ithom
"""


import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime

#import bisect
#from mpl_toolkits.basemap import Basemap
#from scipy import interpolate

from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v02 import get_dataset_contents#,get_hdf5_attributes
#from spice_functions_v01 import convert_hdf5_time_to_spice_utc,find_boresight,find_rad_lon_lat,py_ang
from analysis_functions_v01 import stop#,interpolate_bad_pixel,savitzky_golay,sg_filter,chisquared,find_order,spectral_calibration_simple,write_log,get_filename_list,
from pipeline_config_v03 import figx,figy,BASE_DIRECTORY#,KERNEL_DIRECTORY,AUXILIARY_DIRECTORY

rcParams["axes.formatter.useoffset"] = False
file_level="hdf5_level_0p1c"
#DATA_ROOT_DIRECTORY=os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db")
#FS_DATA_ROOT_DIRECTORY=os.path.normcase(r"X:\projects\NOMAD\data\flight_spare\hdf5_files_cat_split_uvis")

DATA_ROOT_DIRECTORY=os.path.normcase(r"C:\Users\iant\Documents\Python\Data")
FS_DATA_ROOT_DIRECTORY=os.path.normcase(r"C:\Users\iant\Documents\Python\Data\flight_spare\hdf5_files_cat_split_uvis")


save_figs=False
#save_figs=True

#save_files=True
save_files=False

"""Enter location of data"""


#title="UVIS 370BP Spectra"; option=1; file_level="hdf5_level_0p1c"
#title="UVIS D2 Lamp Spectra"; option=1; file_level="hdf5_level_0p1c"
#title="UVIS KV520 Filter Spectra"; option=1; file_level="hdf5_level_0p1c"
#title="UVIS KV370 Filter Spectra"; option=1; file_level="hdf5_level_0p1c"
#title="UVIS Diffraction Grating Model"; option=2
#title="Krypton PFM and FS Comparison"; option=3; file_level="hdf5_level_0p1c"; files_to_compare = [[0,1]]
#title="Hg PFM and FS Comparison"; option=3; file_level="hdf5_level_0p1c"; files_to_compare = [[0,1]]

"""PFM FS comparisons: 2x models, 2x ITs, 3x lamps, 2x temperatures, 10x frames each"""
#PFM 22/03: -15C and 31/03: -5C
#FS 06/09: -15C and 07/09: 0C

#title="UVIS PFM (+20C) and FS (-15C) Comparison"; option=3; file_level="hdf5_level_0p1c"
#title="UVIS PFM (-5C) and FS (-15C) Comparison"; option=3; file_level="hdf5_level_0p1c"
#title="UVIS PFM (+20C) and PFM (-5C) Comparison"; option=3; file_level="hdf5_level_0p1c"
#title="UVIS QTH PFM (+20C,-5C,-15C) Comparison"; option=3; file_level="hdf5_level_0p1c"
#title="UVIS QTH PFM (-5C,-15C) FS (0C,-15C) Comparison"; option=3; file_level="hdf5_level_0p1c"; linestyles = ["-","-","-","-","--","--","--","--"]; files_to_compare = []#[[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]
#title="UVIS RS12 PFM (-5C,-15C) FS (0C,-15C) Comparison"; option=3; file_level="hdf5_level_0p1c"; linestyles = ["-","-","-","-","--","--","--","--"]; files_to_compare = []#[[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]


"""in flight observations"""
title="UVIS Limb Scan MCO2"; option=4; file_level="hdf5_level_0p1c"




obs_dict={"UVIS 370BP Spectra":["20150322_072249_UVIS"],
        "UVIS D2 Lamp Spectra":["20150322_030319_UVIS"],
        "UVIS KV520 Filter Spectra":["20150322_054318_UVIS"],
        "UVIS KV370 Filter Spectra":["20150322_065821_UVIS"],
        "UVIS Diffraction Grating Model":[""],
        "Krypton PFM and FS Comparison":["20150331_182115_UVIS","sinbad_fs_20150907_174926_2_1"],
        "Hg PFM and FS Comparison":["20150331_161719_UVIS","sinbad_fs_20150907_150658_1"],
        
        "UVIS PFM (+20C) and FS (-15C) Comparison":["20150402_080752_UVIS","20150402_081333_UVIS","20150402_105711_UVIS","20150402_110406_UVIS","20150402_122238_UVIS","20150402_123000_UVIS",\
        "sinbad_fs_20150906_144450_0_1","sinbad_fs_20150906_140259_1","sinbad_fs_20150906_185546_1","sinbad_fs_20150906_184322_1","sinbad_fs_20150906_195237_1_1","sinbad_fs_20150906_195237_0_1"],\
        
        "UVIS PFM (-5C) and FS (-15C) Comparison":["20150331_134424_UVIS","20150331_135017_UVIS","20150331_164038_UVIS","20150331_164643_UVIS","20150331_174807_UVIS","20150331_175307_UVIS",\
        "sinbad_fs_20150906_144450_0_1","sinbad_fs_20150906_140259_1","sinbad_fs_20150906_185546_1","sinbad_fs_20150906_184322_1","sinbad_fs_20150906_195237_1_1","sinbad_fs_20150906_195237_0_1"],\
        
        "UVIS PFM (+20C) and PFM (-5C) Comparison":["20150402_080752_UVIS","20150402_081333_UVIS","20150402_105711_UVIS","20150402_110406_UVIS","20150402_122238_UVIS","20150402_123000_UVIS",\
        "20150331_134424_UVIS","20150331_135017_UVIS","20150331_164038_UVIS","20150331_164643_UVIS","20150331_174807_UVIS","20150331_175307_UVIS"],\
        
        "UVIS QTH PFM (+20C,-5C,-15C) Comparison":["20150402_122238_UVIS","20150402_123000_UVIS","20150331_174807_UVIS","20150331_175307_UVIS","20150322_142601_UVIS","20150322_141946_UVIS"],\
        
        "UVIS QTH PFM (-5C,-15C) FS (0C,-15C) Comparison":["20150322_142601_UVIS","20150322_141946_UVIS","20150331_174807_UVIS","20150331_175307_UVIS",\
        "sinbad_fs_20150906_195237_1_1","sinbad_fs_20150906_195237_0_1","sinbad_fs_20150907_174926_0_1","sinbad_fs_20150907_173335_1"],\

        "UVIS RS12 PFM (-5C,-15C) FS (0C,-15C) Comparison":["20150322_120511_UVIS","20150322_121043_UVIS","20150331_164038_UVIS","20150331_164643_UVIS",\
        "sinbad_fs_20150906_185546_1","sinbad_fs_20150906_184322_1","sinbad_fs_20150907_154606_2_1","sinbad_fs_20150907_154606_1_1"],\
        
        "UVIS Limb Scan MCO2":["20161122_134403_UVIS"],\
        }

obspaths=obs_dict[title]


def get_datetime_from_filename(filename):
    year = int(filename[0:4])
    month = int(filename[4:6])
    day = int(filename[6:8])
    hour = int(filename[9:11])
    minute = int(filename[11:13])
    second = int(filename[13:15])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

#note that PFM is in UTC and FS is in local time!!!
ogse_times = [
["20150402_080000","20150402_092000"],["20150331_133800","20150331_150000"],["20150322_174000","20150322_100000"],\
["20150402_104000","20150402_120000"],["20150331_162500","20150331_171500"],["20150322_114000","20150322_124000"],\
["20150402_121000","20150402_124500"],["20150331_173000","20150331_180500"],["20150322_140000","20150322_144000"],\

["20150909_110500","20150909_131500"],["20150907_114000","20150907_140000"],["20150906_133500","20150906_151500"],\
["20150909_140000","20150909_151000"],["20150907_152500","20150907_171500"],["20150906_180500","20150906_191500"],\
["20150909_153000","20150909_160000"],["20150907_172000","20150907_180000"],["20150906_193000","20150906_202000"],\
]
ogse_lamps = [
"D2","D2","D2","RS12","RS12","RS12","QTH","QTH","QTH",\
"D2","D2","D2","RS12","RS12","RS12","QTH","QTH","QTH",\
]

os.chdir(BASE_DIRECTORY)

hdf5_files=[]
models = []
it=[]
lamps=[""] * len(obspaths)
ccd_temps=[]

for file_index,obspath in enumerate(obspaths):
    if obspath != "":
        if "sinbad" in obspath:
            filename=os.path.normcase(FS_DATA_ROOT_DIRECTORY+os.sep+obspath+".h5")
            models.append("FS")
            meastime = get_datetime_from_filename(obspath.replace("sinbad_fs_",""))            
        else:
            year = obspath[0:4]
            month = obspath[4:6]
            day = obspath[6:8]
            filename=os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
            models.append("PFM")
            meastime = get_datetime_from_filename(obspath)            


        hdf5_files.append(h5py.File(filename, "r")) #open file, add to list
        it.append(int(get_dataset_contents(hdf5_files[-1], "IntegrationTime", chosen_group="Channel")[0][0]))
        ccd_temps.append(np.mean(get_dataset_contents(hdf5_files[-1],"TEMP_2_CCD", chosen_group="Housekeeping")[0]))
    
#        print "File %s has the following attributes:" %(filename) #print attributes from file
#        attributes,attr_values=get_hdf5_attributes(hdf5_files[-1])
#        for index in range(len(attributes)):
#            print "%s: %s" %(attributes[index],attr_values[index])
    
        for ogse_index,ogse_time in enumerate(ogse_times):
            start = get_datetime_from_filename(ogse_time[0])
            end = get_datetime_from_filename(ogse_time[1])
            if meastime>start and meastime<end:
                lamps[file_index] = ogse_lamps[ogse_index]
            



    
if option==1:
    """plot UVIS frames"""
    full_frame=True
    rows_to_plot = range(24,97,10)
    frame_to_plot=5
    row_frame_to_plot=range(0,10,2)
    
    xcalcoeffs=[-3.62E-12,-1.64E-09,-2.19E-05,0.47346,194.3127]
    x = range(1048)
    y = range(111)
    x_nm = np.polyval(xcalcoeffs,x)
    
    
    
    
    detector_data_all = get_dataset_contents(hdf5_files[0],"Y")[0] #get data
    frame_name = get_dataset_contents(hdf5_files[0],"Name")
    
    reverse_clock = detector_data_all[0,0:2,:]
    
    i=1;   bias1 = detector_data_all[i:i+8,:,:].reshape((8*15,1048))[0:111,:]
    i=9;   dark1 = detector_data_all[i:i+8,:,:].reshape((8*15,1048))[0:111,:]
    i=97; bias2 = detector_data_all[i:i+8,:,:].reshape((8*15,1048))[0:111,:]
    i=105; dark2 = detector_data_all[i:i+8,:,:].reshape((8*15,1048))[0:111,:]
    
    dark2_binned = np.sum(dark2, axis=0)
    mean_dark = np.mean(np.asfarray([dark1,dark2]),axis=0)

    light = np.zeros((10,111,1048))
    light_sub = np.zeros((10,111,1048))
    light_sub_binned = np.zeros((10,1048))
    starting_indices = range(17,97,8)
    for index,i in enumerate(starting_indices):
        frame = detector_data_all[i:i+8,:,:].reshape((8*15,1048))[0:111,:]
        light[index,:,:] = frame
        light_sub[index,:,:] = frame-mean_dark
        light_sub_binned[index,:] = np.nansum(frame-mean_dark, axis=0)
        
    """plot raw data"""
    fig = plt.figure(figsize=(figx-4,figy))
    ax1 = plt.subplot2grid((3,8),(0,0), colspan=7)
    p1 = plt.imshow(light[frame_to_plot,:,:], aspect=2)
    ax1.set_title(title+" "+obspath+" Raw data")
    ax2 = plt.subplot2grid((3,8),(1,0), colspan=7, rowspan=2)
    for frame_index in row_frame_to_plot:
        for row_to_plot in rows_to_plot:
            plt.plot(light[frame_index,row_to_plot,:], label="Frame %i Row %i" %(frame_index,row_to_plot))
    plt.yscale("log")
    plt.legend()
    
    plt.xlim((0,1048))
    ax3 = plt.subplot2grid((3,8),(0,7), rowspan=3)
    fig.colorbar(p1,cax=ax3)
    plt.tight_layout()
    if save_figs:
        plt.savefig("UVIS_frame_%i_%s_raw_data.png" %(frame_to_plot,obspath))

    """plot approx dark subtracted data"""
    fig = plt.figure(figsize=(figx-4,figy))
    ax1 = plt.subplot2grid((3,8),(0,0), colspan=7)
    p1 = plt.imshow(np.log(np.abs(light_sub[frame_to_plot,:,:])), aspect=2)
    ax1.set_title(title+" "+obspath+" Mean Dark Subtracted")
    ax2 = plt.subplot2grid((3,8),(1,0), colspan=7, rowspan=2)
    for frame_index in row_frame_to_plot:
        for row_to_plot in rows_to_plot:
            plt.plot(light_sub[frame_index,row_to_plot,:], label="Frame %i Row %i" %(frame_index,row_to_plot))
    plt.yscale("log")
    plt.legend()
    
    plt.xlim((0,1048))
    ax3 = plt.subplot2grid((3,8),(0,7), rowspan=3)
    fig.colorbar(p1,cax=ax3)
    plt.tight_layout()
    if save_figs:
        plt.savefig("UVIS_frame_%i_%s_mean_dark_subtracted.png" %(frame_to_plot,obspath))


    """plot binned data"""
    plt.figure(figsize=(figx-4,figy))
    for frame_index in range(len(light_sub_binned[:,0])):
        plt.plot(x_nm,light_sub_binned[frame_index,:],label="Frame %i" %frame_index)
    plt.yscale("log")
    plt.xlim([200,650])
    plt.legend()
    plt.title(title+" "+obspath+" Mean Dark Subtracted Vertically Binned")
    if save_figs:
        plt.savefig("UVIS_%s_mean_dark_subtracted_vertically_binned.png" %(obspath))


if option==2:
    
    """Making diffraction grating model to simulate bouncing straylight"""
    cmap = plt.cm.get_cmap('gist_rainbow_r')
    
    def diff_ang(wavelengths,order): #wavelengths in nm
        if type(order) != int:
            stop
        #716.418 lines/mm = 716.418 e-6 lines/nm
        d = 1.0/(716.418E-6)
        #d sin theta = n lambda
        angles = np.arcsin(np.float(order) * np.asfarray(wavelengths) / d)
        return angles * 180.0 / np.pi

    pix2wav=[-3.62E-12,-1.64E-09,-2.19E-05,0.47346,194.3127]
    pixels = range(1048)
    wavelengths = np.polyval(pix2wav,pixels)
    wav2pix=np.polyfit(wavelengths,pixels,2)

#    chosen_wavelength = 370.0
    chosen_wavelength = 740.0
    chosen_pixel = np.polyval(wav2pix, chosen_wavelength)


    max_pixel = max(pixels)
    min_pixel = min(pixels)
    max_wavelength = max(wavelengths)
    min_wavelength = min(wavelengths)

    orders=[-2,-1,0,1,2]
#    orders=[-1,0,1]
    
#    fig1 = plt.figure(1, figsize=(figx,figy))
    fig2 = plt.figure(2, figsize=(figx,figy))
    plt.ylim([-10,10])
    plt.title(title)
    plt.plot([8,8],[-10,10],"k:")
    plt.plot([1032,1032],[-10,10],"k:")
    plt.ylabel("Diffraction Order")
    plt.xlabel("Pixel Number (8 to 1032 indicated)")
    plt.yticks(orders)
    fig3 = plt.figure(3, figsize=(figx,figy))
    plt.title(title)
    plt.plot([8,8],[-90,90],"k:")
    plt.plot([1032,1032],[-90,90],"k:")
    plt.ylabel("Angle scattered off diffraction grating")
    plt.xlabel("Pixel Number (8 to 1032 indicated)")

    #calculate angle to pixel coeffs using 1st order
    angles_order1 = diff_ang(wavelengths,1)
    ang2pix = np.polyfit(angles_order1,pixels,3)

    max_angle = max(angles_order1)
    min_angle = min(angles_order1)

    for order in orders:
        angles_order = diff_ang(wavelengths,order)
        chosen_angle = diff_ang(chosen_wavelength,order)
        pixels_order = np.polyval(ang2pix,angles_order)
        chosen_pixel = np.polyval(ang2pix,chosen_angle)
        
        if order==2:
            incident_angles_order = np.copy(angles_order)
            incident_pixels_order = np.copy(pixels_order)
            incident_angles_order[pixels_order>max_pixel] = np.nan
            incident_angles_order[pixels_order<min_pixel] = np.nan
            incident_pixels_order[pixels_order>max_pixel] = np.nan
            incident_pixels_order[pixels_order<min_pixel] = np.nan
            
#        plt.figure(1)
#        plt.plot(wavelengths, angles_order)
#        plt.scatter(chosen_wavelength,chosen_angle,c="b")
        
            
        plt.figure(2)
       
#        if order==2:
#            plt.scatter(incident_pixels_order, np.zeros_like(incident_pixels_order)+offset, c=wavelengths, linewidths=0, vmin=min_wavelength, vmax=max_wavelength)
#        else:
        ax2 = plt.scatter(pixels_order, np.zeros_like(pixels_order)+order, c=wavelengths, linewidths=0, vmin=min_wavelength, vmax=max_wavelength, alpha=0.3, cmap=cmap)
        plt.scatter(chosen_pixel,order,c="b",marker="x",s=100,label="%inm" %chosen_wavelength if order==0 else "")

        plt.figure(3)
        ax3 = plt.scatter(wavelengths, angles_order, c=wavelengths, linewidths=0, vmin=min_wavelength, vmax=max_wavelength, alpha=0.3, cmap=cmap)
        plt.scatter(chosen_wavelength,chosen_angle,c="b",marker="x",s=100,label="%inm" %chosen_wavelength if order==0 else "")
       
        
    plt.figure(2)
    cbar = plt.colorbar(ax2)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.set_label("Wavelength (nm)", rotation=270)
    plt.tight_layout()
    plt.legend(scatterpoints = 1)
    if save_figs:
        plt.savefig("%s_diffraction_order_wavelengths_%inm.png" %(title.replace(" ","_"),chosen_wavelength))

    plt.figure(3)
    cbar = plt.colorbar(ax3)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.set_label("Wavelength (nm)", rotation=270)
    plt.tight_layout()
    plt.legend(scatterpoints = 1)
    if save_figs:
        plt.savefig("%s_angle_diffracted_wavelengths_%inm.png" %(title.replace(" ","_"),chosen_wavelength))


if option==3:

    """compare PFM and FS UVIS frames"""
    rows_to_plot = range(24,97,10)
    frames_to_plot=[2]
    row_frame_to_plot=range(0,8,2)
    
    xcalcoeffs=[-3.62E-12,-1.64E-09,-2.19E-05,0.47346,194.3127]
    x = range(1048)
    x_nm = np.polyval(xcalcoeffs,x)
    
    titles=[]
    for file_index,hdf5_file in enumerate(hdf5_files):
        title="%s lamp, %0.1fs, %s, %0.1fC" %(lamps[file_index],it[file_index]/1000.0,models[file_index],ccd_temps[file_index])
        titles.append(title)

    files_to_plot = [0,1,2,3,4,5,6,7]
    
    nfiles = len(hdf5_files)
    
    
    light_sub_binned = np.zeros((nfiles,10,1048))
    for file_index,hdf5_file in enumerate(hdf5_files):
    
        if models[file_index]=="PFM":
            nlines = 123
        elif models[file_index]=="FS":
            nlines = 111
        npackets = int(np.ceil(nlines/15.0))
    
        detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get data
        ccd_temperature = get_dataset_contents(hdf5_file,"TEMP_2_CCD", chosen_group="Housekeeping")[0] #get data        
        frame_name = get_dataset_contents(hdf5_file,"Name")
        hdf5_file.close()
        
       
        reverse_clock = detector_data_all[0,0:2,:]
        
        i=1;                bias1 = detector_data_all[i:i+npackets,:,:].reshape((npackets*15,1048))[0:nlines,:]
        i += npackets;      dark1 = detector_data_all[i:i+npackets,:,:].reshape((npackets*15,1048))[0:nlines,:]
        i += npackets*8;    bias2 = detector_data_all[i:i+npackets,:,:].reshape((npackets*15,1048))[0:nlines,:]
        i += npackets;      dark2 = detector_data_all[i:i+npackets,:,:].reshape((npackets*15,1048))[0:nlines,:]
        
        dark2_binned = np.sum(dark2, axis=0)
        mean_dark = np.mean(np.asfarray([dark1,dark2]),axis=0)
    
        light = np.zeros((10,nlines,1048))
        light_sub = np.zeros((10,nlines,1048))
        starting_indices = range((1+npackets*2),(1+npackets*12),npackets)
        for index,i in enumerate(starting_indices):
            frame = detector_data_all[i:i+npackets,:,:].reshape((npackets*15,1048))[0:nlines,:]
            light[index,:,:] = frame
            light_sub[index,:,:] = frame-mean_dark
            light_sub_binned[file_index,index,:] = np.nansum(frame-mean_dark, axis=0)
            
        linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"]
        if file_index in files_to_plot:
            
            for frame_to_plot in frames_to_plot:
                """plot raw data"""
                fig = plt.figure(figsize=(figx-4,figy))
                ax1 = plt.subplot2grid((3,8),(0,0), colspan=7)
                p1 = plt.imshow(light[frame_to_plot,:,:], aspect=2)
                ax1.set_title(titles[file_index] + " Raw data Frame %i T=%0.1f-%0.1f" %(frame_to_plot,np.min(ccd_temperature),np.max(ccd_temperature)))
                ax2 = plt.subplot2grid((3,8),(1,0), colspan=7, rowspan=2)
                for frame_index in row_frame_to_plot:
                    for row_index,row_to_plot in enumerate(rows_to_plot):
                        plt.plot(light[frame_index,row_to_plot,:], label="Frame %i Row %i" %(frame_index,row_to_plot), color=linecolours[row_index])
                plt.yscale("log")
                plt.legend()
    
                plt.xlim((0,1048))
                ax3 = plt.subplot2grid((3,8),(0,7), rowspan=3)
                fig.colorbar(p1,cax=ax3)
                plt.tight_layout()
                if save_figs:
                    plt.savefig("UVIS_frame_%i_%s_raw_data.png" %(frame_to_plot,obspaths[file_index]))
#
#            """plot approx dark subtracted data"""
#            fig = plt.figure(figsize=(figx-4,figy))
#            ax1 = plt.subplot2grid((3,8),(0,0), colspan=7)
#            p1 = plt.imshow(np.log(np.abs(light_sub[frame_to_plot,:,:])), aspect=2)
#            ax1.set_title(titles[file_index] + " Mean Dark Subtracted T=%0.1f-%0.1f" %(np.min(ccd_temperature),np.max(ccd_temperature)))
#            ax2 = plt.subplot2grid((3,8),(1,0), colspan=7, rowspan=2)
#            for frame_index in row_frame_to_plot:
#                for row_to_plot in rows_to_plot:
#                    plt.plot(light_sub[frame_index,row_to_plot,:], label="Frame %i Row %i" %(frame_index,row_to_plot))
#            plt.yscale("log")
#            plt.legend()
#            
#            plt.xlim((0,1048))
#            ax3 = plt.subplot2grid((3,8),(0,7), rowspan=3)
#            fig.colorbar(p1,cax=ax3)
#            plt.tight_layout()
#            if save_figs:
#                plt.savefig("UVIS_frame_%i_%s_mean_dark_subtracted.png" %(frame_to_plot,obspaths[file_index]))
#        
#        
#            """plot binned data"""
#            plt.figure(figsize=(figx-4,figy))
#            for frame_index in range(10):
#                plt.plot(x_nm,light_sub_binned[file_index,frame_index,:],label="Frame %i" %frame_index)
#            plt.yscale("log")
#            plt.xlim([200,650])
#            plt.legend()
#            ax1.set_title(titles[file_index] + " Mean Dark Subtracted Vertically Binned T=%0.1f-%0.1f" %(np.min(ccd_temperature),np.max(ccd_temperature)))
#            if save_figs:
#                plt.savefig("UVIS_%s_mean_dark_subtracted_vertically_binned.png" %(obspaths[file_index]))
    

    linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"]
    linecolours = ["r","b","r","b","r","b","r","b"]
   
    
    """plot spectra for each lamp separately"""
    for lamp in ["D2","RS12","QTH"]:
        plt.figure(figsize=(figx-4,figy))
        for file_index in range(nfiles):
            if lamps[file_index]==lamp:
                for frame_index in range(7):#range(10):
                    if file_index in range(2):
                        label="%s frame %i" %(titles[file_index],frame_index)
                    else:
                        label=None
                    plt.plot(light_sub_binned[file_index,frame_index,:], label=label, c=linecolours[file_index], ls=linestyles[file_index])
        plt.yscale("log")
#        plt.legend(loc="lower right")
        plt.ylim([2e4,2e6])
        plt.xlabel("Pixel Number")
        plt.ylabel("Signal ADUs")
        plt.title("Mean dark subtracted spectra for %s lamp" %lamp)
        plt.tight_layout()
        if save_figs:
            plt.savefig("UVIS_%s_mean_dark_subtracted_vertically_binned_%s_lamp.png" %(obspath,lamp))


    
    """compare binned spectra"""
    
#    files_to_compare = [[0,6],[1,7],[2,8],[3,9],[4,10],[5,11]]
#    files_to_compare = [[0,2],[0,4],[2,4],[1,3],[1,5],[3,5]]

#    for file_to_compare in files_to_compare:
#        plt.figure(figsize=(figx+4,figy-2))
#        for frame_index in range(10):
##            compared = (light_sub_binned[file_to_compare[0],frame_index,:]-light_sub_binned[file_to_compare[1],frame_index,:])/light_sub_binned[file_to_compare[0],frame_index,:]
#            compared = (light_sub_binned[file_to_compare[0],frame_index,:]/light_sub_binned[file_to_compare[1],frame_index,:])
#            plt.plot(compared, label="%s vs %s frame %i" %(titles[file_to_compare[0]],titles[file_to_compare[1]],frame_index), color=linecolours[frame_index])
#        plt.ylim([0.8,1.2])
#        plt.legend()
#        plt.xlabel("Pixel Number")
#        plt.ylabel("Division of %0.0fC and %0.0fC Spectra" %(ccd_temps[file_to_compare[0]],ccd_temps[file_to_compare[1]]))
#        plt.title("Comparison of mean dark subtracted spectra")
#        plt.tight_layout()
#        if save_figs:
#            plt.savefig("UVIS_comparison_mean_dark_subtracted_vertically_binned_%0.0fC_%0.0fC.png" %(ccd_temps[file_to_compare[0]],ccd_temps[file_to_compare[1]]))

        
#Results: compare three temperatures from PFM calibration. Dark subtraction is rudimentary, so 5% differences seen between +20C spectra and others
#Ignore +20C spectra, look at -5C and -15C only.
#Next do a direct comparison between PFM low temperatures and FS low temperatures
#Is straylight slanted and does it move with 2nd order filter edge?


if option==4:
    
    detector_data_bins = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    name_all = get_dataset_contents(hdf5_files[0],"Name")[0]
    time_data_all = get_dataset_contents(hdf5_files[0],"ObservationTime")[0]
    date_data_all = get_dataset_contents(hdf5_files[0],"ObservationDate")[0]
    hdf5_files[0].close()
    
    pixel_column = 800
    frames_to_plot = range(2,25,1)
    
    detector_data = detector_data_bins[:,0,pixel_column]

    plt.figure(figsize=(figx-4,figy))
    for frame_to_plot in frames_to_plot:
        """plot raw data"""
        plt.plot(detector_data_bins[frame_to_plot,0,:], label="Frame %i" %(frame_to_plot))#, color=linecolours[row_index])
    plt.yscale("log")

    plt.legend()
    plt.xlim((0,1048))
    plt.tight_layout()
    if save_figs:
        plt.savefig("UVIS_frame_%i_%s_raw_data.png" %(frame_to_plot,obspaths[file_index]))




    plt.figure(figsize = (10,8))
    plt.xlabel("Time")
    plt.ylabel("Signal sum")
    plt.plot(detector_data,"o", linewidth=0)
    plt.legend()
    plt.title(title+": Detector column %i versus time" %pixel_column)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+".png")










import os
import numpy as np
from hdf5_functions_v03 import makeFileList

DATA_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\test\iant\hdf5")
#DATA_DIRECTORY = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/")


"""list UVIS acquisiton modes"""
#obspaths = ["*20180601*UVIS*_E"]
fileLevel = "hdf5_level_0p2c"
#hdf5Files, hdf5Filenames, _ = makeFileList(["*201806*UVIS*_I"], fileLevel)
#for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames): 
#    print(hdf5Filename + ": "+str(hdf5File["Channel/AcquisitionMode"][0]))
    
hdf5Files, hdf5Filenames, _ = makeFileList(["*201806*UVIS*_E"], fileLevel)
for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames): 
    print(hdf5Filename + ": "+str(hdf5File["Channel/AcquisitionMode"][0]))
    
    
    



