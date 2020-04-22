# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 07:30:32 2016

@author: iant

"""

import os
import h5py
import numpy as np
import numpy.linalg as la
import gc

import bisect
from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap


from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v02b import getHdf5Attributes,get_dataset_contents,write_to_hdf5
#from spice_functions_v01 import convert_hdf5_time_to_spice_utc,find_boresight,find_rad_lon_lat,py_ang
#from analysis_functions_v01b import interpolate_bad_pixel,sg_filter,fft_filter,fft_filter2,findOrder,spectral_calibration_simple,write_log,get_filename_list,stop,spectral_calibration
from pipeline_config_v04 import DATA_ROOT_DIRECTORY,BASE_DIRECTORY,KERNEL_DIRECTORY,AUXILIARY_DIRECTORY,figx,figy
from pipeline_mappings_v04 import METAKERNEL_NAME

rcParams["axes.formatter.useoffset"] = False


DATA_ROOT_DIRECTORY=os.path.normcase(r"D:\Data\hdf5")
#DATA_ROOT_DIRECTORY=os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db")
"""load spice kernel at start"""
import spiceypy as sp
os.chdir(KERNEL_DIRECTORY)
sp.reset()
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(os.path.normcase(DATA_ROOT_DIRECTORY)) #change directory to NOMAD/Data




option=1
save_figs=False
#save_figs=True
save_files=False
#save_files=True

#checkout="Ground"
#checkout="NEC"
#checkout="MCC"
checkout="MCO1"
#checkout="MCO2"


multiple=False #ignore
file_level="hdf5_level_0p1c"

"""Enter location of data"""

"""Ground calibration"""
#title="LNO Miniscans Ground Cal"; option=46; file_level="hdf5_level_0p1c"
#title="LNO Miniscans Ground Cal"; option=47; file_level="hdf5_level_0p1c"


"""NEC data"""
#title="SO Raster 1"; option=6
#title="LNO Raster 1"
#title="SO Raster 2"
#title="LNO Raster 2"
#title="SO ACS Raster 1"
#title="LNO ACS Raster 1"
#title="UVIS ACS Raster 1"
#title="SO ACS Raster 2"
#title="SO Light Sky"
#title="LNO Light Sky"
#title="SO Miniscan 1" #order 142
#title="LNO Miniscan 1" 
#title="SO Miniscan 2" #order 162
#title="LNO Miniscan 2"
#title="SO Miniscan 3" #order 179
#title="LNO Miniscan 3"
#
#title="SO Fullscan 1 Backup" #not required; slit didn't move
#title="LNO Fullscan 1 Backup" #not required; slit didn't move
#title="SO Fullscan 1"
#title="LNO Fullscan 1"
#title="UVIS Sun Pointing"
#
#title="SO Miniscan 1 Backup" #not required; slit didn't move
#title="SO Miniscan 2 Backup" #not required; slit didn't move

#title="SO Straylight"
#title="LNO Straylight"

"""MCC data"""
#title="SO Raster 1A"; option=6 #team same
#title="LNO Raster 1A"; option=6
#title="SO Raster 1B"; option=6 #team same rotated 90 deg
#title="LNO Raster 1B"; option=6
#title="SO-UVIS Raster 2A"; option=6 #team same uvis
#title="SO-UVIS Raster 3A"; option=6 #team opposite uvis
#title="SO Raster 4A"; option=6 #team opposite
#title="LNO Raster 4A"; option=6
#title="SO Raster 4B"; option=6 #team opposite rotated 90 deg
#title="LNO Raster 4B"; option=6
#title="LNO Dark Sky"

#title="SO Straylight"
#title="LNO Straylight"
#title="SO Fullscan"
#title="LNO Fullscan"
#title="UVIS Sun Pointing"

"""MCO-1 data"""
#title="SO Raster A"; option=35; file_level="hdf5_level_0p1c"
#title="LNO Raster A"; option=4
#title="SO Raster B"; option=4
#title="LNO Raster B"; option=4

#title="LNO Limb Scan 1"; option=29
#title="LNO Limb Scan 2"; option=29
#title="LNO Nadir Dayside"; option=27; file_level="hdf5_level_0p1c"
#title="LNO Nadir Dayside"; option=32
#title="LNO Nadir Dayside"; option=34; file_level="hdf5_level_0p1c"
#title="LNO Nadir Dayside"; option=48; file_level="hdf5_level_0p1c"
#title="UVIS Nadir Dayside"; option=28

#title="LNO Inertial Dayside"; option=27
#title="LNO Phobos"; option=30
#title="UVIS Phobos"; option=31
#title="SO Saturation Time"; option=33; file_level="hdf5_level_0p1c"
#title="LNO Saturation Time"; option=33; file_level="hdf5_level_0p1c"
title="SO ACS Solar Pointing Test"; option=35

#title="LNO Solar Miniscans"; option=39; file_level="hdf5_level_0p1c"
#title="LNO Solar Miniscans"; option=46; file_level="hdf5_level_0p1c"

#title="LNO Solar Miniscans Testing"; option=45; file_level="hdf5_level_0p1c"
#title="LNO Solar Miniscans Testing"; option=47; file_level="hdf5_level_0p1c"

#title="SO LNO Boresight Checks"; option=49; file_level="hdf5_level_1p0a" #check SO/LNO miniscan for boresight position on detector



"""MCO-2 data"""
#title="LNO Limb Scan Spectra for ESA Press Release"; option=36
#title="LNO Nadir Dayside 1"; option=36; file_level="hdf5_level_0p1c"
#title="LNO Nadir Dayside 2"; option=36
#title="LNO Nadir Dayside 4"; option=36
#title="LNO Limb Scan 1"; option=37; file_level="hdf5_level_0p1c"
#title="LNO Limb Scan 2"; option=37; file_level="hdf5_level_0p1c"
#title="LNO Diffraction Order"; option=38; file_level="hdf5_level_0p3a"
#title="UVIS Full Frame"; option=40; file_level="hdf5_level_0p2a"
#title="SO Light to Dark"; option=41; file_level="hdf5_level_0p3a" #not yet done
#title="SO Light to Dark"; option=35; file_level="hdf5_level_0p3a" #check signal levels
#title="LNO MIR Boresight Check"; option=42; file_level="hdf5_level_0p3a"


#"""All"""
#title
#option=43; file_level="hdf5_level_0p1c"  #check temperature perturbations
#option=44; file_level="hdf5_level_0p3a" #plot groundtracks





channel={"SO ":"so", "SO-":"so", "LNO":"lno", "UVI":"uvis"}[title[0:3]]
detector_centre={"so":128, "lno":152, "uvis":0}[channel] #or 152 for lno??
nec_sun_detector_centre={"so":130, "lno":157, "uvis":0}[channel] #for static measurements during NEC using the ground calibration

if checkout=="Ground" or checkout=="MCO1" or checkout=="MCO2":
    if checkout=="Ground":
        obspaths={
#        "LNO Miniscans Ground Cal":["20150427_081547_LNO","20150427_092635_LNO","20150427_095826_LNO","20150427_105912_LNO","20150427_112853_LNO"], \
        "LNO Miniscans Ground Cal":["20150320_081347_LNO","20150320_084352_LNO","20150320_091400_LNO","20150320_110056_LNO"], \
        }[title]
        
        
        
    if checkout=="MCO1":
    
        obspaths={"SO Raster A":"20161120_231420_SO", \
        "SO Raster B":["20161121_012420_SO_1","20161121_012420_SO_2"], \
        "LNO Raster A":"20161121_000420_LNO.", "LNO Raster B":"20161121_021920_LNO", \
        "LNO Nadir Dayside":"20161122_153906_LNO","UVIS Nadir Dayside":"20161122_153906_UVIS","LNO Limb Scan 1":"20161122_134403_LNO", \
        "LNO Inertial Dayside":"20161126_220850_LNO", "LNO Phobos":"20161126_220850_LNO", "UVIS Phobos":"20161126_220850_UVIS", \
        "LNO Limb Scan 2":"20161126_200627_LNO",
        "SO Saturation Time":["20161121_030950_SO","20161121_224950_SO"], \
        "LNO Saturation Time":["20161121_183450_LNO","20161121_233000_LNO"], \
#        "SO ACS Solar Pointing Test":["20161127_211950_SO_1","20161127_211950_SO_2"], \
        "SO ACS Solar Pointing Test":"20161127_211950_SO", \
#        "LNO Solar Miniscans":["20161122_033050_LNO","20161122_225550_LNO","20161122_233550_LNO","20161123_005550_LNO",\
#        "20161123_013550_LNO","20161123_025550_LNO","20161123_033550_LNO","20161123_152550_LNO","20161123_160550_LNO","20161123_172550_LNO",\
#        "20161123_192550_LNO","20161123_200550_LNO","20161123_225550_LNO","20161123_233550_LNO","20161124_005550_LNO",\
#        "20161124_013550_LNO","20161124_025550_LNO","20161124_033550_LNO","20161125_155550_LNO",\
#        "20161125_183550_LNO","20161125_195550_LNO","20161125_203550_LNO","20161127_152550_LNO","20161127_160550_LNO","20161127_172550_LNO",\
#        "20161127_180550_LNO","20161127_192550_LNO","20161127_200550_LNO"], \
        "LNO Solar Miniscans":["20161123_025550_LNO","20161123_192550_LNO","20161122_033050_LNO","20161124_013550_LNO",\
        "20161125_183550_LNO","20161125_195550_LNO","20161125_203550_LNO","20161123_033550_LNO","20161124_025550_LNO",\
        "20161127_152550_LNO","20161127_160550_LNO","20161123_200550_LNO","20161122_225550_LNO","20161124_033550_LNO",\
        "20161123_152550_LNO","20161123_225550_LNO","20161122_233550_LNO","20161123_160550_LNO","20161123_233550_LNO",\
        "20161125_155550_LNO","20161123_005550_LNO","20161123_172550_LNO","20161127_172550_LNO","20161127_180550_LNO",\
        "20161124_005550_LNO","20161123_013550_LNO","20161127_192550_LNO","20161127_200550_LNO"], \
        "LNO Solar Miniscans Testing":["20161123_025550_LNO","20161123_192550_LNO","20161122_033050_LNO","20161124_013550_LNO",\
        "20161125_183550_LNO","20161125_195550_LNO","20161125_203550_LNO","20161123_033550_LNO","20161124_025550_LNO"],\
        "SO LNO Boresight Checks":["20161123_154550_SO_C","20161123_160550_LNO_C"],\
        }[title]

    elif checkout=="MCO2":
    
        obspaths={
        "LNO Limb Scan Spectra for ESA Press Release":"20170306_180300_LNO", \
        "LNO Nadir Dayside 1":["20170228_231658_LNO_1","20170228_231658_LNO_2"], \
        "LNO Nadir Dayside 2":["20170301_231532_LNO_1","20170301_231532_LNO_2"], \
        "LNO Nadir Dayside 4":["20170301_231532_LNO_1","20170301_231532_LNO_2"], \
        "LNO Limb Scan 1":"20170305_180300_LNO", \
        "LNO Limb Scan 2":"20170306_180300_LNO", \
        "LNO Diffraction Order":"20170306_231520_LNO_F", \
        "UVIS Full Frame":"20170305_231920_UVIS_D", \
        "SO Light to Dark":["20170306_064450_SO_I_0","20170306_064450_SO_I_121","20170306_064450_SO_I_134","20170306_064450_SO_I_142","20170306_064450_SO_I_170","20170306_064450_SO_I_190"], \
        "LNO MIR Boresight Check":["20170307_064450_LNO_I_0","20170307_064450_LNO_I_121","20170307_064450_LNO_I_134","20170307_064450_LNO_I_164","20170307_064450_LNO_I_170","20170307_064450_LNO_I_190"], \
        }[title]

        
    if type(obspaths) != list:
        
        """get data"""
        year = obspaths[0:4]
        month = obspaths[4:6]
        day = obspaths[6:8]
        filename=os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+obspaths+".h5") #choose a file
#        filename=os.path.normcase(DATA_DIRECTORY+os.sep+"hdf5_level_0p2a"+os.sep+year+os.sep+month+os.sep+day+os.sep+obspaths+".h5") #choose a file
        hdf5_file = h5py.File(filename, "r") #open file
        
        print("File %s has the following attributes:" %(filename)) #print(attributes from file)
        attributes,values=getHdf5Attributes(hdf5_file)
        for index in range(len(attributes)):
            print("%s: %s" %(list(attributes)[index],list(values)[index]))

    else:
        multiple=True
        hdf5_files=[]
        #loop through filenames, opening each file for reading
        for obspath in obspaths:
            year = obspath[0:4] #get the date from the filename to find the file
            month = obspath[4:6]
            day = obspath[6:8]
#            filename=os.path.normcase(DATA_DIRECTORY+os.sep+"hdf5_level_0p1c"+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
            filename=os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
            hdf5_files.append(h5py.File(filename, "r")) #open file, add to list
    
            print("File %s has the following attributes:" %(filename)) #print(attributes from file
            attributes,values=getHdf5Attributes(hdf5_files[-1])
            for index in range(len(attributes)):
                print("%s: %s" %(list(attributes)[index],list(values)[index]))

    
elif checkout=="NEC" or checkout=="MCC":

    if checkout=="NEC":
        DATA_ROOT_DIRECTORY=BASE_DIRECTORY+os.sep+"Ops"+os.sep+"nec_calibration"+os.sep+"hdf5_data" #directory containing hdf5 files

        obsfolder={"SO Raster 1":"obs1", "LNO Raster 1":"obs2", "SO Raster 2":"obs3", "LNO Raster 2":"obs4", \
                "SO ACS Raster 1":"acs_obs1", "LNO ACS Raster 1":"acs_obs4", "UVIS ACS Raster 1":"acs_obs1", "SO ACS Raster 2":"acs_obs5" ,\
                "SO Light Sky":"obs26", "LNO Light Sky":"obs27", \
                "SO Miniscan 1":"obs14", "LNO Miniscan 1":"obs15", \
                "SO Miniscan 2":"obs16", "LNO Miniscan 2":"obs17", \
                "SO Miniscan 3":"obs18", "LNO Miniscan 3":"obs19", "SO Miniscan 1 Backup":"obs24", "SO Miniscan 2 Backup":"obs25", \
                "SO Fullscan 1 Backup":"obs20", "LNO Fullscan 1 Backup":"obs21", "SO Fullscan 1":"obs22", "LNO Fullscan 1":"obs23", "UVIS Sun Pointing":"obs23", \
                "SO Straylight":"obs28", "LNO Straylight":"obs29"}[title]
        """get data"""
        os.chdir(os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+obsfolder)) #change directory
        filenames=get_filename_list("h5") #print(list of hdf5 files in given folder and all subdirectories
        print("Total files = %i" %len(filenames))
        print("HDF5 Files:")
        print(filenames)
        if channel=="so" or channel=="lno":
            filename=filenames[0] #choose a file
        elif channel=="uvis":
            filename=filenames[1] #choose a file

    elif checkout=="MCC":
#        DATA_ROOT_DIRECTORY=BASE_DIRECTORY+os.sep+"Ops"+os.sep+"mcc"+os.sep+"hdf5_data"+os.sep+"2016"+os.sep+"06" #directory containing hdf5 files
        file_level="hdf5_level_0p1c"
        DATA_ROOT_DIRECTORY=DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+"2016"+os.sep+"06" #directory containing hdf5 files

        obsfolder,number={"SO Raster 1A":["12",0], "UVIS-SO Raster 1A":["12",1], "LNO Raster 1A":["13",0], "UVIS-LNO Raster 1A":["13",1], \
                          "SO Raster 1B":["13",2], "UVIS-SO Raster 1B":["13",3], "LNO Raster 1B":["13",4], "UVIS-LNO Raster 1B":["13",5], \
                          "SO-UVIS Raster 2A":["13",10], "UVIS Raster 2A":["13",11], \
                          "SO-UVIS Raster 3A":["13",12], "UVIS Raster 3A":["13",13], \
                          "SO Fullscan":["15",12], "LNO Fullscan":["15",13], "UVIS Sun Pointing":["15",15], \
                          "SO Straylight":["15",18], "LNO Straylight":["15",20], \
                          "SO Raster 4A":["15",21], "UVIS-SO Raster 4A":["15",22], "LNO Raster 4A":["15",23], "UVIS-LNO Raster 4A":["15",24], \
                          "SO Raster 4B":["16",0], "UVIS-SO Raster 4B":["16",1], "LNO Raster 4B":["16",2], "UVIS-LNO Raster 4B":["16",3], \
                          "LNO Dark Sky":["13",8]}[title]
        """get data"""
        os.chdir(os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+obsfolder)) #change directory
        filenames=get_filename_list("h5") #print(list of hdf5 files in given folder and all subdirectories
        print("Total files = %i" %len(filenames))
        print("HDF5 Files:")
        print(filenames)
        filename=filenames[number] #choose a file
    
    
   
        
    hdf5_file = h5py.File(filename, "r") #open file
    
    print("File %s has the following attributes:" %(filename)) #print(attributes from file
    attributes,values=getHdf5Attributes(hdf5_file)
    for index in range(len(attributes)):
        print("%s: %s" %(attributes[index],values[index]))






if option==1:
    """test functions"""
    """choose an example where dataset name is unique"""
    #dataset_to_search_for1="UVIS_TEMP"
    
    #print("Dataset %s has the following values in file %s:" %(dataset_to_search_for1,filename) #list dataset details
    #data,units,types=get_dataset_contents(hdf5_file,dataset_to_search_for1)
    #print(units
    #print(types
    
    """choose an example where name of group is required"""
    #dataset_to_search_for2="TEMP_1_PROXIMITY_BOARD"
    #data2,_,_=get_dataset_contents(hdf5_file,dataset_to_search_for2,chosen_group="Housekeeping")
    #
    #plt.figure()
    ##plot data
    #plt.plot(data, label=dataset_to_search_for1)
    #plt.plot(data2, label=dataset_to_search_for2)
    #plt.legend()
    #plt.ylabel("TEMPERATURE (%s)" %units)
    
    #hdf5_file.close()

    """test spice functions"""
    import spiceypy as sp
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)

    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
    
    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    print(epoch_times_all)


if option==2:
    """plot intensity vs time"""
    if channel=="so" or channel=="lno":
        detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
        exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
        first_window_top = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0][0]
        hdf5_file.close()
        
        if binning==2: #stretch array
            detector_data=np.repeat(detector_data,2,axis=1)
           
        if checkout=="NEC" or checkout=="MCC":
            nlines=16
            nsteps=16
        elif channel=="lno":
            nlines=24
            nsteps=7
        else:
            print("Error: don't use for MCO SO")

            
            
        sum_centre_all=[]
        exponent_all=[]
        full_frame_all=[]
        vert_slice_all=[]
        for index2 in range(int(detector_data.shape[0]/(nsteps/binning))): #loop through frames
            full_frame = np.zeros((nlines*nsteps,320))
            for index in range(int(nsteps/binning)): #loop through window subframes
                full_frame[(index*nlines*binning):((index+1)*nlines*binning),:]=detector_data[(index+(index2*nsteps/binning)),:,:]
                exponent_all.append(exponent)
    #        sum_centre_all.append(np.sum(full_frame[(detector_centre-4):(detector_centre+4),220:236]))
            sum_centre_all.append(np.sum(full_frame[detector_centre-first_window_top,228]))
            full_frame_all.append(full_frame)
        
        #plt.figure(figsize=(10,8))
        #plt.imshow(full_frame_all[64])
        #plt.colorbar()
        
        #plt.figure(figsize=(10,8))
        #plt.imshow(full_frame_all[64])
        #plt.colorbar()
        
        time=np.arange(int(detector_data.shape[0]/(nsteps/binning)))*(nsteps/binning)
        
        plt.figure(figsize=(10,8))
        plt.plot(time,sum_centre_all)
#        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
        plt.ylabel("Sum signal ADU for pixel %i,228" %(detector_centre))
        plt.xlabel("Approx time after pre-cooling ends (seconds)")
        plt.title(title)
        plt.yscale("log")
        if save_figs: plt.savefig(title+"_intensity_versus_time_raster_scan_log.png")
        
    #    np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
    
        plt.figure(figsize=(10,8))
        plt.plot(time,sum_centre_all)
#        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
        plt.ylabel("Sum signal ADU for pixel %i,228" %(detector_centre))
        plt.xlabel("Approx time after pre-cooling ends (seconds)")
        plt.title(title)
        if save_figs: plt.savefig(title+"_intensity_versus_time_raster_scan.png")
        
        
        plt.figure(figsize=(10,8))
        plt.plot(2**exponent)
        plt.ylabel("Exponent ADU")
        plt.xlabel("Approx time after pre-cooling ends (seconds)")
        plt.title(title)
        if save_figs: plt.savefig(title+"_exponent_versus_time_raster_scan.png")

if option==3:
    """v1 plot animations of chosen raster scan"""
    """this should be changed so a function generates the new frames"""
    detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
    time_data = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
    hdf5_file.close()
    
    if binning==2: #stretch array
        detector_data=np.repeat(detector_data,2,axis=1)
    
    fig=plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    full_frame_all=[]
    for index2 in range(int(len(time_data)/(16/binning))):
        full_frame = np.zeros((256,320))
        for index in range(int(16/binning)):
            full_frame[(index*16*binning):((index+1)*16*binning),:]=detector_data[(index+(index2*16/binning)),:,:]
        frame = ax.imshow(full_frame, vmin=0, vmax=1e4, animated=True)
        t = ax.annotate(time_data[(index+(index2*16/binning))],(50,50),size=50)#time_data[(index+(index2*16))])
        full_frame_all.append([frame,t])
    
    ani = animation.ArtistAnimation(fig, full_frame_all, interval=50, blit=True)
    if save_figs: ani.save(title+" Detector_Frame.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
    plt.show()

    print("Done")
    gc.collect() #clear animation from memory
    
if option==4:
    """plot intensity vs position in raster"""
    import spiceypy as sp
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)


#    plot_both=False
    plot_both=True #flag to store values from orientation A so that results of both A and B can be plotted together.
    
    time_error=1
    if checkout=="NEC":
        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
    elif checkout=="MCC":
        if title=="SO Raster 1A" or title=="SO Raster 1B":
            boresight_to_tgo=(-0.92083,-0.38997,0.00042) #define so boresight in tgo reference frame
        elif title=="SO Raster 4A" or title=="SO Raster 4B": 
            boresight_to_tgo=(-0.92191,-0.38736,0.00608) #define so boresight in tgo reference frame
            if title=="SO Raster 4A":
                time_raster_centre=sp.utc2et("2016JUN15-23:15:00.000") #time of s/c pointing to centre
            elif title=="SO Raster 4B":
                time_raster_centre=sp.utc2et("2016JUN16-01:25:00.000") #time of s/c pointing to centre
            centre_theoretical=find_boresight([time_raster_centre],time_error,boresight_to_tgo)
            centre_theoretical_lat_lon=sp.reclat(centre_theoretical[0][0:3])[1:3]
        elif title=="LNO Raster 1A" or title=="LNO Raster 1B":
            boresight_to_tgo=(-0.92134,-0.38875,0.00076)
        elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
            boresight_to_tgo=(-0.92163,-0.38800,0.00653)
        elif title=="SO-UVIS Raster 2A":
            boresight_to_tgo=(-0.92107,-0.38941,0.00093) #define so boresight in tgo reference frame
        elif title=="SO-UVIS Raster 3A": #team opposite uvis
            boresight_to_tgo=(-0.92207,-0.38696,0.00643) #define so boresight in tgo reference frame
    elif checkout=="MCO":
        if title=="SO Raster A" or title=="SO Raster B":
            boresight_to_tgo=(-0.92156, -0.38819, 0.00618) #define so boresight in tgo reference frame
        elif title=="LNO Raster 1" or title=="LNO Raster 2":
            boresight_to_tgo=(-0.92148, -0.38838, 0.00628) #define so boresight in tgo reference frame

    orientation=title[-1].lower() #find orientation from last letter of title

    #get data
    if multiple:
        hdf5_file = hdf5_files[0]
    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
    window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
    hdf5_file.close()

    if binning==2: #stretch array
        detector_data_all=np.repeat(detector_data_all,2,axis=1)

    #convert data to times and boresights using spice
    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    time_error=1    
    boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
    
#    #sum all detector data
#    detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
#    detector_sum[detector_sum<500000]=500000#np.mean(detector_sum)
    
    #find indices where centre of detector is
    meas_indices=[]
    detector_sum=[]
    for index,window_top in enumerate(window_top_all):
        if detector_centre in range(window_top,window_top+16*binning):
            detector_line=detector_centre-window_top
            meas_indices.append(index)
            pixel_value=detector_data_all[index,detector_line,228]
            if pixel_value<100:
                pixel_value=100
            detector_sum.append(pixel_value)
    detector_sum=np.asfarray(detector_sum)
    chosen_boresights=boresights_all[meas_indices,:]
    lon_lats=np.asfarray([sp.reclat(chosen_boresight)[1:3] for chosen_boresight in list(chosen_boresights)])
    
    if plot_both and orientation=="a":
        detector_sum_a=detector_sum
        chosen_boresights_a=chosen_boresights
        lon_lats_a=lon_lats
        title_a=title
#        centre_theoretical_a=centre_theoretical
#        centre_theoretical_lat_lon_a=centre_theoretical_lat_lon
    if plot_both and orientation=="b":
        detector_sum_b=detector_sum
        chosen_boresights_b=chosen_boresights
        lon_lats_b=lon_lats
        title_b=title
#        centre_theoretical_b=centre_theoretical
#        centre_theoretical_lat_lon_b=centre_theoretical_lat_lon


    if not plot_both:
        marker_colour=np.log(1+detector_sum-min(detector_sum))
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(chosen_boresights[:,0], chosen_boresights[:,1], chosen_boresights[:,2], c=marker_colour, marker='o', linewidth=0)
        ax.azim=-108
        ax.elev=-10
        plt.title(title+": Signal on pixel %i,228" %detector_centre)
        if save_figs: plt.savefig(title+"_Signal_on_pixel_%i,228_in_J2000.png" %detector_centre)

        plt.figure(figsize=(9,9))
        plt.scatter(lon_lats[:,0], lon_lats[:,1], c=marker_colour, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
        plt.scatter(centre_theoretical_lat_lon[0], centre_theoretical_lat_lon[1], c='r', marker='*', linewidth=0, s=120)
        plt.xlabel("Solar System Longitude (degrees)")
        plt.ylabel("Solar System Latitude (degrees)")
        plt.title(title+": Signal on pixel %i,228" %detector_centre)
        if save_figs: plt.savefig(title+"_Signal_on_pixel_%i,228_in_lat_lons.png" %detector_centre)

    
    if plot_both and orientation=="b":
        marker_colour_a=np.log(1+detector_sum_a-min(detector_sum_a))
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(chosen_boresights_a[:,0], chosen_boresights_a[:,1], chosen_boresights_a[:,2], c=marker_colour_a, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
        marker_colour_b=np.log(1+detector_sum_b-min(detector_sum_b))
        ax.scatter(chosen_boresights_b[:,0], chosen_boresights_b[:,1], chosen_boresights_b[:,2], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
        ax.azim=-108
        ax.elev=-10
        plt.gca().patch.set_facecolor('white')
        ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#        plt.title(title_a+" & "+title_b+": Signal on pixel %i,228" %detector_centre)
        plt.title(channel.upper()+" Solar Line Scan: Signal Measured on Detector Centre")

        if save_figs: plt.savefig(title_a+"_"+title_b+"_Signal_on_pixel_%i,228_in_J2000.png" %detector_centre, dpi=600)
        
        
        
        
        plt.figure(figsize=(10,8))
        plt.scatter(lon_lats_a[:,0], lon_lats_a[:,1], c=marker_colour_a, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
        plt.scatter(lon_lats_b[:,0], lon_lats_b[:,1], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#        plt.scatter(centre_theoretical_lat_lon_a[0], centre_theoretical_lat_lon_a[1], c='r', marker='*', linewidth=0, s=120)
#        plt.scatter(centre_theoretical_lat_lon_b[0], centre_theoretical_lat_lon_b[1], c='r', marker='*', linewidth=0, s=120)
        plt.xlabel("Solar System Longitude (degrees)")
        plt.ylabel("Solar System Latitude (degrees)")
#        plt.title(title_a+" & "+title_b+": Signal on pixel %i,228" %detector_centre)
        plt.title(channel.upper()+" Solar Line Scan: Signal Measured on Detector Centre")
        cbar = plt.colorbar()
        cbar.set_label("Log(Signal on Detector)", rotation=270, labelpad=20)
        if save_figs: plt.savefig(title_a+"_"+title_b+"_Signal_on_pixel_%i,228_in_lat_lons.png" %detector_centre, dpi=600)
        
        

if option==5:
    """plot detector counts vs integration time"""
    spectral_line_index=228


    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
    int_time_all = get_dataset_contents(hdf5_file,"INTEGRATION_TIME")[0]
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    hdf5_file.close()
    
    plt.figure(figsize=(9,9))
    
    bins_to_plot=[11,12,13]
    for bin_to_plot in bins_to_plot:
        plt.scatter(int_time_all,detector_data_all[:,bin_to_plot,spectral_line_index],label="Line %i" %bin_to_plot, linewidth=0)
        
    plt.xlabel("Integration Time (ms)")
    plt.ylabel("Sum of detector counts pixel %i" %spectral_line_index)
    plt.legend()
    plt.title(title+r": integration time when detector saturates")
#    plt.xlim(xmax=10)
#    plt.savefig(title+r"- integration time when detector saturates.png")
    
    for index in range(0,256,51):
        plt.figure()
#        plt.imshow(detector_data_all[index,:,:])
#        plt.colorbar()
        if channel=="so":
            bins_to_plot=[11,12,13]
        elif channel=="lno":
            bins_to_plot=[10,11,12]
        for bin_to_plot in bins_to_plot:
            plt.plot(detector_data_all[index,bin_to_plot,:],label="row %i" %bin_to_plot)
        plt.legend()
        plt.title("Int time=%fms" %int_time_all[index]) 

if option==6:
    """make vertical detector plots where sun is seen to determine slit position and time when in centre"""
#    so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
#    lno_nadir_boresight_to_tgo=(-0.00685,-0.99993,0.00945) #define lno boresight in tgo reference frame

    if checkout=="NEC":
        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
        window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
    else:
        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0][:,0]
        date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0][:,0]
        window_top_all = get_dataset_contents(hdf5_file,"WindowTop")[0]
        binning = get_dataset_contents(hdf5_file,"Binning")[0][0]+1
    hdf5_file.close()
    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    time_error=1    
#    boresights_all=find_boresight(epoch_times_all,time_error,so_boresight_to_tgo)
#    detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
#    detector_sum=detector_data_all[:,0,230]
#    detector_sum[detector_sum<500000]=500000 #to plot better

    
    sun_indices1=0
    sun_indices2=0
    if title=="SO ACS Raster 1":
        sun_indices1=range(580,700) #plot all data
        sun_indices2=range(880,990) #plot all data
        window_tops=[96,112,128,144]
#        sun_indices1=range(640,680) #plot limited data
#        sun_indices2=range(920,960) #plot limited data
    elif title=="SO ACS Raster 2":
        sun_indices1=range(100,400)
        sun_indices2=range(100,200)
        window_tops=[96,112,128,144]
    elif title=="LNO ACS Raster 1":
#        sun_indices1=range(405) #plot all data
#        sun_indices2=range(405,1020) #plot all data
        sun_indices1=range(1020) #plot all data on same plot
        window_tops=[64,80,96,112,128,144,160,176,192,208,224] #plot all data
#        sun_indices1=range(475,530) #plot limited data
#        window_tops=[128,144,160]

    elif title=="SO Raster 1" or title=="SO Raster 2":
        sun_indices1=range(880,1150) #plot all data
#        sun_indices1=range(1080,1120) #plot limited data
        window_tops=[96,112,128,144]
    elif title=="LNO Raster 1" or title=="LNO Raster 2":
        sun_indices1=range(0,2100) #plot all data
        window_tops=[80,96,112,128,144,160,176,192,208] #plot all data
#        sun_indices1=range(1040,1110) #plot limited data
#        window_tops=[128,144,160]
    elif title=="SO Raster 4A" or title=="SO Raster 4B":
        sun_indices1=range(0,2100) #plot all data
        window_tops=[96,112,128,144] #plot all data
    elif title=="LNO Raster 1A":
        sun_indices1=range(0,2100) #plot all data
        window_tops=[32,64,96,128,160,192,224] #plot all data
    elif title=="LNO Raster 4A":
        sun_indices1=range(0,2100) #plot all data
        window_tops=[32,64,96,128,160,192,224] #plot all data


    if not sun_indices1==0:
        indices=[]
        for index,window_top in enumerate(window_top_all):
            if window_top in window_tops:
                if index in sun_indices1:
                    indices.append(index)
    
        times=time_data_all[indices]
        dates=date_data_all[indices]
        window_tops_selected=window_top_all[indices]
        epochs=epoch_times_all[indices]
        detector_counts=detector_data_all[indices,:,:]

        xs=[]    
        for window_top in window_tops_selected:
            xs.append(np.arange(16)*binning+window_top)
        xs=np.asarray(xs)
    
        detector_v_centre=230
        vert_slices=detector_counts[:,:,detector_v_centre]
    
        
#        plt.figure(figsize=(10,8))
#        plt.plot(np.transpose(xs),np.transpose(vert_slices))
#        plt.ylabel("Sum signal ADU for pixels in detector column %i" %detector_v_centre)
#        plt.xlabel("Vertical Pixel Number")
#        if channel=="lno":
#            plt.xlim((60,240))
#        elif channel=="so":
#            plt.xlim((100,150))
##        plt.legend(times)
#        plt.title(title+" pass 1: vertical columns on detector where sun is seen")
#        if save_figs: 
#            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_vertical_columns_on_detector_where_sun_is_seen.png")

    if not sun_indices2==0:
        indices=[]
        for index,window_top in enumerate(window_top_all):
            if window_top in window_tops:
                if index in sun_indices2:
                    indices.append(index)
        
        times=time_data_all[indices]
        dates=date_data_all[indices]
        window_tops_selected=window_top_all[indices]
        epochs=epoch_times_all[indices]
        detector_counts=detector_data_all[indices,:,:]
    
        xs=[]    
        for window_top in window_tops_selected:
            xs.append(np.arange(16)*binning+window_top)
        xs=np.asarray(xs)
    
        detector_v_centre=230
        vert_slices=detector_counts[:,:,detector_v_centre]
        
        plt.figure(figsize=(10,8))
        plt.plot(np.transpose(xs),np.transpose(vert_slices))
        plt.ylabel("Sum signal ADU for pixels in detector column %i" %detector_v_centre)
        plt.xlabel("Vertical Pixel Number")
        plt.legend(times)
        plt.title(title+" pass 2: vertical columns on detector where sun is seen")
        if save_figs: 
            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_pass_2_vertical_columns_on_detector_where_sun_is_seen.png")


    """check detector smile"""
    if title=="LNO Raster 4A" or title=="LNO Raster 4B":
        indices = [index for index,window_top in enumerate(window_top_all) if window_top in window_tops]
        continuum_range = [203,209,217,223]
        signal_minimum = 300000
    if title=="SO Raster 4A" or title=="SO Raster 4B":
        indices = [index for index,window_top in enumerate(window_top_all) if window_top in window_tops]
        continuum_range = [210,215,223,228]
        signal_minimum = 200000

    detector_data_selected = detector_data_all[indices,:,:]
    window_top_selected = window_top_all[indices]
        
    plt.figure(figsize = (figx,figy))
    absorption_minima=[]
    detector_rows=[]
    for frame_index in range(len(detector_data_selected[:,0,0])):
        for bin_index in range(len(detector_data_selected[0,:,0])):
            if detector_data_selected[frame_index,bin_index,200]>signal_minimum:
                detector_data_normalised = (detector_data_selected[frame_index,bin_index,:]-np.min(detector_data_selected[frame_index,bin_index,:]))/(np.max(detector_data_selected[frame_index,bin_index,:])-np.min(detector_data_selected[frame_index,bin_index,:]))
                plt.plot(detector_data_normalised)
                
                pixels=np.arange(320)
                
                continuum_pixels = pixels[range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]    
                continuum_spectra = detector_data_selected[frame_index,bin_index,range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]
                
                #fit polynomial to continuum on either side of absorption band
                coefficients = np.polyfit(continuum_pixels,continuum_spectra,2)
                continuum = np.polyval(coefficients,pixels[range(continuum_range[0],continuum_range[3])])
                absorption = detector_data_selected[frame_index,bin_index,range(continuum_range[0],continuum_range[3])]/continuum
                absorption_pixel = np.arange(20)
#                plt.plot(absorption_pixel,absorption)
                
                abs_coefficients = np.polyfit(pixels[range(continuum_range[0],continuum_range[3])][6:12],absorption[6:12],2)
                detector_row = window_top_selected[frame_index]+bin_index*binning
                
                absorption_minima.append((-1*abs_coefficients[1]) / (2*abs_coefficients[0]))
                
                detector_rows.append(detector_row)
                
    plt.figure(figsize = (figx-10,figy-2))
    plt.scatter(detector_rows,absorption_minima,marker="o",linewidth=0,alpha=0.5)
    
    fit_coefficients = np.polyfit(detector_rows,absorption_minima,1)
    fit_line = np.polyval(fit_coefficients,detector_rows)
    
    plt.plot(detector_rows,fit_line,"k", label="Line of best fit, min=%0.1f, max=%0.1f" %(np.min(fit_line),np.max(fit_line)))
    plt.legend()
    plt.ylabel("Pixel column number at minimum of quadratic fit to absorption line")
    plt.xlabel("Detector row")
    plt.title(title+" Detector Smile: Quadratic fits to absorption line")
    plt.ylim((continuum_range[1],continuum_range[2]))
    plt.tight_layout()
    if save_figs: 
        plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_detector_smile.png")


#    plt.figure(figsize=(9,9))
#    plt.plot(boresights_all[:,0],boresights_all[:,1],'b.')

#    plot partially reconstructed frames or raw 16 lines with imshow
#    plt.figure(figsize=(10,8))
#    plt.imshow(full_frame_all[64])
#    plt.colorbar()
#    
#    plt.figure(figsize=(10,8))
#    plt.imshow(full_frame_all[64])
#    plt.colorbar()

if option==7:
    """convert peak sun time to boresight"""
    date1=["2016-04-13"]; time1=["03-25-05.505"]    
    date2=["2016-04-13"]; time2=["03-20-17.509","03-20-33.509"] #half way between these two
    
    so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical

    epoch_time_sun=convert_hdf5_time_to_spice_utc(time1,date1)[0]
    
    cmatrix_sun=sp.ckgp(epoch_time_sun,1)
    
    observer="-143" #tgo
    target="SUN"
    relative_sun_position=sp.spkpos(observer,target,epoch_time_sun)
    sun_distance = la.norm(relative_sun_position)
    sun_pointing_vector = relative_sun_position / sun_distance
    
    new_boresight = tuple(np.dot(cmatrix_sun,sun_pointing_vector))
    print("boresight="+"%.10f "*3 %new_boresight)
    print("angle_difference=%.10f arcmins" %(py_ang(so_boresight_to_tgo,new_boresight) * 180 * 60 / np.pi))
    
    
    
    
if option==8:
    """calculate time when theoretical boresight pointed to sun"""

    step=0.1 #16 minutes #plot limited
    nsteps=40 * 120 
#    step=0.5#40 minutes #plot whole range
    if title=="SO Raster 1":
        date1=["2016-04-11"]; time1=["19-58-24.998"]     #SO calculated
#        epoch_time_start=sp.utc2et("2016APR11-19:40:00.998") #for SO full range
        epoch_time_start=sp.utc2et("2016APR11-19:50:00.998") #for SO limited range
        step=0.1
        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
        nsteps=80 * 120 
    if title=="SO ACS Raster 1":
        date1=["2016-04-13"]; time1=["03-25-05.505"]     #SO calculated
#        epoch_time_start=sp.utc2et("2016APR13-03:05:05.505") #for SO whole time range
        epoch_time_start=sp.utc2et("2016APR13-03:20:05.505") #for SO limited range
        step=0.1
        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
    if title=="LNO Raster 1":
        date1=["2016-04-11"]; time1=["20-48-10.958"]     #LNO calculated
#        epoch_time_start=sp.utc2et("2016APR11-19:40:00.998") #for SO full range
        epoch_time_start=sp.utc2et("2016APR11-20:40:00.958") #for SO limited range
        step=0.1
        boresight_to_tgo=(-0.92126,-0.38890,0.00368) #theoretical
        nsteps=80 * 120 
    if title=="LNO ACS Raster 1":
        date1=["2016-04-13"]; time1=["04-32-55.944"]     #LNO calculated
#        epoch_time_start=sp.utc2et("2016APR13-03:05:05.505") #for SO whole time range
        epoch_time_start=sp.utc2et("2016APR13-04:27:54.944") #for SO limited range
        step=0.1
        boresight_to_tgo=(-0.92126,-0.38890,0.00368) #theoretical
    if title=="UVIS ACS Raster 1":
        date1=["2016-04-13"]; time1=["03-25-08.431"]     #UVIS calculated
        epoch_time_start=sp.utc2et("2016APR13-03:20:08.431") #for UVIS limited range
        step=0.1
        boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000) #theoretical


    epoch_time_peak=convert_hdf5_time_to_spice_utc(time1,date1)[0]
    utc_time_peak=sp.et2utc(epoch_time_peak, "C", 0)
    

    cmatrices=sp.ckgp(epoch_time_start,1,step,nsteps)
    times=sw.et2utcx(epoch_time_start,step,nsteps)
    boresights_all=[]
    for cmatrix in cmatrices:
        boresights_all.append(np.dot(np.transpose(cmatrix),boresight_to_tgo))
    boresights_all=np.asfarray(boresights_all)
    [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)
    
    observer="-143" #tgo
    target="SUN"
    relative_sun_position=sw.spkposx(observer,target,epoch_time_start,step,nsteps)
    sun_distance = la.norm(relative_sun_position[0,:])
    sun_pointing_vector = relative_sun_position / sun_distance
    
    angles=[]
    for boresight,sun in zip(boresights_all,sun_pointing_vector):
        angles.append(py_ang(boresight,sun) * 180 * 60 / np.pi)
    angles=np.asfarray(angles)

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0)
#    ax.scatter(sun_pointing_vector[:,0],sun_pointing_vector[:,1],sun_pointing_vector[:,2], marker='o', linewidth=0, c='y')
    
    index_when_peak_signal=times.index(utc_time_peak) #find index of peak signal
    index_when_centred_on_sun=np.abs(angles-0).argmin() #find index when raster scan is in centre
    print("time of peak signal= %s" %times[index_when_peak_signal])
    print("time of theoretical sun centre= %s" %times[index_when_centred_on_sun])
    ax.scatter(sun_pointing_vector[index_when_peak_signal,0],sun_pointing_vector[index_when_peak_signal,1],sun_pointing_vector[index_when_peak_signal,2], marker='o', linewidth=0, c='r')
    ax.scatter(sun_pointing_vector[index_when_centred_on_sun,0],sun_pointing_vector[index_when_centred_on_sun,1],sun_pointing_vector[index_when_centred_on_sun,2], marker='o', linewidth=0, c='g')
    
    angular_offset=py_ang(boresights_all[index_when_peak_signal,:],boresights_all[index_when_centred_on_sun,:]) * 180 * 60 / np.pi
    print("horizontal offset between theoretical and true boresight=%f" %angular_offset)
    plt.figure()
    plt.plot(angles)

    plt.figure(figsize=(9,9))
    plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c='b')
    plt.scatter(boresight_lons[0:100],boresight_lats[0:100], marker='.', linewidth=0, c='y')
    plt.scatter(boresight_lons[index_when_peak_signal],boresight_lats[index_when_peak_signal],marker='o', linewidth=0, c='r')
    plt.scatter(boresight_lons[index_when_centred_on_sun],boresight_lats[index_when_centred_on_sun],marker='o', linewidth=0, c='g')
    print("%f %f" %(boresight_lons[index_when_centred_on_sun],boresight_lats[index_when_centred_on_sun]))
    
if option==9:
    """find peak for UVIS vertically binned data"""
    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:]
    hdf5_file.close()
    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    time_error=1    
#    boresights_all=find_boresight(epoch_times_all,time_error,so_boresight_to_tgo)
    detector_sum=np.sum(detector_data_all[:,0,:], axis=(1))
    detector_sum[0:2100]=0 #otherwise peak is found in first pass
    index_when_peak_signal=np.abs(detector_sum-0).argmax()
    print("max signal at %s %s" %(time_data_all[index_when_peak_signal],date_data_all[index_when_peak_signal]))
    
    
if option==10:
    """plot proposed boresights on ACS raster scan with detector sum"""
    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
    if channel=="uvis":
        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:]
        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:]
        detector_sum=np.sum(detector_data_all[:,0,:], axis=(1))
        detector_sum[0]=detector_sum[1] #fudge because 1st frame is bias
    elif channel=="so" or channel=="lno":
        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
        #sum all detector data
        detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
        detector_sum[detector_sum<500000]=500000#np.mean(detector_sum)
    hdf5_file.close()
    detector_data_all=[]
    
    print("Calculating times")
    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    marker_colour=np.log(1+detector_sum-min(detector_sum))  
    
    old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
#    lno_nadir_boresight_to_tgo=(-0.00685,-0.99993,0.00945) #define lno boresight in tgo reference frame
    old_lno_boresight_to_tgo=(-0.92126,-0.38890,0.00368)
    old_uvis_boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000)
    
    
    new_uvis_boresight_same_to_tgo=(-0.921039704,-0.389467349,0.001023578) #calculated
    new_uvis_boresight_opposite_to_tgo=(-0.922066875,-0.386977726,0.006396717) #calculated
    new_so_boresight_same_to_tgo=(-0.920827113,-0.389970833,0.000420369) #calculated
    new_so_boresight_opposite_to_tgo=(-0.921909199,-0.387358312,0.006079966) #calculated
    new_lno_boresight_same_to_tgo=(-0.921341439,-0.38875361,0.000764119) #calculated
    new_lno_boresight_opposite_to_tgo=(-0.921634644,-0.388003786,0.006530345) #calculated
    new_mir_boresight_to_tgo=(-0.92148,-0.38842,-0.00112) #calculated
    #convert data to times and boresights using spice
    time_error=1
    print("Calculating boresights")
    boresights_all=find_boresight(epoch_times_all,time_error,old_so_boresight_to_tgo)
    print("Calculating lat lons")
    [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)

    #convert data to times and boresights using spice
    if title=="SO ACS Raster 1":
        time_raster_centre=sp.utc2et("2016APR13-03:25:00.000") #time of s/c pointing to centre
    elif title=="UVIS ACS Raster 1":
        time_raster_centre=sp.utc2et("2016APR13-03:25:00.000") #time of s/c pointing to centre
    elif title=="LNO ACS Raster 1":
        time_raster_centre=sp.utc2et("2016APR13-04:33:00.000") #time of s/c pointing to centre
    time_error=1
    centre_boresight_theoretical=find_boresight([time_raster_centre],time_error,old_so_boresight_to_tgo)
    
    uvis_centre_calc_old=find_boresight([time_raster_centre],time_error,old_uvis_boresight_to_tgo)
    so_centre_calc_old=find_boresight([time_raster_centre],time_error,old_so_boresight_to_tgo)
    lno_centre_calc_old=find_boresight([time_raster_centre],time_error,old_lno_boresight_to_tgo)

    uvis_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_uvis_boresight_same_to_tgo)
    uvis_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_uvis_boresight_opposite_to_tgo)
    so_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_so_boresight_same_to_tgo)
    so_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_so_boresight_opposite_to_tgo)
    lno_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_lno_boresight_same_to_tgo)
    lno_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_lno_boresight_opposite_to_tgo)
    mir_centre_calc_new=find_boresight([time_raster_centre],time_error,new_mir_boresight_to_tgo)

    [_,centre_boresight_lon,centre_boresight_lat]=find_rad_lon_lat(centre_boresight_theoretical)
    [_,uvis_centre_calc_old_lon,uvis_centre_calc_old_lat]=find_rad_lon_lat(uvis_centre_calc_old)
    [_,uvis_centre_calc_new_lon,uvis_centre_calc_new_lat]=find_rad_lon_lat(uvis_centre_calc_same_new)


    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0, c=marker_colour)

    ax.scatter(uvis_centre_calc_old[:,0],uvis_centre_calc_old[:,1],uvis_centre_calc_old[:,2], marker='o', linewidth=0, c='r')
    ax.scatter(so_centre_calc_old[:,0],so_centre_calc_old[:,1],so_centre_calc_old[:,2], marker='o', linewidth=0, c='k')
    ax.scatter(lno_centre_calc_old[:,0],lno_centre_calc_old[:,1],lno_centre_calc_old[:,2], marker='o', linewidth=0, c='c')

    ax.scatter(uvis_centre_calc_same_new[:,0],uvis_centre_calc_same_new[:,1],uvis_centre_calc_same_new[:,2], marker='o', linewidth=0, c='r')
    ax.scatter(uvis_centre_calc_opposite_new[:,0],uvis_centre_calc_opposite_new[:,1],uvis_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='r')
    ax.scatter(so_centre_calc_same_new[:,0],so_centre_calc_same_new[:,1],so_centre_calc_same_new[:,2], marker='o', linewidth=0, c='k')
    ax.scatter(so_centre_calc_opposite_new[:,0],so_centre_calc_opposite_new[:,1],so_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='k')
    ax.scatter(lno_centre_calc_same_new[:,0],lno_centre_calc_same_new[:,1],lno_centre_calc_same_new[:,2], marker='o', linewidth=0, c='c')
    ax.scatter(lno_centre_calc_opposite_new[:,0],lno_centre_calc_opposite_new[:,1],lno_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='c')
    ax.scatter(mir_centre_calc_new[:,0],mir_centre_calc_new[:,1],mir_centre_calc_new[:,2], marker='o', linewidth=0, c='g')


    ax.set_title(title+" sum of detector counts during raster scan plotted in solar system coordinates")
    ax.set_xlabel("X in S.S. reference frame")
    ax.set_ylabel("Y in S.S. reference frame")
    ax.set_zlabel("Z in S.S. reference frame")

    plt.figure(figsize=(9,9))
    plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c=marker_colour)
    plt.scatter(centre_boresight_lon,centre_boresight_lat, marker='o', linewidth=0, c='b')
    plt.scatter(uvis_centre_calc_old_lon,uvis_centre_calc_old_lat, marker='o', linewidth=0, c='k')
#    plt.scatter(uvis_centre_calc_new_lon,uvis_centre_calc_new_lat, marker='o', linewidth=0, c='r')
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title(title+" sum of detector counts during raster scan plotted in solar system lat/lon")
#    plt.savefig(title+"_sum_detector_counts__lat_lon.png")

if option==11:
    """try to figure out which way to move the boresights by plotting tgo coordinates in 3d and using model"""
    hdf5_file.close()

    time=sp.utc2et("2016APR13-03:25:00.000") #ACS raster scan centre
    cmatrix=sw.ckgp1(time,1)
    tgo_x_ss = np.dot(np.transpose(cmatrix),(1,0,0))
    tgo_y_ss = np.dot(np.transpose(cmatrix),(0,1,0))
    tgo_z_ss = np.dot(np.transpose(cmatrix),(0,0,1))
    
    old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325)
    old_so_bs_ss = np.dot(np.transpose(cmatrix),old_so_boresight_to_tgo)

    observer="-143" #tgo
    target="SUN"
    sun_pos_ss = sw.spkpos1(observer,target,time)
    sun_distance = la.norm(sun_pos_ss)
    sun_vector_ss = sun_pos_ss/sun_distance
    
    [tgo_x_radius,tgo_x_lon,tgo_x_lat]=sw.reclat(tgo_x_ss)
    [tgo_y_radius,tgo_y_lon,tgo_y_lat]=sw.reclat(tgo_y_ss)
    [tgo_z_radius,tgo_z_lon,tgo_z_lat]=sw.reclat(tgo_z_ss)
    [sun_vector_radius,sun_vector_lon,sun_vector_lat]=sw.reclat(sun_vector_ss)
    
    #actual observed sun locations during SO ACS Raster 1: lon=21.40907405, lat=10.84657821

    sun_observed_lon=21.2846285676596+(21.40907405-21.2846285676596)*100
    sun_observed_lat=10.973230250359+(10.84657821-10.973230250359)*100
    sun_obs_vector_ss=sw.latrec(1,sun_observed_lon,sun_observed_lat)

    plt.figure(figsize=(9,9))
    plt.scatter(tgo_x_lon,tgo_x_lat, marker='o', linewidth=0, c='r')
    plt.scatter(tgo_y_lon,tgo_y_lat, marker='o', linewidth=0, c='g')
    plt.scatter(tgo_z_lon,tgo_z_lat, marker='o', linewidth=0, c='b')
    plt.scatter(sun_vector_lon,sun_vector_lat, marker='o', linewidth=0, c='y')
    plt.scatter(sun_observed_lon,sun_observed_lat, marker='o', linewidth=0, c='orange')
    plt.plot((0,tgo_x_lon),(0,tgo_x_lat), c='r')
    plt.plot((0,tgo_y_lon),(0,tgo_y_lat), c='g')
    plt.plot((0,tgo_z_lon),(0,tgo_z_lat), c='b')
    plt.plot((0,sun_vector_lon),(0,sun_vector_lat), c='y')
    plt.plot((0,sun_observed_lon),(0,sun_observed_lat), c='orange')
 
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot((0,tgo_x_ss[0]),(0,tgo_x_ss[1]),(0,tgo_x_ss[2]), c='r')
    ax.plot((0,tgo_y_ss[0]),(0,tgo_y_ss[1]),(0,tgo_y_ss[2]), c='g')
    ax.plot((0,tgo_z_ss[0]),(0,tgo_z_ss[1]),(0,tgo_z_ss[2]), c='b')
    ax.plot((0,sun_vector_ss[0]),(0,sun_vector_ss[1]),(0,sun_vector_ss[2]), c='y')
    ax.plot((0,old_so_bs_ss[0]/2),(0,old_so_bs_ss[1]/2),(0,old_so_bs_ss[2]/2), c='k')
    ax.plot((0,sun_obs_vector_ss[0]),(0,sun_obs_vector_ss[1]),(0,sun_obs_vector_ss[2]), c='orange')
    
    ax.text(tgo_x_ss[0],tgo_x_ss[1],tgo_x_ss[2],"TGO X")
    ax.text(tgo_y_ss[0],tgo_y_ss[1],tgo_y_ss[2],"TGO Y")
    ax.text(tgo_z_ss[0],tgo_z_ss[1],tgo_z_ss[2],"TGO Z")
    ax.text(sun_vector_ss[0],sun_vector_ss[1],sun_vector_ss[2],"Sun")
    ax.text(old_so_bs_ss[0]/2,old_so_bs_ss[1]/2,old_so_bs_ss[2]/2,"SO Old BS")
    ax.text(sun_obs_vector_ss[0],sun_obs_vector_ss[1],sun_obs_vector_ss[2],"Sun Observed")
    
    rectan=sw.latrec(1,20,15)
    out=sw.reclat(rectan)


if option==12:
    """make detector and line plot animations of the miniscan/fullscan data"""
    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    hdf5_file.close()
    
    max_value=np.max(detector_data_all)
    
    if channel=="so":
#        ybin_centre=12 #for YBins data
#        ybins=[10,11,12,13,14] #for YBins data
        ybin_centre=detector_centre #for Y data
        ybins=[124,126,128,130,132] #for Y data
    if channel=="lno":
#        ybin_centre=11 #for YBins data
#        ybins=[9,10,11,12,13] #for YBins data
        ybin_centre=detector_centre #for Y data
        ybins=[140,146,152,158,164] #for Y data
    
    print("Plotting detector frame:")

    fig=plt.figure(figsize=(10,8))
    num=0
    
    im = plt.imshow(detector_data_all[num,:,:], vmin=0, vmax=max_value, animated=True)
    imtitle = plt.title("%s: AOTF Frequency = %f kHz" %(title,aotf_freq_all[num]))
    imbar = plt.colorbar()

    def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
        global detector_data_all
        if np.mod(num,50)==0:
            print(num)
        im.set_array(detector_data_all[num,:,:])
        imtitle.set_text("%s: AOTF Frequency = %f kHz" %(title,aotf_freq_all[num]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=600, interval=50, blit=True)
    ani.save(title+" Detector_Frame.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
    plt.show()

    print("Plotting detector centre lines:")

    fig2=plt.figure(figsize=(9,9))
    ax = plt.axes(xlim=(0,320),ylim=(0,max_value))
    num=0
    linetitle = plt.title("%s: AOTF Frequency = %f kHz" %(title,aotf_freq_all[num]))
    line1, = ax.plot([],[])
    line2, = ax.plot([],[])
    line3, = ax.plot([],[])
    line4, = ax.plot([],[])
    line5, = ax.plot([],[])

    def updatefig2(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
        global detector_data_all
        if np.mod(num,50)==0:
            print(num)
        line1.set_data(range(320),detector_data_all[num,ybins[0],:])
        line2.set_data(range(320),detector_data_all[num,ybins[1],:])
        line3.set_data(range(320),detector_data_all[num,ybins[2],:])
        line4.set_data(range(320),detector_data_all[num,ybins[3],:])
        line5.set_data(range(320),detector_data_all[num,ybins[4],:])
        linetitle.set_text("%s: AOTF Frequency = %f kHz" %(title,aotf_freq_all[num]))
        return line1,line2,line3,line4,line5,
    
    ani2 = animation.FuncAnimation(fig2, updatefig2, frames=detector_data_all.shape[0], interval=50, blit=True)
    ani2.save(title+" Centre_Line.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
    plt.show()
    
    print("Done")
    gc.collect() #clear animation from memory

if option==13:
    """calculate optimum integration time"""
    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    binning_all = get_dataset_contents(hdf5_file,"BINNING")[0]
    nacc_all = get_dataset_contents(hdf5_file,"NUMBER_OF_ACCUMULATIONS")[0]
    inttime_all = get_dataset_contents(hdf5_file,"INTEGRATION_TIME")[0]
    backsub_all = get_dataset_contents(hdf5_file,"BACKGROUND_SUBTRACTION")[0]
    hdf5_file.close()
    
    pixel_saturation_value=12000    
    index=10 #take frame x (best not to choose 1st, as sometimes data is corrupted)
    obs_saturation_value=pixel_saturation_value*(binning_all[index]+1)*(nacc_all[index]/(backsub_all[index]+1))
    max_value=np.max(detector_data_all)
#    print("For frame %i: observation saturation value = %i; max value in dataset = %i" %(index,obs_saturation_value,max_value)
    
    min_sat_inttime=10000000.0
    saturation_inttime_all=[]
    for index in range(len(aotf_freq_all)):
        frame_max_value=np.max(detector_data_all[index])
        obs_saturation_value=pixel_saturation_value*(binning_all[index]+1)*(nacc_all[index]/(backsub_all[index]+1))
        saturation_inttime=(np.float(obs_saturation_value)/np.float(frame_max_value))*np.float(inttime_all[index])
        saturation_inttime_all.append(saturation_inttime)
        if saturation_inttime < min_sat_inttime:
            min_sat_inttime=saturation_inttime
            index_of_minimum=index
        print("For frame %i: integration time can be increased from %i ms to %.4f ms" %(index,inttime_all[index],saturation_inttime))
    saturation_inttime_all=np.asfarray(saturation_inttime_all)
    print("Lowest integration time is for frame %i, where it can be increased from %i ms to %.4f ms" %(index_of_minimum,inttime_all[index],min_sat_inttime))
    
    dark_centre_lines = detector_data_all[aotf_freq_all==0,nec_sun_detector_centre,:]
    stdev_dark = np.std(dark_centre_lines, axis=0)

    chosen_frame=223
    light_line = detector_data_all[chosen_frame,nec_sun_detector_centre,:]
    snr = light_line / stdev_dark
    
    plt.figure(figsize=(10,8))
    plt.plot(snr,'.')
    plt.yscale("log")
    plt.title(title + ": Estimated Signal-to-Noise Ratio for frame %i" %chosen_frame)
    plt.xlabel("Pixel Number")
    plt.ylabel("Centre line / Stdev of dark centre lines")
    
    plt.figure(figsize=(10,8))
    plt.plot(aotf_freq_all[aotf_freq_all>0],saturation_inttime_all[aotf_freq_all>0],'.')
    plt.yscale("log")
    plt.title(title + ": Integration Time Required for Saturation")
    plt.xlabel("AOTF Frequency (kHz)")
    plt.ylabel("Saturation integration time (ms)")
    
    detector_data_all=[]
    gc.collect()
    
if option==14:
    """make 2d plot of aotf versus miniscan absorption lines. need to adapt for full scan data"""

    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    hdf5_file.close()
    x=np.arange(320)

    if channel=="so":
        ybin_centre=12 #for YBins data
        ybins=[9,10,11,12,13,14,15,16] #for YBins data
        if title=="SO Fullscan 1":
            npoints=132 #255 steps then go back to beginning
        else:
            npoints=256 #255 steps then go back to beginning
    if channel=="lno":
        ybin_centre=11 #for YBins data
        ybins=[9,10,11,12,13] #for YBins data
        if title=="LNO Fullscan 1":
            npoints=132 #255 steps then go back to beginning
        else:
            npoints=256 #255 steps then go back to beginning
    order={"SO Miniscan 1":142, "LNO Miniscan 1":142, "SO Miniscan 2":162, "LNO Miniscan 2":162, "SO Miniscan 3":179, "LNO Miniscan 3":179, \
        "SO Fullscan 1":0, "LNO Fullscan 1":0}[title]

    for frame_loop in range(detector_data_all.shape[0]): #interpolate bad pixels for whole dataset
        detector_data_all[frame_loop,15,:]=interpolate_bad_pixel(detector_data_all[frame_loop,15,:],211)
        detector_data_all[frame_loop,12,:]=interpolate_bad_pixel(detector_data_all[frame_loop,12,:],84)
        detector_data_all[frame_loop,12,:]=interpolate_bad_pixel(detector_data_all[frame_loop,12,:],269)
    
    for frame_loop in range(2):
        if frame_loop==0: frames_to_view=[0,npoints]
        if frame_loop==1: frames_to_view=[npoints,npoints*2]
        
        detector_slices = np.sum(detector_data_all[frames_to_view[0]:frames_to_view[1],ybins,:],axis=1) #sum lines containing peak data to make set of slices
        
        sg_lines=np.zeros(detector_slices.shape)
        for line_loop in range(detector_slices.shape[0]):
            
            detector_line = detector_slices[line_loop,:]
            sg_line = sg_filter(detector_line)
            sg_lines[line_loop,:] = detector_line/sg_line
    
        
        plt.figure(figsize=(10,8))
        plt.pcolormesh(aotf_freq_all[frames_to_view[0]:frames_to_view[1]],x,np.transpose(sg_lines))
        plt.axis([aotf_freq_all[frames_to_view[0]:frames_to_view[1]].min(), aotf_freq_all[frames_to_view[0]:frames_to_view[1]].max(), x.min(), x.max()])
        plt.title(title + " order %i: frames %i-%i" %(order,frames_to_view[0],frames_to_view[1]-1))
        plt.xlabel("AOTF frequency (kHz)")
        plt.ylabel("Pixel number")
        cbar = plt.colorbar()
        cbar.set_label("Signal ADU / Savitsky Golay Filter: 49,2", rotation=270, labelpad=20)
        plt.savefig(title + "_order_%i_frames_%i-%i.png" %(order,frames_to_view[0],frames_to_view[1]-1))
        np.savetxt(title + "_order_%i_frames_%i-%i.txt" %(order,frames_to_view[0],frames_to_view[1]-1),sg_lines,delimiter=",")
   
if option==15:
    """make line plots of given orders from the fullscan results with spectral calibration"""
    
    if channel=="so" or channel=="lno":
        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    elif channel=="uvis":
        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
    
    
    hdf5_file.close()

    if channel=="so":
        starting_order=95
        ybins=[9,10,11,12,13,14,15,16] #for YBins data
        for frame_loop in range(detector_data_all.shape[0]): #interpolate bad pixels for whole dataset
            detector_data_all[frame_loop,15,:]=interpolate_bad_pixel(detector_data_all[frame_loop,15,:],211)
            detector_data_all[frame_loop,12,:]=interpolate_bad_pixel(detector_data_all[frame_loop,12,:],84)
            detector_data_all[frame_loop,12,:]=interpolate_bad_pixel(detector_data_all[frame_loop,12,:],269)
    elif channel=="lno":
        starting_order=107
        ybins=[9,10,11,12,13] #for YBins data
    
    if channel=="so" or channel=="lno":
        plt.figure(figsize=(10,8))
        orders=np.arange(172,178)
        for order in orders:
            detector_line = np.sum(detector_data_all[order-starting_order,ybins,:],axis=0) #sum lines containing peak data to make set of slices
            calc_order = findOrder(channel,aotf_freq_all[order-starting_order])
            wavenumbers = spectral_calibration_simple(channel,calc_order)
    
            plt.plot(wavenumbers[-240::],detector_line[-240::])
        plt.xlabel("Wavenumber (cm-1)")
        plt.ylabel("Signal on detector (counts in ADU)")
        plt.title(checkout+" "+title + ": Diffraction Orders %i-%i" %(min(orders),max(orders)))
        if save_figs: plt.savefig(checkout+" "+title + "_orders_%i__%i.png" %(min(orders),max(orders)) ,dpi=900)
    elif channel=="uvis":
        detector_line = detector_data_all[10,0,:]
        wavelengths = spectral_calibration_simple(channel,0)
        plt.figure(figsize=(10,8))
        plt.plot(wavelengths[8:-16],detector_line[8:-16])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Signal on detector (counts in ADU)")
        plt.title(checkout+" "+title + ": Sun Observation")
        if save_figs: plt.savefig(checkout+" "+title + "_sun_observation.png",dpi=900)
        
    
    
#    sg_line = sg_filter(detector_line,window_size=19, order=1)
#    divided_line = detector_line/sg_line - 1
#
#    gradient_line = np.gradient(detector_line)
#    gradient_sg = sg_filter(gradient_line,window_size=19, order=5)
#
#    plt.figure(figsize=(10,8))
#    plt.plot(gradient_line)
#    plt.plot(gradient_sg)
#    plt.figure(figsize=(10,8))
#    plt.plot(gradient_line/gradient_sg)
#    plt.figure(figsize=(10,8))
#    plt.plot(detector_line)
#    plt.plot(sg_line)





    
#    plt.figure(figsize=(10,8))
#    plt.pcolormesh(aotf_freq_all[frames_to_view[0]:frames_to_view[1]],x,np.transpose(sg_lines))
#    plt.axis([aotf_freq_all[frames_to_view[0]:frames_to_view[1]].min(), aotf_freq_all[frames_to_view[0]:frames_to_view[1]].max(), x.min(), x.max()])
#    plt.title(title + " order %i: frames %i-%i" %(order,frames_to_view[0],frames_to_view[1]-1))
#    plt.xlabel("AOTF Frequency (kHz)")
#    plt.ylabel("Pixel Number")
#    plt.savefig(title + "_order_%i_frames_%i-%i.png" %(order,frames_to_view[0],frames_to_view[1]-1))
#    np.savetxt(title + "_order_%i_frames_%i-%i.txt" %(order,frames_to_view[0],frames_to_view[1]-1),sg_lines,delimiter=",")
#

#    detector_line = af.noise_filter(detector_line,1.0)
    
#    plt.plot(detector_line)
    
    
if option==16:
    """extrapolate sun shape to find pixel containing sun centre"""
    """1. plot vertical slices, define dark and light values and sun width to calculate position of sun on detector and whether fully illuminated
    2. define crossing time indices to detect when sun crosses detector centre (forward scans) or when sun illumination peaks (for reverse scans) 
    3. plot straight line through crossing points to define measured sun position
    4. 
    """
#    from scipy.interpolate import UnivariateSpline
    detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
    window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
    hdf5_file.close()
    
    if binning==2: #stretch array
        detector_data=np.repeat(detector_data,2,axis=1)
    window_size=16*binning

    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
    time_error=1    
    nframes=detector_data.shape[0]
    
    fine_time_step=0.1
    if title=="LNO ACS Raster 1":
        illuminated_value=150000
        dark_value=100000
        half_max_value=100000
        spectral_line_index=228
#        sun_width=22
#        smoothing=100
        sun_width=25
        smoothing=200
        chosen_window_top=144
        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
    elif title=="SO ACS Raster 1" or title=="SO ACS Raster 2":
        illuminated_value=100000
        dark_value=50000
        half_max_value=75000
        spectral_line_index=228
#        sun_width=22
#        smoothing=100
        sun_width=25
        smoothing=200
        chosen_window_top=128
        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
    elif title=="SO Raster 4A" or title=="SO Raster 4B":
#        illuminated_value=100000
#        dark_value=50000
#        half_max_value=75000
        illuminated_value=250000
        dark_value=50000
        half_max_value=150000
        spectral_line_index=228
        smoothing=100

        sun_width=25
        chosen_window_top=128
        boresight_to_tgo=(-0.92191,-0.38736,0.00608) #theoretical
        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)

        fine_position_step=fine_time_step/3.0
        
        if title=="SO Raster 4A":       
            time_indices=[[0,4],[5,8],[9,13],[14,18],[19,22]]
            centre_indices=[56,57]
            reverse=False
        elif title=="SO Raster 4B":
            reverse=True
            time_indices=[[24,36],[37,49],[51,62],[63,75],[76,88],[89,101]]
            centre_indices=[55,56]

    elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
#        illuminated_value=100000
#        dark_value=50000
#        half_max_value=75000
        illuminated_value=300000
        dark_value=50000
        half_max_value=200000
        spectral_line_index=228
        smoothing=100

        sun_width=25
        chosen_window_top=128
        boresight_to_tgo=(-0.92163,-0.38800,0.00653) #theoretical
        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)

        fine_position_step=0.08/2.0
        
        if title=="LNO Raster 4A":       
            time_indices=[[0,17],[18,48],[50,73]]
            centre_indices=[111,112]
            reverse=False
        elif title=="LNO Raster 4B":
            reverse=True
            time_indices=[[70,90],[95,125],[130,160],[165,195]]
            centre_indices=[111,112]

    
    detector_vlines=detector_data[:,:,spectral_line_index]
    
    centres=[]
    boresights=[]
    times=[]
    plt.figure(figsize=(9,9))
    for frame_index in range(nframes):
        detector_vline = detector_vlines[frame_index,:]
        vpixel_number = np.asfarray(range(window_top_all[frame_index],window_top_all[frame_index]+window_size))
        if detector_vline[0]<dark_value and detector_vline[window_size-1]>illuminated_value: #if rising
            """interpolate to find pixel value where rising edge is seen"""
            vpix1=max(vpixel_number[np.where(detector_vline<half_max_value)[0]])
            pixval1=detector_vline[int(vpix1-window_top_all[frame_index])]
            vpix2=min(vpixel_number[np.where(detector_vline>half_max_value)[0]])
            pixval2=detector_vline[int(vpix2-window_top_all[frame_index])]
            vpix_interp = np.interp(half_max_value,[pixval1,pixval2],[vpix1,vpix2])
#            print(vpix_interp
            centres.append(vpix_interp+sun_width/2)
            boresights.append(boresights_all[frame_index])
            times.append(epoch_times_all[frame_index])
            plt.scatter(vpix_interp,half_max_value)
            print("rising %i" %frame_index)
            plt.plot(vpixel_number,detector_vline)
        elif detector_vline[0]>illuminated_value and detector_vline[window_size-1]<dark_value: #if falling
            vpix1=max(vpixel_number[np.where(detector_vline>half_max_value)[0]])
            pixval1=detector_vline[int(vpix1-window_top_all[frame_index])]
            vpix2=min(vpixel_number[np.where(detector_vline<half_max_value)[0]])
            pixval2=detector_vline[int(vpix2-window_top_all[frame_index])]
            vpix_interp = np.interp(half_max_value,[pixval2,pixval1],[vpix2,vpix1])
#            print(vpix_interp
            centres.append(vpix_interp-sun_width/2)
            boresights.append(boresights_all[frame_index])
            times.append(epoch_times_all[frame_index])
            plt.scatter(vpix_interp,half_max_value)
            print("falling %i" %frame_index)
            plt.plot(vpixel_number,detector_vline)
    plt.xlabel("Vertical Pixel Number")
    plt.ylabel("Signal ADU for horizontal pixel %i" %spectral_line_index)
    plt.title(title+": Vertical detector slices where Sun is seen")    
    if save_figs: plt.savefig(title+"_vertical_detector_slices_where_Sun_is_seen.png")
    
    #find indices where centre of detector is
    meas_indices=[]
    for index,window_top in enumerate(window_top_all):
        if window_top==chosen_window_top:
            meas_indices.append(index)
    detector_sum=np.sum(detector_data[:,detector_centre-chosen_window_top,:],axis=1)

    marker_colour=np.log(1+detector_sum[meas_indices]-min(detector_sum[meas_indices]))
    

    if not reverse:
        fine_centres=[]
        fine_times=[]
        for time_groups in range(len(time_indices)):
            m,x=np.polyfit(times[time_indices[time_groups][0]:time_indices[time_groups][1]],centres[time_indices[time_groups][0]:time_indices[time_groups][1]],1)
            fine_time=np.arange(times[time_indices[time_groups][0]],times[time_indices[time_groups][1]],fine_time_step)
            fine_centres.extend(fine_time*m + x)
            fine_times.extend(fine_time)

#        fine_centres=30.0*np.sin(fine_times/63.0-12.5)+128.0
        fine_centres = np.asfarray(fine_centres)
        fine_times = np.asfarray(fine_times)

        detector_times=fine_times[np.where((fine_centres>detector_centre-fine_position_step) & (fine_centres<detector_centre+fine_position_step))[0]]
        plt.figure(figsize=(9,9))
#        plt.plot(np.asfarray(times)[np.abs(np.asfarray(centres)-122.5)>0.5],np.asfarray(centres)[np.abs(np.asfarray(centres)-122.5)>0.5],'*')
        plt.plot(times,centres,'*')
        plt.plot(fine_times,fine_centres)
        plt.plot([times[0],times[-1]],[detector_centre,detector_centre])
        plt.plot([detector_times,detector_times],[min(centres),max(centres)])
        plt.xlabel("Ephemeris time")
        plt.ylabel("Calculated pixel row where sun is centred")
        plt.title(title+": Position of sun centre on detector vs time")
        if save_figs: plt.savefig(title+"_position_of_sun_centre_on_detector_vs_time.png")
    
        previous_time=0
        crossing_times=[]
        for time_loop in range(len(detector_times)):
            if (detector_times[time_loop]-previous_time)>5: #remove values that are very close together
                print("Calculated peak sun on detector row %i at %s" %(detector_centre,sw.et2utc(detector_times[time_loop])))
                plt.text(detector_times[time_loop],110.0+time_loop,"%s" %sw.et2utc(detector_times[time_loop]))
                crossing_times.append(detector_times[time_loop])
            previous_time=detector_times[time_loop]

    if reverse:
#        spl = UnivariateSpline(times,centres)
#        spl.set_smoothing_factor(smoothing)
#        fine_times=np.arange(times[0],times[-1],fine_time_step)
#        fine_centres=spl(fine_times)

        crossing_times=[]
        plt.figure(figsize=(9,9))
        plt.title(title+" sum of detector row %i during raster scan" %chosen_window_top)
        plt.xlabel("Time")
        plt.ylabel("Sum of signal ADU")
        for time_groups in range(len(time_indices)):
#            plt.figure(figsize=(9,9))
#            plt.title(title+" detector row %i for vertical lines during raster scan" %chosen_window_top)
#            plt.xlabel("Vertical pixel")
#            plt.ylabel("Signal ADU")
#            for plot_loop in range((detector_vlines[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]],:]).shape[0]):
#                plt.plot(np.arange(16), detector_vlines[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]],:][plot_loop,:], marker='o', linewidth=0)#,label=sw.et2utc(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]][plot_loop]))
#            plt.legend()
#            if save_figs: plt.savefig(title+"_detector_row_%i_for_vertical_lines_during_raster_scan.png" %chosen_window_top)
 
            """for non-gaussian shapes
            spl = UnivariateSpline(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),detector_sum[meas_indices[time_indices[0][0]:time_indices[0][1]]])
            spl.set_smoothing_factor(0.001)
            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[0][0]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),epoch_times_all[meas_indices[time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),fine_time_step)
            fine_centres=spl(fine_times)
    
            m2,m1,x = np.polyfit(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),detector_sum[meas_indices[time_indices[0][0]:time_indices[0][1]]],2)
            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[0][0]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),epoch_times_all[meas_indices[time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),fine_time_step)
            fine_centres=fine_times**2 * m2 + fine_times * m1 + x
            """
            
            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[time_groups][0]]],epoch_times_all[meas_indices[time_indices[time_groups][1]]],fine_time_step)
            #Gaussian function
            def gaussian(x, a, x0, sigma):
                return a*np.exp(-(x-x0)**2/(2*sigma**2))
            from scipy.optimize import curve_fit
            mean = np.mean(epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]])
            sigma = 100
            popt, pcov = curve_fit(gaussian, epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], p0 = [np.mean(detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]]), mean, sigma])
            fine_centres =  gaussian(fine_times, *popt)
            crossing_times.append(popt[1])
            
            plt.plot(epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], marker='o', linewidth=0)
            plt.plot(fine_times, fine_centres)
            plt.text(popt[1],popt[0],sw.et2utc(popt[1]))
            if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_gaussian.png" %chosen_window_top)



    
    observer="-143" #tgo
    target="SUN"
    angles=np.zeros(nframes)
    sun_pointing_vector=np.zeros((nframes,3))
    
#old matlab version
#    for time_loop in range(nframes):
#        relative_sun_position=sw.spkpos1(observer,target,epoch_times_all[time_loop])
#        sun_distance = la.norm(relative_sun_position)
#        sun_pointing_vector = relative_sun_position / sun_distance
    relative_sun_position=sw.spkposx(observer,target,epoch_times_all)
    for time_loop in range(nframes):
        sun_distance = la.norm(relative_sun_position[time_loop])
        sun_pointing_vector[time_loop,:] = relative_sun_position[time_loop] / sun_distance #normalise

        angles[time_loop] = py_ang(boresights_all[time_loop],sun_pointing_vector[time_loop,:]) * 180 * 60 / np.pi
        
    print("Raster %f arcmins from centre at %s" %(angles[angles.argmin()],sw.et2utc(epoch_times_all[angles.argmin()])))
    print("Raster started at %s" %sw.et2utc(epoch_times_all[0]))
    print("Raster ended at %s" %sw.et2utc(epoch_times_all[-1]))


    plt.figure(figsize=(9,9))
    plt.plot(angles)

#    plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
#    plt.xlabel("Approx time after pre-cooling ends (seconds)")
#    plt.title(title)
#    plt.yscale("log")
    
#    np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
#    plt.savefig(title+"_intensity_versus_time_raster_scan_log.png")
    

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(boresights_all[meas_indices,0],boresights_all[meas_indices,1],boresights_all[meas_indices,2], c=marker_colour, marker='o', linewidth=0)
    ax.text(boresights_all[0,0],boresights_all[0,1],boresights_all[0,2], "Raster Start")
    ax.text(boresights_all[-1,0],boresights_all[-1,1],boresights_all[-1,2], "Raster End")
    centre_boresight=find_boresight([epoch_times_all[angles.argmin()]],time_error,boresight_to_tgo)
    crossing_boresights=find_boresight(crossing_times,time_error,boresight_to_tgo)
    ax.scatter(crossing_boresights[:,0],crossing_boresights[:,1],crossing_boresights[:,2], marker='s', linewidth=0, c='b')
    ax.scatter(centre_boresight[:,0],centre_boresight[:,1],centre_boresight[:,2], marker='*', s=100, linewidth=0, c='k')
    ax.set_title(title+" sum of detector row %i during raster scan in J2000 coordinates" %chosen_window_top)
    ax.set_xlabel("X in S.S. reference frame")
    ax.set_ylabel("Y in S.S. reference frame")
    ax.set_zlabel("Z in S.S. reference frame")
    ax.azim=-120
    ax.elev=1
    if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_in_J2000_coordinates.png" %chosen_window_top)

    boresight_lat_lons=sw.reclatx(boresights_all[meas_indices,:])
    centre_lat_lons=sw.reclatx(centre_boresight)
    crossing_lat_lons=sw.reclatx(crossing_boresights)
    sun_lat_lons=sw.reclatx(sun_pointing_vector)
    
    print(crossing_lat_lons)

    plt.figure(num=100,figsize=(9,9))
    plt.scatter(boresight_lat_lons[:,1],boresight_lat_lons[:,2], c=marker_colour, marker='o', linewidth=0)
    plt.scatter(sun_lat_lons[:,1],sun_lat_lons[:,2], marker='*', s=100, linewidth=0, c='r')
    plt.scatter(sun_lat_lons[angles.argmin(),1],sun_lat_lons[angles.argmin(),2], marker='*', s=100, linewidth=0, c='g')
    plt.scatter(centre_lat_lons[0][1],centre_lat_lons[0][2], marker='*', s=100, linewidth=0, c='k')
    plt.scatter(crossing_lat_lons[:,1],crossing_lat_lons[:,2], marker='s', linewidth=0, c='b')
    y1=crossing_lat_lons[0,2]
    y2=crossing_lat_lons[-1,2]
    m,dy=np.polyfit(crossing_lat_lons[:,2],crossing_lat_lons[:,1],1)
    plt.plot([y1*m+dy,y2*m+dy],[y1,y2])
    plt.plot([boresight_lat_lons[centre_indices[0],1],boresight_lat_lons[centre_indices[1],1]], \
             [boresight_lat_lons[centre_indices[0],2],boresight_lat_lons[centre_indices[1],2]],'b')
        
    plt.title(title+" sum of detector row %i during raster scan in lat lons" %chosen_window_top)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_in_lat_lons.png" %chosen_window_top)

    print("_calculated_lon_lats=[%f,%f]" %(sun_lat_lons[angles.argmin(),1],sun_lat_lons[angles.argmin(),2]))
    print("_sun_time=%f" %epoch_times_all[angles.argmin()])


    """add calculated values here"""
    if title=="SO Raster 4A" or title=="SO Raster 4B":
        #left right and up down
        so_measured_lon_lats = [73.3200,26.3243] #found from measurement data
        so_calculated_lon_lats = [73.376989,26.335803]
        so_sun_time=519304568.137519
        
        lon_lat_misalignment = [so_calculated_lon_lats[0] - so_measured_lon_lats[0],so_calculated_lon_lats[1] - so_measured_lon_lats[1]]
        print("Misalignment= %0.2f arcmins lon, %0.2f arcmins lat" %(lon_lat_misalignment[0] * 60.0,lon_lat_misalignment[1] * 60.0))
        
        so_new_lon_lat = [so_calculated_lon_lats[0] + lon_lat_misalignment[0],so_calculated_lon_lats[1] + lon_lat_misalignment[1]]
        
        so_new_vector = sw.latrec(1.00000, so_new_lon_lat[0],so_new_lon_lat[1])
        print("SO new vector= "+"%0.7f "*3 %(so_new_vector[0],so_new_vector[1],so_new_vector[2]))
        so_cmatrix = sw.ckgp1(so_sun_time, 1)
        so_new_boresight=np.dot(so_cmatrix,so_new_vector)
        print("SO new boresight= "+"%0.7f "*3 %(so_new_boresight[0],so_new_boresight[1],so_new_boresight[2]))
        
        #now reverse calcualation to check numbers
        so_vector_recalc = find_boresight([so_sun_time,so_sun_time+1],1.00000,so_new_boresight) #use two boresights to run functions correctly
        so_lon_lat_recalc = sw.reclatx(so_vector_recalc)[0,:]
        ax.scatter(so_vector_recalc[0,0],so_vector_recalc[0,1],so_vector_recalc[0,2],marker="*",s=150,c="c")
        plt.scatter(so_lon_lat_recalc[1],so_lon_lat_recalc[2],marker="*",s=150,c="c")
    elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
        lno_measured_lon_lats = [73.3754,26.3505] #found from measurement data
        lno_calculated_lon_lats=[73.403552,26.338780]
        lno_sun_time=519307568.191518

        lon_lat_misalignment = [lno_calculated_lon_lats[0] - lno_measured_lon_lats[0],lno_calculated_lon_lats[1] - lno_measured_lon_lats[1]]
        print("Misalignment= %0.2f arcmins lon, %0.2f arcmins lat" %(lon_lat_misalignment[0] * 60.0,lon_lat_misalignment[1] * 60.0))
        
        lno_new_lon_lat = [lno_calculated_lon_lats[0] + lon_lat_misalignment[0],lno_calculated_lon_lats[1] + lon_lat_misalignment[1]]
        
        lno_new_vector = sw.latrec(1.00000, lno_new_lon_lat[0],lno_new_lon_lat[1])
        print("LNO new vector= "+"%0.7f "*3 %(lno_new_vector[0],lno_new_vector[1],lno_new_vector[2]))
        lno_cmatrix = sw.ckgp1(lno_sun_time, 1)
        lno_new_boresight=np.dot(lno_cmatrix,lno_new_vector)
        print("LNO new boresight= "+"%0.7f "*3 %(lno_new_boresight[0],lno_new_boresight[1],lno_new_boresight[2]))
        
        #now reverse calcualation to check numbers
        lno_vector_recalc = find_boresight([lno_sun_time,lno_sun_time+1],1.00000,lno_new_boresight) #use two boresights to run functions correctly
        lno_lon_lat_recalc = sw.reclatx(lno_vector_recalc)[0,:]
        ax.scatter(lno_vector_recalc[0,0],lno_vector_recalc[0,1],lno_vector_recalc[0,2],marker="*",s=150,c="c")
        plt.scatter(lno_lon_lat_recalc[1],lno_lon_lat_recalc[2],marker="*",s=150,c="c")

    

    
if option==17:
    """display frame"""
    detector_data_all = get_dataset_contents(hdf5_file,"YBins")[0] #get data
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    hdf5_file.close()
    
    chosen_frame=100
    
    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],84)
    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],269)
    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],199)
    detector_data_all[chosen_frame,8,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,8,:],256)

    calc_order = findOrder(channel,aotf_freq_all[chosen_frame])
    wavenumbers = spectral_calibration_simple(channel,calc_order)


    plt.figure(figsize=(10,8))
    plt.imshow(detector_data_all[chosen_frame,6:20,:],interpolation='none',cmap=plt.cm.gray, aspect=2.4, extent=[wavenumbers[0],wavenumbers[-1],6,20])
    plt.colorbar()
    plt.title("Solar spectrum taken on 15th April 2016")
#    plt.xlabel("Horizontal pixel number (spectral direction)")
    plt.xlabel("Wavenumbers (cm-1)")
    plt.ylabel("Vertical pixel number (spatial direction)")
    if save_figs: plt.savefig("Solar_spectrum_taken_on_14th_April_2016.png", dpi=400)


if option==18:
    """plot straylight intensity vs time"""
    if channel=="so" or channel=="lno":
        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
#        exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
        window_top = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0][0]
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
        hdf5_file.close()

        frame_indices=[]
#        exponent_all=[]
        chosen_aotf_freq=aotf_freq_all[0]
        dark_indices=[]
        times=[]
        dates=[]
        for frame_loop in range(detector_data_all.shape[0]): #loop through all frames
            if aotf_freq_all[frame_loop]==0:
                dark_indices.append(frame_loop)
            if aotf_freq_all[frame_loop]==chosen_aotf_freq:
                frame_indices.append(frame_loop)
                times.append(time_data_all[frame_loop])
                dates.append(date_data_all[frame_loop])
                
#        print("Calculating times"
        epoch_times=convert_hdf5_time_to_spice_utc(times,dates)
        
        sum_corrected_frame_centre=[]
        dark_indices_before=[]
        dark_indices_after=[]
        corrected_frames=np.zeros((len(frame_indices),detector_data_all.shape[1],detector_data_all.shape[2]))
        for frame_loop,light_frame_index in enumerate(frame_indices): #loop through chosen light frames
            found_dark=bisect.bisect_left(dark_indices, light_frame_index) #find index of next dark frame
            dark_index_after=dark_indices[found_dark] #find index of next dark frame
            dark_indices_after.append(dark_index_after)
            if found_dark==0: #if first dark frame
                dark_index_before=dark_index_after #set index to same value
            else:
                dark_index_before=dark_indices[found_dark-1]
            dark_indices_before.append(dark_index_before)
            #subtract mean of dark frames on either side from light frame
            corrected_frames[frame_loop,:,:]=detector_data_all[light_frame_index,:,:] - np.mean([detector_data_all[dark_index_before,:,:],detector_data_all[dark_index_after,:,:]], axis=0)
            sum_corrected_frame_centre.append(np.mean(corrected_frames[frame_loop,1:2,200:250]))
        
#        plt.figure(figsize=(10,8))
#        plt.imshow(detector_data_all[64,:,:], aspect=binning, interpolation='none')
#        plt.colorbar()
#        
#        plt.figure(figsize=(10,8))
#        plt.imshow(corrected_frames[64,:,:], aspect=binning, interpolation='none')
#        plt.colorbar()
        
        
        time=np.arange(len(frame_indices))
        
        plt.figure(figsize=(10,8))
        plt.plot(time,sum_corrected_frame_centre)
#        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
        plt.ylabel("Sum signal ADU for pixel bin 1:2,columns 200:250")
        plt.xlabel("Approx time after pre-cooling ends (seconds)")
        plt.title(title+": after background subtraction")
        plt.yscale("log")
        if save_figs: plt.savefig(checkout+" "+title+"_intensity_versus_time_log.png")
        
#        np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
    
        plt.figure(figsize=(10,8))
        plt.plot(time,sum_corrected_frame_centre)
#        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
        plt.ylabel("Sum signal ADU for pixel bin 1:2,columns 200:250")
        plt.xlabel("Approx time after pre-cooling ends (seconds)")
        plt.title(title+": after background subtraction")
        if save_figs: plt.savefig(checkout+" "+title+"_intensity_versus_time.png")

        marker_colour=np.log(1+np.asfarray(sum_corrected_frame_centre)-min(np.asfarray(sum_corrected_frame_centre)))  
        
        old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
        old_lno_boresight_to_tgo=(-0.92126,-0.38890,0.00368)
        old_uvis_boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000)
    
#        print("Calculating boresights"
        time_error=1
        boresights_all=find_boresight(epoch_times,time_error,old_so_boresight_to_tgo)
    
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0, c=marker_colour)
        ax.set_title(title+" background subtracted ADU bins 1:2,columns 200:250 in J2000")
        ax.set_xlabel("X in S.S. reference frame")
        ax.set_ylabel("Y in S.S. reference frame")
        ax.set_zlabel("Z in S.S. reference frame")
        ax.azim=-130
        ax.elev=-0
        if save_figs: plt.savefig(checkout+" "+title+"_background_subtracted_ADU_bins_1-2_columns_200-250_in_J2000.png")

#        print("Calculating lat lons"
        [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)

        plt.figure(figsize=(9,9))
        plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c=marker_colour)
        plt.text(boresight_lons[0],boresight_lats[0],"Raster Start")
        plt.text(boresight_lons[-1],boresight_lats[-1],"Raster End")

        plt.xlabel("Longitude (degrees)")
        plt.ylabel("Latitude (degrees)")
        plt.title(title+" background subtracted ADU bins 1:2,columns 200:250 in J2000")
        if save_figs: plt.savefig(checkout+" "+title+"background_subtracted_ADU_bins_1-2_columns_200-250_solar_lat_lons.png")

if option==19:
    """print(angular differences between all FOV centres"""
    lno_old_boresight=[-0.92163,-0.38800,0.00653] #pre MCC
    so_old_boresight=[-0.92191,-0.38736,0.00608] #pre MCC
    lno_new_boresight=[ -0.9214767, -0.3883830, 0.0062766 ] #post MCC
    so_new_boresight=[ -0.9215576, -0.3881924, 0.0061777 ] #post MCC
    uvis_new_boresight=[-0.92207,-0.38696,0.00643 ] #UVIS no change
    
    print("lno_new_vs_uvis = %0.7f" %(py_ang(lno_new_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
    print("lno_new_vs_so_new = %0.7f" %(py_ang(lno_new_boresight,so_new_boresight) * 180.0 / np.pi * 60.0))
    print("so_new_vs_uvis = %0.7f" %(py_ang(so_new_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
    print("lno_new_vs_lno_old = %0.7f" %(py_ang(lno_new_boresight,lno_old_boresight) * 180.0 / np.pi * 60.0))
    print("so_new_vs_so_old = %0.7f" %(py_ang(so_new_boresight,so_old_boresight) * 180.0 / np.pi * 60.0))
    print("lno_old_vs_uvis = %0.7f" %(py_ang(lno_old_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
    print("so_old_vs_uvis = %0.7f" %(py_ang(so_old_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))

    boresight_vector=lno_new_boresight
    
    time=519304568.137519
    cmatrix = sw.ckgp1(time, 1)
    vector=np.dot(cmatrix,boresight_vector)
    lon_lat = sw.reclat(vector)
    
    
    
if option==21:
    """nomad data workshop script: plot saturation time for all orders and estimate non-shot noise"""
    
#    if channel=="so":
#        filename=os.path.normpath("demonstration1\SO_1_SCI__DNMD__03000070_2016-107T01-59-58__00001") #SO integration time stepping
#    elif channel=="lno":
#        filename=os.path.normpath("demonstration2\\LNO_1_SCI__DNMD__03000071_2016-107T02-29-58__00001") #LNO integration time stepping
    
#    """open file and print(file attributes"""
#    hdf5_file = h5py.File(filename+".h5", "r")
#    print("File %s has the following attributes:" %(filename)
#    attributes,values = getHdf5Attributes(hdf5_file) #get attributes and values from hdf5 file
#    for index in range(len(attributes)): #print(all of the attributes and their values
#        print("%s: %s" %(attributes[index],values[index])
    
    if title=="SO Light Sky" or title=="LNO Light Sky":
        
        """get data from file, plot single detector frame"""
        detector_data,_,_ = get_dataset_contents(hdf5_file,"Y") #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        frame_number = 10 #choose a single frame
        frame_data = detector_data[frame_number,:,:] #get data for chosen frame only
        
        plt.figure(figsize = (10,8))
        plt.imshow(frame_data)
        plt.title("Frame %i" %frame_number)
        plt.xlabel("Detector horizontal (spectral) direction")
        plt.ylabel("Detector vertical (spatial) direction")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_typical_detector_frane_NEC_int_time_stepping.png" %channel)
        
        detector_centre_lines = detector_data[:,detector_centre,:] #get data for detector vertical line 130 for each frame
        max_pixel_binned = np.max(detector_centre_lines, axis=1) #find highest pixel on each line
        
        
        """get integration times from file, plot maximum value vs. integration time"""
        integration_times = get_dataset_contents(hdf5_file,"INTEGRATION_TIME")[0] #get integration times from file for each frame
        plt.figure(figsize = (10,8))
        plt.plot(integration_times,max_pixel_binned,'.')
        plt.title("Maximum pixel value vs. integration time for line %i" %detector_centre)
        plt.xlabel("Integration time (ms)")
        plt.ylabel("Maximum pixel value (ADU)")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_NEC_int_time_stepping_binned_max_value_each_frame.png" %channel)
        
        
        """get binning factor, divide maximum pixel by no. of pixels per bin"""
        binning_factors = get_dataset_contents(hdf5_file,"BINNING")[0] #get binning factor from file
        pixels_per_bin = binning_factors + 1
        max_per_pixel = max_pixel_binned / pixels_per_bin #find highest pixel on each line
        plt.figure(figsize = (10,8))
        plt.plot(integration_times,max_per_pixel,'.')
        plt.title("Maximum pixel value vs. integration time for line %i" %detector_centre)
        plt.xlabel("Integration time (ms)")
        plt.ylabel("Maximum pixel value (ADU)")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_NEC_int_time_stepping_max_value_each_frame.png" %channel)
        
        
        hdf5_file.close() #close hdf5 file
        del detector_data #clear data from memory
        gc.collect() #clear memory allocation

    elif title=="SO Fullscan 1" or title=="LNO Fullscan 1" or title=="SO Fullscan" or title=="LNO Fullscan":
    
    
#        """now open new file containing SO full scan data"""
#        if channel=="so":
#            filename=os.path.normpath("demonstration1\\SO_1_SCI__DNMD__03000066_2016-106T23-58-49__00001") #SO full scan data
#        elif channel=="lno":
#            filename=os.path.normpath("demonstration2\\LNO_1_SCI__DNMD__03000067_2016-107T00-28-58__00001") #LNO full scan data
#        hdf5_file = h5py.File(filename+".h5", "r") #open new file
        
        
        """get needed data from file then close file"""
        detector_data_fullscan,_,_ = get_dataset_contents(hdf5_file,"Y") #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        integration_times = get_dataset_contents(hdf5_file,"INTEGRATION_TIME")[0] #get integration times from file for each frame
        aotf_frequencies = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
        number_of_accumulations = get_dataset_contents(hdf5_file,"NUMBER_OF_ACCUMULATIONS")[0]
        binning_factors = get_dataset_contents(hdf5_file,"BINNING")[0] #get binning factor from file
        background_subtractions = get_dataset_contents(hdf5_file,"BACKGROUND_SUBTRACTION")[0]
        pixels_per_bin = binning_factors + 1
        
        orders = np.asfarray([findOrder(channel,aotf_frequency,silent=True) for aotf_frequency in list(aotf_frequencies)])
        
        """plot single frame"""
        frame_number = 100 #choose a single frame
        frame_data = detector_data_fullscan[frame_number,:,:] #get data for chosen frame only
        plt.figure(figsize = (10,8))
        plt.imshow(frame_data)
        plt.colorbar()
        plt.title("Frame %i" %frame_number)
        plt.xlabel("Detector horizontal (spectral) direction")
        plt.ylabel("Detector vertical (spatial) direction")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_fullscan_typical_frame_from_NEC.png" %channel)
        
        
        """plot single line"""
        line_data = detector_data_fullscan[frame_number,detector_centre,:]
        plt.figure(figsize = (10,8))
        plt.plot(line_data)
        plt.title("Frame %i Vertical Line %i" %(frame_number,detector_centre))
        plt.xlabel("Detector horizontal (spectral) direction")
        plt.ylabel("Signal ADU")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_fullscan_centre_line_spectrum_from_NEC.png" %channel)
        
        
        """remove bad pixels from line"""
        new_line_data = interpolate_bad_pixel(line_data,84)
        new_line_data = interpolate_bad_pixel(new_line_data,269)
        detector_data_fullscan,_,_ = get_dataset_contents(hdf5_file,"Y")
        hdf5_file.close() #close hdf5 file
        old_line_data = detector_data_fullscan[frame_number,detector_centre,:]
        
        """replot single line before and after bad pixel removal"""
        plt.figure(figsize = (10,8))
        plt.plot(old_line_data, label="Before")
        plt.plot(new_line_data, label="After")
        plt.legend()
        plt.title("Frame %i Vertical Line %i" %(frame_number,detector_centre))
        plt.xlabel("Detector horizontal (spectral) direction")
        plt.ylabel("Signal ADU")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_fullscan_centre_line_spectrum_bad_pixel_removal.png" %channel)
        
        
        """choose saturation value"""
        pixel_saturation_value = 12000
        
        
        """calculate required integration time so that pixel value equals saturation value"""
        max_pixel_fullscan = np.asfarray(np.max(detector_data_fullscan, axis=(1,2))) #find highest value pixel for each frame
        calculated_saturation_values = np.asfarray(pixel_saturation_value*(pixels_per_bin)*(number_of_accumulations/(background_subtractions+1))) #calculate pixel saturation level
        saturation_integration_times = (calculated_saturation_values/max_pixel_fullscan)*integration_times #calculate required integration time until pixel value = saturated value
        
        
        """plot integration time to saturation vs. aotf frequency"""
        plt.figure(figsize=(10,8))
        plt.plot(aotf_frequencies[aotf_frequencies>0],saturation_integration_times[aotf_frequencies>0],'.') #there are some dark frames in the file where aotf frequency=0. Don't plot these
        plt.yscale("log")
        plt.title("Integration Time Required for Saturation")
        plt.xlabel("AOTF Frequency (kHz)")
        plt.ylabel("Saturation integration time (ms)")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_saturation_integration_time.png" %channel)

        """plot integration time to saturation vs. order"""
        plt.figure(figsize=(10,8))
        plt.plot(orders[orders>50],saturation_integration_times[orders>50],'.') #there are some dark frames in the file where aotf frequency=0. Don't plot these
        plt.ylim((4,20))
        plt.title("Integration Time Required for Saturation")
        plt.xlabel("Diffraction Order")
        plt.ylabel("Saturation integration time (ms)")
        plt.grid()
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_saturation_integration_time_diff_order.png" %channel)
        
        
        """there are also some dark frames in the file. Use these to estimate SNR"""
        dark_centre_lines = detector_data_fullscan[aotf_frequencies==0,detector_centre,:] #get horizontal line from each dark frame where aotf frequency=0
        stdev_dark = np.std(dark_centre_lines, axis=0) #find standard deviation for each horizontal (spectral) pixel
        stdev_dark = interpolate_bad_pixel(stdev_dark,269) #interpolate over bad pixel
        stdev_dark = interpolate_bad_pixel(stdev_dark,84) #interpolate over bad pixel
        
        
        """plot snr for a typical frame"""
        chosen_frame=200
        light_line = detector_data_fullscan[chosen_frame,detector_centre,:]
        light_line = interpolate_bad_pixel(light_line,269) #interpolate over bad pixel
        light_line = interpolate_bad_pixel(light_line,84) #interpolate over bad pixel
        snr = light_line / stdev_dark
        
        plt.figure(figsize=(10,8))
        plt.plot(snr,'.')
        plt.yscale("log")
        plt.title("Estimated Signal-to-Noise Ratio for frame %i" %chosen_frame)
        plt.xlabel("Pixel Number")
        plt.ylabel("Centre line / Stdev of dark centre lines")
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_approx_SNR_without_shot_noise.png" %channel)
        
        del detector_data_fullscan #clear data from memory
        gc.collect() #clear memory allocation
        
if option==22:
    """analyse first nadir obs - prepared using a blackbody measurement done in air"""
    
    """get data from file, plot single detector frame"""
    detector_data = get_dataset_contents(hdf5_file,"Y")[0][0:116,:,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    sbsf_all = get_dataset_contents(hdf5_file,"BACKGROUND_SUBTRACTION")[0]

#    """correct for exponent"""
#    exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#    for index,frame in enumerate(detector_data):
#        detector_data[index,:,:] = frame[:,:]*2**exponent[index]
        
        
    apply_offset=False
    
    frame_numbers = [4]
    for frame_number in frame_numbers:
        frame_data = detector_data[frame_number,:,:] #get data for chosen frame only
    
        plt.figure(figsize = (10,8))
        plt.imshow(frame_data)
        plt.title("Frame %i" %frame_number)
        plt.xlabel("Detector horizontal (spectral) direction")
        plt.ylabel("Detector vertical (spatial) direction")
        plt.colorbar()
        #plt.show() #you may need to uncomment this, especially if working from the command line
        if save_figs: plt.savefig("%s_typical_detector_frane_NEC_int_time_stepping.png" %channel)

    """get data from file, plot single detector frame"""
    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0][0:116,:,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    """correct for exponent"""
#    exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#    for index,frame in enumerate(detector_data_bins):
#        detector_data_bins[index,:,:] = frame[:,:]*2**exponent[index]

    binned_detector_data = np.sum(detector_data_bins, axis=1) #sum all lines vertically
    zero_indices=range(0,20)+range(300,320) #assume mean of first and last 20 values are centred on zero. this will become offset
    offset_data = np.mean(binned_detector_data[:,zero_indices], axis=1) #calculate offset for every frame
    
    if apply_offset:
        for index,offset_value in enumerate(offset_data):
            binned_detector_data[index,:] = binned_detector_data[index,:] - offset_value #subtract offset from every summed detector line
    
    plt.figure(figsize = (10,8))
    
    plt.title("Vertically binned spectrum with offset subtracted")
    plt.xlabel("Pixel number")
    plt.ylabel("Signal value")
    plt.plot(np.transpose(binned_detector_data[8:10,:]))

    centre_indices=range(180,220)
    
    frame_means = np.mean(binned_detector_data[:,centre_indices], axis=1)
#    frame_means = np.max(binned_detector_data[:,:], axis=1)

    plt.figure(figsize = (10,8))
    plt.title("Sum of centre of vertically binned spectrum with offset subtracted")
    plt.xlabel("Frame number")
    plt.ylabel("Signal value")
    plt.plot(frame_means)
    
    if sum(sbsf_all)==0:
        dark_indices = np.where(aotf_freq_all==0)[0]
        dark_mean = np.mean(binned_detector_data[dark_indices,:], axis=0)
        dark_stdev = np.std(binned_detector_data[dark_indices,:], axis=0)
        light_sub = binned_detector_data[:,:]-dark_mean[:]    

    hdf5_file.close()
    
if option==23:
    """make animations"""
    from scipy.signal import argrelextrema
#    from scipy.interpolate import UnivariateSpline as spline
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter as sg
    """animate frames or plots"""
    
#    what_to_animate="frames"
    what_to_animate="lines"
    
    variable_changing="aotf"
    
    sum_vertically=True
#    sum_vertically=False
    sum_frames=True
#    sum_frames=False
#    animate=True
    animate=False
    
    line_number=11
    nframes_to_average=5

    """get data from file, plot single detector frame"""
    if what_to_animate=="frames":
        detector_data = get_dataset_contents(hdf5_file,"Y")[0][0:116,:,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        
    elif what_to_animate=="lines":
        if title=="LNO Fullscan":
            detector_data_sun = get_dataset_contents(hdf5_file,"YBins")[0][[59,174,289,519,404],:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
            exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
            for index,frame in enumerate(detector_data_sun):
                detector_data_sun[index,:,:] = frame[:,:]*2**exponent[index]
            detector_data_sun = np.sum(detector_data_sun, axis=(0,1))
            stop()

        detector_data = get_dataset_contents(hdf5_file,"YBins")[0][0:116,:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
    
    exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
    for index,frame in enumerate(detector_data):
        detector_data[index,:,:] = frame[:,:]*2**exponent[index]
    
    if variable_changing=="aotf":
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
        orders_all = [findOrder(channel,freq,silent=True) for index,freq in enumerate(aotf_freq_all)]
        wavenumbers = [spectral_calibration_simple(channel,order,silent=True) for order in orders_all]
    hdf5_file.close()

            
    if sum_vertically and what_to_animate != "frames":
        line_number=0
        detector_data = np.sum(detector_data, axis=1)

    
    if sum_frames and what_to_animate=="lines":
        detector_data_averaged = np.zeros((int(np.ceil(len(detector_data)/nframes_to_average)),320))
        for index in range(int(np.ceil(len(detector_data)/nframes_to_average))):
            index_start=index*nframes_to_average
            index_end=(index+1)*nframes_to_average
            detector_data_averaged[index,:] = np.mean(detector_data[index_start:index_end,:], axis=0)
#            plt.plot(wavenumbers[0],detector_data_1_frame, color="k")
#            plt.xlabel("Wavenumber cm-1")
#            plt.ylabel("Detector signal ADU")
#            plt.title("%s: Mean of all frames" %(title))

        detector_data = detector_data_averaged
    n_frames = len(detector_data)
    max_value=np.nanmax(detector_data)
    fig=plt.figure(1, figsize=(10,8))
    num=0
    
    if what_to_animate=="frames": 
        plot = plt.imshow(detector_data[num,:,:], vmin=0, vmax=max_value, animated=True, interpolation=None, extent=[min(wavenumbers[num]),max(wavenumbers[num]),255,0], aspect=0.1)
        plotbar = plt.colorbar()
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Detector vertical (spatial) direction")
    if what_to_animate=="lines":
        plt.ylim((0,max_value))
        if sum_vertically:
            if animate:
                plot, = plt.plot(wavenumbers[num],detector_data[num,:], color="k", animated=True)
            else:
                plot, = plt.plot(wavenumbers[num],detector_data[num,:], color="k")
        else:
            if animate:
                plot, = plt.plot(wavenumbers[num],detector_data[num,line_number,:], color="k", animated=True)
            else:
                plot, = plt.plot(wavenumbers[num],detector_data[num,line_number,:], color="k")
        
    if variable_changing=="aotf":
        plottitle = plt.title("%s: Frame %i AOTF %0.0f kHz" %(title,num,aotf_freq_all[num]))

    if animate:
        def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
            global plot,plottitle#,detector_data,variable_changing,what_to_animate,line_number,sum_vertically
            if np.mod(num,50)==0:
                print(num)
            if what_to_animate=="frames": 
                plot.set_array(detector_data[num,:,:])
            elif what_to_animate=="lines": 
                if sum_vertically:
                    plot.set_data(wavenumbers[num], detector_data[num,:])
                else:
                    plot.set_data(wavenumbers[num], detector_data[num,line_number,:])
            if variable_changing=="aotf":
                plottitle.set_text("%s: Frame %i Exponent %i AOTF %0.0f kHz" %(title,num,exponent[num],aotf_freq_all[num]))
            return plot,
                
        ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=50, blit=True)
        if save_figs: ani.save(title+"_detector_%s.mp4" %what_to_animate, fps=20, extra_args=['-vcodec', 'libx264'])
        plt.show()
        
   
    sg_lengths = [5,11]    
    interp_kinds = ["cubic","linear","slinear"]
#    y 
    
    plt.figure(2, figsize=(10,8))
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Normalised radiance")
#    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
    plt.title("Typical nadir spectrum from LNO using various fitting types")
    
    plt.figure(3, figsize=(10,8))
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Signal ADUs")
#    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
    plt.title("Fitting points to nadir curves")
    

    for sg_length in sg_lengths:
        for interp_kind in interp_kinds:
    
            #remove very noisy wings from spectrum
            detector_line = detector_data[0,:]
            wavenumber_range = wavenumbers[0]
            adu_cutoff = 3000
            detector_line_centre = detector_line[np.where(detector_line[:]>adu_cutoff)[0]]
            wavenumber_centre = wavenumber_range[np.where(detector_line[:]>adu_cutoff)[0]]
            #smooth data, remove gross variations
            detector_line_presmooth = sg(detector_line_centre, window_length=sg_length, polyorder=3)
            #then find local maxima and fit to these
            loc_max_indices = list(argrelextrema(detector_line_presmooth, np.greater)[0]) #find indices of local maxima
           
            plt.figure(3)
            plt.scatter(wavenumber_centre[loc_max_indices],detector_line_presmooth[loc_max_indices])
            
        #    spl = spline(wavenumbers[0][loc_max_indices],detector_data[0,loc_max_indices], w=detector_data[0,loc_max_indices])
        #    y_new = spl(wavenumbers[0])
        #    plt.plot(wavenumbers[0],y_new)
            
            spl = interp1d(wavenumber_centre[loc_max_indices],detector_line_centre[loc_max_indices], kind=interp_kind)
            wavenumbers_bounded = wavenumber_centre[min(loc_max_indices):max(loc_max_indices)]    
            detector_data_bounded = detector_line_centre[min(loc_max_indices):max(loc_max_indices)]
        
            y_new = spl(wavenumbers_bounded)
            plt.plot(wavenumbers_bounded,y_new, label="%s-%s" %(interp_kind,sg_length))
            
            y_above = y_new[:]
            for index in range(len(y_above)):
                if y_above[index] < detector_data_bounded[index]:
                    y_above[index] = detector_data_bounded[index]
            plt.plot(wavenumbers_bounded,y_above)
            
            plt.figure(2)
            plt.plot(wavenumbers_bounded,detector_data_bounded/y_above)
    
    plt.legend()

    plt.figure(4, figsize=(10,8))
    for index in range(len(detector_data[:,0]))[1::]: #first spectrum bad
        plt.plot(wavenumbers[0],detector_data[index,:]/detector_data_sun)
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Radiance factor vs Sun")
#    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
    plt.title("Typical nadir spectrum from LNO divided by solar spectrum")
    
    if save_figs:
        os.chdir(BASE_DIRECTORY)
        plt.figure(2)
        plt.savefig("Typical_nadir_spectrum_from_LNO.png")
        plt.figure(3)
        plt.savefig("Fitting_points_to_nadir_curves.png")
        plt.figure(4)
        plt.savefig("Typical_nadir_spectrum_from_LNO_divided_by_solar_spectrum.png")
#    poly_factors = np.polyfit(wavenumbers[0][loc_max_indices], detector_data[0,loc_max_indices], 5, w=detector_data[0,loc_max_indices]*detector_data[0,loc_max_indices])
#    poly = np.poly1d(poly_factors)
#    y_new = poly(wavenumbers[0])
#    plt.plot(wavenumbers[0],y_new)

#    filtered_data = sg_filter(detector_data[0,loc_max_indices],window_size=25,order=1)
#    plt.plot(wavenumbers[0][loc_max_indices],filtered_data)


    
if option==24:
    """plot solar fullscan spectrum for Severine simulation"""
    indices=[123,228]
    line=16
    
    detector_data = get_dataset_contents(hdf5_file,"YBins")[0][indices[0]:(indices[1]+1),:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
    exponent = get_dataset_contents(hdf5_file,"EXPONENT")[0][indices[0]:(indices[1]+1)]
    for index,frame in enumerate(detector_data):
        detector_data[index,:,:] = frame[:,:]*2**exponent[index]
    
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0][indices[0]:(indices[1]+1)]
    orders_all = [findOrder(channel,freq,silent=True) for index,freq in enumerate(aotf_freq_all)]
    wavenumbers = [spectral_calibration_simple(channel,order,silent=True) for order in orders_all]
    hdf5_file.close()
    
#    plt.figure()
#    plt.plot(detector_data[80,:,200])
    
    plt.figure(figsize=(10,8))
    for index,wavenumber_range in enumerate(wavenumbers): #first spectrum bad
        plt.plot(wavenumber_range,detector_data[index,line,:])
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Sun signal ADUs for spectrum line %i" %line)
    title = "%s: Orders %i to %i" %(title,min(orders_all),max(orders_all)) 
    plt.title(title)
    if save_figs:
        os.chdir(BASE_DIRECTORY)
        plt.savefig(title.replace(" ","_").replace(":","").replace(",","")+".png")
    if save_files:
        os.chdir(BASE_DIRECTORY)
        write_log(title.replace(" ","_").replace(":","").replace(",",""),"Wavenumber,Signal on line %i" %line, silent=True)
        for order_index,wavenumber_range in enumerate(wavenumbers): #first spectrum bad
            for pixel_index,wavenumber in enumerate(wavenumber_range):
                write_log(title.replace(" ","_").replace(":","").replace(",",""),"%0.3f,%0.1f" %(wavenumber,detector_data[order_index,line,pixel_index]), silent=True)
    
if option==25:
    """compare expected radiances of a calibration bb measurement and a radiance simulation"""
    input_file = r"C:\Users\iant\Documents\Python\snr\dataRad_IR_from_Ann_Carine\input_files_ACV\LNO_nadir_LS251_SzaVar_HighAlb.dat"   #in W/cm2/(cm-1)/sr
    
    k=1.38065040000E-23
    h=6.62607550000E-34
    c_speed=2.99792458000E+8 #m/s

    def typical_radiances_waven(): #output a typical radiance file in cm-1 and W/cm2/(cm-1)/sr
        global input_file
        inputradiancefile=np.loadtxt(input_file)
        input_radiance=inputradiancefile[:,1]
        input_radiance_waven=inputradiancefile[:,0]
        return input_radiance_waven,input_radiance
   
    def planck_waven2(wavencm,temp): #cm-1 in planck function W/cm2/(cm-1)/sr
        global h, c_speed, k
        wavenm = wavencm * 100.0
        return ((2.0 * h * c_speed**2.0) * wavenm**3.0) /(np.exp(h * c_speed * wavenm / (k * temp)) - 1.0) * 100 / 10000.0
        
    detector_data = get_dataset_contents(hdf5_file,"YBins")[0][0:116,:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
    
    exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
    for index,frame in enumerate(detector_data):
        detector_data[index,:,:] = frame[:,:]*2**exponent[index]
    
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    orders_all = [findOrder(channel,freq,silent=True) for index,freq in enumerate(aotf_freq_all)]
    wavenumbers = [spectral_calibration_simple(channel,order,silent=True) for order in orders_all]
    integration_time = get_dataset_contents(hdf5_file,"INTEGRATION_TIME")[0]
    n_acc = get_dataset_contents(hdf5_file,"NUMBER_OF_ACCUMULATIONS")[0]
    hdf5_file.close()
    
    print("Integration time = %0.1f" %integration_time[0])
    print("Number of accumulations = %0.1f" %n_acc[0])
    print(n_acc[0]*integration_time[0])
    
    
    bb=planck_waven2(wavenumbers[0],150+273)
    mars_waven,mars=typical_radiances_waven()
    
    plt.figure(figsize=(10,8))
    plt.plot(wavenumbers[0],bb)
    plt.scatter(mars_waven[1728:1758],mars[1728:1758])
    
    
if option==26:
    """read in MCC temperatures"""
    import spiceypy as sp
    from scipy.interpolate import UnivariateSpline

    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    
   
    os.chdir(r"C:\Users\iant\Documents\Python\Ops\mcc\temperatures")
    filenames = get_filename_list("tab")
    
    time_strings=[]
    utc_times=[]
    so_nom=[]
    lno_nom=[]
    so_red=[]
    lno_red=[]
    uvis_red=[]
    for filename in filenames:
        with open(filename) as f:
            for index,line in enumerate(f):
                content = line.strip("\n").split()
                time_strings.append(content[0])
                utc_times.append(content[1])
                so_nom.append(content[2])
                lno_nom.append(content[3])
                so_red.append(content[4])
                lno_red.append(content[5])
                uvis_red.append(content[6])

    utc_times = np.asfarray([sp.utc2et(time_string.replace("T"," ") + " UTC") for time_string in time_strings])
    so_nom = np.asfarray(so_nom)
    lno_nom = np.asfarray(lno_nom)
    so_red = np.asfarray(so_red)
    lno_red = np.asfarray(lno_red)
    uvis_red = np.asfarray(uvis_red)
    
    plt.figure(figsize=(14,8))
    plt.plot(utc_times,so_nom,label="SO NOM")
    plt.plot(utc_times,so_red,label="SO RED")
    plt.plot(utc_times,lno_nom,label="LNO NOM")
    plt.plot(utc_times,lno_red,label="LNO RED")
    plt.plot(utc_times,uvis_red,label="UVIS RED")
    plt.legend()

    spl = UnivariateSpline(utc_times,so_nom)
    spl.set_smoothing_factor(100)
    so_nom_smooth=spl(utc_times)
    
    minmax = np.diff(np.sign(np.diff(so_nom_smooth))).nonzero()[0] + 1 # local min+max
    for index in list(minmax):
        plt.text(utc_times[index],so_nom[index], time_strings[index])


if option==27:
    """analyse lno dayside nadir"""
    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_files[0],"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_files[0],"AOTF_FREQUENCY")[0]
    time_data_all = get_dataset_contents(hdf5_files[0],"Observation_Time")[0][:,0]
    hdf5_files[0].close()

    binned_detector_data = np.sum(detector_data_bins, axis=1) #sum all lines vertically
#    aotf_freq_subd = aotf_freq_all[0]
#    aotf_freq_subd = aotf_freq_all[1]
#    aotf_freq_subd = aotf_freq_all[2]
    aotf_freq_subd = aotf_freq_all[3]
    binned_data_subd = np.asfarray([frame for index,frame in enumerate(list(binned_detector_data)) if aotf_freq_all[index]==aotf_freq_subd])
    time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd]

    order = findOrder(channel,aotf_freq_subd,silent=True)
    wavenumbers = spectral_calibration_simple(channel,order,silent=True)
    
#    zero_indices=range(0,20)+range(300,320) #assume mean of first and last 20 values are centred on zero. this will become offset
#    offset_data = np.mean(binned_detector_data[:,zero_indices], axis=1) #calculate offset for every frame
#    
#    if apply_offset:
#        for index,offset_value in enumerate(offset_data):
#            binned_detector_data[index,:] = binned_detector_data[index,:] - offset_value #subtract offset from every summed detector line
    
#    plt.figure(figsize = (10,8))
#    plt.title("Vertically binned spectra")
#    plt.xlabel("Pixel number")
#    plt.ylabel("Signal value")
#    plt.plot(np.transpose(binned_data_subd[:,:]))

#    plt.figure(figsize = (10,8))
#    plt.title("Vertically binned spectra frame 140")
#    plt.xlabel("Pixel number")
#    plt.ylabel("Signal value")
#    plt.plot(np.transpose(binned_data_subd[140,:]))

    plt.figure(figsize = (10,8))
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Signal value")
#    frame_ranges=[[90,190]]
#    frame_ranges=[[70,90],[90,110],[110,130],[130,150],[150,170],[170,190],[190,210]] #lno dayside
    frame_ranges=[[110,120],[120,130],[130,140],[140,150],[150,151],[150,160],[160,170],[170,180]] #lno dayside
#    frame_ranges=[[50,70],[70,90],[90,110],[110,130],[130,150]] #lno inertial dayside

#    frame_ranges=[[40,45],[45,50],[50,55],[55,60],[60,65],[65,70]] #lno limb 1 (day)
#    frame_ranges=[[140,145],[155,160],[160,165],[165,170]] #lno limb 2 (night)
    range_title = "Summed vertically binned spectra order %i\nframes " %order
    for frame_range in frame_ranges:
        range_title=range_title + "%i-%i," %(frame_range[0],frame_range[1])
        summed_binned_frames_subd = np.mean(binned_data_subd[range(frame_range[0],frame_range[1]),:],axis=0)
#        plt.plot(wavenumbers,summed_binned_frames_subd, label="Frames %i-%i" %(frame_range[0],frame_range[1]))
        plt.plot(wavenumbers,summed_binned_frames_subd, label=time_data_subd[int(np.mean([frame_range[0],frame_range[1]]))])
    plt.title(range_title)
#    plt.title("NOMAD LNO Infrared Spectra of Mars, 22 November 2016")
    plt.legend()
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO_spectra_mars_xx_November_2016.png", dpi=400)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"order %i.png" %order)
    

    """try to calibrate with lno fullscan measurement"""
    """get needed data from file then close file"""
    detector_data_fullscan,_,_ = get_dataset_contents(hdf5_files[1],"YBins") #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    integration_times = get_dataset_contents(hdf5_files[1],"INTEGRATION_TIME")[0] #get integration times from file for each frame
    aotf_frequencies = get_dataset_contents(hdf5_files[1],"AOTF_FREQUENCY")[0]
    number_of_accumulations = get_dataset_contents(hdf5_files[1],"NUMBER_OF_ACCUMULATIONS")[0]
    binning_factors = get_dataset_contents(hdf5_files[1],"BINNING")[0] #get binning factor from file
    background_subtractions = get_dataset_contents(hdf5_files[1],"BACKGROUND_SUBTRACTION")[0]
    pixels_per_bin = binning_factors + 1

    fullscan_frames = np.asfarray([frame for index,frame in enumerate(list(detector_data_fullscan)) if aotf_frequencies[index]==aotf_freq_subd])
    fullscan_line = fullscan_frames[10,8,:]
    
    plt.figure(figsize = (10,8))
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Signal value")
    for frame_range in frame_ranges:
        summed_binned_frames_subd = np.mean(binned_data_subd[range(frame_range[0],frame_range[1]),:],axis=0)
        plt.plot(wavenumbers,summed_binned_frames_subd/fullscan_line, label=time_data_subd[int(np.mean([frame_range[0],frame_range[1]]))])
    plt.title(range_title)
    plt.legend()
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"_rad_factor_order_%i.png" %order)
   

if option==28:
    """analyse uvis dayside nadir"""
    
    """get data from file"""
    detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get data
#    data_name_all = get_dataset_contents(hdf5_file,"Name")[0] #get data
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0]
    hdf5_file.close()
    
    reverse_clock = detector_data_all[0,0:2,:]
    
    i=1;   bias1 = detector_data_all[i,1,:]
    i=2;   dark1 = detector_data_all[i,1,:]
    i=260; bias2 = detector_data_all[i,1,:]
    i=43; dark2 = detector_data_all[i,1,:]

#    i=range(3,43)[::5]
    i=range(3,15)[::2]
    
    dark_mean = np.mean([dark1,dark2],axis=0)
    
    wavelengths = spectral_calibration_simple(channel,"")    

    light = np.zeros((len(i),1048))
    light_sub = np.zeros((len(i),1048))
    light_norm = np.zeros((len(i),1048))
    light_sub_norm = np.zeros((len(i),1048))
    light_sub_norm_diff = np.zeros((len(i),1048))
    fig1 = plt.figure(figsize=(18,10))
    ax1 = plt.subplot2grid((4,1),(0,0))
    ax2 = plt.subplot2grid((4,1),(1,0))
    ax3 = plt.subplot2grid((4,1),(2,0))
    ax4 = plt.subplot2grid((4,1),(3,0))
    for index,frame_no in enumerate(i):
        light[index,:] = detector_data_all[frame_no,1,:]
        light_sub[index,:] = light[index,:] - dark_mean
        light_norm[index,:] = (light[index,:]-min(light[index,:]))/(max(light[index,:])-min(light[index,:]))
        light_sub_norm[index,:] = (light_sub[index,:]-min(light_sub[index,:]))/(max(light_sub[index,:])-min(light_sub[index,:]))
        light_sub_norm_diff[index,:] = light_sub_norm[index,:] - light_sub_norm[0,:]
        ax1.plot(wavelengths,light[index,:], label=time_data_all[index])
        ax2.plot(wavelengths,light_sub[index,:], label=time_data_all[index])
        ax3.plot(wavelengths,light_sub_norm[index,:], label=time_data_all[index])
        ax4.plot(wavelengths,light_sub_norm_diff[index,:], label=time_data_all[index])
    ax1.set_title(title+ ": raw light frames")
    ax2.set_title(title+ ": light frames - mean of 2 closest dark frames")
    ax3.set_title(title+ ": as above, normalised between 0 and 1")
    ax4.set_title(title+ ": as above, subtracted from first normalised frame")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.set_xlim((200,750))
    ax2.set_xlim((200,750))
    ax3.set_xlim((200,750))
    ax4.set_xlim((200,750))
    if save_figs: fig1.savefig(BASE_DIRECTORY+os.sep+title+".png")
        
    plt.figure(figsize=(12,6))
    for index,frame_no in enumerate([10]):
        plt.plot(wavelengths,light_sub_norm[index,:], label=time_data_all[index])
#    plt.legend()
    plt.title("NOMAD UVIS Spectra of Mars, 22 November 2016")
    plt.ylabel("Signal from Mars, normalised to 1")
    plt.xlabel("Wavelength (nanometres)")
    plt.xlim((205,650))
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"UVIS_spectra_mars_22_November_2016.png", dpi=400)
        
if option==29:
    """analyse lno limb scan 1"""
    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    data_name_all = get_dataset_contents(hdf5_file,"Name")[0] #get data
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0]
    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0]
    hdf5_file.close()
    
    chosen_range = [50,270]
    zero_indices = range(0,20)+range(300,320)
    
    mean_offsets = np.zeros_like(detector_data_bins)
    
    mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
    for column_index in range(320):
        mean_offsets[:,:,column_index] = mean_offset
    
    offset_data_bins = detector_data_bins - mean_offsets
    
    spec_summed_data = np.sum(detector_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
    offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
    
    frame_number = 0
    plt.figure(figsize = (10,8))
    plt.imshow(detector_data_bins[frame_number,:,:])
    plt.title("Frame %i" %frame_number)
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Detector vertical (spatial) direction")
    plt.colorbar()

    frame_number = 0
    plt.figure(figsize = (10,8))
    plt.imshow(offset_data_bins[frame_number,:,:])
    plt.title("Frame %i" %frame_number)
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Detector vertical (spatial) direction")
    plt.colorbar()

#    binned_detector_data = np.sum(detector_data_bins, axis=1) #sum all lines vertically
#    detector_data_sum = np.sum(binned_detector_data[:,], axis=1)

#    plt.figure(figsize = (10,8))
#    plt.xlabel("Time")
#    plt.ylabel("Signal sum")
#    for detector_row in list(np.transpose(spec_summed_data)): #plot each row separately
#        plt.plot(detector_row)
#    plt.title(title+": Detector rows versus time")
##    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+".png")

    plt.figure(figsize = (10,8))
    plt.xlabel("Time")
    plt.ylabel("Signal sum")
    for row_index in [2,8,14,20]: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
        plt.plot(offset_spec_summed_data[:,row_index],"o", linewidth=0, label="%i" %row_index)
    plt.legend()
    plt.title(title+": Detector rows versus time")
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+".png")



if option==30:
    """try to find phobos in the LNO data"""
    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0][450:500,:,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0][450:500]
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0][450:500]
    hdf5_file.close()

#    aotf_freq_subd = aotf_freq_all[0]
    aotf_freq_subd = aotf_freq_all[1]
    order = findOrder(channel,aotf_freq_subd,silent=True)

    detector_data_subd = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
    time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd]

    chosen_range = [50,270]
    zero_indices = range(0,20)+range(300,320)
    
    mean_offsets = np.zeros_like(detector_data_subd)
    
    mean_offset = np.mean(detector_data_subd[:,:,zero_indices], axis=2)
    for column_index in range(320):
        mean_offsets[:,:,column_index] = mean_offset
    
    offset_data_bins = detector_data_subd - mean_offsets
    
    spec_summed_data = np.sum(detector_data_subd[:,:,chosen_range[0]:chosen_range[1]], axis=2)
    offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)


    plt.figure(figsize=(10,8))
    plt.xlabel("Frame")
    plt.ylabel("Signal sum")
    for row_index in range(len(offset_spec_summed_data[0,:])): #plot each row separately
#        plt.plot(offset_spec_summed_data[:,row_index],"o", linewidth=0, label="%i" %row_index)
        plt.plot(offset_spec_summed_data[:,row_index], label="%i" %row_index)
    for index,time_data in enumerate(time_data_subd):
        if np.mod(index,5)==0:
            plt.text(index,-1000,time_data)
    plt.legend()
    plt.title(title+": detector rows versus time order %i" %order)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"_detector_rows_versus_time_order_%i.png" %order)
    


if option==31:
    """try to find phobos in the UVIS data"""
    
    """get data from file"""
    start=200
    detector_data = get_dataset_contents(hdf5_file,"Y")[0][start:250,0,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    name_all = get_dataset_contents(hdf5_file,"Name")[0][200:260] #get data
    time_data = get_dataset_contents(hdf5_file,"Observation_Time")[0][start:250]
#    hdf5_file.close()
    
    summed_detector_data = np.sum(detector_data, axis=1)

    wavelengths = spectral_calibration_simple(channel,"")    


    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((1,1),(0,0))
    fig2 = plt.figure(figsize=(10,8))
    ax2 = plt.subplot2grid((1,1),(0,0))
    fig3 = plt.figure(figsize=(10,8))
    ax3 = plt.subplot2grid((1,1),(0,0))
    fig4 = plt.figure(figsize=(10,8))
    ax4 = plt.subplot2grid((1,1),(0,0))
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Signal")
    for index,time_point in enumerate(time_data):
#        ax1.text(index,1.94e8,time_point)
        if index in [208-start,209-start,210-start]:
            ax2.plot(wavelengths, detector_data[index,:]-detector_data[208-start,:], label="frame %i, %s" %(start+index,time_point))
        if index in [222-start,223-start,224-start,225-start]:
            ax3.plot(wavelengths, detector_data[index,:]-detector_data[222-start,:], label="frame %i, %s" %(start+index,time_point))
        if index in [236-start,237-start,238-start]:
            ax4.plot(wavelengths, detector_data[index,:]-detector_data[236-start,:], label="frame %i, %s" %(start+index,time_point))
    ax1.plot(range(start,start+len(summed_detector_data)),summed_detector_data)
    ax1.set_title(title+": sum of signal versus time")
    if save_figs: fig1.savefig(BASE_DIRECTORY+os.sep+title+" sum of signal versus time.png")

    ax2.set_title(title+": spectra in blip 1 subtracted from spectrum before blip 1")
    ax2.set_ylim((-500,5000))
    ax2.legend()
    if save_figs: fig2.savefig(BASE_DIRECTORY+os.sep+title+" spectra in blip 1 subtracted from spectrum before blip 1.png")

    ax3.set_title(title+": spectra in blip 2 subtracted from spectrum before blip 2")
    ax3.set_ylim((-500,5000))
    ax3.legend()
    if save_figs: fig3.savefig(BASE_DIRECTORY+os.sep+title+" spectra in blip 2 subtracted from spectrum before blip 2.png")

    ax4.set_title(title+": spectra in blip 3 subtracted from spectrum before blip 3")
    ax4.set_ylim((-500,5000))
    ax4.legend()
    if save_figs: fig4.savefig(BASE_DIRECTORY+os.sep+title+" spectra in blip 3 subtracted from spectrum before blip 3.png")
    



if option==32:
    """check lno dayside nadir dataset"""
    
    """get data from file"""
    start_frame=145 #frames counted after aotf selection
    end_frame=150
    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
    hdf5_file.close()

#    aotf_freq_subd = aotf_freq_all[0]
    aotf_freq_subd = aotf_freq_all[1]
#    aotf_freq_subd = aotf_freq_all[2]
#    aotf_freq_subd = aotf_freq_all[3]
    binned_data_subd = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd][start_frame:end_frame])
    time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd][start_frame:end_frame]

    order = findOrder(channel,aotf_freq_subd,silent=True)
    wavenumbers = spectral_calibration_simple(channel,order,silent=True)
    
    binned_data_subd_corrected = np.zeros_like(binned_data_subd)
    binned_data_subd_corrected[:,:,:] = binned_data_subd[:,:,:]
    for frame_number in range(binned_data_subd_corrected.shape[0]):
        for line_number in range(binned_data_subd_corrected.shape[1]):
            if line_number==0:
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],82)
            elif line_number==1:
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],29)
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],64)
            if line_number==4:
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],83)
            elif line_number==5:
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],47)
                binned_data_subd_corrected[frame_number,line_number,:] = interpolate_bad_pixel(binned_data_subd_corrected[frame_number,line_number,:],76)
    
#    zero_indices=range(0,20)+range(300,320) #assume mean of first and last 20 values are centred on zero. this will become offset
#    offset_data = np.mean(binned_detector_data[:,zero_indices], axis=1) #calculate offset for every frame
#    
#    if apply_offset:
#        for index,offset_value in enumerate(offset_data):
#            binned_detector_data[index,:] = binned_detector_data[index,:] - offset_value #subtract offset from every summed detector line
    
    fig1 = plt.figure(figsize=(12,12))
    for frame_number in range(binned_data_subd.shape[0]):
        ax = plt.subplot2grid((binned_data_subd.shape[0]+1,1),(frame_number,0),)
        ax.plot(np.transpose(binned_data_subd[frame_number,:,:]), label="%i" %frame_number)
#    ax.legend()
        ax.set_title(title+": Prior to Bad Pixel Correction")
    plt.tight_layout()
    if save_figs: plt.savefig(title.replace(" ","_")+ "_prior_to_bad_pixel_correction.png")

    fig2 = plt.figure(figsize=(12,12))
    for frame_number in range(binned_data_subd.shape[0]):
        ax = plt.subplot2grid((binned_data_subd.shape[0]+1,1),(frame_number,0),)
        ax.plot(np.transpose(binned_data_subd_corrected[frame_number,:,:]))
        ax.set_title(title+": After Bad Pixel Correction")
    plt.tight_layout()
    if save_figs: plt.savefig(title.replace(" ","_")+ "_after_bad_pixel_correction.png")

    binned_frame = np.sum(binned_data_subd, axis=1)
    binned_frame_range = np.sum(binned_frame, axis=0)

    binned_frame_corrected = np.sum(binned_data_subd_corrected, axis=1)
    binned_frame_range_corrected = np.sum(binned_frame_corrected, axis=0)
    
    plt.figure(figsize=(10,8))
    plt.plot(np.transpose(binned_frame))
    plt.plot(np.transpose(binned_frame_corrected)+1000)

    plt.figure(figsize=(10,8))
    plt.plot(binned_frame_range)
    plt.plot(binned_frame_range_corrected+5000)








if option==33:
    """nomad data workshop script: plot saturation time for all orders and estimate non-shot noise
    read in multiple datasets at one time"""
       
    """get data from file, plot single detector frame"""
    detector_data,_,_ = get_dataset_contents(hdf5_files[0],"Y") #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    frame_number = 10 #choose a single frame
    frame_data = detector_data[frame_number,:,:] #get data for chosen frame only
    detector_centre = 23
    
    plt.figure(figsize = (10,8))
    plt.imshow(frame_data)
    plt.title("Frame %i" %frame_number)
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Detector vertical (spatial) direction")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_typical_detector_frane_int_time_stepping.png")
    
    detector_centre_lines = detector_data[:,detector_centre,:] #get data for detector vertical line 130 for each frame
    max_pixel_binned = np.max(detector_centre_lines, axis=1) #find highest pixel on each line
    
        
    """get integration times from file, plot maximum value vs. integration time"""
    integration_times = get_dataset_contents(hdf5_files[0],"IntegrationTime")[0] #get integration times from file for each frame
    plt.figure(figsize = (10,8))
    plt.plot(integration_times,max_pixel_binned,'.')
    plt.title("Maximum pixel value vs. integration time for line %i" %detector_centre)
    plt.xlabel("Integration time (ms)")
    plt.ylabel("Maximum pixel value (ADU)")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_int_time_stepping_binned_max_value_each_frame.png")
    
    
    """get binning factor, divide maximum pixel by no. of pixels per bin"""
    binning_factors = get_dataset_contents(hdf5_files[0],"Binning")[0] #get binning factor from file
    pixels_per_bin = binning_factors + 1
    max_per_pixel = max_pixel_binned / pixels_per_bin #find highest pixel on each line
    plt.figure(figsize = (10,8))
    plt.plot(integration_times,max_per_pixel,'.')
    plt.title("Maximum pixel value vs. integration time for line %i" %detector_centre)
    plt.xlabel("Integration time (ms)")
    plt.ylabel("Maximum pixel value (ADU)")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_int_time_stepping_max_value_each_frame.png")
    
    
    hdf5_files[0].close() #close hdf5 file
    del detector_data #clear data from memory
    gc.collect() #clear memory allocation

        
    """get needed data from file then close file"""
    detector_data_fullscan,_,_ = get_dataset_contents(hdf5_files[1],"Y") #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    integration_times = get_dataset_contents(hdf5_files[1],"IntegrationTime")[0] #get integration times from file for each frame
    aotf_frequencies = get_dataset_contents(hdf5_files[1],"AOTFFrequency")[0]
    number_of_accumulations = get_dataset_contents(hdf5_files[1],"NumberOfAccumulations")[0]
    binning_factors = get_dataset_contents(hdf5_files[1],"Binning")[0] #get binning factor from file
    background_subtractions = get_dataset_contents(hdf5_files[1],"BackgroundSubtraction")[0]
    pixels_per_bin = binning_factors + 1
    detector_centre = 12
    
    orders = np.asfarray([findOrder(channel,aotf_frequency,silent=True) for aotf_frequency in list(aotf_frequencies)])
    
    """plot single frame"""
    frame_number = 100 #choose a single frame
    frame_data = detector_data_fullscan[frame_number,:,:] #get data for chosen frame only
    plt.figure(figsize = (10,8))
    plt.imshow(frame_data)
    plt.colorbar()
    plt.title("Frame %i" %frame_number)
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Detector vertical (spatial) direction")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_fullscan_typical_frame.png")
    
    
    """plot single line"""
    line_data = detector_data_fullscan[frame_number,detector_centre,:]
    plt.figure(figsize = (10,8))
    plt.plot(line_data)
    plt.title("Frame %i Vertical Line %i" %(frame_number,detector_centre))
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Signal ADU")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_fullscan_centre_line_spectrum.png")
    
    
    """remove bad pixels from line"""
    new_line_data = interpolate_bad_pixel(line_data,84)
    new_line_data = interpolate_bad_pixel(new_line_data,269)
    detector_data_fullscan,_,_ = get_dataset_contents(hdf5_files[1],"Y")
    hdf5_files[1].close() #close hdf5 file
    old_line_data = detector_data_fullscan[frame_number,detector_centre,:]
    
    """replot single line before and after bad pixel removal"""
    plt.figure(figsize = (10,8))
    plt.plot(old_line_data, label="Before")
    plt.plot(new_line_data, label="After")
    plt.legend()
    plt.title("Frame %i Vertical Line %i" %(frame_number,detector_centre))
    plt.xlabel("Detector horizontal (spectral) direction")
    plt.ylabel("Signal ADU")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_fullscan_centre_line_spectrum_bad_pixel_removal.png")
    
    
    """choose saturation value"""
    pixel_saturation_value = 12000
    
    
    """calculate required integration time so that pixel value equals saturation value"""
    max_pixel_fullscan = np.asfarray(np.max(detector_data_fullscan[:,detector_centre,:], axis=1)) #find highest value pixel for each frame
    calculated_saturation_values = np.asfarray(pixel_saturation_value*(pixels_per_bin)*(number_of_accumulations/(background_subtractions+1))) #calculate pixel saturation level
    saturation_integration_times = (calculated_saturation_values/max_pixel_fullscan)*integration_times #calculate required integration time until pixel value = saturated value
    
    
    """plot integration time to saturation vs. aotf frequency"""
    plt.figure(figsize=(10,8))
    plt.plot(aotf_frequencies[aotf_frequencies>0],saturation_integration_times[aotf_frequencies>0],'.') #there are some dark frames in the file where aotf frequency=0. Don't plot these
    plt.yscale("log")
    plt.title("Integration Time Required for Saturation")
    plt.xlabel("AOTF Frequency (kHz)")
    plt.ylabel("Saturation integration time (ms)")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_saturation_integration_time.png")
    if save_files: np.savetxt(title.replace(" ","_")+ "_saturation_integration_time.txt", np.transpose(np.asfarray([aotf_frequencies[aotf_frequencies>0],saturation_integration_times[aotf_frequencies>0]])), delimiter=",", header="AOTF frequency kHz,Milliseconds to saturation")

    """plot integration time to saturation vs. order"""
    plt.figure(figsize=(14,8))
    plt.plot(orders[orders>50],saturation_integration_times[orders>50],'.') #there are some dark frames in the file where aotf frequency=0. Don't plot these
    plt.ylim((2,20))
    plt.title("Integration Time Required for Saturation")
    plt.xlabel("Diffraction Order")
    plt.ylabel("Saturation integration time (ms)")
    plt.grid()
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_saturation_integration_time_diff_order.png")
    if save_files: np.savetxt(title.replace(" ","_")+ "_saturation_integration_time_diff_order.txt", np.transpose(np.asfarray([orders[orders>50],saturation_integration_times[orders>50]])), delimiter=",", header="Diffraction order,Milliseconds to saturation")
    
    
    """there are also some dark frames in the file. Use these to estimate SNR"""
    dark_centre_lines = detector_data_fullscan[aotf_frequencies==0,detector_centre,:] #get horizontal line from each dark frame where aotf frequency=0
    stdev_dark = np.std(dark_centre_lines, axis=0) #find standard deviation for each horizontal (spectral) pixel
    stdev_dark = interpolate_bad_pixel(stdev_dark,269) #interpolate over bad pixel
    stdev_dark = interpolate_bad_pixel(stdev_dark,84) #interpolate over bad pixel
    
    
    """plot snr for a typical frame"""
    chosen_frame=200
    light_line = detector_data_fullscan[chosen_frame,detector_centre,:]
    light_line = interpolate_bad_pixel(light_line,269) #interpolate over bad pixel
    light_line = interpolate_bad_pixel(light_line,84) #interpolate over bad pixel
    snr = light_line / stdev_dark
    
    plt.figure(figsize=(10,8))
    plt.plot(snr,'.')
    plt.yscale("log")
    plt.title("Estimated Signal-to-Noise Ratio for frame %i" %chosen_frame)
    plt.xlabel("Pixel Number")
    plt.ylabel("Centre line / Stdev of dark centre lines")
    #plt.show() #you may need to uncomment this, especially if working from the command line
    if save_figs: plt.savefig(title.replace(" ","_")+ "_approx_SNR_without_shot_noise.png")
    
    del detector_data_fullscan #clear data from memory
    gc.collect() #clear memory allocation



if option==34:
    """plot lno dayside animation with ground track"""
    import spiceypy as sp
    
    ref="J2000"
    abcorr="None"
    tolerance="1"
    method="Intercept: ellipsoid"
    formatstr="C"
    prec=3
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)

    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_files[0],"AOTFFrequency")[0]
    time_data_all = get_dataset_contents(hdf5_files[0],"ObservationTime")[0][:,0]
    date_data_all = get_dataset_contents(hdf5_files[0],"ObservationDate")[0][:,0]
    hdf5_files[0].close()

    binned_detector_data = np.sum(detector_data_bins, axis=1) #sum all lines vertically
    
#    aotf_freq_subd = aotf_freq_all[0]
#    aotf_freq_subd = aotf_freq_all[1]
#    aotf_freq_subd = aotf_freq_all[2]
    aotf_freq_subd = aotf_freq_all[3]
    data_subd = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
    binned_data_subd = np.asfarray([frame for index,frame in enumerate(list(binned_detector_data)) if aotf_freq_all[index]==aotf_freq_subd])
    time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd]
    date_data_subd = [date_out for index,date_out in enumerate(list(date_data_all)) if aotf_freq_all[index]==aotf_freq_subd]

    order = findOrder(channel,aotf_freq_subd,silent=True)
    wavenumbers = spectral_calibration_simple(channel,order,silent=True)

    times=convert_hdf5_time_to_spice_utc(time_data_subd,date_data_subd)
    target="MARS"
    observer="-143"
    subpoints=[sp.subpnt("Intercept: ellipsoid",target,eachtime,"IAU_MARS",abcorr,observer)[0] for eachtime in list(times)]
    coords = np.asfarray([sp.reclat(subpoint) for subpoint in subpoints])
    lonlats=coords[:,1:3] * sp.dpr()


    
#    n_frames = binned_data_subd.shape[0]
#    max_value=np.nanmax(binned_data_subd)
    n_frames = data_subd.shape[0]
    max_value=1000#np.nanmax(data_subd)
    mean_values=np.mean(data_subd, axis=(1,2)).astype(int)
    colors = cm.rainbow(np.linspace(0, 1, max(mean_values)+1))

    fig=plt.figure(1, figsize=(10,8))
    num=0

    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0))
    
    if 'dem32' not in locals():
        dem32=np.zeros((1152,576))+65535
        file_size=1152,5760 #reads in 10 horiz at a time, actually 11520 x 5760
        with open(os.path.normcase(r'C:\Users\iant\Documents\Python\orbit\megt90n000fb.img'), mode='rb') as dem32_file:
            for loop in range(file_size[1]):
                for loop2 in range(file_size[0]):
                    temp=struct.unpack('>HHHHHHHHHH', dem32_file.read(20))[9]
                    if (loop % 10):
                        if (temp > 2**15):
                            dem32[loop2][loop/10]=float(temp)-(2.0**16)#((float(temp)/2.0**16) * (8206.0+21181.0)-8206.0)
                        else:
                            dem32[loop2][loop/10]=float(temp)#((float(temp)/2.0**16) * (8206.0+21181.0)-8206.0)
        dem32 = np.transpose(np.fliplr(dem32))

    lons = np.arange(0.0, 360.0, 0.3125)
    lats = np.arange(-90, 90.0, 0.3125)
    lons, lats = np.meshgrid(lons, lats)

    m = Basemap(llcrnrlon=-180,llcrnrlat=-20,urcrnrlon=10,urcrnrlat=20,projection='mill')
    m.drawparallels(np.arange(-80,81,10),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])    
    m.contourf(lons, lats, dem32, shading='flat', latlon=True, cmap='terrain')
    x, y = m(lonlats[:,0],lonlats[:,1])
    plot2, = m.plot(x[num],y[num], marker='o', c=colors[mean_values[num]], animated=True)

    ax1.set_ylim((0,max_value))
#    plot, = ax1.plot(wavenumbers,binned_data_subd[num,:], color="k", animated=True)
    plota, = ax1.plot(wavenumbers,data_subd[num,0,:], animated=True)
    plotb, = ax1.plot(wavenumbers,data_subd[num,1,:], animated=True)
    plotc, = ax1.plot(wavenumbers,data_subd[num,2,:], animated=True)
    plotd, = ax1.plot(wavenumbers,data_subd[num,3,:], animated=True)
    plote, = ax1.plot(wavenumbers,data_subd[num,4,:], animated=True)
    plotf, = ax1.plot(wavenumbers,data_subd[num,5,:], animated=True)

    def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
        if np.mod(num,50)==0:
            print(num)
        plota.set_data(wavenumbers,data_subd[num,0,:])
        plotb.set_data(wavenumbers,data_subd[num,1,:])
        plotc.set_data(wavenumbers,data_subd[num,2,:])
        plotd.set_data(wavenumbers,data_subd[num,3,:])
        plote.set_data(wavenumbers,data_subd[num,4,:])
        plotf.set_data(wavenumbers,data_subd[num,5,:])

        plot2.set_data(x[0:num], y[0:num])
        plot2.set_color(colors[mean_values[num]])
#        plottitle.set_text("%s" %time_data_subd[num])
        return plot2,plota,plotb,plotc,plotd,plote,plotf,

    ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=50, blit=True)
#    if save_figs: ani.save(title+"_spectra_variability_order_%i.mp4" %order, fps=20, extra_args=['-vcodec', 'libx264'])
    if save_figs: ani.save(title+"_spectra_variability_order_%i.mp4" %order, fps=20, bitrate=1000)
    plt.show()

if option==35:
    """check ACS solar pointing test"""
    """get data from file"""
    dark_detector_data_bins = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    hdf5_files[0].close()

    file_number=1
    light_detector_data_bins = get_dataset_contents(hdf5_files[file_number],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    exponent_all = get_dataset_contents(hdf5_files[file_number],"Exponent")[0]
    hdf5_files[file_number].close()

    pixel_number=100
    bin_number=1
    
    exponent_values = 2.0**np.asfarray(list(exponent_all)[bin_number::][::4])

#    light_line1 = detector_data_bins[:,23,:]
#    dark_sum1=np.mean(dark_line1, axis=1)
#    light_sum1=np.mean(light_line1, axis=1)


    light1 = np.asfarray(list(light_detector_data_bins[:,pixel_number])[bin_number::][::4])
    dark1 = np.asfarray(list(dark_detector_data_bins[:,pixel_number])[bin_number::][::4])
    
    sub1 = light1-dark1
    sub_all = np.asfarray(list(light_detector_data_bins)[bin_number::][::4])-np.asfarray(list(dark_detector_data_bins)[bin_number::][::4])

    plt.figure(figsize=(10,8))
    plt.plot(dark1, label="Dark pixel %i" %pixel_number)
    plt.plot(light1, label="Light pixel %i" %pixel_number)
    plt.title(title+" "+obspaths[file_number])
    plt.legend()
    plt.xlabel("Frame Number")
    plt.ylabel("Signal on pixel %i" %pixel_number)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_dark_and_light.png")

    plt.figure(figsize=(10,8))
    plt.plot(exponent_values)
    plt.xlabel("Frame Number")
    plt.ylabel("Exponent")
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_exponent_value.png")

    plt.figure(figsize=(10,8))
#    plt.plot(sub1)
    plt.errorbar(range(len(sub1)),sub1,yerr=exponent_values, ecolor="r")
    plt.xlabel("Frame Number")
    plt.ylabel("Dark subtracted signal on pixel %i" %pixel_number)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_dark_sub.png")

    

    if title=="SO ACS Solar Pointing Test":
        frames_to_plot=range(50,200,20)
    elif title=='SO Raster A':
        frames_to_plot=range(100,120,2)+range(1040,1060,2)
    elif title=="SO Light to Dark":
        frames_to_plot=range(50,500,50)

    plt.figure(figsize=(10,8))
    for frame_to_plot in frames_to_plot:
        for subframe in range((frame_to_plot*4),(frame_to_plot*4+4),1):
            plt.plot(light_detector_data_bins[subframe,:], label="Frame=%i-%i" %(frame_to_plot,subframe))
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Detector Signal")
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+ "_spectra_comparison.png")


        
    plt.figure(figsize=(10,8))
    for frame_to_plot in frames_to_plot:
        plt.plot(sub_all[frame_to_plot], label="Frame=%s" %frame_to_plot)
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Background Subtracted Detector Signal")
#    plt.yscale("log")
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+ "_bg_sub_spectra.png")

if option==36:

    for hdf5_file in hdf5_files:

#        binned_detector_data = np.squeeze(get_dataset_contents(hdf5_file,"Y")[0]) #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        binned_detector_data = np.mean(get_dataset_contents(hdf5_file,"Y")[0],axis=1) #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0][:,0]
        hdf5_file.close()
        
        pxstart=160
        pxend=200
    
        for aotf_freq_subd in aotf_freq_all[1:4]:
#        for aotf_freq_subd in [aotf_freq_all[1]]:
            binned_data_subd = np.asfarray([frame for index,frame in enumerate(list(binned_detector_data)) if aotf_freq_all[index]==aotf_freq_subd])
            time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd]
        
            order = spectral_calibration("aotf2order",channel,aotf_freq_subd,0)
        #        plt.plot(np.sum(binned_data_subd[:,pxstart:pxend], axis=1), label=order)
    #        plt.legend()
        
            wavenumbers = spectral_calibration("pixel2waven",channel,order,-15.0)
            
        #    zero_indices=range(0,20)+range(300,320) #assume mean of first and last 20 values are centred on zero. this will become offset
        #    offset_data = np.mean(binned_detector_data[:,zero_indices], axis=1) #calculate offset for every frame
        #    
        #    if apply_offset:
        #        for index,offset_value in enumerate(offset_data):
        #            binned_detector_data[index,:] = binned_detector_data[index,:] - offset_value #subtract offset from every summed detector line
            
        #    plt.figure(figsize = (10,8))
        #    plt.title("Vertically binned spectra")
        #    plt.xlabel("Pixel number")
        #    plt.ylabel("Signal value")
        #    plt.plot(np.transpose(binned_data_subd[:,:]))
        
        #    plt.figure(figsize = (10,8))
        #    plt.title("Vertically binned spectra frame 140")
        #    plt.xlabel("Pixel number")
        #    plt.ylabel("Signal value")
        #    plt.plot(np.transpose(binned_data_subd[140,:]))
        
            plt.figure(figsize = (figx/2,figy/2))
            plt.xlabel("Wavenumber (cm-1)")
            plt.ylabel("Signal value")
#            frame_ranges=[[90,110],[130,150],[170,190]] #lno inertial dayside
            frame_ranges=[[10,30],[30,50],[50,70],[70,90],[90,110],[110,130]] #lno inertial dayside
        
            range_title = "Summed vertically binned spectra order %i\nframes " %order
            for frame_range in frame_ranges:
                range_title=range_title + "%i-%i," %(frame_range[0],frame_range[1])
                summed_binned_frames_subd = np.mean(binned_data_subd[range(frame_range[0],frame_range[1]),:],axis=0)
        #        plt.plot(wavenumbers,summed_binned_frames_subd, label="Frames %i-%i" %(frame_range[0],frame_range[1]))
                plt.plot(wavenumbers,summed_binned_frames_subd, label=time_data_subd[int(np.mean([frame_range[0],frame_range[1]]))])
            plt.title(title+": "+range_title)
#            plt.title("NOMAD LNO Infrared Spectra of Mars, 22 November 2016")
#            plt.title("NOMAD LNO Infrared Spectra of Mars, 6th March 2017")
            plt.legend()
        #    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO_spectra_mars_xx_November_2016.png", dpi=400)
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"order %i.png" %order)
    
    
    
    
    
if option==37:
    """analyse lno limb scan 1 and 2"""
   
    import spiceypy as sp
    ref="J2000"
    abcorr="None"
    tolerance="1"
    method="Intercept: ellipsoid"
    formatstr="C"
    prec=3
    
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)
    
    use_both_subdomains=1
    subdomain=1

    if title=="LNO Limb Scan 1":
        bins_to_use=range(12)
        scaler=4.0 #conversion factor between the two diffraction orders
        signal_cutoff=6000.0
        signal_peak=23000.0
    elif title=="LNO Limb Scan 2":
        bins_to_use=range(1,12)
        scaler=3.8 #conversion factor between the two diffraction orders
        if subdomain==0:
            signal_cutoff=2000.0
            signal_peak=8000.0
        elif subdomain==1:
            signal_cutoff=7000.0
            signal_peak=30000.0

    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
    time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
    date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
    binning_all = get_dataset_contents(hdf5_file,"Binning")[0]
    bins_all = get_dataset_contents(hdf5_file,"Bins")[0]
    hdf5_file.close()
    
    """assumes all bins identical!"""
    bins_mid = np.mean(bins_all[0,:,:],axis=1)
    offset_from_centre = bins_mid - detector_centre
    bin_size = float(binning_all[0]+1)
   
    epoch_times_start=convert_hdf5_time_to_spice_utc(list(time_data_all[:,0]),list(date_data_all[:,0]))
    epoch_times_end=convert_hdf5_time_to_spice_utc(list(time_data_all[:,1]),list(date_data_all[:,1]))
    epoch_times_mid = np.mean(np.asarray([epoch_times_start,epoch_times_end]),axis=0)
    
    
    aotf_freq_subd=aotf_freq_all[0]
    if use_both_subdomains==1:
        detector_data_bins = np.asfarray([frame*scaler if aotf_freq_all[index]==aotf_freq_subd else frame for index,frame in enumerate(list(detector_data_bins))])
    else:
        if title=="LNO Limb Scan 1":
            detector_data_bins = np.asfarray([frame*scaler for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
        elif title=="LNO Limb Scan 2":
            if subdomain==0:
                detector_data_bins = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
                epoch_times_mid = np.asfarray([et for index,et in enumerate(list(epoch_times_mid)) if aotf_freq_all[index]==aotf_freq_subd])
            elif subdomain==1:
                detector_data_bins = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]!=aotf_freq_subd])
                epoch_times_mid = np.asfarray([et for index,et in enumerate(list(epoch_times_mid)) if aotf_freq_all[index]!=aotf_freq_subd])
    

#    chosen_range = [50,270]
    chosen_range = [140,200]
    zero_indices = range(0,20)+range(300,320) #for scaling all spectra to a common zero level on first and last pixels
    
    mean_offsets = np.zeros_like(detector_data_bins)
    
    mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
    for column_index in range(320):
        mean_offsets[:,:,column_index] = mean_offset
    
    offset_data_bins = detector_data_bins - mean_offsets
    
    spec_summed_data = np.sum(detector_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
    offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
    frame_range = np.arange(len(offset_spec_summed_data))
    
    colours=["bo","go","ro","co","mo","yo","ko","bs","gs","rs","cs","ms","ys","ks","bd","gd"]
    line_colours=["b-","g-","r-","c-","m-","y-","k-","b-","g:","r:","c:","m:","y:","k:","b--","g--"]

    plt.figure(figsize = (figx,figy))
    plt.xlabel("Frame Number")
    plt.ylabel("Sum of signal from chosen detector region")
    
    limb_time=np.zeros((12,3))
    time_offset=np.zeros((12,3))
    offset_from_centre_corrected=np.zeros((12,3))
    offset_from_centre_uncorrected=np.zeros((12,3))

        
    for row_index in bins_to_use: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
    #    for row_index in [1,4,7,10]: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
    #    for row_index in [3,4]:
        summed_row = offset_spec_summed_data[:,row_index]
        plt.scatter(epoch_times_mid,summed_row, linewidth=0, label="Detector region %i" %row_index)
    
        start_index1=0
        start_index2=0
        start_range=np.min(np.where(summed_row[start_index1:]<signal_cutoff)[0])+start_index1
        end_range=np.min(np.where(summed_row[start_index2:]>signal_cutoff)[0])+start_index2
        dark_range1 = range(start_range,end_range)
    
        start_index1=85*(use_both_subdomains+1)
        start_index2=130*(use_both_subdomains+1)
        start_range=np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
        end_range=np.min(np.where(summed_row[start_index2:]<signal_cutoff)[0])+start_index2
        light_range1 = range(start_range,end_range)
        plt.plot(epoch_times_mid[light_range1],summed_row[light_range1],line_colours[row_index], label="Mars line 1 %i" %row_index)

        start_index1=170*(use_both_subdomains+1)
        start_index2=200*(use_both_subdomains+1)
        start_range=np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
        end_range=np.min(np.where(summed_row[start_index2:]<signal_cutoff)[0])+start_index2
        light_range3 = range(start_range,end_range)
        plt.plot(epoch_times_mid[light_range3],summed_row[light_range3],line_colours[row_index], label="Mars line %i" %row_index)

        start_index1=0
        limb_index1 = np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
        plt.plot(epoch_times_mid[limb_index1],summed_row[limb_index1],colours[row_index], label="Limb crossing line %i" %row_index)
        limb_ratio = summed_row[limb_index1]/signal_peak
        limb_time[row_index,0] = epoch_times_mid[limb_index1]
        print(limb_ratio)
        offset_from_centre_uncorrected[row_index,0] = offset_from_centre[row_index]
        offset_from_centre_corrected[row_index,0] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
        
        start_index1=130*(use_both_subdomains+1)
        if row_index==0: start_index1=140*(use_both_subdomains+1) #fudge to make first row work        
        
        limb_index2 = np.min(np.where(summed_row[start_index1:]<signal_cutoff)[0])+start_index1 -1
        plt.plot(epoch_times_mid[limb_index2],summed_row[limb_index2],colours[row_index], label="Limb crossing line %i" %row_index)
        limb_ratio = summed_row[limb_index2]/signal_peak
        limb_time[row_index,1] = epoch_times_mid[limb_index2]
        print(limb_ratio)
        offset_from_centre_uncorrected[row_index,1] = offset_from_centre[row_index]
        offset_from_centre_corrected[row_index,1] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
        
        start_index1=170*(use_both_subdomains+1)
        limb_index3 = np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
        plt.plot(epoch_times_mid[limb_index3],summed_row[limb_index3],colours[row_index], label="Limb crossing line %i" %row_index)
        limb_ratio = summed_row[limb_index3]/signal_peak
        limb_time[row_index,2] = epoch_times_mid[limb_index3]
        print(limb_ratio)
        offset_from_centre_uncorrected[row_index,2] = offset_from_centre[row_index]
        offset_from_centre_corrected[row_index,2] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
            
    if 0 not in bins_to_use:
        offset_from_centre_uncorrected=np.delete(offset_from_centre_uncorrected,0,axis=0)
        offset_from_centre_corrected=np.delete(offset_from_centre_corrected,0,axis=0)
        limb_time=np.delete(limb_time,0,axis=0)
    
    
#    plt.legend()
    plt.title("LNO Channel Limb Scan during Mars Capture Orbit Calibration Part 2")
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO Channel Limb Scan during Mars Capture Orbit Calibration Part 2.png", dpi=300)

    for crossing_index in range(3):

        plt.figure(figsize = (figx-7,figy-3))
        plt.xlabel("Offset in arcmins of detector bin from centre of FOV (negative is towards detector line 0)")
        plt.ylabel("Time of limb crossing (seconds)")
#        plt.plot(offset_from_centre_uncorrected[:,crossing_index],limb_time[:,crossing_index],label="Limb crossing time without signal correction")
        plt.plot(offset_from_centre_corrected[:,crossing_index],limb_time[:,crossing_index],label="Limb crossing time with signal correction")
        plt.ylim((min(limb_time[:,crossing_index]-1),max(limb_time[:,crossing_index]+2)))
    
        fit=np.polyfit(offset_from_centre_corrected[:,crossing_index],limb_time[:,crossing_index],1)
        fit_residual = np.sum(np.sqrt((limb_time[:,crossing_index] - np.polyval(fit,offset_from_centre_corrected[:,crossing_index]))**2))
        print(fit_residual)
        
        fitx = np.arange(-50,50,1)
        fity = np.polyval(fit,fitx)
        plt.plot(fitx,fity,label="Fit to corrected crossing time")
        plt.legend()
        
        crossing_et = np.polyval(fit,0)
        plt.plot(0,crossing_et,"ko")
        crossing_time = sp.et2utc(crossing_et,formatstr,prec)
        print(crossing_time)
        if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO Limb Scan Linear fit vs limb crossing time %i for each detector bin.png" %crossing_index)
        

if option==38:
    """analyse lno diffraction order stepping measurement"""
    
    """get data from file"""
    detector_data_bins = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
#    time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
#    date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
    incidence_angle_all = get_dataset_contents(hdf5_file,"IncidenceAngle",chosen_group="Geometry/Point0")[0]
    hdf5_file.close()

    incidence_angle_start = incidence_angle_all[:,0]
    lambertian_intensity = np.cos(np.deg2rad(incidence_angle_start))
    detector_data_binned = np.sum(detector_data_bins,axis=1)
    orders = np.asfarray([findOrder(channel,aotf_frequency,silent=True) for aotf_frequency in list(aotf_freq_all)])

    #sometimes the spectra in each bin are offset from one another. Normalise all offsets to zero, then do the vertical binning
    zero_indices = range(0,20)+range(300,320) #for scaling all spectra to a common zero level on first and last pixels
    mean_offsets = np.zeros_like(detector_data_bins)
    mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
    for column_index in range(320):
        mean_offsets[:,:,column_index] = mean_offset #make a 3d matrix of offsets
    offset_data_bins = detector_data_bins - mean_offsets
    offset_detector_data_binned = np.sum(offset_data_bins, axis=1)

    chosen_range = [150,250]

    detector_data_summed = np.sum(detector_data_binned[:,chosen_range], axis=1)
    offset_detector_data_summed = np.sum(offset_detector_data_binned[:,chosen_range], axis=1)
    
    lambertian_normalised = offset_detector_data_summed/lambertian_intensity
    
#    frames_to_plot=[80,130,180,200]
    frames_to_plot=range(25,30)
    plot_colours = ["b"]*114 + ["r"]*114 + ["g"]*52
    
    plt.figure(figsize = (10,8))
    plt.xlabel("Pixel number")
    plt.ylabel("Signal value")

    offset=-300
    for frame_to_plot in frames_to_plot:
        offset += 300
        plt.plot(detector_data_binned[frame_to_plot,:]+offset,label="%s" %orders[frame_to_plot])
    plt.legend()
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_raw_spectra_from_frames_%i_to_%i.png" %(min(frames_to_plot),max(frames_to_plot)))

#    plt.figure(figsize = (10,8))
#    plt.xlabel("AOTF frequency")
#    plt.ylabel("Summed signal value pixels %i to %i" %(chosen_range[0],chosen_range[1]))
#    plt.plot(aotf_freq_all,detector_data_summed,'b')
#
#    plt.figure(figsize = (10,8))
#    plt.xlabel("AOTF frequency")
#    plt.ylabel("Summed zero offset signal value pixels %i to %i" %(chosen_range[0],chosen_range[1]))
#    plt.plot(aotf_freq_all,offset_detector_data_summed,'b')
#
#    plt.figure(figsize = (10,8))
#    plt.xlabel("Frame Number")
#    plt.ylabel("Summed signal value pixels %i to %i" %(chosen_range[0],chosen_range[1]))
#    plt.plot(range(len(detector_data_summed)),detector_data_summed)

    plt.figure(figsize = (10,8))
    plt.xlabel("Diffraction Order")
    plt.ylabel("Summed zero offset signal value pixels %i to %i" %(chosen_range[0],chosen_range[1]))
    plt.plot(range(len(detector_data_summed)),offset_detector_data_summed)
    plt.plot(range(len(detector_data_summed)),orders*10.0)
    plt.plot(range(len(detector_data_summed)),lambertian_intensity*3000.0)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_signal_diffraction_orders_and_lambertian_surface_reflectance.png")
    
    plt.figure(figsize = (10,8))
    plt.xlabel("Diffraction Order")
    plt.ylabel("Lambertian Normalised pixels %i to %i" %(chosen_range[0],chosen_range[1]))
    
    frame_range = [116,226]
    plt.scatter(orders[frame_range[0]:frame_range[1]],lambertian_normalised[frame_range[0]:frame_range[1]],c=plot_colours[frame_range[0]:frame_range[1]],linewidth=0)
    plt.xlim([100,225])
    plt.plot(orders[frame_range[0]:frame_range[1]],sg_filter(lambertian_normalised[frame_range[0]:frame_range[1]], window_size=31, order=2))
    plot_title = title+" relative signal vs diffraction order for lambertian surface"
    plt.title(plot_title)
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+plot_title.replace(" ","_")+".png")


if option==39:
    """check and plot random LNO miniscans"""
    """read in files, correct wrong aotf frequencies, average detector data when aotfs repeat, interpolate onto 1khz grid
    find overlap with next file, scale data to match signal levels, stitch all data into 1khz grid, generate file""" 
    
    #make empty arrays to hold all data
    all_data_interpolated = np.zeros((len(obspaths),2*255,320))
    all_aotfs_interpolated = np.zeros((len(obspaths),2*255))
    #now loop through files, reading in the detector data and aotf frequencies from each and storing them in the empty arrays
    for file_index,hdf5_file in enumerate(hdf5_files):
    
        """get data from file"""
        print("Reading in file %i: %s" %(file_index,obspaths[file_index]))
        detector_data_bins = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
        print("AOTF range %i to %i" %(min(aotf_freq_all),max(aotf_freq_all)))
        if aotf_freq_all[0]<16000: print("Warning: AOTFs in %s are too low - small signal" %obspaths[file_index])
        hdf5_file.close()
        
        #code to correct incorrect telecommand stuck onto detector data by SINBAD
        aotf_freq_range = np.arange(min(aotf_freq_all),min(aotf_freq_all)+2*256,2)
        aotf_freq_corrected = np.append(np.append(aotf_freq_range,aotf_freq_range),aotf_freq_range[0:28])
        if max(aotf_freq_all)-min(aotf_freq_all) != 510:
            print("Error: AOTFs may not be correct") #print(error and stop program if there is a problem
            stop()
            
        
        detector_data_binned = np.mean(detector_data_bins[:,6:18,:],axis=1) #average detector spatial data to make one spectrum per AOTF frequency. Bins will be done later
        #each file contains two and a bit sweeps through AOTF frequencies. Average the first two sweeps together
        detector_data_binned_mean = np.mean(np.asarray([detector_data_binned[0:256,:],detector_data_binned[256:512,:]]),axis=0)
        #interpolate aotf frequencies and detector data into 1kHz steps
        all_aotfs_interpolated[file_index,:] = np.arange(aotf_freq_range[0],aotf_freq_range[-1],1.0)
        for pixel_index in range(detector_data_binned_mean.shape[1]):
            all_data_interpolated[file_index,:,pixel_index] = np.interp(all_aotfs_interpolated[file_index,:],aotf_freq_range,detector_data_binned_mean[:,pixel_index])


        
    """v3: analyse individual miniscans containing absorption lines"""
    pixels = np.arange(320)

    """now select some solar lines. Each item in each list is a different solar line"""
    
    #list of file numbers to be plotted
    file_indices = [5,8,16,17,18,19]
    #for each file, tell the program which frame contains the biggest absorption line. This is used only for plotting the Pixel Number vs Signal graph
    line_centres=[230,185,410,120,485,25]
    #for each file, give the range of absorptions to fit a polynomial to. This is used to find the frame number and AOTF frequency where the absorption is the greatest
    #values are of the form [starting_frame_number,ending_frame_number]
    fitting_ranges = [[0,460],[90,270],[320,510],[0,80],[380,550],[0,120]]
    #for each file, specify the pixel numbers for the fitting of a polynomial to the continuum on either side of an absorption line
    #values are of the form [pixel number at start of continuum fit to left of absorption line,pixel number at end of continuum fit to left of absorption line,pixel number at start of continuum fit to right of absorption line,pixel number at end of continuum fit to right of absorption line,
    continuum_ranges = [[285,290,310,315],[210,215,225,230],[105,111,127,133],[109,115,131,137],[226,228,241,243],[224,226,244,249]]
    animate_index=5
#    animate=True
    animate=False

    #loop through chosen solar lines, doing calculations and plotting each one.
    for file_index,fitting_range,line_centre,continuum_range in zip(file_indices,fitting_ranges,line_centres,continuum_ranges):
        fig = plt.figure(figsize = (figx,figy))
        ax1 = plt.subplot2grid((2,1),(0,0))
        ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)

        frames = range(0,2*255-1) #frame numbers in each miniscan file
        frame_aotfs = all_aotfs_interpolated[file_index,0:2*255-1] #AOTF frequencies in each miniscan file
        
    
        absorption_depths = []
        
        #make empty arrays for storing values
        selected_detector_data = np.zeros((len(frames),320))
        continuum_pixels = np.zeros((len(frames),len(range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3]))))
        continuum_spectra = np.zeros((len(frames),len(range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3]))))
        absorption_pixels = np.zeros((len(frames),len(range(continuum_range[1],continuum_range[2]))))
        absorption_spectra = np.zeros((len(frames),len(range(continuum_range[1],continuum_range[2]))))
        absorption_continua = np.zeros((len(frames),len(range(continuum_range[1],continuum_range[2]))))
        absorptions = np.zeros((len(frames),len(range(continuum_range[1],continuum_range[2]))))
        
        #now loop through each file in a miniscan file
        for frame_index,frame in enumerate(frames):
        
            #store raw data
            selected_detector_data[frame_index,:] = all_data_interpolated[file_index,frame,:]
            #calculate continuum pixels and data
            continuum_pixels[frame_index,:] = pixels[range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]    
            continuum_spectra[frame_index,:] = all_data_interpolated[file_index,frame,range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]
            
            #fit polynomial to continuum on either side of absorption band
            coefficients = np.polyfit(continuum_pixels[frame_index,:],continuum_spectra[frame_index,:],2)
            
            #store pixel numbers and data from part of spectrum containing absorption line
            absorption_pixels[frame_index,:] = pixels[range(continuum_range[1],continuum_range[2])]
            absorption_spectra[frame_index,:] = all_data_interpolated[file_index,frame,range(continuum_range[1],continuum_range[2])]
            #calculate continuum in this region
            absorption_continua[frame_index,:] = np.polyval(coefficients, absorption_pixels[frame_index,:])
            
            #calculate absorption by dividing by continuum
            absorptions[frame_index,:] = absorption_spectra[frame_index,:]/absorption_continua[frame_index,:]
        
            #record minimum depth of absorption band
            absorption_depths.append(min(absorptions[frame_index,:]))
            
            #plot when absorption band depth is maximum (set manually!)
            if frame==line_centre:
                ax1.plot(selected_detector_data[frame_index,:])
                ax1.scatter(continuum_pixels[frame_index,:],continuum_spectra[frame_index,:],c="g")
                ax1.scatter(absorption_pixels[frame_index,:],absorption_continua[frame_index,:],c="r")
                ax2.plot(absorption_pixels[frame_index,:],absorptions[frame_index,:])
        
    
        #now plot Absorption depth vs AOTF frequency
        plt.figure(figsize = (figx,figy))
        plt.plot(frame_aotfs,absorption_depths)
        
        #fit a polynomial to the data in the manually chosen fitting range, to find AOTF where absorption peaks
        frame_coefficients = np.polyfit(frames[fitting_range[0]:fitting_range[1]],absorption_depths[fitting_range[0]:fitting_range[1]],2)
        frame_fit = np.polyval(frame_coefficients, frames)

        def func(x, a, b, c, d):
            return 1.0 - (a * np.exp(-((x-b)**2.0) / (2.0 * c**2.0)) + d)

        popt, pcov = curve_fit(func, np.asfarray(frame_aotfs[fitting_range[0]:fitting_range[1]]),np.asfarray(absorption_depths[fitting_range[0]:fitting_range[1]]), p0=np.array([0.22,frame_aotfs[np.int(np.mean(fitting_range))],60.,0.]), maxfev=10000)
        frame_gauss = func(np.asfarray(frame_aotfs[fitting_range[0]:fitting_range[1]]), popt[0], popt[1], popt[2], popt[3])
        
#        plt.plot(frame_aotfs[fitting_range[0]:fitting_range[1]],frame_fit[fitting_range[0]:fitting_range[1]],'g')
        plt.plot(frame_aotfs[fitting_range[0]:fitting_range[1]],frame_gauss,'r')
    
        #write on graph
        x = frame_aotfs[np.int(np.mean(fitting_range))]
        y = frame_fit[np.int(np.mean(fitting_range))]
        plt.text(x, y+0.005, "%0.3f" %x)
        print("centre at %0.1f" %x)
    
        
#        old_tuning = np.polyval(tuning,frame_aotfs)
    
#        plt.figure(figsize = (figx,figy))
#        plt.plot(old_tuning,absorption_depths)


        """code to animate the plot"""
        if animate and animate_index==file_index:
            fig=plt.figure(figsize = (figx,figy))
            frame_index=0
        
            ax1 = plt.subplot2grid((2,1),(0,0))
            ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)
        
            max_value = np.max(selected_detector_data)
            ax1.set_ylim((0,max_value))
            ax2.set_ylim((np.min(absorptions),np.max(absorptions)))
            n_frames = len(frames)
            plot1, = ax1.plot(range(320),selected_detector_data[frame_index,:], color="k", animated=True)
            plot2, = ax1.plot(continuum_pixels[frame_index,:],continuum_spectra[frame_index,:],color="g",marker="o",linewidth=0,alpha = 0.5, animated=True)
            plot3, = ax1.plot(absorption_pixels[frame_index,:],absorption_continua[frame_index,:],color="r",marker="o",linewidth=0,alpha = 0.5, animated=True)
            plot4, = ax2.plot(absorption_pixels[frame_index,:],absorptions[frame_index,:],color="k", animated=True)
            
            text1 = ax1.text(10,max_value-20000,"")#"AOTF Frequency = %ikHz\nDiffraction Order = %i" %(miniscan_aotfs_sorted[num],miniscan_orders_sorted[num]))
        
            def updatefig(frame_index): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
                if np.mod(frame_index,500)==0:
                    print(frame_index)
                plot1.set_data(range(320),selected_detector_data[frame_index,:])
                plot2.set_data(continuum_pixels[frame_index,:],continuum_spectra[frame_index,:])
                plot3.set_data(absorption_pixels[frame_index,:],absorption_continua[frame_index,:])
                plot4.set_data(absorption_pixels[frame_index,:],absorptions[frame_index,:])
                text1.set_text("")#"AOTF Frequency = %ikHz\nDiffraction Order = %i" %(miniscan_aotfs_sorted[num],miniscan_orders_sorted[num]))
        #        plottitle.set_text("%s" %time_data_subd[num])
                return plot1,plot2,plot3,plot4,text1,
        
            ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=20, blit=True)
        #    if save_figs: ani.save(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_mean_spectrum.mp4", fps=5, extra_args=['-vcodec', 'libx264'])


    """v2: all interpolated data read in at start, then match overlapping regions and scale data. Doesn't work well in merging different files"""
#    sg_window_size=29
#    file_lengths = 255
#    miniscan_spectra = np.zeros((file_lengths*len(obspaths),320))
#    miniscan_aotfs = np.zeros((file_lengths*len(obspaths)))
#    for file_index in range(len(hdf5_files)-1):
#        overlapping_indices1 = [index for index,aotf_freq in enumerate(list(all_aotfs_interpolated[file_index,:])) if aotf_freq in list(all_aotfs_interpolated[file_index+1,:])]
#        overlapping_indices2 = range(len(overlapping_indices1))
#        overlapping_ratio = [value1/value2 for value1,value2 in zip(all_data_interpolated[file_index,overlapping_indices1,160],all_data_interpolated[file_index+1,overlapping_indices2,160])]
#
#        mean_overlap = np.mean(overlapping_ratio)    
#        print(mean_overlap
#        all_data_interpolated[file_index+1,:,:] = all_data_interpolated[file_index+1,:,:] * mean_overlap
#
#        if file_index==0:
#            print("adding %i to %i" %(all_aotfs_interpolated[file_index,0],all_aotfs_interpolated[file_index,overlapping_indices1[0]])
#            #add unique region
#            miniscan_aotfs = all_aotfs_interpolated[file_index,0:overlapping_indices1[0]]
#            miniscan_spectra = all_data_interpolated[file_index,0:overlapping_indices1[0],:]
#            #append overlapping region
#            print("adding %i to %i" %(all_aotfs_interpolated[file_index,overlapping_indices1[0]],all_aotfs_interpolated[file_index,overlapping_indices1[-1]])
#            miniscan_aotfs = np.append(miniscan_aotfs, all_aotfs_interpolated[file_index,overlapping_indices1])
#            miniscan_spectra = np.append(miniscan_spectra, np.mean(np.asarray([all_data_interpolated[file_index,overlapping_indices1,:],all_data_interpolated[file_index+1,overlapping_indices2,:]]), axis=0), axis=0)
#            final_index = max(overlapping_indices2)
#        else:
#            print("adding %i to %i" %(all_aotfs_interpolated[file_index,final_index],all_aotfs_interpolated[file_index,overlapping_indices1[0]])
#            miniscan_aotfs = np.append(miniscan_aotfs, all_aotfs_interpolated[file_index,final_index:overlapping_indices1[0]])
#            miniscan_spectra = np.append(miniscan_spectra, all_data_interpolated[file_index,final_index:overlapping_indices1[0],:], axis=0)
#            #append overlapping region
#            print("adding %i to %i" %(all_aotfs_interpolated[file_index,overlapping_indices1[0]],all_aotfs_interpolated[file_index,overlapping_indices1[-1]])
#            miniscan_aotfs = np.append(miniscan_aotfs, all_aotfs_interpolated[file_index,overlapping_indices1])
#            miniscan_spectra = np.append(miniscan_spectra, np.mean(np.asarray([all_data_interpolated[file_index,overlapping_indices1,:],all_data_interpolated[file_index+1,overlapping_indices2,:]]), axis=0), axis=0)
#            final_index = max(overlapping_indices2)
#    #add the last file at the end
#    file_index = len(hdf5_files)-1
#    print("adding %i to %i" %(all_aotfs_interpolated[file_index,final_index],all_aotfs_interpolated[file_index,-1])
#    miniscan_aotfs = np.append(miniscan_aotfs, all_aotfs_interpolated[file_index,final_index:-1])
#    miniscan_spectra = np.append(miniscan_spectra, all_data_interpolated[file_index,final_index:-1,:], axis=0)
#
#
#    plt.figure(figsize = (figx,figy))
#    for file_index in range(len(hdf5_files)):
#        plt.scatter(all_aotfs_interpolated[file_index,:],all_data_interpolated[file_index,:,160])
#        plt.plot(all_aotfs_interpolated[file_index,overlapping_indices1],all_data_interpolated[file_index,overlapping_indices1,160],"r")
#
#    plt.figure(figsize = (figx,figy))
#    plt.scatter(miniscan_aotfs,miniscan_spectra[:,160])
#
#    miniscan_spectra_smoothed = np.zeros_like(miniscan_spectra)
#    for pixel_index in range(miniscan_spectra.shape[1]):
#        miniscan_spectra_smoothed[:,pixel_index] = sg_filter(miniscan_spectra[:,pixel_index], window_size=sg_window_size, order=2)
#
#    if save_files:
#        from datetime import datetime
#        output_filename = "%s_smoothing=%i" %(title.replace(" ","_"),sg_window_size)
#        #write bad pixel map to file
#        hdf5_file_out = h5py.File(BASE_DIRECTORY+os.sep+output_filename+".h5", "w")
#        write_to_hdf5(hdf5_file_out,miniscan_spectra,"MiniscanSpectraInterpolated",np.float,frame="None")
#        write_to_hdf5(hdf5_file_out,miniscan_spectra_smoothed,"MiniscanSpectraSmoothed",np.float,frame="None")
#        write_to_hdf5(hdf5_file_out,miniscan_aotfs,"MiniscanAOTFsInterpolated",np.float,frame="None")
#        
#        comments = "Files Used: "
#        for obspath in obspaths:
#            comments = comments + obspath + "; "
#        hdf5_file_out.attrs["Comments"] = comments
#        hdf5_file_out.attrs["Date_Created"] = str(datetime.now())
#        hdf5_file_out.close()


    """v1: read in all files first, don't scale to match signal values. Produces large jumps in the data"""                
#        miniscan_aotfs[file_index*file_lengths:file_index*file_lengths+file_lengths] = aotf_freq_corrected[0:file_lengths]
#        miniscan_spectra[file_index*file_lengths:file_index*file_lengths+file_lengths,:] = detector_data_binned_mean[0:file_lengths]
#
#    sort_indices = miniscan_aotfs.argsort()
#    miniscan_aotfs_sorted = miniscan_aotfs[sort_indices]
#    
#    
#    miniscan_spectra_sorted = miniscan_spectra[sort_indices,:]
#    miniscan_orders_sorted = np.asfarray([findOrder(channel,aotf_frequency,silent=True) for aotf_frequency in list(miniscan_aotfs_sorted)])
#
#    #next interpolate to make 1kHz resolution grid
#    miniscan_aotfs_interpolated = np.arange(miniscan_aotfs_sorted[0],miniscan_aotfs_sorted[-1],1.0)
#      
#    miniscan_spectra_interpolated = np.zeros((miniscan_aotfs_interpolated.shape[0],miniscan_spectra_sorted.shape[1]))
#    miniscan_spectra_smoothed = np.zeros((miniscan_aotfs_interpolated.shape[0],miniscan_spectra_sorted.shape[1]))
#    for pixel_index in range(miniscan_spectra_sorted.shape[1]):
#        miniscan_spectra_interpolated[:,pixel_index] = np.interp(miniscan_aotfs_interpolated,miniscan_aotfs_sorted,miniscan_spectra_sorted[:,pixel_index])
#        miniscan_spectra_smoothed[:,pixel_index] = sg_filter(miniscan_spectra_interpolated[:,pixel_index], window_size=sg_window_size, order=2)
#
#    plt.figure(figsize = (figx,figy))
#    plt.plot(miniscan_aotfs_interpolated,miniscan_spectra_interpolated[:,160])
#    plt.plot(miniscan_aotfs_interpolated,miniscan_spectra_smoothed[:,160])

#    if save_files:
#        from datetime import datetime
#        output_filename = "%s_smoothing=%i_offset=%i" %(title.replace(" ","_"),sg_window_size,file_offset)
#        #write bad pixel map to file
#        hdf5_file_out = h5py.File(BASE_DIRECTORY+os.sep+output_filename+".h5", "w")
#        write_to_hdf5(hdf5_file_out,miniscan_spectra_smoothed,"MiniscanSpectraInterpolated",np.float,frame="None")
#        write_to_hdf5(hdf5_file_out,miniscan_aotfs_interpolated,"MiniscanAOTFsInterpolated",np.float,frame="None")
#        
#        comments = "Files Used: "
#        for obspath in obspaths:
#            comments = comments + obspath + "; "
#        hdf5_file_out.attrs["Comments"] = comments
#        hdf5_file_out.attrs["Date_Created"] = str(datetime.now())
#        hdf5_file_out.close()


    """code to animate the plot"""
#    fig=plt.figure(1, figsize=(10,8))
#    num=0
#
#    ax1 = plt.subplot2grid((1,1),(0,0))
#
#    max_value = np.max(miniscan_spectra_sorted)
#    ax1.set_ylim((0,max_value))
#    n_frames = len(miniscan_aotfs_sorted)
#    plot1, = ax1.plot(range(320),miniscan_spectra_sorted[num,:], color="k", animated=True)
#    text1 = ax1.text(10,max_value-20000,"AOTF Frequency = %ikHz\nDiffraction Order = %i" %(miniscan_aotfs_sorted[num],miniscan_orders_sorted[num]))
#
#    def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
#        if np.mod(num,500)==0:
#            print(num)
#        plot1.set_data(range(320),miniscan_spectra_sorted[num,:])
#        text1.set_text("AOTF Frequency = %ikHz\nDiffraction Order = %i" %(miniscan_aotfs_sorted[num],miniscan_orders_sorted[num]))
##        plottitle.set_text("%s" %time_data_subd[num])
#        return plot1,text1,
#
#    ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=5, blit=True)
#    if save_figs: ani.save(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_mean_spectrum.mp4", fps=5, extra_args=['-vcodec', 'libx264'])


 
    """code to plot selected spectra""" 
#    if file_index==0:
#        frames_to_plot=range(0,100,4)
#    else:
#        frames_to_plot=range(0,100,4)
#        
#    
#    plt.figure(figsize = (10,8))
#    plt.xlabel("Pixel number")
#    plt.ylabel("Signal value")
#    plt.title(obspaths[file_index])
#
#    loop=-1
#    for frame_to_plot in frames_to_plot:
#        loop += 1
#        plt.plot(range(320),detector_data_binned[frame_to_plot,:],linestyles[loop],label="%s" %aotf_freq_all[frame_to_plot])
#    plt.legend()
#    plt.plot([160,160],[0,max(detector_data_binned[frame_to_plot,:])])
#
#


if option==40:
    """quick analysis of UVIS full frame Mars nadir from MCO-2"""
    
    detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get data
    v_start = int(get_dataset_contents(hdf5_file,"VStart")[0][0])
    v_end = int(get_dataset_contents(hdf5_file,"VEnd")[0][0])
    obs_time_all = get_dataset_contents(hdf5_file,"ObservationDateTime")[0]
    hdf5_file.close()
    
    n_packets_per_frame = 9
    n_lines = v_end - v_start + 1
    print(detector_data_all.shape[0])
    n_frames = 210 #int(np.floor(detector_data_all.shape[0]/n_packets_per_frame - 1))
    
    light = np.zeros((n_frames,n_lines,1048), dtype=int)
    starting_indices = range(n_packets_per_frame*2+1,(n_packets_per_frame*2+1)+n_frames*n_packets_per_frame,n_packets_per_frame)

    for index,i in enumerate(starting_indices):
        frame = detector_data_all[i:i+n_packets_per_frame,:,:].reshape((n_packets_per_frame*15,1048))[0:n_lines,:]
        light[index,:,:] = frame
        
    obs_times = obs_time_all[starting_indices]
    detector_data_all=0

    row=55
    frames_to_plot = [30]
    for frame_to_plot in frames_to_plot:
        fig = plt.figure(figsize = (figx-9,figy))

        ax1 = plt.subplot2grid((3,8),(0,0), colspan=7)
        p1 = plt.imshow(light[frame_to_plot,:,:], aspect=2)
    
        ax1.set_title("Frame %i raw counts and slice of row %i" %(frame_to_plot,row))
        ax2 = plt.subplot2grid((3,8),(1,0), colspan=7, rowspan=2)
        p2 = plt.plot(light[frame_to_plot,row,:])
        plt.xlim((0,1048))
        ax3 = plt.subplot2grid((3,8),(0,7), rowspan=3)
        fig.colorbar(p1,cax=ax3)
        if save_figs:
            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_frame_%i_counts_and_slice_of_row_%i.png" %(frame_to_plot,row))
            
        plt.figure(figsize = (figx,figy))
        plt.imshow(light[frame_to_plot,:,:], aspect=2)
        plt.title("Signal from full frame %i at %s" %(frame_to_plot,obs_times[frame_to_plot,0]))
        if save_figs:
            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_raw_full_frame_%i.png" %(frame_to_plot))
    light=0
    gc.collect()



if option==41:
    """SO sun to dark slew (occultation simulation)"""

    bintops = [116,122,128,134]
    n_orders = len(hdf5_files)
    n_bins = len(bintops)
    full_sun_region = [900,1100]
    pixel_range = [100,102]


    linestyles = ["-", "--", "-.", ":"]
    colours = ["","r","g","b","k","c"]
    dark_detector_data = np.zeros((detector_data_all.shape[0]/n_bins,n_bins))

    fig = plt.figure(figsize = (10,8))
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax1.set_title(title+": pixel range %i to %i" %(pixel_range[0],pixel_range[1]))
    ax2 = plt.subplot2grid((2,1),(1,0))


    for file_index,hdf5_file in enumerate(hdf5_files):
    
        """get data from file"""
        print("Reading in file %s" %obspaths[file_index])
        detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        binning_all = get_dataset_contents(hdf5_file,"Binning")[0]
        bins_all = get_dataset_contents(hdf5_file,"Bins")[0]
        diffractionorder_all = get_dataset_contents(hdf5_file,"DiffractionOrder")[0]
        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
        date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
        hdf5_file.close()

        binning = binning_all[0]
        diffractionorder = diffractionorder_all[0]
        
   
        for bin_index,chosen_bintop in enumerate(bintops):
            mean_detector_data = np.asfarray([np.mean(detector_data_all[index,pixel_range[0]:pixel_range[1]]) for index,value in enumerate(list(bins_all[:,0])) if value==chosen_bintop])
            if file_index==0:
                dark_detector_data[:,bin_index] = mean_detector_data[:]
            else:
                mean_detector_data = mean_detector_data - dark_detector_data[:,bin_index]
                full_sun_value = np.mean(mean_detector_data[full_sun_region[0]:full_sun_region[1]])
                transmittance = mean_detector_data/full_sun_value
    
                ax1.plot(mean_detector_data,linestyle=linestyles[bin_index],color=colours[file_index], label="Order %i BinTop %i" %(diffractionorder,chosen_bintop))
                ax2.plot(transmittance,linestyle=linestyles[bin_index],color=colours[file_index], label="Order %i BinTop %i" %(diffractionorder,chosen_bintop))


    plt.legend()
    
    frames_to_plot=range(100,150,1)#+range(1050,1060,2)
    plt.figure(figsize=(10,8))
    for frame_to_plot in frames_to_plot:
        plt.plot(detector_data_all[frame_to_plot,:], label="Frame=%s" %frame_to_plot)
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Detector Signal")
#    plt.yscale("log")
    

    frames_to_plot=range(1000,1050,1)#+range(1050,1060,2)
    plt.figure(figsize=(10,8))
    for frame_to_plot in frames_to_plot:
        plt.plot(detector_data_all[frame_to_plot,:], label="Frame=%s" %frame_to_plot)
    plt.legend()
    plt.xlabel("Pixel Number")
    plt.ylabel("Detector Signal")
#    plt.yscale("log")


if option==42:
    """LNO sun to dark slew (occultation simulation)"""
    
    bintops = [140,146,152,158]
    n_orders = len(hdf5_files)
    n_bins = len(bintops)
    full_sun_region = [900,1100]
    pixel_range = [100,102]


    linestyles = ["-", "--", "-.", ":"]
    colours = ["","r","g","b","k","c"]
    dark_detector_data = np.zeros((detector_data_all.shape[0]/n_bins,n_bins))

    fig = plt.figure(figsize = (figx,figy))
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax1.set_title(title+": pixel range %i to %i" %(pixel_range[0],pixel_range[1]))
    ax2 = plt.subplot2grid((2,1),(1,0))


    for file_index,hdf5_file in enumerate(hdf5_files):
    
        """get data from file"""
        print("Reading in file %s" %obspaths[file_index])
        detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        binning_all = get_dataset_contents(hdf5_file,"Binning")[0]
        bins_all = get_dataset_contents(hdf5_file,"Bins")[0]
        diffractionorder_all = get_dataset_contents(hdf5_file,"DiffractionOrder")[0]
        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
        date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
        hdf5_file.close()

        binning = binning_all[0]
        diffractionorder = diffractionorder_all[0]
        
   
        for bin_index,chosen_bintop in enumerate(bintops):
            mean_detector_data = np.asfarray([np.mean(detector_data_all[index,pixel_range[0]:pixel_range[1]]) for index,value in enumerate(list(bins_all[:,0])) if value==chosen_bintop])
            if file_index==0:
                dark_detector_data[:,bin_index] = mean_detector_data[:]
            else:
                mean_detector_data = mean_detector_data - dark_detector_data[:,bin_index]
                full_sun_value = np.mean(mean_detector_data[full_sun_region[0]:full_sun_region[1]])
                transmittance = mean_detector_data/full_sun_value
    
                ax1.plot(mean_detector_data,linestyle=linestyles[bin_index],color=colours[file_index], label="Order %i BinTop %i" %(diffractionorder,chosen_bintop))
                ax2.plot(transmittance,linestyle=linestyles[bin_index],color=colours[file_index], label="Order %i BinTop %i" %(diffractionorder,chosen_bintop))


    plt.legend()

if option==43:
    """plot measured detector temperatures for any file"""
    
    if multiple:
        hdf5_file = hdf5_files[0]
    if channel=="so":
        detector_temperature = get_dataset_contents(hdf5_file,"FPA1_FULL_SCALE_TEMP_SO")[0]
    elif channel=="lno":
        detector_temperature = get_dataset_contents(hdf5_file,"FPA1_FULL_SCALE_TEMP_LNO")[0]
    hdf5_file.close()

    plt.figure(figsize = (10,6))
    plt.title(title)
    plt.ylabel("Detector Temperature K")
    plt.xlabel("Frame Number")
    plt.plot(detector_temperature)
    plt.ylim([80,90])
    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"-detector_temperature.png")

if option==44:
    
    """plot groundtrack and some spectra (not animated)"""
    
    lons_all = get_dataset_contents(hdf5_files[0],"Lon",chosen_group="Geometry/Point0")[0]
    lats_all = get_dataset_contents(hdf5_files[0],"Lat",chosen_group="Geometry/Point0")[0]
    detector_data_bins = get_dataset_contents(hdf5_files[0],"Y")[0]
    wavenumbers = get_dataset_contents(hdf5_files[0],"X")[0][0,:]
    hdf5_files[0].close()
    
    data_subd=detector_data_bins
    fig=plt.figure(1, figsize=(10,8))
    num=100

    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0))
    
    if 'dem32' not in locals():
        dem32=np.zeros((1152,576))+65535
        file_size=1152,5760 #reads in 10 horiz at a time, actually 11520 x 5760
        with open(os.path.normcase(r'D:\megt90n000fb.img'), mode='rb') as dem32_file:
            for loop in range(file_size[1]):
                for loop2 in range(file_size[0]):
                    temp=struct.unpack('>HHHHHHHHHH', dem32_file.read(20))[9]
                    if (loop % 10):
                        if (temp > 2**15):
                            dem32[loop2][loop/10]=float(temp)-(2.0**16)#((float(temp)/2.0**16) * (8206.0+21181.0)-8206.0)
                        else:
                            dem32[loop2][loop/10]=float(temp)#((float(temp)/2.0**16) * (8206.0+21181.0)-8206.0)
        dem32 = np.transpose(np.fliplr(dem32))

    lons = np.arange(0.0, 360.0, 0.3125)
    lats = np.arange(-90, 90.0, 0.3125)
    lons, lats = np.meshgrid(lons, lats)

    m = Basemap(llcrnrlon=-180,llcrnrlat=-80,urcrnrlon=180,urcrnrlat=80,projection='mill')
    m.drawparallels(np.arange(-80,81,10),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])    
    m.contourf(lons, lats, dem32, shading='flat', latlon=True, cmap='terrain')
    x, y = m(lons_all[:,0],lats_all[:,1])
    plot2, = m.plot(x,y, marker='o')

    ax1.set_ylim((0,max_value))
#    plot, = ax1.plot(wavenumbers,binned_data_subd[num,:], color="k", animated=True)
    plota, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    plotb, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    plotc, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    plotd, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    plote, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    plotf, = ax1.plot(wavenumbers,data_subd[num,:], animated=True)
    


if option==45:
    """test methods of joining files together"""
    
    #make empty arrays to hold all data
    all_data_interpolated = np.zeros((len(obspaths),2*255,320))
    all_aotfs_interpolated = np.zeros((len(obspaths),2*255))
    #now loop through files, reading in the detector data and aotf frequencies from each and storing them in the empty arrays
    for file_index,hdf5_file in enumerate(hdf5_files):
    
        """get data from file"""
        print("Reading in file %i: %s" %(file_index,obspaths[file_index]))
        detector_data_bins = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
        measurement_temperature = np.mean(get_dataset_contents(hdf5_file,"AOTF_TEMP_%s" %channel.upper())[0][2:10])
        print("AOTF range %i to %i at %0.1fC" %(min(aotf_freq_all),max(aotf_freq_all),measurement_temperature))
        if aotf_freq_all[0]<16000: print("Warning: AOTFs in %s are too low - small signal" %obspaths[file_index])
        hdf5_file.close()
        
        #code to correct incorrect telecommand stuck onto detector data by SINBAD
        aotf_freq_range = np.arange(min(aotf_freq_all),min(aotf_freq_all)+2*256,2)
        aotf_freq_corrected = np.append(np.append(aotf_freq_range,aotf_freq_range),aotf_freq_range[0:28])
        if max(aotf_freq_all)-min(aotf_freq_all) != 510:
            print("Error: AOTFs may not be correct") #print(error and stop program if there is a problem
            stop()
            
        
        detector_data_binned = np.mean(detector_data_bins[:,6:18,:],axis=1) #average detector spatial data to make one spectrum per AOTF frequency. Bins will be done later
        #each file contains two and a bit sweeps through AOTF frequencies. Average the first two sweeps together
        detector_data_binned_mean = np.mean(np.asarray([detector_data_binned[0:256,:],detector_data_binned[256:512,:]]),axis=0)
        #interpolate aotf frequencies and detector data into 1kHz steps
        all_aotfs_interpolated[file_index,:] = np.arange(aotf_freq_range[0],aotf_freq_range[-1],1.0)
        for pixel_index in range(detector_data_binned_mean.shape[1]):
            all_data_interpolated[file_index,:,pixel_index] = np.interp(all_aotfs_interpolated[file_index,:],aotf_freq_range,detector_data_binned_mean[:,pixel_index])


        
    
#    linestyles = ["-", "--", "-", "--","-", "--","-", "--","-", "--","-", "--"] * 100
    linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"] * 100
    alpha=0.3
    
    file_index1=0
    file_index2=1
#    frame_indices1 = range(468,469,1)#510,1)
#    frame_indices2 = range(0,1,1)#510,1)
    frame_indices1 = range(0,510,1)
    frame_indices2 = range(0,510,1)
    index_shift=468
    continuum_range1 = [261,272]
    line_centres1=[267]
    spectra_to_plot = [120,220]
    
    
    overlapping_indices1 = range(index_shift,469,1)
    overlapping_indices2 = range(0,469-index_shift,1)

    pixels1 = range(320)

    xshift=-3
    if xshift==0:
        pixels2 = pixels1
    elif xshift>0:
        pixels2 = pixels1[xshift::]
    elif xshift<0:
        pixels2 = [pixels1[0]]*(xshift*-1)+pixels1[:xshift:]
#    yscale=1.13237
#    yshift=872.834

    continuum_range2 = [continuum_range1[0]-xshift,continuum_range1[1]-xshift]
    line_centres2=[line_centres1[0]-xshift]

    mean_yscale=[]
    mean_yshift=[]
    
#    plt.figure(figsize = (figx,figy))
    for index,(spectrum_index1,spectrum_index2) in enumerate(zip(overlapping_indices1,overlapping_indices2)):
        spectrum1 = all_data_interpolated[file_index1,spectrum_index1,range(len(pixels1))]
        spectrum2 = all_data_interpolated[file_index2,spectrum_index2,range(len(pixels2))]
        
        absorption_pixels1 = pixels1[continuum_range1[0]:continuum_range1[1]]
        absorption_pixels2 = pixels2[continuum_range2[0]:continuum_range2[1]]
        absorption_spectrum1 = spectrum1[continuum_range1[0]:continuum_range1[1]]
        absorption_spectrum2 = spectrum2[continuum_range2[0]:continuum_range2[1]]
                    
#        def func(params, spectrum_in1, spectrum_in2): #least squares doesn't give right answer
#            yshift_in = params[1]
#            yscale_in = params[0]
#            return (spectrum_in1 - spectrum_in2*yscale_in + yshift_in)**2
#        yscale,yshift = leastsq(func, [1.22,-3900], args=(absorption_spectrum1,absorption_spectrum2))[0]
#        def fity(spectrum_in1, spectrum_in2): #attempt to write own version, too hard
#            fit=1e7
#            yscale = 1000.
#            yscale_increment = 10.0
#            yshift = 1.1
#            yshift_increment = 0.01
#            
#            for attempt in range(1000):
#                yscale = yscale+yscale_increment
#                new_fit = (spectrum_in1 - spectrum_in2*yscale + yshift)**2
#                if new_fit > fit:
#                    yscale_increment = -1 * yscale_increment
#        def fity(spectrum_in1, spectrum_in2): #attempt to write own version, too hard
#            spectrum[0]
                    
            
#            return 
#        yscale,yshift = fity(absorption_spectrum1,absorption_spectrum2)
        yscale = 1.22
        yshift = -3900
        print("yscale=%0.3f, yshift=%0.3f: target=%.0f, was=%.0f, now=%.0f" %(yscale,yshift,absorption_spectrum1[0],absorption_spectrum2[0],absorption_spectrum2[0]*yscale+yshift))
        mean_yscale.append(yscale)
        mean_yshift.append(yshift)

#        plt.plot(pixels1,spectrum1, color=linecolours[index], linestyle="-")
#        plt.plot(pixels2,spectrum2, color=linecolours[index], linestyle="--")
#        plt.scatter(absorption_pixels2,absorption_spectrum2*yscale+yshift, color=linecolours[index], linewidth=0, alpha=alpha)


    mean_yscale=np.mean(mean_yscale)    
    mean_yshift=np.mean(mean_yshift)    
    
#    plt.figure(figsize = (figx,figy))

    absorption_depths1 = np.zeros(len(frame_indices1))
    absorption_depths2 = np.zeros(len(frame_indices2))
    for frame_index,spectrum_index1 in enumerate(frame_indices1):
        spectrum1 = all_data_interpolated[file_index1,spectrum_index1,range(len(pixels1))]
        aotfs1 = all_aotfs_interpolated[file_index1,frame_indices1]
        
        absorption_pixels1 = [pixels1[continuum_range1[0]]]+[pixels1[continuum_range1[1]]]
        continuum_spectra1 = [spectrum1[continuum_range1[0]]]+[spectrum1[continuum_range1[1]]]
        
        #fit polynomial to continuum on either side of absorption band
        absorption_continua1= np.polyval(np.polyfit(absorption_pixels1,continuum_spectra1,1), pixels1[continuum_range1[0]:continuum_range1[1]])
        
        #store pixel numbers and data from part of spectrum containing absorption line
        absorption_spectra1 = spectrum1[continuum_range1[0]:continuum_range1[1]]
        #calculate continuum in this region

        #calculate absorption by dividing by continuum
        absorption1 = absorption_spectra1/absorption_continua1
    
        #record minimum depth of absorption band
        absorption_depths1[frame_index] = min(absorption1)
            
#        if frame_index in spectra_to_plot:
#            plt.plot(pixels1,spectrum1, color=linecolours[frame_index], linestyle="-")
#            plt.scatter(pixels1[continuum_range1[0]:continuum_range1[1]],absorption_continua1, color=linecolours[frame_index], linewidth=0, alpha=alpha)


#    plt.figure(figsize = (figx,figy))

    for frame_index,spectrum_index2 in enumerate(frame_indices2):
        spectrum2 = all_data_interpolated[file_index2,spectrum_index2,range(len(pixels2))]*(1.+mean_yscale)/2.+mean_yshift
        aotfs2 = all_aotfs_interpolated[file_index2,frame_indices2]

        absorption_pixels2 = [pixels2[continuum_range2[0]]]+[pixels2[continuum_range2[1]]]
        continuum_spectra2 = [spectrum2[continuum_range2[0]]]+[spectrum2[continuum_range2[1]]]
        
        #fit polynomial to continuum on either side of absorption band
        absorption_continua2= np.polyval(np.polyfit(absorption_pixels2,continuum_spectra2,1), pixels2[continuum_range2[0]:continuum_range2[1]])
        
        #store pixel numbers and data from part of spectrum containing absorption line
        absorption_spectra2 = spectrum2[continuum_range2[0]:continuum_range2[1]]
        #calculate continuum in this region

        #calculate absorption by dividing by continuum
        absorption2 = absorption_spectra2/absorption_continua2
    
        #record minimum depth of absorption band
        absorption_depths2[frame_index] = min(absorption2)

#        if frame_index in spectra_to_plot:
#            plt.plot(pixels2,spectrum2, color=linecolours[frame_index], linestyle="--")
#            plt.scatter(pixels2[continuum_range2[0]:continuum_range2[1]],absorption_continua2, color=linecolours[frame_index], linewidth=0, alpha=alpha)



#    plt.figure(figsize = (figx,figy))
#    plt.plot(aotfs1,absorption_depths1)
#    plt.plot(aotfs2,absorption_depths2)

    absorption_range = 6
    
    spectra_uncorrected = np.zeros((len(frame_indices1),len(pixels1)))
    spectra_corrected = np.zeros((len(frame_indices1),len(pixels1)))
    spectrum1_sg = np.zeros((len(frame_indices1),len(pixels1)))
    
    absorption_indices_all=[]

    for frame_index,spectrum_index1 in enumerate(frame_indices1):
        spectrum1 = all_data_interpolated[file_index1,spectrum_index1,range(len(pixels1))]
        
        """bad pixel removal"""
        spectrum1[40] = np.mean([spectrum1[39],spectrum1[41]])
        spectra_uncorrected[frame_index,:] = spectrum1

        """1st attempt at removing continuum shape"""
        spectrum1_sg[frame_index,:] = sg_filter(spectrum1, window_size=29, order=2)
        spectrum1_div = (spectrum1 - spectrum1_sg[frame_index,:])/spectrum1
        spectrum1_abs_div = np.abs((spectrum1 - spectrum1_sg[frame_index,:])/spectrum1)

        """find local minima, maxima"""        
        local_maxima = (np.diff(np.sign(np.diff(spectrum1_div))) < 0).nonzero()[0] + 1 # local max
        local_minima = (np.diff(np.sign(np.diff(spectrum1_div))) > 0).nonzero()[0] + 1 # local min
        
        """find points where divided initial spectrum deviates from filtered line and are also local minima"""
        absorption_indices = [local_minimum for local_minimum in local_minima if spectrum1_abs_div[local_minimum]>0.015 and local_minimum<311]
        absorption_indices_all.append(absorption_indices)
        absorption_indices=[57,89,94,99,100,108,251,260,267,306]
        absorption_regions = [[absorption_index-absorption_range,absorption_index+absorption_range] for absorption_index in absorption_indices]
                
        """for each absorption found, make linear fit across region bounded by absorption"""
        spectrum2[:] = spectrum1[:]
        for absorption_index in absorption_indices:
            pixel_number_absorption = [absorption_index-absorption_range,absorption_index+absorption_range] #find two points in x
            pixel_absorption = [spectrum1[absorption_index-absorption_range],spectrum1[absorption_index+absorption_range]] #find two points in y
            
            """generate linear spectrum between the two points"""
            coeffs = np.polyfit(pixel_number_absorption,pixel_absorption,1)
            spectrum2[(absorption_index-absorption_range):(absorption_index+absorption_range)] = np.polyval(coeffs, range(absorption_index-absorption_range,absorption_index+absorption_range,1))
        
        
        """now that absorptions have been interpolated over, re-run continuum fitting"""
        spectrum2_sg = sg_filter(spectrum2, window_size=29, order=2)
        spectrum2_div = (spectrum2 - spectrum2_sg)/spectrum2
        spectrum2_abs_div = np.abs((spectrum2 - spectrum2_sg)/spectrum2)

        spectra_corrected[frame_index,:] = spectrum2_sg
        

#        if frame_index in spectra_to_plot:
#            plt.figure(figsize = (figx,figy))
#            plt.plot(pixels1,spectrum1)
#            plt.plot(pixels1,spectrum2)
#            plt.plot(pixels1,spectrum1_sg[frame_index,:])
#            plt.plot(pixels1,spectrum2_sg)
#            plt.title("Spectrum 1, 2 Raw and SG Fit")
##            plt.scatter(np.asfarray(pixels1)[local_maxima],spectrum1[local_maxima])
#
#            for absorption_index in absorption_indices:
#                pixel_number_absorption = [absorption_index-absorption_range,absorption_index+absorption_range]
#                pixel_absorption = [spectrum1[absorption_index-absorption_range],spectrum1[absorption_index+absorption_range]]
#                plt.scatter(pixel_number_absorption,pixel_absorption)
#                plt.scatter(pixel_number_absorption,pixel_absorption)
#                plt.scatter(np.asfarray(pixels1)[absorption_index],spectrum1[absorption_index],marker="*")
#            
#            plt.figure(figsize = (figx,figy))
#            plt.plot(pixels1,spectrum1_div)
#            plt.plot(pixels1,spectrum2_div)
#            plt.scatter(np.asfarray(pixels1)[absorption_indices],np.asfarray(spectrum1_div)[absorption_indices])
#            plt.scatter(np.asfarray(pixels1)[absorption_indices],np.asfarray(spectrum2_div)[absorption_indices])
#            plt.title("Spectrum 1, 2 Divided with Absorption Centres Labelled")

#    animate=True
    animate=False


    aotfs = all_aotfs_interpolated[file_index1,frame_indices1]
    absorption_plot_indices=[251]#89,99,260] #test some absorptions
    for absorption_index in absorption_plot_indices:
        absorption_depth = np.zeros(len(frame_indices1))
        absorption = np.zeros((len(frame_indices1),absorption_range*2))
        absorption_continua = np.zeros((len(frame_indices1),absorption_range*2))
        
        for frame_index,spectrum_index1 in enumerate(frame_indices1):
            continuum_points = [absorption_index-absorption_range,absorption_index+absorption_range]
            continuum_spectrum = [spectra_corrected[frame_index,continuum_points[0]],spectra_corrected[frame_index,continuum_points[1]]]
            
            #fit linear to continuum on either side of absorption band
            absorption_continua[frame_index,:] = np.polyval(np.polyfit(continuum_points,continuum_spectrum,1), pixels1[continuum_points[0]:continuum_points[1]])
            centre_continuum = np.polyval(np.polyfit(continuum_points,continuum_spectrum,1), absorption_index)
            absorption[frame_index,:] = spectra_uncorrected[frame_index,continuum_points[0]:continuum_points[1]]/absorption_continua[frame_index,:]
            absorption_depth[frame_index] = spectra_uncorrected[frame_index,absorption_index]/centre_continuum
            
#            if frame_index in spectra_to_plot:
#                plt.figure(figsize = (figx,figy))
#                plt.plot(pixels1,spectra_uncorrected[frame_index,:])
#                plt.scatter(pixels1[continuum_points[0]:continuum_points[1]],absorption_continua)
#                plt.plot(pixels1[continuum_points[0]:continuum_points[1]],spectra_corrected[frame_index,continuum_points[0]:continuum_points[1]])
#                plt.scatter(pixels1[absorption_index],spectra_uncorrected[frame_index,absorption_index])

        if animate:
            
            frame_index=0
            """code to animate the plot"""
            fig=plt.figure(figsize = (figx,figy))
        
            ax1 = plt.subplot2grid((2,1),(0,0))
            ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)
        
            max_value = np.max(spectra_uncorrected)
            ax1.set_ylim((0,max_value))
            ax2.set_ylim((np.min(absorption_depth),np.max(absorption_depth)))
            n_frames = len(frame_indices1)
            plot1, = ax1.plot(pixels1,spectra_uncorrected[frame_index,:], color="k", animated=True)
            plot2, = ax1.plot(pixels1[continuum_points[0]:continuum_points[1]],absorption_continua[frame_index,:],color="g",marker="o",linewidth=0,alpha = 0.5, animated=True)
            plot3, = ax1.plot(pixels1,spectra_corrected[frame_index,:],color="r", animated=True)
            plot4, = ax1.plot(pixels1,spectrum1_sg[frame_index,:],color="b", animated=True)
            plot5, = ax2.plot(pixels1[continuum_points[0]:continuum_points[1]],absorption[frame_index,:],color="m", animated=True)
            text1 = ax1.text(10,max_value-20000,"AOTF Frequency = %ikHz\nIndex=%i" %(aotfs[frame_index],frame_index))
            
   
            def updatefig(frame_index): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
                if np.mod(frame_index,500)==0:
                    print(frame_index)
                plot1.set_data(pixels1,spectra_uncorrected[frame_index,:])
                plot2.set_data(pixels1[continuum_points[0]:continuum_points[1]],absorption_continua[frame_index,:])
                plot3.set_data(pixels1,spectra_corrected[frame_index,:])
                plot4.set_data(pixels1,spectrum1_sg[frame_index,:])
                plot5.set_data(pixels1[continuum_points[0]:continuum_points[1]],absorption[frame_index,:])
                text1.set_text("AOTF Frequency = %ikHz\nIndex=%i" %(aotfs[frame_index],frame_index))
                return plot1,plot2,plot3,plot4,text1,
        
            ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=20, blit=True)

                
            
            
        plt.figure(figsize = (figx,figy))
        plt.plot(aotfs,absorption_depth)
        plt.title("Absorption Index %i" %absorption_index)

        plt.figure(figsize = (figx,figy))
        plt.imshow(spectra_corrected)
        
if option==46:
    """test methods of analysing miniscans part 2"""
#    animate=False
    animate=True; file_to_animate=[2]

#    fitting_type="fft"
    fitting_type="linear"
#    fitting_type="quadratic"
    filter_end=20
    
    if fitting_type=="fft":
        fitting_type_name=fitting_type+"_cutoff=%i" %filter_end
    else:
        fitting_type_name=fitting_type

    xshifts=[0,3,0,0,0,0,0,0,0,0,0,0,0,5,5,0]
    
    line=15
    absorption_range=200
    
    half_filter=True
#    half_filter=False
    
#    yscale=[1.0,2.13237,1.13237,1.13237,1.13237]
#    yshift=[0.,872.834,872.834,872.834,872.834]

    subtitle=""
    if line==1:
        line_centre=267
        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(0,[350,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
        frame_ranges=[(0,[0,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
        files_not_to_fit = [0,1]
        subtitle="_good"
    elif line==2:
        line_centre=95
        continuum_range = [line_centre-6,line_centre+6]
        frame_ranges=[(0,[250,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
        files_not_to_fit = [0,1]
#    elif line==3:
#        line_centre=199
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(0,[0,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
#    elif line==4:
#        line_centre=309
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(0,[0,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
#    elif line==5:
#        line_centre=78
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(0,[0,510]),(1,[0,510]),(2,[0,510]),(3,[0,510])]
#        files_not_to_fit = [2,3]
    elif line==6:
        line_centre=302
        continuum_range = [line_centre-6,line_centre+6]
        frame_ranges=[(5,[0,510]),(6,[0,510]),(7,[0,510]),(8,[0,510]),(9,[0,510])]
        files_not_to_fit = [0]
        subtitle="_good"
#    elif line==7:
#        line_centre=204
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(4,[0,510]),(5,[0,510]),(6,[0,510]),(7,[0,510])]
    elif line==8:
        line_centre=247
        continuum_range = [line_centre-6,line_centre+6]
        frame_ranges=[(4,[0,510]),(5,[0,510]),(6,[0,400])] #done
        subtitle="_good"
        files_not_to_fit = [0]
        absorption_range=300
#    elif line==9:
#        line_centre=76
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(5,[0,510]),(6,[0,510]),(7,[0,510]),(8,[0,510]),(9,[0,510]),(10,[0,510]),(11,[0,510])]
    elif line==10:
        line_centre=221
        continuum_range = [line_centre-6,line_centre+6]
        frame_ranges=[(7,[0,510]),(8,[0,510]),(9,[0,510]),(10,[0,510]),(11,[0,510])]
        subtitle="_ok"
        files_not_to_fit = [1]
#    elif line==11:
#        line_centre=235
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(5,[0,510]),(6,[0,510]),(7,[0,510]),(8,[0,510]),(9,[0,510]),(10,[0,510]),(11,[0,510])]
#    elif line==12:
#        line_centre=241
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(9,[0,510]),(10,[0,510]),(11,[0,510]),(12,[0,510]),(13,[0,510]),(14,[0,510])]
#    elif line==13:
#        line_centre=154
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(12,[0,510]),(13,[0,510]),(14,[0,510])]
#    elif line==14:
#        line_centre=302
#        continuum_range = [line_centre-6,line_centre+6]
#        frame_ranges=[(5,[0,510])]
#        subtitle="_testing"
    elif line==15:
        line_centre=241
        continuum_range = [line_centre-8,line_centre+8]
        frame_ranges=[(2,[0,510])]
        subtitle="_CH4_testing"
        files_not_to_fit = [0,1]
    elif line==16:
        line_centre=200
        continuum_range = [line_centre-8,line_centre+8]
        frame_ranges=[(2,[0,510]),(3,[0,510])]
        subtitle="_CH4_testing"
        files_not_to_fit = [0]
    elif line==17:
        line_centre=155
        continuum_range = [line_centre-6,line_centre+6]
        frame_ranges=[(1,[0,510]),(2,[0,510]),(3,[0,510])]
        subtitle="_CH4_testing"
        files_not_to_fit = [0,1]
        
    #make list of required files
    file_indices = [frame_range[0] for frame_range in frame_ranges]
    if animate==True:
        file_indices=file_to_animate

    pixels = range(320)

    
    #make empty arrays to hold all data
    all_data_interpolated = np.zeros((len(file_indices),2*255,320))
    all_aotfs_interpolated = np.zeros((len(file_indices),2*255))
    #now loop through required files, reading in the detector data and aotf frequencies from each and storing them in the empty arrays
    for loop_index,file_index in enumerate(file_indices):
    
        """get data from file"""
        print("Reading in file %i:\t%s" %(file_index,obspaths[file_index]))
        detector_data_bins = get_dataset_contents(hdf5_files[file_index],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotf_freq_all = get_dataset_contents(hdf5_files[file_index],"AOTFFrequency")[0]
        measurement_temperature = np.mean(get_dataset_contents(hdf5_files[file_index],"AOTF_TEMP_%s" %channel.upper())[0][2:10])
        print("AOTF range at %0.1fC:\t%i\t%i " %(measurement_temperature,min(aotf_freq_all),max(aotf_freq_all)))
        if aotf_freq_all[0]<16000: print("Warning: AOTFs in %s are too low - small signal" %obspaths[file_index])
        hdf5_files[file_index].close()
        
        #code to correct incorrect telecommand stuck onto detector data by SINBAD
        aotf_freq_range = np.arange(min(aotf_freq_all),min(aotf_freq_all)+2*256,2)
        aotf_freq_corrected = np.append(np.append(aotf_freq_range,aotf_freq_range),aotf_freq_range[0:28])
        if max(aotf_freq_all)-min(aotf_freq_all) != 510:
            print("Error: AOTFs may not be correct") #print(error and stop program if there is a problem
            stop()
            
        
        detector_data_binned = np.mean(detector_data_bins[:,6:18,:],axis=1) #average detector spatial data to make one spectrum per AOTF frequency. Bins will be done later
        """choose temporal binning"""
#        #each file contains two and a bit sweeps through AOTF frequencies. Average the first two sweeps together
#        detector_data_binned_mean = np.mean(np.asarray([detector_data_binned[0:256,:],detector_data_binned[256:512,:]]),axis=0)
        #or just take first 256 values
        detector_data_binned_mean = detector_data_binned[0:256,:]


        #interpolate aotf frequencies and detector data into 1kHz steps
        all_aotfs_interpolated[loop_index,:] = np.arange(aotf_freq_range[0],aotf_freq_range[-1],1.0)
        for pixel_index in range(detector_data_binned_mean.shape[1]):
            all_data_interpolated[loop_index,:,pixel_index] = np.interp(all_aotfs_interpolated[loop_index,:],aotf_freq_range,detector_data_binned_mean[:,pixel_index])

    print("Checking pixels:\t%i\t%i\t%i" %(continuum_range[0],line_centre,continuum_range[1]))
    if animate:
        
        frame_index=0
        """code to animate the plot"""
        fig_anim = plt.figure(figsize = (figx,figy))
        ax_anim = fig_anim.add_subplot(111)
        
        data_in = all_data_interpolated[0,:,:]
        x_in = pixels
        number_in = all_aotfs_interpolated[0,:]
    
        max_value = np.max(data_in)
        ax_anim.set_ylim((0,max_value))
        n_frames = len(data_in[:,0])
        plot1, = ax_anim.plot(x_in,data_in[frame_index,:], color="k", animated=True)
        text1 = ax_anim.text(10,max_value-20000,"AOTF Frequency = %ikHz\nIndex=%i" %(number_in[frame_index],frame_index))

   
        def updatefig(frame_index): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
            if np.mod(frame_index,500)==0:
                print(frame_index)
            plot1.set_data(x_in,data_in[frame_index,:])
            text1.set_text("AOTF Frequency = %ikHz\nIndex=%i" %(number_in[frame_index],frame_index))
            return plot1,text1,
    
        ani = animation.FuncAnimation(fig_anim, updatefig, frames=n_frames, interval=20, blit=True)

    else:
    #    linestyles = ["-", "--", "-", "--","-", "--","-", "--","-", "--","-", "--"] * 100
        linecolours = ["r","orange","y","lime","g","c","b","k","m","pink"] * 100
        alpha=0.6 #make more readable.
        
        absorption_depths = []
        aotfs = []
    
        colour_loop=-1
        
        fig0 = plt.figure(figsize = (figx-5,figy-3))
        ax0 = fig0.add_subplot(111)
        ax0.set_ylim([0,60000])
        ax0.set_xlabel("Pixel number")
        ax0.set_ylabel("Signal ADUs")
        fig1 = plt.figure(figsize = (figx-5,figy))
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel("AOTF frequency")
        ax1.set_ylabel("Absorption depth")
#        fig2 = plt.figure(figsize = (figx,figy))
#        ax2 = fig2.add_subplot(111)
#        fig3 = plt.figure(figsize = (figx-5,figy-3))
#        ax3 = fig3.add_subplot(111)
#        ax3.set_xlabel("FFT frequency")
#        ax3.set_ylabel("FFT amplitude")
        
        file_start_indices=[]
        spectrum_loop = -1
        
        for loop_index,file_index in enumerate(file_indices):
            xshift = xshifts[file_index]
            print("xshift=%i" %(xshift))
            
            frame_indices = range(frame_ranges[loop_index][1][0],frame_ranges[loop_index][1][1],1)
            spectra_to_plot = frame_indices[::100]
                 
                
            for frame_index,spectrum_index in enumerate(frame_indices):
        
                spectrum = all_data_interpolated[loop_index,spectrum_index,range(len(pixels))] # * yscale[file_index] + yshift[file_index]
    
                if xshift==0:
                    spectrum = spectrum
                elif xshift>0.:
                    spectrum = np.asfarray(list(spectrum[xshift::]) + [spectrum[-1]]*xshift)
                elif xshift<0.:
                    spectrum = np.asfarray([spectrum[0]]*(xshift*-1)+list(spectrum[:xshift:]))
                
                
                #fit polynomial to continuum on either side of absorption band
                if fitting_type=="linear":
                    absorption_pixels = [pixels[continuum_range[0]]]+[pixels[continuum_range[1]]]
                    continuum_spectra = [spectrum[continuum_range[0]]]+[spectrum[continuum_range[1]]]
                    absorption_continua = np.polyval(np.polyfit(absorption_pixels,continuum_spectra,1), pixels[continuum_range[0]:continuum_range[1]+1])
                elif fitting_type=="quadratic":
                    absorption_pixels = list(pixels[(continuum_range[0]-5):(continuum_range[0])])+list(pixels[(continuum_range[1]):(continuum_range[1]+5)])
                    continuum_spectra = list(spectrum[(continuum_range[0]-5):(continuum_range[0])])+list(spectrum[(continuum_range[1]):(continuum_range[1]+5)])
                    absorption_continua = np.polyval(np.polyfit(absorption_pixels,continuum_spectra,2), pixels[continuum_range[0]:continuum_range[1]+1])
                
                #make spectrum without absorptions
                cut_spectrum = fft_filter(pixels,spectrum,filter_end=filter_end)
                    
                #calculate absorption by dividing by continuum
                if fitting_type=="linear" or fitting_type=="quadratic":
                    absorption = spectrum[continuum_range[0]:continuum_range[1]+1]/absorption_continua
                elif fitting_type=="fft":
                    absorption = spectrum[continuum_range[0]:continuum_range[1]+1]/cut_spectrum[continuum_range[0]:continuum_range[1]+1]
                    absorption_continua = cut_spectrum[continuum_range[0]:continuum_range[1]+1]
    
                #record minimum depth of absorption band
                absorption_depths.append(absorption[line_centre-continuum_range[0]])
                aotfs.append(all_aotfs_interpolated[loop_index,spectrum_index])
                
                spectrum_loop += 1
                    
                if spectrum_index in spectra_to_plot:
                    
#                    ax2.plot(spectrum,label="Original", color=linecolours[colour_loop], linestyle="--")
##                    ax2.plot(uncut_spectrum,label="Uncut", color=linecolours[colour_loop], linestyle="-")
#                    ax2.plot(cut_spectrum,label="W lt 6", color=linecolours[colour_loop], linestyle=":")
#                    ax2.legend()
                    
                    if colour_loop==-1:
                        
                        f_spectrum,cut_f_spectrum = fft_filter(pixels,spectrum,filter_end=filter_end,output_freqs=True)
                            
#                        ax3.plot(f_spectrum)
#                        ax3.plot(cut_f_spectrum)
#                        ax3.set_ylim([-1e6,1e6])
                        
                        
                    colour_loop += 1
                    ax0.plot(pixels,spectrum, color=linecolours[colour_loop], label="%i" %spectrum_index)
                    ax0.scatter(pixels[continuum_range[0]:continuum_range[1]+1],absorption_continua, color=linecolours[colour_loop], linewidth=0, alpha=alpha)
                    ax0.legend()
            #record joins between spectra
            file_start_indices.append(spectrum_loop)
            #add NaN to stop joining of spectra
#            aotfs.append(np.nan)
#            absorption_depths.append(np.nan)
            
        absorption_min_index = np.where(absorption_depths== np.nanmin(absorption_depths))[0][0]
        absorption_min_aotf = aotfs[absorption_min_index]
        absorption_min_value = absorption_depths[absorption_min_index]
        
        def func1(x, a, b, c): #inverted gaussian
            return 1.0 - (a * np.exp(-(((x-b))**2.0) / (2.0 * c**2.0)))
        starting_values1 = [1.0-absorption_min_value,absorption_min_aotf+0.001,60.]
        def func2(x, a, b, c, d): #inverted sinc2
            return 1.0 - (a * (np.sin((x-b)/c)**2.0) / (((x-b)/c)**2.0) + d)
        starting_values2 = [1.0-absorption_min_value,absorption_min_aotf+0.001,60.0,0.0]
        def func3(x, a, b, c, d, e): #damped inverted sinc2
            return 1.0 - ((a * np.exp(-(((x-b))**2.0) / (2.0 * c**2.0)) * (np.sin((x-b)/e))**2.0) / (((x-b)/e)**2.0) + d)
        starting_values3 = [1.0-absorption_min_value,absorption_min_aotf+0.001,60.0,-0.005,100.0]
        def func_continuum4(x, a, b, c, d): #sine
#            return a * np.sin((x-b)*np.pi/(2*c)) + d
            return a * np.sin((x-b)/c) + d
        starting_values4 = [0.2,0.0,40.0,1.0]
            
        absorption_range_start = np.max([0,absorption_min_index-absorption_range])
        plot_range_start = np.max([0,absorption_min_index-absorption_range-500])
        absorption_range_end = np.min([len(aotfs),absorption_min_index+absorption_range])
        plot_range_end = np.min([len(aotfs),absorption_min_index+absorption_range+500])

        ax1.plot(aotfs,absorption_depths,label="Absorption Depth")
        corrected_absorption_depths = absorption_depths.copy()

        try:
            popt1,_= curve_fit(func1, np.asfarray(aotfs[absorption_range_start:absorption_range_end]),np.asfarray(absorption_depths[absorption_range_start:absorption_range_end]), p0=np.array(starting_values1), maxfev=100000)
        except RuntimeError:
            print("Warning func1 convergence not found, reducing range")
            popt1,_= curve_fit(func1, np.asfarray(aotfs[absorption_range_start+100:absorption_range_end-100]),np.asfarray(absorption_depths[absorption_range_start+100:absorption_range_end-100]), p0=np.array(starting_values1), maxfev=100000)
            
        fit_func1 = func1(np.asfarray(aotfs[plot_range_start:plot_range_end]), popt1[0], popt1[1], popt1[2])
        fit_goodness1 = np.sum((absorption_depths[absorption_range_start:absorption_range_end] - func1(np.asfarray(aotfs[absorption_range_start:absorption_range_end]), popt1[0], popt1[1], popt1[2])) **2)
        ax1.plot(aotfs[plot_range_start:plot_range_end],fit_func1,"r",label="Gaussian fit: Depth=%0.2f Freq=%0.0f FWHM=%0.1f, chi=%0.3f" %(popt1[0],popt1[1],(2.0*np.sqrt(2.0*np.log(2.0))*popt1[2]),fit_goodness1))
        
        try:
            popt2,_= curve_fit(func2, np.asfarray(aotfs[absorption_range_start:absorption_range_end]),np.asfarray(absorption_depths[absorption_range_start:absorption_range_end]), p0=np.array(starting_values2), maxfev=100000)
        except RuntimeError:
            print("Warning func1 convergence not found, reducing range")
            popt2,_= curve_fit(func2, np.asfarray(aotfs[absorption_range_start+100:absorption_range_end-100]),np.asfarray(absorption_depths[absorption_range_start+100:absorption_range_end-100]), p0=np.array(starting_values2), maxfev=100000)

        fit_func2 = func2(np.asfarray(aotfs[plot_range_start:plot_range_end]), popt2[0], popt2[1], popt2[2], popt2[3])
        fit_goodness2 = np.sum((absorption_depths[absorption_range_start:absorption_range_end] - func2(np.asfarray(aotfs[absorption_range_start:absorption_range_end]), popt2[0], popt2[1], popt2[2], popt2[3])) **2)
        ax1.plot(aotfs[plot_range_start:plot_range_end],fit_func2,"g",label="Sinc squared: Depth=%0.2f Freq=%0.0f Coeff=%0.1f, chi=%0.3f" %(popt2[0],popt2[1],popt2[2],fit_goodness2))

        local_maxima = (np.diff(np.sign(np.diff(fit_func2))) < 0).nonzero()[0] + 1 # local max
        print("Sinc squared maximum aotfs:")
        for local_maximum in list(local_maxima):
            print(aotfs[local_maximum])

        """"calculate FWHM of the sinc squared"""
        
        

#        try:
#            popt3,_= curve_fit(func3, np.asfarray(aotfs[absorption_range_start:absorption_range_end]),np.asfarray(absorption_depths[absorption_range_start:absorption_range_end]), p0=np.array(starting_values3), maxfev=100000)
#        except RuntimeError:
#            print("Warning func1 convergence not found, reducing range")
#            popt3,_= curve_fit(func3, np.asfarray(aotfs[absorption_range_start+100:absorption_range_end-100]),np.asfarray(absorption_depths[absorption_range_start+100:absorption_range_end-100]), p0=np.array(starting_values3), maxfev=100000)
#
#        fit_func3 = func3(np.asfarray(aotfs[plot_range_start:plot_range_end]), popt3[0], popt3[1], popt3[2], popt3[3], popt3[4])
#        fit_goodness3 = np.sum((absorption_depths[absorption_range_start:absorption_range_end] - func3(np.asfarray(aotfs[absorption_range_start:absorption_range_end]), popt3[0], popt3[1], popt3[2], popt3[3], popt3[4])) **2)
#        ax1.plot(aotfs[plot_range_start:plot_range_end],fit_func3,"k",label="Exponential sinc squared: %0.2f %0.0f %0.1f %0.1f, %0.3f" %(popt3[0],popt3[1],popt3[2],popt3[4],fit_goodness3))




        sine_ranges=[]
        if not(0 < absorption_min_index < file_start_indices[0]):
            if 0 not in files_not_to_fit:
                sine_ranges.append([0,file_start_indices[0]-1])
        if len(file_start_indices)>1:
            if not(file_start_indices[0] < absorption_min_index < file_start_indices[1]):
                if 1 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[0]+1,file_start_indices[1]-1])
        if len(file_start_indices)>2:
            if not(file_start_indices[1] < absorption_min_index < file_start_indices[2]):
                if 2 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[1]+1,file_start_indices[2]-1])
        if len(file_start_indices)>3:
            if not(file_start_indices[2] < absorption_min_index < file_start_indices[3]):
                if 3 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[2]+1,file_start_indices[3]-1])
        if len(file_start_indices)>4:
            if not(file_start_indices[3] < absorption_min_index < file_start_indices[4]):
                if 4 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[3]+1,file_start_indices[4]-1])
        if len(file_start_indices)>5:
            if not(file_start_indices[4] < absorption_min_index < file_start_indices[5]):
                if 5 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[4]+1,file_start_indices[5]-1])
        if len(file_start_indices)>6:
            if not(file_start_indices[5] < absorption_min_index < file_start_indices[6]):
                if 6 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[5]+1,file_start_indices[6]-1])
        if len(file_start_indices)>7:
            if not(file_start_indices[6] < absorption_min_index < file_start_indices[7]):
                if 7 not in files_not_to_fit:
                    sine_ranges.append([file_start_indices[6]+1,file_start_indices[7]-1])
        
        
        
        for sine_range in sine_ranges:
    #        popt4,_= curve_fit(func_continuum4, np.asfarray(aotfs[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])],np.asfarray(absorption_depths[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])], p0=np.array(starting_values4), maxfev=100000)
    #        fit_continuum_func4 = func_continuum4(np.asfarray(aotfs[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])], popt4[0], popt4[1], popt4[2], popt4[3])
    #        fit_goodness4 = np.sum((np.asfarray(absorption_depths[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])] - func_continuum4(np.asfarray(aotfs[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])], popt4[0], popt4[1], popt4[2], popt4[3])) **2)
    #        ax1.plot(np.asfarray(aotfs[sine_range[0]:sine_range[1]])[np.isfinite(aotfs[sine_range[0]:sine_range[1]])],fit_continuum_func4,label="Sinusoidal: Amp=%0.4f Wavelength=%0.1f, chi=%0.3f" %(popt4[0],(popt4[2]*2.0*np.pi),fit_goodness4))
    #        #remove sinusoidal shape
    #        corrected_absorption_depths[sine_range[0]:sine_range[1]] = absorption_depths[sine_range[0]:sine_range[1]]/fit_continuum_func4
            popt4,_= curve_fit(func_continuum4, np.asfarray(aotfs[sine_range[0]:sine_range[1]]),np.asfarray(absorption_depths[sine_range[0]:sine_range[1]]), p0=np.array(starting_values4), maxfev=100000)
            fit_continuum_func4 = func_continuum4(np.asfarray(aotfs[sine_range[0]:sine_range[1]]), popt4[0], popt4[1], popt4[2], popt4[3])
            fit_goodness4 = np.sum((absorption_depths[sine_range[0]:sine_range[1]] - func_continuum4(np.asfarray(aotfs[sine_range[0]:sine_range[1]]), popt4[0], popt4[1], popt4[2], popt4[3])) **2)
            ax1.plot(aotfs[sine_range[0]:sine_range[1]],fit_continuum_func4,label="Sinusoidal: Amp=%0.4f Wavelength=%0.1f, chi=%0.3f" %(popt4[0],(popt4[2]*2.0*np.pi),fit_goodness4))
            #remove sinusoidal shape
            corrected_absorption_depths[sine_range[0]:sine_range[1]] = absorption_depths[sine_range[0]:sine_range[1]]/fit_continuum_func4


        ax1.plot(aotfs,corrected_absorption_depths,"k",label="Corrected absorption depth after sine function removal")
        
#        plt.figure(); plt.plot(fft_filter2(absorption_depths[1533:2042], filter_start=0, filter_end=100))

        new_title = "Files=%i-%i_absorption_freq=%i_fitting_type=%s" %(min(file_indices),max(file_indices),absorption_min_aotf,fitting_type_name)
        ax0.set_title(new_title)
        
        if save_figs: fig0.savefig(BASE_DIRECTORY+os.sep+new_title+subtitle+"_1.png")
        ax1.set_title(new_title)
        ax1.legend(loc="lower right")
        if save_figs: fig1.savefig(BASE_DIRECTORY+os.sep+new_title+subtitle+"_2.png")
#        if save_figs: fig3.savefig(BASE_DIRECTORY+os.sep+new_title+subtitle+"_4.png")


if option==47:
    """plot 2d grid of correct aotf vs sg filter abs depth"""
    
    #aotf to wavenumber coefficients from arnaud
    aotf2wvn = [-4.7146921e-7,0.168101589,0.001357194]
    
    
    #make empty arrays to hold all data
    nfiles = len(obspaths)
    nframes = len(obspaths)*510
    pixels = range(320)
    all_data_interpolated = np.zeros((nframes,320))
    all_aotfs_interpolated = np.zeros(nframes)
    aotf_starts=[]
    aotf_ends=[]

#    xshifts=[0,3,3,3,3,3,3,0,0,0,0,0,0,5,5,0] 
    xshifts=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #shift the spectra in each file by x pixels, so that they match up with the 
    absorption_range = 6 #when interpolating a continuum across an absorption band, 
    


    #now loop through files, reading in the detector data and aotf frequencies from each and storing them in the empty arrays
    for file_index,hdf5_file in enumerate(hdf5_files):
    
        """get data from file"""
        print("Reading in file %i: %s" %(file_index,obspaths[file_index]))
        detector_data_bins = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
        measurement_temperature = np.mean(get_dataset_contents(hdf5_file,"AOTF_TEMP_%s" %channel.upper())[0][2:10])
        print("AOTF range %i to %i at %0.1fC" %(min(aotf_freq_all),max(aotf_freq_all),measurement_temperature))
        if aotf_freq_all[0]<16000: print("Warning: AOTFs in %s are too low - small signal" %obspaths[file_index])
        hdf5_file.close()
        
        #code to correct incorrect telecommand stuck onto detector data by SINBAD
        aotf_freq_range = np.arange(min(aotf_freq_all),min(aotf_freq_all)+2*256,2)
        aotf_freq_corrected = np.append(np.append(aotf_freq_range,aotf_freq_range),aotf_freq_range[0:28])
        if max(aotf_freq_all)-min(aotf_freq_all) != 510:
            print("Error: AOTFs may not be correct") #print(error and stop program if there is a problem
            stop()
            
        #average vertical detector data from rows 6 to 18 to make one spectrum per AOTF frequency. Bins will be done later
        detector_data_binned = np.mean(detector_data_bins[:,6:18,:],axis=1) 
        #each file contains two and a bit sweeps through AOTF frequencies. Just take first run through
        detector_data_binned_mean = detector_data_binned[0:256,:]

        """shift spectra in x direction to match files"""
        for frame_index in range(256):
            xshift = xshifts[file_index]
            spectrum = detector_data_binned_mean[frame_index]
            if xshift==0:
                spectrum_shifted = spectrum
            elif xshift>0.:
                spectrum_shifted = np.asfarray(list(spectrum[xshift::]) + [spectrum[-1]]*xshift)
            elif xshift<0.:
                spectrum_shifted = np.asfarray([spectrum[0]]*(xshift*-1)+list(spectrum[:xshift:]))
            detector_data_binned_mean[frame_index] = spectrum_shifted


        #interpolate aotf frequencies and detector data into 1kHz steps and store frequencies and spectra in a big array for all files
        all_aotfs_interpolated[file_index*510:(file_index+1)*510] = np.arange(aotf_freq_range[0],aotf_freq_range[-1],1.0)
        for pixel_index in range(detector_data_binned_mean.shape[1]):
            all_data_interpolated[file_index*510:(file_index+1)*510,pixel_index] = np.interp(all_aotfs_interpolated[file_index*510:(file_index+1)*510],aotf_freq_range,detector_data_binned_mean[:,pixel_index])

        aotf_starts.append(int(aotf_freq_range[0])) #record first and last aotf frequency in each file
        aotf_ends.append(int(aotf_freq_range[-1]))


    """determine start and end points of overlapping region"""
    #make a list of all overlapping indices i.e. where a given aotf frequency is present in 2 files. The overlapping region will be deleted from the 2d array so that the files are continuous
    range_starts=[0]
    range_ends=[0]
    overlapping_indices=[]
    for index in range(nfiles-1):
        range_starts.append(aotf_ends[index] - aotf_starts[index] + range_starts[-1])
        range_ends.append(range_starts[-1] + aotf_ends[index] - aotf_starts[index+1])
        overlapping_indices.extend(range(range_starts[-1],range_ends[-1]))
    

    #make more empty array to hold data
    spectra_uncorrected = np.zeros((nframes,len(pixels)))
    spectra_corrected = np.zeros((nframes,len(pixels)))
    spectrum1_sg = np.zeros((nframes,len(pixels)))
    spectra_absorptions = np.zeros((nframes,len(pixels)))
    
#    absorption_indices_all=[]
    
  
    """Loop through each frame"""
    for frame_index in range(nframes):
        spectrum1 = all_data_interpolated[frame_index,:]



        
        """pixel 40 is always bad, and can affect the fitting routine, so remove it by interpolation of pixels 39 and 40"""
        spectrum1[40] = np.mean([spectrum1[39],spectrum1[41]])
        spectra_uncorrected[frame_index,:] = spectrum1

        """fit savitsky golay filter (running mean) to spectrum"""
        spectrum1_sg[frame_index,:] = sg_filter(spectrum1, window_size=29, order=2)
#        spectrum1_div = (spectrum1 - spectrum1_sg[frame_index,:])/spectrum1
#        spectrum1_abs_div = np.abs((spectrum1 - spectrum1_sg[frame_index,:])/spectrum1)

#        """find local minima, maxima"""        
#        local_maxima = (np.diff(np.sign(np.diff(spectrum1_div))) < 0).nonzero()[0] + 1 # local max
#        local_minima = (np.diff(np.sign(np.diff(spectrum1_div))) > 0).nonzero()[0] + 1 # local max
        
#        """find points where divided initial spectrum deviates from filtered line and are also local minima. Assume these are absorption bands"""
#        absorption_indices = [local_minimum for local_minimum in local_minima if spectrum1_abs_div[local_minimum]>0.015 and local_minimum<311 and local_minimum>10]
#        absorption_indices_all.append(absorption_indices)
#        absorption_indices=[57,89,94,99,100,108,251,260,267,306] #manually select absorption bands
#        absorption_regions = [[absorption_index-absorption_range,absorption_index+absorption_range] for absorption_index in absorption_indices]
                
#        """for each absorption found, make linear fit across region bounded by absorption"""
#        spectrum2 = spectrum1[:]
#        for absorption_index in absorption_indices:
#            pixel_number_absorption = [absorption_index-absorption_range,absorption_index+absorption_range] #find two points in x
#            pixel_absorption = [spectrum1[absorption_index-absorption_range],spectrum1[absorption_index+absorption_range]] #find two points in y
            
#            """generate linear spectrum between the two points"""
#            coeffs = np.polyfit(pixel_number_absorption,pixel_absorption,1)
#            spectrum2[(absorption_index-absorption_range):(absorption_index+absorption_range)] = np.polyval(coeffs, range(absorption_index-absorption_range,absorption_index+absorption_range,1))
        
        
#        """now that absorptions have been interpolated over, re-run continuum fitting"""
#        spectrum2_sg = sg_filter(spectrum2, window_size=29, order=2)
#        spectrum2_div = (spectrum2 - spectrum2_sg)/spectrum2
#        spectrum2_abs_div = np.abs((spectrum2 - spectrum2_sg)/spectrum2)
        
        """divide original spectrum by this filtered spectrum to remove the general shape, leaving only the absorption bands"""
        spectrum2_absorption = spectrum1/spectrum1_sg[frame_index,:] 

#        spectra_corrected[frame_index,:] = spectrum2_sg
        spectra_absorptions[frame_index,:] = spectrum2_absorption #store in variable

    """delete overlapping regions"""
    spectra_absorptions = np.delete(spectra_absorptions,overlapping_indices,axis=0)
    aotfs_corrected = np.delete(all_aotfs_interpolated,overlapping_indices)
                
    wavenumbers = np.polyval(aotf2wvn, aotfs_corrected)
        
    x = wavenumbers
    y = pixels
    X,Y=np.meshgrid(x,y)

    plt.figure(figsize = (figx,figy))
    plt.pcolormesh(X,Y,1.0-(np.transpose(spectra_absorptions)))#, aspect=np.int(nfiles/1.5), interpolation=None)
    plt.axis([x.min(), x.max(), y[0], y[-1]])


if option == 48:
    
    """a few line plots from MCO1 nadir data for SWT11"""
    
    detectorDataAll = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotfFrequencyAll = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
    measurementTemperature = np.mean(get_dataset_contents(hdf5_file,"AOTF_TEMP_%s" %channel.upper())[0][2:10])
    hdf5_file.close()
    
    chosenIndex = 559
    chosenAotfFrequency = aotfFrequencyAll[chosenIndex]
    chosenFrame = detectorDataAll[chosenIndex,:,:]
    plt.figure(figsize = (figx-5,figy-3))
    for chosenLine in chosenFrame:
        plt.plot(chosenLine)



if option ==49:
    
    soDetectorDataAll = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    lnoDetectorDataAll = get_dataset_contents(hdf5_files[1],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    hdf5_files[0].close()
    hdf5_files[1].close()
    
    frameNumber = 20
    columnNumber = 180
    vPixels = np.arange(24)
    soStartPixel = 2
    lnoStartPixel = 0
    
    plt.figure(figsize = (figx-5,figy-3))
    plt.imshow(soDetectorDataAll[frameNumber,:,:], aspect=4)
    plt.figure(figsize = (figx-5,figy-3))
    plt.imshow(lnoDetectorDataAll[frameNumber,:,:], aspect=4)
    
    soDetectorLine = soDetectorDataAll[frameNumber,:,columnNumber]
    
    soPolyCoeffs = np.polyfit(vPixels[soStartPixel::],soDetectorLine[soStartPixel::], 2)
    soPoly = np.polyval(soPolyCoeffs, vPixels[soStartPixel::])
    




    lnoDetectorLine = lnoDetectorDataAll[frameNumber,:,columnNumber]
    
    lnoPolyCoeffs = np.polyfit(vPixels[lnoStartPixel::],lnoDetectorLine[lnoStartPixel::], 2)
    lnoPoly = np.polyval(lnoPolyCoeffs, vPixels[lnoStartPixel::])
    
    plt.figure(figsize = (figx-5,figy-3))
    plt.plot(vPixels[soStartPixel::],soDetectorLine[soStartPixel::], label="SO Vertical Slice")
    plt.plot(vPixels[soStartPixel::],soPoly, label="SO Polynomial")
    plt.plot(vPixels[lnoStartPixel::],lnoDetectorLine[lnoStartPixel::], label="LNO Vertical Slice")
    plt.plot(vPixels[lnoStartPixel::],lnoPoly, label="LNO Polynomial")
    plt.legend()

    print(soPolyCoeffs)
    print("SO centre at pixel %0.1f" %(-1.0*soPolyCoeffs[1]/(2.0*soPolyCoeffs[0])))
    print(lnoPolyCoeffs)
    print("LNO centre at pixel %0.1f" %(-1.0*lnoPolyCoeffs[1]/(2.0*lnoPolyCoeffs[0])))







