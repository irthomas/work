# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:25:28 2018

@author: iant


DETERMINE OCCULTATION BORESIGHT BASED ON OBSERVATION TIME
"""

import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
#from scipy import stats
#import scipy.optimize

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

#from datetime import datetime
#from matplotlib import rcParams
#import matplotlib.pyplot as plt
#import matplotlib as mpl


import spiceypy as sp


BORESIGHT_VECTOR_FILE_PATH = os.path.join(r"C:\Users\iant\Dropbox\NOMAD\Python", "boresight_vectors.txt")


#load spiceypy kernels if required
KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
#KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
#METAKERNEL_NAME = "em16_plan_win.tm"
METAKERNEL_NAME = "em16_ops_win.tm"
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)

#BORESIGHTS UPDATED FOR MTP005 (11 AUG)

fileNames_in = {"20180421_202111_0p3a_SO_1_E_134":"UVIS OLD",
                "20180430_020404_0p3a_SO_1_I_134":"SO OLD",
                "20180606_013055_0p3a_SO_1_E_134":"SO OLD",
                "20180806_010959_0p3a_SO_1_I_134":"SO OLD",
                "20180831_004625_0p3a_SO_1_I_134":"SO NEW",
                "20180925_154510_0p3a_SO_1_I_134":"SO NEW",
                "20181006_185748_0p3a_SO_2_I_134":"UVIS NEW",
                "20181027_165550_0p3a_SO_2_I_134":"UVIS NEW",
                
                "20180521_212349_0p3a_SO_1_I_134":"MIR",
                "20180521_214930_0p3a_SO_1_E_134":"MIR",
                "20180522_031715_0p3a_SO_1_S":"MIR",
                "20180522_034405_0p3a_SO_1_E_134":"MIR",
                "20180527_030547_0p3a_SO_1_I_134":"TIRVIM",
                "20180527_034419_0p3a_SO_1_E_134":"TIRVIM",
                "20180617_004358_0p3a_SO_1_E_134":"MIR",
                "20180617_055441_0p3a_SO_1_I_134":"MIR",
                "20180617_095025_0p3a_SO_1_I_164":"MIR",
}

def readBoresightFile(boresight_vector_file_path):

    with open(boresight_vector_file_path) as f:
        lines = f.readlines()
        
    
    all_boresight_vectors = []
    all_boresight_names = []
    
    for line in lines:
        content = line.split(",")
        boresight_name_in = content[0].strip()
        boresight_vector_in = [
                np.float32(content[1].strip()),
                np.float32(content[2].strip()),
                np.float32(content[3].strip()),
                ]
        if boresight_name_in == "SO_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("SO_BORESIGHT")
        elif boresight_name_in == "UVIS_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("UVIS_BORESIGHT")
        elif boresight_name_in == "LNO_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("LNO_BORESIGHT")
        elif boresight_name_in == "MIR_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("MIR_BORESIGHT")
        elif boresight_name_in == "TIRVIM_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("TIRVIM_BORESIGHT")
        else:
            logger.error("Boresight file cannot be read in correctly (%s boresight unknown)" %boresight_name_in)
    
    return all_boresight_vectors, all_boresight_names
     


def findBoresightUsed(et, boresight_vectors, boresight_names):

    obs2SunVector = sp.spkpos("SUN", et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)
    
    v_norm = obs2SunVector[0]/sp.vnorm(obs2SunVector[0])
    
    v_sep_min = 999.0
    for boresight_vector, boresight_name in zip(boresight_vectors, boresight_names):
        v_sep = sp.vsep(v_norm, boresight_vector) * sp.dpr() * 60.0 #arcmins difference
        if v_sep < v_sep_min:
            v_sep_min = v_sep
            boresight_found = boresight_name
    return boresight_found, v_sep_min

    


all_boresight_vectors, all_boresight_names = readBoresightFile(BORESIGHT_VECTOR_FILE_PATH)

for fileName, realBoresightUsed in fileNames_in.items():

    hdf5file_path = os.path.join(
            r"C:\Users\iant\Documents\DATA\hdf5_copy\hdf5_level_0p3a",
            fileName[0:4], fileName[4:6], fileName[6:8],
            fileName+".h5")
    
    hdf5FileIn = h5py.File(hdf5file_path, "r")
    observationDTimes = hdf5FileIn["Geometry/ObservationDateTime"][...]
    
    dref = "TGO_NOMAD_SO"
    channelId = sp.bods2c(dref) #find channel id number
    [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4) 
    
    SPICE_REFERENCE_FRAME = "TGO_SPACECRAFT"
    SPICE_ABERRATION_CORRECTION = "None"
    SPICE_OBSERVER = "-143"
    
    
    observationDTimesStart = [str(observationDTime[0]) for observationDTime in observationDTimes]
    observationDTimesEnd = [str(observationDTime[1]) for observationDTime in observationDTimes]
    obsTimesStart = [sp.utc2et(datetime.strip("<b>").strip("'")) for datetime in observationDTimesStart]
    obsTimesEnd = [sp.utc2et(datetime.strip("<b>").strip("'")) for datetime in observationDTimesEnd]
    
    #find times in occultation
    obsTimeMids = [obsTimesStart[0], np.mean([obsTimesStart[0],obsTimesStart[-1]]), obsTimesStart[-1]]

    
    print(fileName)
    for obsTimeMid in obsTimeMids:
        boresight_found, v_sep_min = findBoresightUsed(obsTimeMid, all_boresight_vectors, all_boresight_names)
        print("Real boresight is %s, script thinks it is %s, vsep = %0.3f" %(realBoresightUsed, boresight_found, v_sep_min))












