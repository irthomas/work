# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:50:47 2022

@author: iant

TEST SOLAR CALIBRATION FILENAME CHANGE
"""


import os
import glob
import h5py

from tools.file.paths import paths




SOLAR_DATA_DIRECTORY = os.path.join(paths["DATA_DIRECTORY"], "hdf5_level_0p2a")




def get_solar_calibration_letter(hdf5FileIn):

    integrationTime = hdf5FileIn["Channel/IntegrationTime"][...]
    aotfFrequency = hdf5FileIn["Channel/AOTFFrequency"][...]
    windowTop = hdf5FileIn["Channel/WindowTop"][...]
        
    letter = "N" #if unknown, use N for normal observation
    
    windowTops = sorted(list(set(windowTop)))
    if len(windowTops)>1: #window stepping (FOV calibration)
        letter = "L"

    integrationTimes = sorted(list(set(integrationTime)))
    if len(integrationTimes)>1: #integration time stepping
        letter = "D"
    
    aotfFrequencies = sorted(list(set(aotfFrequency)))
    if len(aotfFrequencies)>1: #miniscan or fullscan
        if aotfFrequencies[1] - aotfFrequencies[0] < 50.0: #miniscan
        
            letter = "M"

        else: #fullscan
            letter = "F"
            
    if len(windowTops)==1 and len(integrationTimes)==1 and len(aotfFrequencies)==1: #FOV calibration normal obs
        letter = "L"

        
    return letter



def make_new_basename(hdf5_basename, letter):
    
    basename_split = hdf5_basename.split("_")
    basename_split[4] = letter
    
    return "_".join(basename_split)
    


calFilepathList = sorted(glob.glob(SOLAR_DATA_DIRECTORY+"/**/*_C.h5", recursive=True))
#get rid of UVIS files
calFilepathList = [filepath for filepath in calFilepathList if ("SO" in filepath) or ("LNO" in filepath)]
calFilenameList = [os.path.split(filename)[1] for filename in calFilepathList]


for calFilename, calFilepath in zip(calFilenameList[0:20], calFilepathList[0:20]):
    
    
    
    with h5py.File(calFilepath, "r") as hdf5FileIn:
        letter = get_solar_calibration_letter(hdf5FileIn)
        
    new_basename = make_new_basename(calFilename, letter)


    print("########")
    print(calFilename)
    print(new_basename)
    print(letter)