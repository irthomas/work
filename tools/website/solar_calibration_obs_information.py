# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:25:37 2020

@author: iant

GET INFO FROM ALL SOLAR CALIBRATION OBS

"""
import os
import glob
import datetime
#import numpy as np
import h5py
import platform

from hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, DATASTORE_ROOT_DIRECTORY

SOLAR_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "hdf5_level_0p2a")
if platform.system() == "Windows":
    COP_TABLE_DIRECTORY = os.path.join(r"C:\Users\iant\Dropbox\NOMAD\Python\data\cop_tables")
else:
    COP_TABLE_DIRECTORY = os.path.join(DATASTORE_ROOT_DIRECTORY, "cop_tables")

HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"



calFilepathList = sorted(glob.glob(SOLAR_DATA_DIRECTORY+"/**/*_C.h5", recursive=True))
#get rid of UVIS files
calFilepathList = [filepath for filepath in calFilepathList if ("SO" in filepath) or ("LNO" in filepath)]
calFilenameList = [os.path.split(filename)[1] for filename in calFilepathList]

copDirs = sorted(os.listdir(COP_TABLE_DIRECTORY))
copDirsGood = [dirname for dirname in copDirs if len(dirname)==15]

copDirsDatetime = [datetime.datetime.strptime(dirname, HDF5_FILENAME_FORMAT) for dirname in copDirsGood]

h = "<table border=2><tr><th>Filename</th><th>COP Table Version</th><th>Observation Type</th><th>Description</th></tr>\n"

#for calFilename, calFilepath in zip(calFilenameList[0:1], calFilepathList[0:1]):
for calFilename, calFilepath in zip(calFilenameList, calFilepathList):
    
    calDatetime = datetime.datetime.strptime(calFilename[:15], HDF5_FILENAME_FORMAT)
    
    timedeltas = [(patchDatetime - calDatetime).total_seconds() for patchDatetime in copDirsDatetime]    
    
    correctCopIndex = [i for i, x in enumerate(timedeltas) if x > 0][0]
    copVersion = copDirsGood[correctCopIndex-1]
    
    with h5py.File(calFilepath, "r") as hdf5_file:
        integrationTime = hdf5_file["Channel/IntegrationTime"][...]
        aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
        binning = hdf5_file["Channel/Binning"][...]
        windowTop = hdf5_file["Channel/WindowTop"][...]
        naccs = hdf5_file["Channel/NumberOfAccumulations"][...]
        
    channel = calFilename.split("_")[3]
    obsType = "%s " %channel
        
    windowTops = sorted(list(set(windowTop)))
    if len(windowTops)>1: #window stepping
        obsType += "window stepping (FOV calibration)"
        description = "%i steps" %len(windowTops)

    integrationTimes = sorted(list(set(integrationTime)))
    if len(integrationTimes)>1: #int time stepping
        obsType += "integration time stepping"
        description = "%i-%ims in steps of %ims" %(min(integrationTimes), max(integrationTimes), integrationTimes[1]-integrationTimes[0])
    
    aotfFrequencies = sorted(list(set(aotfFrequency)))
    if len(aotfFrequencies)>1: #miniscan or fullscan
        if aotfFrequencies[1] - aotfFrequencies[0] < 50.0: #miniscan
            obsType += "miniscan"
            description = "%i-%ikHz in steps of %ikHz" %(min(aotfFrequencies), max(aotfFrequencies), aotfFrequencies[1]-aotfFrequencies[0])
        else:
            obsType += "fullscan"
            description = "%i orders" %len(aotfFrequencies)
            
    if len(windowTops)==1 and len(integrationTimes)==1 and len(aotfFrequencies)==1:
        obsType += "FOV calibration"
        description = "Normal observation"
    
#    if len(description) < 20:
#        print("Error: calibration type not found")
    
    description += ". %i accumulations" %naccs[0]
        
    copVersionString = "%s-%s-%s" %(copVersion[0:4], copVersion[4:6], copVersion[6:8])
    h += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" %(calFilename, copVersionString, obsType, description)
#    print(calFilename, description)
    
h += "</table>"

with open(os.path.join(BASE_DIRECTORY, "calibration_log.html"), "w") as f:
    f.write(h)