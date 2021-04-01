# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:25:37 2020

@author: iant

GET INFO FROM ALL SOLAR CALIBRATION OBS AND WRITE TO HTML PAGE TO COPY TO WEBSITE

"""
import os
import glob
import datetime
#import numpy as np
import h5py
#import platform

from tools.file.paths import paths
from instrument.nomad_lno_instrument import order_nu0p as lno_order
from instrument.nomad_lno_instrument import nu0_aotf as lno_centre

from instrument.nomad_so_instrument import order_nu0p as so_order
from instrument.nomad_so_instrument import nu0_aotf as so_centre



HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"


SOLAR_DATA_DIRECTORY = os.path.join(paths["DATA_DIRECTORY"], "hdf5_level_0p2a")




calFilepathList = sorted(glob.glob(SOLAR_DATA_DIRECTORY+"/**/*_C.h5", recursive=True))
#get rid of UVIS files
calFilepathList = [filepath for filepath in calFilepathList if ("SO" in filepath) or ("LNO" in filepath)]
calFilenameList = [os.path.split(filename)[1] for filename in calFilepathList]

copDirs = sorted(os.listdir(paths["COP_TABLE_DIRECTORY"]))
copDirsGood = [dirname for dirname in copDirs if len(dirname)==15 and dirname[-1]=="0"]

copDirsDatetime = [datetime.datetime.strptime(dirname, HDF5_FILENAME_FORMAT) for dirname in copDirsGood]
#add future cop table datetime so last value in table can be found
copDirsDatetime.append(datetime.datetime(2050, 1, 1))

h = "<html><head></head><body>\n"
h += "<table border=2><tr><th>Filename</th><th>Observation Type</th><th>Description</th><th>COP Table Version</th></tr>\n"

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
        
            if channel == "SO":
                nu0_min = so_centre(min(aotfFrequencies))
                nu0_max = so_centre(max(aotfFrequencies))
                orders = [so_order(nu0_min, 160, 0.0), so_order(nu0_max, 160, 0.0)]
            elif channel == "LNO":
                nu0_min = lno_centre(min(aotfFrequencies))
                nu0_max = lno_centre(max(aotfFrequencies))
                orders = [lno_order(nu0_min, 160, 0.0), lno_order(nu0_max, 160, 0.0)]
                
                
            obsType += "miniscan"
            description = "%i-%ikHz (approx. orders %i-%i) in steps of %ikHz" %(min(aotfFrequencies), max(aotfFrequencies), min(orders), max(orders), aotfFrequencies[1]-aotfFrequencies[0])
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
    h += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" %(calFilename, obsType, description, copVersionString)
#    print(calFilename, description)
    
h += "</table></body></html>"

print("Writing output")
with open(os.path.join(paths["BASE_DIRECTORY"], "calibration_log.html"), "w") as f:
    f.write(h)