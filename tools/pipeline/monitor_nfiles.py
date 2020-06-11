# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:33:22 2019

@author: iant

MONITOR N FILES
"""
import os
import re
import argparse
import time
import glob
import platform
from datetime import datetime

if platform.system() == "Windows":
    DATA_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
else:
    DATA_DIRECTORY = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"


parser = argparse.ArgumentParser()
parser.add_argument('regex', type=str, help='enter regex')
parser.add_argument('--level', default="all", help='enter a level e.g. hdf5_l10a')
parser.add_argument('-list', action="store_true", help='list filenames')
args = parser.parse_args()

regex = args.regex
level = args.level

if level != "all":
    levels = {level:0}
else:
    levels = {
            "hdf5_l01a":0,
            "hdf5_l01d":0,
            "hdf5_l01e":0,
            "hdf5_l02a":0,
            "hdf5_l02b":0,
            "hdf5_l02c":0,
            "hdf5_l03a":0,
            "hdf5_l03b":0,
            "hdf5_l03c":0,
            "hdf5_l03i":0,
            "hdf5_l03j":0,
            "hdf5_l03k":0,
            "hdf5_l10a":0,
              }

list_filenames = False
if args.list:
    list_filenames = True




#regex = "201804[0-9][0-9]_"
#regex = "20180501_"
print("Searching for ", regex)

r = re.compile(regex)

while True:
    
    filename_list = []
    
    previousLevels = {key: value for key, value in levels.items()}
    for level in levels.keys():
        
        folderName = "hdf5_level_%sp%s" %(level[-3:-2], level[-2:])
    
        folderPath = os.path.join(DATA_DIRECTORY, folderName)
        
#        nfiles = 0
#        for root, dirs, files in os.walk(folderPath):
#            matchingFiles = list(filter(r.match, files))
#            for filename in matchingFiles:
#                if ".h5" in filename:
#                    nfiles += 1
#        levels[level] = nfiles
        
        files = [os.path.split(f)[1] for f in glob.glob(folderPath + os.sep + "**" + os.sep + "*.h5", recursive=True)]
        matchingFiles = list(filter(r.match, files))
        levels[level] = len(matchingFiles)

        if list_filenames:
            filename_list.extend(matchingFiles)


    print("######### %s #########" %str(datetime.now())[:-7:])
    for (level, nFiles), (_, previousNFiles) in zip(levels.items(), previousLevels.items()):
        if nFiles-previousNFiles != 0:
            print(level, nFiles, "(%+i)" %(nFiles-previousNFiles))
        else:
            print(level, nFiles)
            
    if list_filenames:
        for filename in filename_list:
            print(filename)
    
    time.sleep(10)
    
