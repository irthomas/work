# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:11:59 2020

@author: iant


CHECK PSA CAL PRODUCTS FOR ERRONEOUS PAR REFERENCES

STEPS:
1. UNZIP EACH CAL PRODUCT IN ARCHIVE
2. FIND PAR REF IN EACH XML FILE
3. CHECK DELTA TIME BETWEEN PAR AND CAL START TIME (COULD ALSO CHECK DB EITHER)
4. IF BAD, FLAG FILE

"""

import os
import glob
import zipfile
import re
import datetime
import platform

if platform.system() == "Windows":
#    PSA_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\archive\1.0\data_calibrated"
    PSA_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\psa\data_calibrated"
else:
    
    PSA_DIRECTORY = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/psa/data_calibrated"


FORMAT_STR_ZIP = "%Y%m%d-%H%M%S%f"
FORMAT_STR_PAR = "%Y%m%dT%H%M%S"

folderPath = os.path.join(PSA_DIRECTORY)
    
zipFilenames = sorted([os.path.split(f)[1] for f in glob.glob(folderPath + os.sep + "**" + os.sep + "*.zip", recursive=True)])

for zipFilename in zipFilenames[0:10]:

    zipStartString = zipFilename.split("_")[4][:17]
    zipStart = datetime.datetime.strptime(zipStartString, FORMAT_STR_ZIP)
    
    year = "%04i" %zipStart.year
    month = "%02i" %zipStart.month
    day = "%02i" %zipStart.day
    
    
    zipFilepath = os.path.join(folderPath, year, month, day, zipFilename)
    
    found = False
    with zipfile.ZipFile(zipFilepath) as z:
        for filename in z.namelist():
            if re.search("nmd_cal_sc_.*_20.*.xml", filename):
                # read the file
                with z.open(filename) as f:
                    for line in f:
                        if "<psa:file_name>nmd_par_sc" in line.decode():
                            parRefLine = line.decode().strip()
#                            print(parRefLine)
                            
                            parStartString = parRefLine.split("_")[5][:15]
                            
                            parStart = datetime.datetime.strptime(parStartString, FORMAT_STR_PAR)
                            found = True
                            
                            if zipStart > parStart:
                                calParTimeDifference = zipStart - parStart
                            else:
                                calParTimeDifference = parStart - zipStart
                                
                            
                            if calParTimeDifference > datetime.timedelta(minutes=20):
                                print("##############", zipFilename, parStartString, str(calParTimeDifference))
                            else:
                                print(zipFilename, str(calParTimeDifference))
    if not found:
        print("Error")