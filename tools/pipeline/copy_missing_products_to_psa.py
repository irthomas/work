# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:39:42 2020

@author: iant


GET PSA LOG FILES FROM ESA SERVER
LIST ALL FILES THAT PASSED AND FAILED
COMPARE TO LIST OF HDF5 1.0A FILES AND PSA CAL ARCHIVE
CHECK PSA ERROR LOGS

MAKE LIST OF FILES TO BE REDELIVERED
MAKE LIST OF FILES THAT ARE NOT CONVERTED TO CAL PRODUCTS
"""

import os
import glob
#import datetime
#import numpy as np
import platform
#import codecs
import subprocess
#import sys
import tarfile
from urllib.parse import urlparse
from datetime import datetime#, timedelta


"""REMEMBER TO RUN TRANSFER ESA_TO_BIRA FIRST TO GET THE LATEST LOG FILES!"""


PSA_VERSION = "1.0"

if platform.system() == "Windows":
    from hdf5_functions_v04 import BASE_DIRECTORY, DATASTORE_ROOT_DIRECTORY
    LOG_FILE_DIR = os.path.join(BASE_DIRECTORY, "psa_cal") #where are PSA logs located
    MAKE_PSA_LOG_DIR = os.path.join(BASE_DIRECTORY, "psa_cal") #for writing output logs
    PSA_FILE_DIR = os.path.join(DATASTORE_ROOT_DIRECTORY, "archive", "1.0", "data_calibrated")
else:
    os.environ["NMD_OPS_PROFILE"] = "default"
    os.environ["FS_MODEL"] = "False"
    from nomad_ops.config import ROOT_DATASTORE_PATH
    #set paths so that any user can run PSA update and write results to master log
    LOG_FILE_DIR = os.path.join(ROOT_DATASTORE_PATH, "nomadlsy/datastore/pds/logs/psa_cal/")
    MAKE_PSA_LOG_DIR = os.path.join(ROOT_DATASTORE_PATH, "logs", "psa_cal")
    PSA_FILE_DIR = os.path.join(ROOT_DATASTORE_PATH, "archive", "1.0", "data_calibrated")


BIRA_URL = "file:/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/"
ESA_URL = "ssh://exonmd@exoops01.esac.esa.int/home/exonmd/"

BIRA_PSA_CAL_URL = BIRA_URL + "archive/1.0/data_calibrated/"
ESA_PSA_CAL_URL = ESA_URL + "nmd/tmp0/"


logFilepathList = sorted(glob.glob(LOG_FILE_DIR+"/**/*nmd-pi*", recursive=True))
if len(logFilepathList)==0:
    print("Error: Log files not found")

successList = []
for logFile in logFilepathList:
    
    with open(logFile, "r") as f:
        logFileLines = f.readlines()
        

    for line in logFileLines:
        if "PASS" in line:
            lineSplit = line.split("/")
            psaXmlFilename = lineSplit[-1].replace("\n", "")
            if "browse" not in psaXmlFilename:
                successList.append(psaXmlFilename)
#            print(logFile, line)

successList = sorted(successList)
ingestedXmlFilenameList = sorted(list(set(successList)))

        

localZipFilepathList = sorted(glob.glob(PSA_FILE_DIR+"/**/*.zip", recursive=True))
localZipFilenameList = [os.path.split(filename)[1] for filename in localZipFilepathList]

with open(os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_log_ingested.txt"), "w") as f:
    for xmlFilename in ingestedXmlFilenameList:
        f.write("%s\n" %xmlFilename)

with open(os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_log_local.txt"), "w") as f:
    for zipFilename in localZipFilenameList:
        f.write("%s\n" %zipFilename)

def psaTransferLog(lineToWrite):
    dt = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S")
    logPath = os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_transfer.log")
    with open(logPath, 'a') as logFile:
        logFile.write(dt + "\t" + lineToWrite + '\n')


def convertXmlFilenameToZip(xmlFilename):
    xmlFilenameSplit = xmlFilename.split("_")
    start = xmlFilenameSplit[0]+"_"+xmlFilenameSplit[1]+"_"+xmlFilenameSplit[2]+"_"+xmlFilenameSplit[3]+"_"+xmlFilenameSplit[4][:8]
    mid = "-"+xmlFilenameSplit[4][9:15]+"00"
    end = xmlFilenameSplit[4][31:].split(".")[0]+"_"+PSA_VERSION+".zip"
    return start+mid+end

ingestedZipFilenameList = sorted([convertXmlFilenameToZip(xmlFilename) for xmlFilename in ingestedXmlFilenameList])

def returnNotMatching(a, b):
    """compare list 1 to list 2 and return non matching entries"""
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

psaNotInLocal, localNotInPsa = returnNotMatching(ingestedZipFilenameList, localZipFilenameList)
print("PSA files not in local directory %s:" %PSA_FILE_DIR)
print(psaNotInLocal)
print("Local files not in the PSA:")
print(localNotInPsa)

with open(os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_log_mismatches.txt"), "w") as f:
    f.write("PSA files not in local directory %s\n" %PSA_FILE_DIR)
    for filename in psaNotInLocal:
        f.write("%s\n" %filename)
    f.write("\n\n\nLocal files not in the PSA\n")
    for filename in localNotInPsa:
        f.write("%s\n" %filename)



esa_p_url = urlparse(ESA_PSA_CAL_URL)


localNotInPsaPath = []
error = False
for localFilename in localNotInPsa:
    if localFilename in localZipFilenameList:
        index = [i for i,x in enumerate(localZipFilenameList) if x==localFilename]
        if len(index)==1:
            localNotInPsaPath.append(localZipFilepathList[index[0]])
        else:
            print("Error: Multiple files found")
            error = True
    else:
        print("Error: File not found")
        error = True

if not error:
    #transfer to ESA mismatching files
    transferFileList = localNotInPsaPath
    if len(transferFileList)==0:
        print("All files are up to date on the PSA server. No transfer required")
    else:
        # Run a 'tar' on ESA server
        tar_cmd =  "tar xz -b 32 -C %s" % (esa_p_url.path)
        ssh_cmd = ["ssh", esa_p_url.netloc, tar_cmd]
        # Run a 'tar' extract on ESA server
        with subprocess.Popen(ssh_cmd,
                        shell=False,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE) as ssh:
            # Create a tar stream connected to the tar-extract @ ESA
            tar = tarfile.open(fileobj=ssh.stdin, mode="w|gz", bufsize=32*512)
            # Write files to the stream
            for path in transferFileList:
#                path = codecs.decode(path)
                n = os.path.getsize(path)
                arcname = os.path.basename(path)
                tar.add(path, arcname)
                print("File added to TAR archive: %s (%.1f kB)" %(arcname, n/1000))
                psaTransferLog(arcname)
            tar.close()
        ssh_cmd2 = ["ssh", esa_p_url.netloc, "mv nmd/tmp0/* nmd/"]
        subprocess.Popen(ssh_cmd2,
                        shell=False,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)            
        
       
        