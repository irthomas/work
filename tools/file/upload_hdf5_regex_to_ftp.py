# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:23:09 2019

@author: iant


SCRIPT TO TRANSFER SELECTED FILES TO AN FTP SERVER - E.G. CALIBRATION, ETC.
"""


import os
import re
from ftplib import FTP

from tools.file.hdf5_functions_v04 import makeFileList
from tools.file.paths import paths
from tools.file.passwords import passwords

regex = re.compile(".*(_SO_1_C|_SO_2_C|_LNO_1_C|_LNO_2_C).*")
fileLevel = "hdf5_level_0p2a"

FTP_SERVER = ["ftp-ae.oma.be", "nomadadm", passwords["nomadadm"]]
FTP_DIRECTORY = "Data"



def putFilesOnFtp(hdf5_filenames, file_level):
    for hdf5_filename in hdf5_filenames:
        year_in = hdf5_filename[0:4]
        month_in = hdf5_filename[4:6]
        day_in = hdf5_filename[6:8]
        putFileOnFtp(file_level, year_in, month_in, day_in, hdf5_filename, FTP_SERVER, FTP_DIRECTORY)
    
def cdTree(ftp, path):
    print("entering folder {0}".format(path))
    try:
        ftp.cwd(path)
    except:
        print("failed to enter, creating")
        cdTree(ftp, path="/".join(path.split("/")[:-1]))
        ftp.mkd(path)
        ftp.cwd(path)


def putFileOnFtp(file_level, year_in, month_in, day_in, filename_in, ftp_server, ftp_directory):
    silent = False
    source_path = os.path.join(paths["DATA_DIRECTORY"], file_level, year_in, month_in, day_in)
    os.chdir(source_path)
    
    ftp = FTP(ftp_server[0], ftp_server[1], ftp_server[2])

    pathToFtpFile = "/%s/%s/%s/%s/%s" %(ftp_directory, file_level, year_in, month_in, day_in)
    if not silent: print(pathToFtpFile)
    cdTree(ftp, pathToFtpFile)
    
    ftp.cwd(pathToFtpFile) #change directory

    if ("%s" %(filename_in+".h5")) in ftp.nlst(): #if file already exists, don't copy
        print("File %s already exists on ftp server. Ignoring" %filename_in)
    else: # if file not already on ftp, copy it in binary copy mode
        print("Copying file %s to NOMAD Science FTP server" %filename_in)
        fileToUpload = open(filename_in+".h5",'rb')
        ftp.storbinary("STOR %s" %(filename_in+".h5"), fileToUpload)
        fileToUpload.close()


    os.chdir(paths["BASE_DIRECTORY"])


hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel, open_files=False)

if len(hdf5Filenames) == 0:
    print("No matching files found. Nothing uploaded to ftp")
else:
    print("The following files will be uploaded to the ftp")
    for hdf5Filename in hdf5Filenames:
        print(hdf5Filename)
    
    input("Press any key to continue")
    
    putFilesOnFtp(hdf5Filenames, fileLevel)