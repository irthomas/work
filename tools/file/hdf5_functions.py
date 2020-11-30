# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Mon Nov 16 10:56:01 2015

@author: iant

FUNCTIONS AND CONSTANTS FOR READING HDF5 FILES AND EXTRACTING DATA/ATTRIBUTES
VERSION 4 NOW USES REGEX FOR SELECTING FILENAMES E.G. obspaths = re.compile("201905.*LNO.*_D_.*|201906.*LNO.*_D_.*")

"""

#import numpy as np
#from datetime import datetime
import os
import collections
import h5py
import re
import sys

from tools.file.paths import paths, SYSTEM

DATASTORE_PATHS = paths["DATASTORE"]

if SYSTEM == "Windows":
    print("Running on windows")
    # import getpass
    import pysftp
#    import spiceypy as sp

PASSWORD = "" #make global variable so it doesn't ask for pword for every file

if sys.version_info[0] == 3 and sys.version_info[1] > 6:
    RE_PATTERN_TYPE = re.Pattern
else:
    RE_PATTERN_TYPE = re._pattern_type



#remove these files automatically from wildcard searches only
BAD_FILE_DICTIONARY = {
        "20180521_212349":"ACS MIR Pointing",
        "20180521_214930":"ACS MIR Pointing",
        "20180522_031715":"ACS MIR Pointing",
        "20180522_034405":"ACS MIR Pointing",
        "20180527_030547":"ACS TIRVIM Pointing",
        "20180527_034419":"ACS TIRVIM Pointing",
        "20180617_004358":"ACS MIR Pointing",
        "20180617_055441":"ACS MIR Pointing",
        "20180617_095025":"ACS MIR Pointing",
        
        "20180709_000545":"Nadir off-planet",
        "20180504_150953":"Nadir off-planet",
        "20180504_165745":"Nadir off-planet",
        "20180529_150145":"Nadir off-planet",
        "20180625_191417":"Nadir off-planet",
        
#        "20150315_235950":"Wrong day",
        "20180628_235204":"Wrong day",
        "20190130_235926":"Wrong day",
        "20190308_235913":"Wrong day",
        "20190712_235957":"Wrong day",
        "20191016_235331":"Wrong day",

        }


"""make list of all hdf5 files in given folder and all subdirectories"""
def get_hdf5_filename_list(root_directory,check_for_calibration_file=False): 
    
    data_filenames=[]
    for path, subfolders, files in os.walk(root_directory):
    #    for folder in subfolders:    
        for each_file in files:
            if '.h5' in each_file:
                if check_for_calibration_file:
                    hdf5_file = h5py.File(each_file, "r") #read file, check in calibration file or not
                    ignore=False
                    if "Calibration_File" in hdf5_file.attrs.keys(): #check if calibration file
                        if hdf5_file.attrs["Calibration_File"]=="True":
                            ignore=True
                    hdf5_file.close()
                    if not ignore: data_filenames.append(each_file[:-3]) #if not calibration file, add to list of hdf5 files
                else:
                    data_filenames.append(each_file[:-3])
     #warn if duplicates found
    duplicate_search=collections.Counter(data_filenames)
    duplicates= [i for i in duplicate_search if duplicate_search[i]>1]
    if len(duplicates) > 0:
        print("Warning, duplicates found:")
        print(duplicates)
        return ""
    data_filenames_sorted=sorted(data_filenames)
    return data_filenames_sorted


def get_filepath(hdf5_filename):
    """get full file path from name"""

    import os
    from tools.file.paths import paths
    
    file_level = "hdf5_level_%s" %hdf5_filename[16:20]
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]

    filepath = os.path.join(paths["DATA_DIRECTORY"], file_level, year, month, day, hdf5_filename+".h5") #choose a file
    
    return filepath




def get_files_from_datastore(hdf5_filenames):

    for hdf5_filename in hdf5_filenames:
        file_level = "hdf5_level_%s" %hdf5_filename[16:20]
        year_in = hdf5_filename[0:4]
        month_in = hdf5_filename[4:6]
        day_in = hdf5_filename[6:8]
        get_file_from_datastore(file_level, year_in, month_in, day_in, hdf5_filename, DATASTORE_PATHS["DATASTORE_SERVER"], DATASTORE_PATHS["DATASTORE_DIRECTORY"])
    

def get_file_from_datastore(file_level, year_in, month_in, day_in, filename_in, server, server_directory):
    silent = False
    global PASSWORD

    if PASSWORD == "":
        # PASSWORD = getpass.getpass('Password:')
        PASSWORD = input('Password:')

    future_path = os.path.join(paths["DATA_DIRECTORY"], file_level, year_in, month_in, day_in)
    os.makedirs(future_path, exist_ok=True)
    os.chdir(future_path)
    with pysftp.Connection(server[0], username=server[1], password=PASSWORD) as sftp:
        with sftp.cd(server_directory+"/"+file_level): # temporarily chdir to public
            pathToDatastoreFile = "%s/%s/%s/%s/%s/%s.h5" %(server_directory, file_level, year_in, month_in, day_in, filename_in)
            if not silent: print(pathToDatastoreFile)
            sftp.get(pathToDatastoreFile) # get a remote file
    os.chdir(paths["BASE_DIRECTORY"])


def get_file(obspath, file_level, count, model="INFLIGHT", silent=False, open_files=True):
    """check if file exists in data directory; if not download from server"""

#    if model == "INFLIGHT":
#        DATA_DIRECTORY_IN = DATA_DIRECTORY
#        DATASTORE_DIRECTORY_IN = DATASTORE_DIRECTORY
#    elif model == "PFM":
#        DATA_DIRECTORY_IN = DATA_DIRECTORY_PFM
#        DATASTORE_DIRECTORY_IN = DATASTORE_DIRECTORY_PFM
    if model == "FS":
        DATA_DIRECTORY_IN = paths["DATA_DIRECTORY_FS"]
#        DATASTORE_DIRECTORY_IN = DATASTORE_DIRECTORY_FS


    year = obspath[0:4] #get the date from the filename to find the file
    month = obspath[4:6]
    day = obspath[6:8]
    
    """secondary check if model is defined correctly"""
#    if year == "2015" and month in ["03","04"] and model != "PFM":
#        print("Warning: model is %s but observation is during PFM ground calibration period. Switching to PFM" %model)
#        model = "PFM"
#        DATA_DIRECTORY_IN = DATA_DIRECTORY_PFM
#        DATASTORE_DIRECTORY_IN = DATASTORE_DIRECTORY_PFM
    if year == "2015" and month == "09" and model != "FS":
        print("Warning: model is %s but observation is during FS ground calibration period. Switching to FS" %model)
        model = "PFM"
        DATA_DIRECTORY_IN = paths["DATA_DIRECTORY_FS"]
#        DATASTORE_DIRECTORY_IN = DATASTORE_DIRECTORY_FS
    else:
        DATA_DIRECTORY_IN = paths["DATA_DIRECTORY"]
        DATASTORE_DIRECTORY_IN = DATASTORE_PATHS["DATASTORE_DIRECTORY"]


    
    if DATASTORE_PATHS["DIRECTORY_STRUCTURE"]:
        filename = os.path.join(DATA_DIRECTORY_IN, file_level, year, month, day, obspath+".h5") #choose a file
    else:
        filename = os.path.join(DATA_DIRECTORY_IN, obspath+".h5") #choose a file

    if os.path.exists(filename):
        if not silent: print("%i: File %s found" %(count, filename))
    else:
        if DATASTORE_PATHS["SEARCH_DATASTORE"]:
            print("File %s not found. Getting from datastore (%s, %s, %s)" %(filename, DATASTORE_PATHS["DATASTORE_SERVER"][0], DATASTORE_PATHS["DATASTORE_SERVER"][1], DATASTORE_DIRECTORY_IN))
            get_file_from_datastore(file_level, year, month, day, obspath, DATASTORE_PATHS["DATASTORE_SERVER"], DATASTORE_DIRECTORY_IN)
            if DATASTORE_PATHS["DIRECTORY_STRUCTURE"]:
                filename = os.path.join(DATA_DIRECTORY_IN, file_level, year, month, day, obspath+".h5") #choose a file
            else:
                filename = os.path.join(DATA_DIRECTORY_IN, obspath+".h5") #choose a file
        else:
            print("File %s not found." %filename)
    if open_files:
        hdf5_file = h5py.File(filename, "r") #open file
    else:
        hdf5_file = filename
    return filename, hdf5_file


def open_hdf5_file(hdf5_filename):

    year = hdf5_filename[0:4] #get the date from the filename to find the file
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]
    file_level = "hdf5_level_%s" %hdf5_filename[16:20]
    hdf5_filepath = os.path.join(paths["DATA_DIRECTORY"], file_level, year, month, day, hdf5_filename+".h5")
    hdf5_file = h5py.File(hdf5_filepath, "r")
    
    return hdf5_file


def make_filelist(obs_paths, file_level, model="INFLIGHT", silent=False, open_files=True, path=None):
    """make list of filenames containing matching attributes and datasets"""
    """new version uses regex"""
    if path:
        DATA_DIRECTORY_IN = path
    else:
        if model == "INFLIGHT":
            DATA_DIRECTORY_IN = paths["DATA_DIRECTORY"]
        elif model == "PFM":
            DATA_DIRECTORY_IN = paths["DATA_DIRECTORY"]
        elif model == "FS":
            DATA_DIRECTORY_IN = paths["DATA_DIRECTORY_FS"]

    
    """new regex type check"""
    if isinstance(obs_paths, RE_PATTERN_TYPE):
        obspathsList = get_hdf5_filename_list(os.path.join(DATA_DIRECTORY_IN, file_level))
        hdf5Filenames = list(filter(obs_paths.search, obspathsList))
        print("%i matching files found for regex %s" %(len(hdf5Filenames), obs_paths.pattern))
        obsTitles = hdf5Filenames

    else:
        hdf5Filenames = obs_paths
        obsTitles = obs_paths
        
    
    hdf5_files_out = []
    hdf5_filenames_out = []
    titles = []
    for fileIndex, (obspath, title) in enumerate(zip(hdf5Filenames, obsTitles)):
        

        badFile = False
        for bad_file_prefix in list(BAD_FILE_DICTIONARY.keys()):
            if bad_file_prefix in obspath:
                badFile = True
                found_bad_file = bad_file_prefix
        if badFile:
            print("Warning: Bad file %s (%s) not added to list" %(obspath,BAD_FILE_DICTIONARY[found_bad_file]))
        else:
            filename, hdf5_file = get_file(obspath, file_level, fileIndex, model=model, silent=silent, open_files=open_files)
            hdf5_files_out.append(hdf5_file) #add open file to list
            hdf5_filenames_out.append(obspath) #add open file to list
            titles.append(title)
            
    

    return hdf5_files_out, hdf5_filenames_out, titles


def write_file(file_name, lines_to_write):
    """function to write text file"""
    txtFile = open(file_name, 'w')
    for line_to_write in lines_to_write:
        txtFile.write(line_to_write+'\n')
    txtFile.close()


def write_filelist(file_level):
    
    outputFilelist = []
    
    hdf5Filenames = get_hdf5_filename_list(os.path.join(paths["DATA_DIRECTORY"], file_level))
    for hdf5Filename in hdf5Filenames:
        outputFilelist.append(hdf5Filename)

    write_file(os.path.join(paths["BASE_DIRECTORY"], "hdf5_filename_list_%s.txt" %file_level), outputFilelist)




