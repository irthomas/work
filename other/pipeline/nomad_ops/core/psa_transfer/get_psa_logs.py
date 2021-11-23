# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:20:23 2021

@author: iant

READ IN PSA LOGS. WRITE LOG FILE TO SQLITE WITH MD5 TO CHECK FOR CONSISTENCY

READ IN NEW LOGS
IF ANY MD5S CHANGED, DELETE ALL ENTRIES FROM THAT LOG AND REREAD

"""


import os
import glob
import re
import numpy as np
from datetime import datetime

from nomad_ops.core.psa_transfer.config import PATH_DB_PSA_CAL_LOG, UPLOAD_DATES, LOG_FORMAT_STR

from nomad_ops.core.psa_transfer.functions import \
    convert_lid_to_short_filename


def get_log_list():
    """get all logs"""
    
    log_filepath_list = sorted(glob.glob(PATH_DB_PSA_CAL_LOG+"/**/*nmd-pi-delivery.log*", recursive=True))
    if len(log_filepath_list)==0:
        print("Error: Log files not found")
    
    return log_filepath_list




def extract_log_info(log_filepath):
    """From list of log filepaths, read in and extract all the required information, saving it to a dictionary
    Nomenclature: 
        zip filenames contain one datetime string only with version number: nmd_cal_sc_uvis_20180529-11060000-d_1.0
        shortened filename is same without version number: nmd_cal_sc_uvis_20180529-11060000-d
    
        xml/tab/png filenames are lids with full start/end times: nmd_cal_sc_so_20181007t064522-20181007t073102-a-i-149
    """

    
    date_string = ("date_string", "S30")
    zip_filename = ("zip_filename", "S45")
    short_filename = ("short_filename", "S100")
    lid_string = ("lid_string", "S100")
    error_string = ("error_string", "S200")
    
    
    #parse all logs into memory
    log_dict = {
    }
    
    log_filename = os.path.basename(log_filepath)
    print("Reading", log_filename)
    
    files_received = np.fromregex(log_filepath, "(\S+\s\S+) INFO  Checking file received: (\S+)[.]zip", [date_string, zip_filename])
    files_transferred_to_staging = np.fromregex(log_filepath, "(\S+\s\S+) INFO  File (\S+)[.]zip transferred to \S+staging\/(\S+)", [date_string, zip_filename, short_filename])
    zip_file_expanded = np.fromregex(log_filepath, "(\S+\s\S+) INFO  Expanding zip file: \S+staging\/(\S+)\/(\S+)[.]zip", [date_string, short_filename, zip_filename])
    validator_pass = np.fromregex(log_filepath, "PASS: \S+Orbit_\d+\/(\S+)[.]xml", [lid_string])
    validator_fail = np.fromregex(log_filepath, "FAIL: \S+Orbit_\d+\/(\S+)[.]xml", [lid_string])
    validator_error = np.fromregex(log_filepath, "ERROR: (.+)\S+\n\s+file\S+Orbit_\d+\/(\S+)[.]xml.+", [error_string, lid_string])


    zip_filenames_received_temp = [i.decode() for i in files_received["zip_filename"]]
    zip_filenames_transferred_temp = [i.decode() for i in files_transferred_to_staging["zip_filename"]]
    zip_filenames_expanded_temp = [i.decode() for i in zip_file_expanded["zip_filename"]]
    
    validator_lids_pass_temp = [i.decode() for i in validator_pass["lid_string"]]
    validator_lids_fail_temp = [i.decode() for i in validator_fail["lid_string"]]
    validator_lids_error_temp = [i.decode() for i in validator_error["lid_string"]]
    validator_errors_temp = [i.decode() for i in validator_error["error_string"]]

    log_dict["zip_filenames_received"] = zip_filenames_received_temp
    log_dict["zip_filenames_transferred"] = zip_filenames_transferred_temp
    log_dict["zip_filenames_expanded"] = zip_filenames_expanded_temp
    log_dict["validator_lids_pass"] = validator_lids_pass_temp
    log_dict["validator_lids_fail"] = validator_lids_fail_temp
    log_dict["validator_lids_error"] = validator_lids_error_temp
    log_dict["validator_errors"] = validator_errors_temp

        
    #convert validator lids to short filenames
    log_dict["validator_short_filenames_pass"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_pass"]]
    log_dict["validator_short_filenames_fail"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_fail"]]
    log_dict["validator_short_filenames_error"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_error"]]
    
    return log_dict



def get_versions_from_zip(zip_filenames):
    
    regex = re.compile("nmd_cal_sc_\w*_\d*-\d*-\S*_(\d*.\d*)")
    versions = [regex.findall(i)[0] for i in zip_filenames]
    unique_versions = sorted(list(set(versions)))
    
    return unique_versions




def get_log_datetime(log_path):
    """get first datetime from PSA log file contents"""
    with open(log_path, "r") as f:
        #check first 100 lines of file for datetime
        for i in range(100):
            info_datetime = re.match("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO.*", next(f))
            if info_datetime:
                log_datetime = datetime.strptime(info_datetime.groups()[0], LOG_FORMAT_STR)
                return log_datetime
    print("Error: datetime not found in log file %s" %log_path)
    return False



def get_log_version(log_datetime):
    """get version from log file datetime"""
    for upload_date in UPLOAD_DATES:
        if (log_datetime > upload_date[0]) & (log_datetime < upload_date[1]):
            log_version = upload_date[2]
            return log_version
    
    print("Error: log datetime does not correspond to any time period. Update UPLOAD_DATES")
    return False



def last_process_datetime():
    """get last datetime from PSA log file contents"""
    open_log_path = os.path.join(PATH_DB_PSA_CAL_LOG, "nmd-pi-delivery.log")
    with open(open_log_path, "r") as f:
        lines = f.readlines()
        #read lines backwards to find a datetime
        for i in range(-1, -100, -1):
            info_datetime = re.match("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO.*", lines[i])
            if info_datetime:
                log_datetime = datetime.strptime(info_datetime.groups()[0], LOG_FORMAT_STR)
                return log_datetime
        print("Error: datetime not found in log file %s" %open_log_path)
        return False
        