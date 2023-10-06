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


def get_log_list():
    """get all logs"""
    
    log_filepath_list = sorted(glob.glob(PATH_DB_PSA_CAL_LOG+"/**/*nmd-pi-delivery.log*", recursive=True))
    if len(log_filepath_list)==0:
        print("Error: Log files not found")
    
    return log_filepath_list




def extract_log_info(log_filepath):
    """From list of log filepaths, read in and extract all the required information, saving it to a dictionary
    lids with full start/end times: nmd_cal_sc_so_20181007t064522-20181007t073102-a-i-149
    """

    
     
    #parse all logs into memory
    
    log_filename = os.path.basename(log_filepath)
    print("Reading", log_filename)
    
    #2022 version 3.0 products: the format of the output log from ESA has changed - extract ingestion time and lid from the logs
    #all older versions in the logs can be skipped
    validator_lids_pass_text = np.fromregex(log_filepath, "\[PI_Packager\] NMD: PASS: \S+em16_tgo_nmd-(\d+T\d+)\S+Orbit_\d+\/(\S+)[.]xml", dtype={'names': ('dts', 'lids'), 'formats': ((np.str_,18), (np.str_,100))})
    validator_lids_fail_text = np.fromregex(log_filepath, "\[PI_Packager\] NMD: FAIL: \S+em16_tgo_nmd-(\d+T\d+)\S+Orbit_\d+\/(\S+)[.]xml", dtype={'names': ('dts', 'lids'), 'formats': ((np.str_,18), (np.str_,100))})
    
    pass_dict = {str(s[1]):{"dt":datetime.strptime(s[0][:-3], "%Y%m%dT%H%M%S")} for  s in validator_lids_pass_text}
    fail_dict = {str(s[1]):{"dt":datetime.strptime(s[0][:-3], "%Y%m%dT%H%M%S")} for  s in validator_lids_fail_text}
    
    #add the version number to the dictionary
    for key in pass_dict.keys():
        pass_dict[key]["version"] = get_log_version(pass_dict[key]["dt"])
    for key in fail_dict.keys():
        fail_dict[key]["version"] = get_log_version(fail_dict[key]["dt"])
    
        
    return pass_dict, fail_dict




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
    ongoing_log_filepath = os.path.join(PATH_DB_PSA_CAL_LOG, "nmd-pi-delivery.log")
    
    if os.path.exists(ongoing_log_filepath):
        with open(ongoing_log_filepath, "r") as f:
            lines = f.readlines()
            #read lines backwards to find a datetime
            #new ESA log format for 2022 version 3.0 onwards
            for i in range(-1, -100, -1):
                info_datetime = re.match("\[PI_Packager\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO.*", lines[i])
                if info_datetime:
                    log_datetime = datetime.strptime(info_datetime.groups()[0], LOG_FORMAT_STR)
                    return log_datetime
            print("Error: datetime not found in log file %s" %ongoing_log_filepath)
            return False
    else:
        print("Log file not found %s, continuing" %ongoing_log_filepath)
        