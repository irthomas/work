# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:50:17 2020

@author: iant
"""


import os
import glob
import sys
import re
import numpy as np
import platform
from datetime import datetime
import xlsxwriter
#import codecs
import subprocess
#import sys
import tarfile
from urllib.parse import urlparse


"""REMEMBER TO RUN TRANSFER ESA_TO_BIRA FIRST TO GET THE LATEST LOG FILES!"""


# TRANSFER_MISSING_FILES = True
TRANSFER_MISSING_FILES = False



PSA_CAL_VERSION = "1.0"

if platform.system() == "Windows":
    from tools.file.paths import paths
    ROOT_DATASTORE_PATH = paths["DATASTORE_ROOT_DIRECTORY"]

else:
    os.environ["NMD_OPS_PROFILE"] = "default"
    os.environ["FS_MODEL"] = "False"
    if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
        sys.path.append(".")
    from nomad_ops.config import ROOT_DATASTORE_PATH


PATH_DB_PSA_CAL_LOG = os.path.join(ROOT_DATASTORE_PATH, "db", "psa", "cal_logs") #where the PSA ingestion logs are moved to
MAKE_PSA_LOG_DIR = os.path.join(ROOT_DATASTORE_PATH, "logs", "psa_cal")
PSA_FILE_DIR = os.path.join(ROOT_DATASTORE_PATH, "archive", "psa", "1.0", "data_calibrated")

BIRA_URL = "file:/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/"
ESA_URL = "ssh://exonmd@exoops01.esac.esa.int/home/exonmd/"

BIRA_PSA_CAL_URL = BIRA_URL + "archive/1.0/data_calibrated/"
ESA_PSA_CAL_URL = ESA_URL + "nmd/tmp0/"




#get all logs
# log_filepath_list = sorted(glob.glob(PATH_DB_PSA_CAL_LOG+"/**/*nmd-pi-delivery.log.2*", recursive=True))
log_filepath_list = sorted(glob.glob(PATH_DB_PSA_CAL_LOG+"/**/*nmd-pi-delivery*", recursive=True))
if len(log_filepath_list)==0:
    print("Error: Log files not found")




def extract_log_info(log_filepath_list):
    """From list of log filepaths, read in and extract all the required information, saving it to a dictionary
    Nomenclature: 
        zip filenames contain one datetime string only with version number: nmd_cal_sc_uvis_20180529-11060000-d_1.0
        shortened filename is same without version number: nmd_cal_sc_uvis_20180529-11060000-d
    
        xml/tab/png filenames are lids with full start/end times: nmd_cal_sc_so_20181007t064522-20181007t073102-a-i-149
    """

    def convert_lid_to_short_filename(lid):
        matches = re.search("(nmd_cal_sc_\D+_\d+)T(\d+)-\d+T\d+(\S+)", lid).groups()
        return matches[0] + "-" + matches[1] + "00" + matches[2]
    
    date_string = ("date_string", "S30")
    zip_filename = ("zip_filename", "S45")
    short_filename = ("short_filename", "S100")
    lid_string = ("lid_string", "S100")
    error_string = ("error_string", "S200")
    
    
    #parse all logs into memory
    log_dict = {
    "zip_filenames_received":[],
    "zip_dates_received":[],
    "zip_filenames_transferred":[],
    "zip_short_filenames_transferred":[],
    "zip_dates_transferred":[],
    "zip_filenames_expanded":[],
    "zip_short_filenames_expanded":[],
    "zip_dates_expanded":[],
    "validator_lids_pass":[],
    "validator_lids_fail":[],
    "validator_lids_error":[],
    "validator_errors":[],
    }
    
    for log_filepath in log_filepath_list:
        
        
        log_filename = os.path.basename(log_filepath)
        print("Reading", log_filename)
        
        files_received = np.fromregex(log_filepath, "(\S+\s\S+) INFO  Checking file received: (\S+)[.]zip", [date_string, zip_filename])
        files_transferred_to_staging = np.fromregex(log_filepath, "(\S+\s\S+) INFO  File (\S+)[.]zip transferred to \S+staging\/(\S+)", [date_string, zip_filename, short_filename])
        zip_file_expanded = np.fromregex(log_filepath, "(\S+\s\S+) INFO  Expanding zip file: \S+staging\/(\S+)\/(\S+)[.]zip", [date_string, short_filename, zip_filename])
        validator_pass = np.fromregex(log_filepath, "PASS: \S+Orbit_\d+\/(\S+)[.]xml", [lid_string])
        validator_fail = np.fromregex(log_filepath, "FAIL: \S+Orbit_\d+\/(\S+)[.]xml", [lid_string])
        validator_error = np.fromregex(log_filepath, "ERROR: (.+)\S+\n\s+file\S+Orbit_\d+\/(\S+)[.]xml.+", [error_string, lid_string])
    
    
        zip_filenames_received_temp = [i.decode() for i in files_received["zip_filename"]]
        zip_dates_received_temp = [i.decode() for i in files_received["date_string"]]
        
        zip_filenames_transferred_temp = [i.decode() for i in files_transferred_to_staging["zip_filename"]]
        zip_short_filenames_transferred_temp = [i.decode() for i in files_transferred_to_staging["short_filename"]]
        zip_dates_transferred_temp = [i.decode() for i in files_transferred_to_staging["date_string"]]
        
        zip_filenames_expanded_temp = [i.decode() for i in zip_file_expanded["zip_filename"]]
        zip_short_filenames_expanded_temp = [i.decode() for i in zip_file_expanded["short_filename"]]
        zip_dates_expanded_temp = [i.decode() for i in zip_file_expanded["date_string"]]
        
        validator_lids_pass_temp = [i.decode() for i in validator_pass["lid_string"]]
        validator_lids_fail_temp = [i.decode() for i in validator_fail["lid_string"]]
        validator_lids_error_temp = [i.decode() for i in validator_error["lid_string"]]
        validator_errors_temp = [i.decode() for i in validator_error["error_string"]]

        log_dict["zip_filenames_received"].extend(zip_filenames_received_temp)
        log_dict["zip_dates_received"].extend(zip_dates_received_temp)
        log_dict["zip_filenames_transferred"].extend(zip_filenames_transferred_temp)
        log_dict["zip_short_filenames_transferred"].extend(zip_short_filenames_transferred_temp)
        log_dict["zip_dates_transferred"].extend(zip_dates_transferred_temp)
        log_dict["zip_filenames_expanded"].extend(zip_filenames_expanded_temp)
        log_dict["zip_short_filenames_expanded"].extend(zip_short_filenames_expanded_temp)
        log_dict["zip_dates_expanded"].extend(zip_dates_expanded_temp)
        log_dict["validator_lids_pass"].extend(validator_lids_pass_temp)
        log_dict["validator_lids_fail"].extend(validator_lids_fail_temp)
        log_dict["validator_lids_error"].extend(validator_lids_error_temp)
        log_dict["validator_errors"].extend(validator_errors_temp)
    
    
        
    #convert validator lids to short filenames
    log_dict["validator_short_filenames_pass"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_pass"]]
    log_dict["validator_short_filenames_fail"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_fail"]]
    log_dict["validator_short_filenames_error"] = [convert_lid_to_short_filename(i) for i in log_dict["validator_lids_error"]]
    
    return log_dict



def match_filenames_to_log_entries(log_dict):
    """make dictionary, with one list for each filename received"""
    psa_dict = {}
    
    for filename_index, zip_filename_received in enumerate(log_dict["zip_filenames_received"]):
        
        #check if entry already exists
        if zip_filename_received not in psa_dict.keys(): #set up blank dictionary for entries
            psa_dict[zip_filename_received] = {
                "zip_date_received":[], 
                "zip_filename_transferred":[], 
                "zip_short_filename_transferred":[],
                "zip_date_transferred":[],
                "zip_filename_expanded":[],
                "zip_short_filename_expanded":[],
                "zip_date_expanded":[],
                "validator_short_filename_pass":[],
                "validator_lid_pass":[],
                "validator_short_filename_fail":[],
                "validator_lid_fail":[],
                "validator_short_filename_error":[],
                "validator_lid_error":[],
                "validator_error":[],
                }
        
    
        
            entry_dict = psa_dict[zip_filename_received]
    
            #find zip filename in lists
            indices = [i for i, value in enumerate(log_dict["zip_filenames_received"]) if value == zip_filename_received]
            for i, index in enumerate(indices):
                entry_dict["zip_date_received"].append(log_dict["zip_dates_received"][index])
            
            #find zip filename in lists
            indices = [i for i, value in enumerate(log_dict["zip_filenames_transferred"]) if value == zip_filename_received]
            for i, index in enumerate(indices):
                entry_dict["zip_filename_transferred"].append(log_dict["zip_filenames_transferred"][index])
                entry_dict["zip_short_filename_transferred"].append(log_dict["zip_short_filenames_transferred"][index])
                entry_dict["zip_date_transferred"].append(log_dict["zip_dates_transferred"][index])
        
            #find zip filename in lists
            indices = [i for i, value in enumerate(log_dict["zip_filenames_expanded"]) if value == zip_filename_received]
            for i, index in enumerate(indices):
                entry_dict["zip_filename_expanded"].append(log_dict["zip_filenames_expanded"][index])
                entry_dict["zip_short_filename_expanded"].append(log_dict["zip_short_filenames_expanded"][index])
                entry_dict["zip_date_expanded"].append(log_dict["zip_dates_expanded"][index])
        
            zip_short_filename = log_dict["zip_short_filenames_expanded"][index]
        
            #find zip short filename in list of validator pass short filenames
            name_indices = [i for i,value in enumerate(log_dict["validator_short_filenames_pass"]) if value == zip_short_filename]
            for i, name_index in enumerate(name_indices):
                entry_dict["validator_short_filename_pass"].append(log_dict["validator_short_filenames_pass"][name_index])
                entry_dict["validator_lid_pass"].append(log_dict["validator_lids_pass"][name_index])
        
            #find zip short filename in list of validator fail short filenames
            name_indices = [i for i,value in enumerate(log_dict["validator_short_filenames_fail"]) if value == zip_short_filename]
            for i, name_index in enumerate(name_indices):
                entry_dict["validator_short_filename_fail"].append(log_dict["validator_short_filenames_fail"][name_index])
                entry_dict["validator_lid_fail"].append(log_dict["validator_lids_fail"][name_index])
        
        
            #find zip short filename in list of validator error short filenames
            name_indices = [i for i,value in enumerate(log_dict["validator_short_filenames_error"]) if value == zip_short_filename]
            for i, name_index in enumerate(name_indices):
                entry_dict["validator_short_filename_error"].append(log_dict["validator_short_filenames_error"][name_index])
                entry_dict["validator_lid_error"].append(log_dict["validator_lids_error"][name_index])
                entry_dict["validator_error"].append(log_dict["validator_errors"][name_index])

    return psa_dict

print("Making log dictionary")
log_dict = extract_log_info(log_filepath_list)
print("Organising PSA dictionary")
psa_dict = match_filenames_to_log_entries(log_dict)




#make a list of all filenames that pass validation
pass_list = []
for short_filename, entry_dict in psa_dict.items():
    search_dict = entry_dict["validator_short_filename_pass"]
    if len(search_dict) > 0:
        pass_list.append(short_filename+".zip")
pass_list = sorted(pass_list)

#check for duplications
if len(pass_list) != len(set(pass_list)):
    print("Error: duplications found")
else:
    ingested_zip_filename_list = pass_list
        

local_zip_filepath_list = sorted(glob.glob(PSA_FILE_DIR+"/**/*.zip", recursive=True))
local_zip_filename_list = [os.path.split(filename)[1] for filename in local_zip_filepath_list]



def psaTransferLog(lineToWrite):
    dt = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S")
    logPath = os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_transfer.log")
    with open(logPath, 'a') as logFile:
        logFile.write(dt + "\t" + lineToWrite + '\n')


# def convertXmlFilenameToZip(xmlFilename):
#     xmlFilenameSplit = xmlFilename.split("_")
#     start = xmlFilenameSplit[0]+"_"+xmlFilenameSplit[1]+"_"+xmlFilenameSplit[2]+"_"+xmlFilenameSplit[3]+"_"+xmlFilenameSplit[4][:8]
#     mid = "-"+xmlFilenameSplit[4][9:15]+"00"
#     end = xmlFilenameSplit[4][31:].split(".")[0]+"_"+PSA_CAL_VERSION+".zip"
#     return start+mid+end

# ingestedZipFilenameList = sorted([convertXmlFilenameToZip(xmlFilename) for xmlFilename in ingestedXmlFilenameList])

def returnNotMatching(a, b):
    """compare list 1 to list 2 and return non matching entries"""
    return [[x for x in a if x not in b], [x for x in b if x not in a]]


# def writeFileLog(filepath, filenameList):
#     """write list of filenames to log files"""
#     with open(filepath, "w") as f:
#         for filename in filenameList:
#             f.write("%s\n" %filename)


print("Comparing contents of PSA and local directory")
psa_not_in_local, local_not_in_psa = returnNotMatching(ingested_zip_filename_list, local_zip_filename_list)
print("There are %i PSA files not in local directory %s" %(len(psa_not_in_local), PSA_FILE_DIR))
#print(psaNotInLocal)
print("There are %i local files not in the PSA" %(len(local_not_in_psa)))
#print(localNotInPsa)



def write_xlsx_report(log_filepath_list, psa_dict):
    """write to xlsx"""
    current_time_string = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    with xlsxwriter.Workbook(os.path.join(MAKE_PSA_LOG_DIR, "psa_transfer_report_%s.xlsx" %current_time_string)) as workbook:
    
        cell_format = workbook.add_format()
        cell_format.set_bg_color('red')
    
    
        worksheet1 = workbook.add_worksheet("log_filelist")
        worksheet1.set_column('A:A', 800/7)
        worksheet1.write(0, 0, "PSA Delivery Log Path")
        
        for row,value in enumerate(log_filepath_list):
            worksheet1.write(row+1, 0, value)
        
        
        worksheet2 = workbook.add_worksheet("zip_files_received")
        worksheet2.set_column('A:A', 320/7)
        worksheet2.set_column('B:C', 260/7)
        worksheet2.write(0, 0, "Product filename and version")
        worksheet2.write(0, 1, "PSA zip file received time")
        worksheet2.write(0, 2, "PSA zip file received time (duplicate)")
                          	
        row = 1
        for short_filename, entry_dict in psa_dict.items():
            worksheet2.write(row, 0, short_filename)
            for i, date in enumerate(entry_dict["zip_date_received"]):
                worksheet2.write(row, i+1, date)
            row += 1
    
        worksheet3 = workbook.add_worksheet("zip_files_transferred")
        worksheet3.set_column('A:A', 320/7)
        worksheet3.set_column('B:C', 260/7)
        worksheet3.write(0, 0, "Product filename and version")
        worksheet3.write(0, 1, "PSA product transferred to staging time")
        worksheet3.write(0, 2, "PSA product transferred to staging time (duplicate)")
        row = 1
        for short_filename, entry_dict in psa_dict.items():
            worksheet3.write(row, 0, short_filename)
            for i, date in enumerate(entry_dict["zip_date_transferred"]):
                worksheet3.write(row, i+1, date)
            row += 1
    
        worksheet4 = workbook.add_worksheet("zip_files_expanded")
        worksheet4.set_column('A:A', 320/7)
        worksheet4.set_column('B:C', 260/7)
        worksheet4.write(0, 0, "Product filename and version")
        worksheet4.write(0, 1, "PSA zip file expanded time")
        worksheet4.write(0, 2, "PSA zip file expanded time (duplicate)")
        row = 1
        for short_filename, entry_dict in psa_dict.items():
            worksheet4.write(row, 0, short_filename)
            for i, date in enumerate(entry_dict["zip_date_expanded"]):
                worksheet4.write(row, i+1, date)
            row += 1
    
    
        worksheet5 = workbook.add_worksheet("validation_results")
        worksheet5.set_column('A:A', 320/7)
        worksheet5.set_column('B:D', 150/7)
        worksheet5.write(0, 0, "Product filename and version")
        worksheet5.write(0, 1, "Pass validation")
        worksheet5.write(0, 2, "Fail validation")
        worksheet5.write(0, 3, "Validation error")
        row = 1
        for short_filename, entry_dict in psa_dict.items():
            worksheet5.write(row, 0, short_filename)
    
            pass_text = ""
            for i, _ in enumerate(entry_dict["validator_short_filename_pass"]):
                pass_text += "PASS, "
            worksheet5.write(row, 1, pass_text)
    
            fail_text = ""
            for i, _ in enumerate(entry_dict["validator_short_filename_fail"]):
                fail_text += "FAIL, "
            if fail_text == "":
                worksheet5.write(row, 2, fail_text)
            else:
                worksheet5.write(row, 2, fail_text, cell_format)
    
            error_text = ""
            for i, error in enumerate(entry_dict["validator_error"]):
                error_text += error + ", "
            if error_text == "":
                worksheet5.write(row, 3, error_text)
            else:
                worksheet5.write(row, 3, error_text, cell_format)
    
            row += 1
                
    
    
        worksheet6 = workbook.add_worksheet("psa_not_in_local")
        worksheet6.set_column('A:A', 430/7)
        worksheet6.write(0, 0, "List of zip files that are present in the PSA but not in local directory")
        for row,value in enumerate(sorted(psa_not_in_local)):
            worksheet6.write(row+1, 0, value)
    
        worksheet7 = workbook.add_worksheet("local_not_in_psa")
        worksheet7.set_column('A:A', 430/7)
        worksheet7.write(0, 0, "List of zip files that are present locally but not yet in the PSA (or did not yet pass validation)")
        for row,value in enumerate(sorted(local_not_in_psa)):
            worksheet7.write(row+1, 0, value)


print("Writing results to file")
write_xlsx_report(log_filepath_list, psa_dict)




if TRANSFER_MISSING_FILES:
    esa_p_url = urlparse(ESA_PSA_CAL_URL)
    
    
    localNotInPsaPath = []
    error = False
    for localFilename in local_not_in_psa:
        if localFilename in local_zip_filename_list:
            index = [i for i,x in enumerate(local_zip_filename_list) if x==localFilename]
            if len(index)==1:
                localNotInPsaPath.append(local_zip_filepath_list [index[0]])
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
            
            print("%i files will be copied to the PSA server, from %s to %s" \
                  %(len(transferFileList), os.path.basename(list(sorted(transferFileList))[0]), os.path.basename(list(sorted(transferFileList))[-1])))
            input("Press any key to continue")
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
        
       