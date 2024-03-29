# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:30:34 2021

@author: iant

STEP 1: MOVE/COPY PSA DATA TO NEW DIR IN ARCHIVE

STEP 2: RUN PROGRAM:
    CHECK FOR ACTIVITY ON ESA SERVER
    COMPARE LIST OF FILES IN PSA TO FILES
    UNZIP AND MOVE MISSING FILES TO STAGING AREA
    ZIP ALL MISSING FILES TOGETHER
    TRANSFER TO ESA TMP DIR
    MOVE FROM TMP TO NMD

TO DO:
    CHECK FILES IN PSA ROOT FOLDER VS ARCHIVE


"""

import os
import sys
import re
import glob
import platform
from datetime import datetime
import argparse
import time

if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    sys.path.append(".")


from nomad_ops.core.psa_transfer.config import \
    ESA_PSA_CAL_URL, LOG_FORMAT_STR, PSA_CAL_VERSION, PSA_FILE_DIR, LOCAL_UNZIP_TMP_DIR, N_FILES_PER_ZIP

from nomad_ops.core.psa_transfer.get_psa_logs import \
    last_process_datetime

from nomad_ops.core.psa_transfer.db_functions import \
    get_lids_of_version, make_db

from nomad_ops.core.psa_transfer.functions import \
    psaTransferLog, returnNotMatching, convert_lid_to_short_filename, remove_version_extension, \
    get_zip_version, unzip_files_to_dir, files_to_zip, n_products_in_queue, make_big_zip_filename, \
    transfer_files, move_zip_files_on_esac_server


from nomad_ops.core.psa_transfer.transfer_psa_logs_from_esa import psa_logs_esa_to_bira

windows = platform.system() == "Windows"


version = PSA_CAL_VERSION


MAKE_ZIPS = True #make big zip files by unzipping individual products then delete unzipped files
# MAKE_ZIPS = False

MAKE_TRANSFER = True #transfer to ESAC
# MAKE_TRANSFER = False

WAIT_FOR_USER = True #user must press a key to start transfer. Overruled by --now or --transfer_now
# WAIT_FOR_USER = False

MOVE_FROM_TEMP = True #move from ESAC temp dir to main staging area once transfer complete
# MOVE_FROM_TEMP = False


# MANUAL_SELECTION = True #select files based on regex. Overruled is arg given
MANUAL_SELECTION = False

MANUAL_SELECTION_REGEX = re.compile(".*")



if __name__ == "__main__":
    
    #if --now keyword given, don't wait for user to confirm transfer
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_queue", help="Check if ESA ingestion queue is empty", action="store_true")
    parser.add_argument("--transfer", help="Transfer missing products to ESA", action="store_true")
    parser.add_argument("--transfer_now", help="Transfer missing products to ESA automatically", action="store_true")
    parser.add_argument("--filter", help="Transfer missing products based on regex filter", metavar="RE")
    args = parser.parse_args()
    if args.transfer_now:
        WAIT_FOR_USER = False
    
    
    
    if args.filter:
        MANUAL_SELECTION = True
        MANUAL_SELECTION_REGEX = re.compile(args.filter)


    
    
    if args.check_queue:
        queue_size = 1
        last_process_delta = 1.0
        
        #before starting transfer, check that processing is not ongoing and that there are no products waiting
        #loop until ESA server is ready
        while queue_size != 0 or last_process_delta < 3600.:
        
            #first check for new PSA logs on ESA server
            if not windows:
                psa_logs_esa_to_bira()
            
            
            
            #check active PSA log for datetime of last entry. If within last hour, or files still in queue, wait 1 hour
            if windows:
                queue_size = 0
            else:
                queue_size = n_products_in_queue()
            
            last_process = last_process_datetime()
            last_process_delta = (datetime.now() - last_process).total_seconds()
        
            #wait an hour
            now = datetime.strftime(datetime.now(), LOG_FORMAT_STR)
            if queue_size != 0:
                print("Time is %s; there are %i files remaining in the ESA server queue" %(now, queue_size))
            if last_process_delta < 3600.:
                print("Time is %s; files were being processed on ESA server %i minutes ago" %(now, last_process_delta / 60.))
            
            if queue_size != 0 or last_process_delta < 3600.:
                for i in range(10):
                    time.sleep(60)
    
    
    if args.transfer or args.transfer_now:
        print("ESA server is ready for transfer: checking local log db")
        make_db()
        
        
        #get PSA files from log db
        pass_lids_in_psa = get_lids_of_version("pass", version)
        
        #convert lids to filenames (without version number)
        filenames_in_psa = sorted(list(set([convert_lid_to_short_filename(i) for i in pass_lids_in_psa if "browse" not in i])))
        
        #get list of local psa files to transfer
        local_zip_filepath_list = sorted(glob.glob(PSA_FILE_DIR+"/**/*.zip", recursive=True))
        local_filename_dict = {remove_version_extension(i):i for i in local_zip_filepath_list}
        
        #check local PSA files all have the correct version
        local_zip_versions = sorted(list(set([get_zip_version(i) for i in local_zip_filepath_list])))
        local_versions_str = ",".join(local_zip_versions)
        
        if local_versions_str != version:
            print("Error: the %s files in the local directory are not of the correct version %s" %(local_versions_str, version))
        
        
        #compare contents of PSA vs local directory
        psa_not_in_local, local_not_in_psa = returnNotMatching(filenames_in_psa, local_filename_dict.keys())
        
        print("There are %i v%s PSA files not in local directory %s\n" %(len(psa_not_in_local), version, PSA_FILE_DIR))
        print("There are %i v%s local files not in the PSA" %(len(local_not_in_psa), version))
        # for version, n_files in count_versions(local_not_in_psa).items():
            # print("%i are of PSA data version %s" %(n_files, version))    
            # print(local_not_in_psa)
        
        
        if MAKE_ZIPS:
            
            if MANUAL_SELECTION:
                print("Manually selecting files based on regex filter %s" %(MANUAL_SELECTION_REGEX.pattern))
                filtered_zip_filenames = list(filter(MANUAL_SELECTION_REGEX.search, local_filename_dict.keys()))

                #retrieve filepaths of local zip files that match the filter
                zip_filepaths = [local_filename_dict[i] for i in filtered_zip_filenames]
                print("%i files are to be transferred" %(len(zip_filepaths)))
            
            else:
                #retrieve filepaths of local zip files not in PSA
                zip_filepaths = [local_filename_dict[i] for i in local_not_in_psa]
    
    
            unzip_tmp_dir = os.path.join(LOCAL_UNZIP_TMP_DIR, "unzip")
            
            big_zip_prefix = make_big_zip_filename(zip_filepaths, version)
            
            big_zip_filepaths = []
            
            #unzip all to staging area in groups of N_FILES_PER_ZIP
            for ix, n in enumerate(range(0, len(zip_filepaths), N_FILES_PER_ZIP)):
                
                #get partial file list
                zips_split = zip_filepaths[n:(n+N_FILES_PER_ZIP)]
        
                big_zip_filename = "%s_%0.1f" %(big_zip_prefix, (ix+1))
                print("Making zip file %s.zip" %big_zip_filename)
                
                #unzip partial list to staging area
                unzip_files_to_dir(zips_split, unzip_tmp_dir)
            
                #rezip into big zip file
                big_zip_filepath = os.path.join(LOCAL_UNZIP_TMP_DIR, big_zip_filename)
                files_to_zip(unzip_tmp_dir, big_zip_filepath)
                
                #delete unzipped files from temp directory
                for filepath in os.scandir(unzip_tmp_dir):
                    os.remove(filepath)
                    
                big_zip_filepaths.append(big_zip_filepath + ".zip")
                
            
        
        if MAKE_ZIPS and MAKE_TRANSFER:
            print("%i zip files will be copied to the PSA server, from %s to %s" \
              %(len(big_zip_filepaths), sorted(local_not_in_psa)[0], sorted(local_not_in_psa)[-1]))
            
            if WAIT_FOR_USER:
                input("Press any key to continue")
    
            if not windows:
                    transfer_files(big_zip_filepaths)
                    
                    if MOVE_FROM_TEMP:
                        move_zip_files_on_esac_server(big_zip_filepaths)
                    else:
                        print("Not moving files from ESAC temp directory")
                    
            else:
                print("Warning: Running on windows; no files transferred")
        else:
            print("Transfer cancelled")
        
        
