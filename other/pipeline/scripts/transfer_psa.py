# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:27:35 2022

@author: iant

SCRIPT TO PERFORM ALL DATA TRANSFERS TO PSA

ARGUMENTS:
    
    archive copy from psa dir to archive dir
    transfer copy from archive dir to ESAC
    
    --beg filename start date
    --end filename end date
    --filter regex filter for product filenames
    
    --all (re)transfer all products even if already present at destination
    
    --n number of products to put in zip file to transfer to ESA (default=500)
    
    {archive, transfer} --beg --end --filter --n --all
    
"""

import os
import sys
import re
import glob
import platform
from datetime import datetime
import subprocess
import tarfile
from urllib.parse import urlparse
import argparse
import shutil
import time

if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    sys.path.append(".")


from nomad_ops.core.psa_transfer.config import \
    ESA_PSA_CAL_URL, LOG_FORMAT_STR, PSA_CAL_VERSION, PSA_DATASTORE_DIR, PSA_ARCHIVE_DIR, LOCAL_UNZIP_TMP_DIR


from nomad_ops.core.psa_transfer.db_functions import \
    get_lids_of_version, make_db

from nomad_ops.core.psa_transfer.functions import \
    psaTransferLog, returnNotMatching, \
    convert_lid_to_short_filename, remove_version_extension, get_zip_version, unzip_files_to_dir, files_to_zip, \
    get_datetime_from_lid, get_datetime_from_psa_filename, get_psa_filename_from_fullpath

from nomad_ops.core.psa_transfer.esac_server import \
    wait_until_no_activity


windows = platform.system() == "Windows"




def make_datetime(s):
    format = "%Y-%m-%dT%H:%M:%S"
    for end_ix in (100, -3, -6, -9):
        try:
            return datetime.strptime(s, format[:end_ix])
        except ValueError as e:
            if end_ix == -9:
                raise e



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NOMAD PSA data transfer")
    parser.add_argument('task', choices=["archive", "transfer"])
    parser.add_argument('--beg', dest='beg_dtime', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="Begin time")
    parser.add_argument('--end', dest='end_dtime', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="End time")
    parser.add_argument('--filter', help="Process only files with name matching this RE", metavar="RE")
    parser.add_argument('--list', dest='list', action='store_true', help='List files to be copied')
    parser.add_argument('--all', dest="all", action='store_true', help="Transfer all products even if already present at destination")
    parser.add_argument('--n', dest="n", type=int, default=500, help="The number products to put in zip file to transfer to ESA (default=500)")
    args = parser.parse_args()





if args.task == "archive":
    """transfer products from datastore folder to the archive"""
    
    #check what files are already in the directories
    
    #get list of datastore psa files
    datastore_zip_filepath_list = sorted(glob.glob(PSA_DATASTORE_DIR+"/**/*.zip", recursive=True))
    datastore_filename_dict = {remove_version_extension(i):{"dt":get_datetime_from_psa_filename(remove_version_extension(i)), "path":i} for i in datastore_zip_filepath_list}

    #get list of archive psa files
    archive_zip_filepath_list = sorted(glob.glob(PSA_ARCHIVE_DIR+"/**/*.zip", recursive=True))
    archive_filename_dict = {remove_version_extension(i):{"dt":get_datetime_from_psa_filename(remove_version_extension(i)), "path":i} for i in archive_zip_filepath_list}

    #select filenames by begin/end times if given
    if args.beg_dtime and args.end_dtime:
        datastore_filenames = [k for k,v in datastore_filename_dict.items() if args.beg_dtime < v["dt"] < args.end_dtime]

    elif args.beg_dtime:
        datastore_filenames = [k for k,v in datastore_filename_dict.items() if v["dt"] > args.beg_dtime]
        
    elif args.end_dtime:
        datastore_filenames = [k for k,v in datastore_filename_dict.items() if v["dt"] < args.end_dtime]
        
    else:
        datastore_filenames = list(datastore_filename_dict.keys())

    

    #then check filter and remove non matching filenames
    if args.filter:
        re_filter = re.compile(args.filter)
        datastore_filenames = [s for s in datastore_filenames if re_filter.match(s)]
        
        
    #check if overwriting existing or not
    if args.all:
        transfer_list = datastore_filenames
        
    else:
        transfer_list = [s for s in datastore_filenames if s not in archive_filename_dict.keys()]
        
        
    #check if only listing or actually transferring
    if args.list:
        print("### Files to be transferred to the archive ###")
        for s in transfer_list:
            print(s)
            
    else:
        print("Transferring %i files to the archive" %len(transfer_list))
        for s in transfer_list:
            print("Moving %s" %s)
            src_path = datastore_filename_dict[s]["path"]
            dest_path = src_path.replace(PSA_DATASTORE_DIR, PSA_ARCHIVE_DIR)
   
            #make destination path
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                print("Making directory %s" %dest_dir)
                os.makedirs(dest_dir, exist_ok=True)
            
            #move file
            shutil.move(src_path, dest_path)
            
            
            #if source directories are now empty, delete them (day / month / year only)
            source_dir = os.path.dirname(src_path)
            if len(os.listdir(source_dir)) == 0: #if dir is empty
                print("Deleting directory %s" %source_dir)
                shutil.rmtree(source_dir)

                #repeat for month            
                source_dir = os.path.dirname(source_dir)
                if len(os.listdir(source_dir)) == 0: #if dir is empty
                    print("Deleting directory %s" %source_dir)
                    shutil.rmtree(source_dir)
                
                    #repeat for year - don't delete more than this
                    source_dir = os.path.dirname(source_dir)
                    if len(os.listdir(source_dir)) == 0: #if dir is empty
                        print("Deleting directory %s" %source_dir)
                        shutil.rmtree(source_dir)







if args.task == "transfer":
    """transfer products from archive folder to the ESAC server"""

    #wait for ESAC server
    wait_until_no_activity()
    
    #add ESAC log entries to db
    print("Checking local log db")
    make_db()


    
    #get list of archive psa files, this time with version numbers
    archive_zip_filepath_list = sorted(glob.glob(PSA_ARCHIVE_DIR+"/**/*.zip", recursive=True))
    archive_filename_dict = {get_psa_filename_from_fullpath(s):{"dt":get_datetime_from_psa_filename(get_psa_filename_from_fullpath(s)), "path":s} for s in archive_zip_filepath_list}


    #get list of psa files already processed by ESAC from db
    version = "2.0"
    pass_lids_in_psa = get_lids_of_version("pass", version)
    
    #convert lids to filenames (without version number)
    filenames_in_psa = sorted(list(set([convert_lid_to_short_filename(i)+"_%s" %version for i in pass_lids_in_psa if "browse" not in i])))
        


    #select filenames by begin/end times if given
    if args.beg_dtime and args.end_dtime:
        archive_filenames = [k for k,v in archive_filename_dict.items() if args.beg_dtime < v["dt"] < args.end_dtime]

    elif args.beg_dtime:
        archive_filenames = [k for k,v in archive_filename_dict.items() if v["dt"] > args.beg_dtime]
        
    elif args.end_dtime:
        archive_filenames = [k for k,v in archive_filename_dict.items() if v["dt"] < args.end_dtime]
        
    else:
        archive_filenames = list(archive_filename_dict.keys())

    

    #then check filter and remove non matching filenames
    if args.filter:
        re_filter = re.compile(args.filter)
        archive_filenames = [s for s in archive_filenames if re_filter.match(s)]
        
        
    #check if overwriting existing or not
    if args.all:
        transfer_list = archive_filenames
        
    else:
        transfer_list = [s for s in archive_filenames if s not in filenames_in_psa]
        
        
    #check if only listing or actually transferring
    if args.list:
        print("### Files to be transferred to ESAC ###")
        for s in transfer_list:
            print(s)






    n_files_per_zip = args.n


    zip_filepaths = [archive_filename_dict[s]["path"] for s in transfer_list]
    
    unzip_tmp_dir = os.path.join(LOCAL_UNZIP_TMP_DIR, "unzip")
    
    datetime_now = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    
    big_zip_filepaths = []
    
    #unzip all to staging area in groups of N_FILES_PER_ZIP
    for ix, n in enumerate(range(0, len(zip_filepaths), n_files_per_zip)):
        big_zip_filename = "%s_%s_%i" %(datetime_now, PSA_CAL_VERSION, ix)
        
        print("Making zip file %s.zip" %big_zip_filename)
        #get partial file list
        zips_split = zip_filepaths[n:(n+n_files_per_zip)]
    
        #unzip partial list to staging area
        unzip_files_to_dir(zips_split, unzip_tmp_dir)
    
        #rezip into big zip file
        big_zip_filepath = os.path.join(LOCAL_UNZIP_TMP_DIR, big_zip_filename)
        files_to_zip(unzip_tmp_dir, big_zip_filepath)
        
        #delete unzipped files from temp directory
        for filepath in os.scandir(unzip_tmp_dir):
            os.remove(filepath)
            
        big_zip_filepaths.append(big_zip_filepath)
        
    

    print("%i zip files will be copied to the PSA server, from %s to %s" \
          %(len(big_zip_filepaths), sorted(transfer_list)[0], sorted(transfer_list)[-1]))



    #transfer big zip files to ESA tmp directory
    esa_p_url = urlparse(ESA_PSA_CAL_URL)
        
    if not windows:
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
            for path in big_zip_filepaths:
                n = os.path.getsize(path)
                big_zip_name = os.path.basename(path)
                tar.add(path, big_zip_name)
                print("File added to TAR archive: %s (%.1f kB)" %(big_zip_name, n/1000))
                # psaTransferLog(big_zip_name)
            tar.close()
            time.sleep(2)

        print("Moving files from /nmd/tmp0 to nmd/ directory")
        ssh_cmd2 = ["ssh", esa_p_url.netloc, "mv nmd/tmp0/* nmd/"]
        subprocess.Popen(ssh_cmd2,
                        shell=False,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print("Transfer completed")
    else:
        print("Warning: Running on windows; no files transferred")

