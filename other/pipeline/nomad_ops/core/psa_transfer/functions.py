# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:22:27 2021

@author: iant

PSA GENERIC FUNCTIONS
"""

import os
import re
from datetime import datetime
import subprocess
from urllib.parse import urlparse
import shutil

from nomad_ops.core.psa_transfer.config import \
    MAKE_PSA_LOG_DIR, PSA_CAL_VERSION, OLD_CAL_VERSIONS, ESA_PSA_CAL_URL




def psaTransferLog(lineToWrite):
    dt = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S")
    logPath = os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_transfer.log")
    with open(logPath, 'a') as logFile:
        logFile.write(dt + "\t" + lineToWrite + '\n')



def returnNotMatching(a, b):
    """compare list 1 to list 2 and return non matching entries"""
    return [[x for x in a if x not in b], [x for x in b if x not in a]]



def count_versions(file_list):
    """given a list of zip filenames, count how many are in each version"""
    versions = {version:0 for version in OLD_CAL_VERSIONS + [PSA_CAL_VERSION]}
    for version in versions.keys():
        ending = "%s.zip" %version
        for filename in file_list:
            if ending in filename:
                versions[version] += 1
    return versions



def convert_lid_to_short_filename(lid):
    matches = re.search("(nmd_cal_sc_\D+_\d+)T(\d+)-\d+T\d+(\S+)", lid).groups()
    return matches[0] + "-" + matches[1] + "00" + matches[2]



def remove_version_extension(filename):
    filename_out = os.path.splitext(os.path.basename(filename))[0].rsplit("_", 1)[0]
    return filename_out



def get_zip_version(zip_filepath):
    version = os.path.splitext(os.path.basename(zip_filepath))[0].rsplit("_", 1)[1]
    return version



def unzip_files_to_dir(zip_filepaths, unzip_dir):
    for zip_filepath in zip_filepaths:
        shutil.unpack_archive(zip_filepath, unzip_dir)



def files_to_zip(unzipped_file_dir, zip_filepath):
    shutil.make_archive(zip_filepath, "zip", unzipped_file_dir)
  
    
    
def n_products_in_queue():
    """first check for any products not yet moved from nmd directory to PSA staging area"""
 
    esa_p_url = urlparse(ESA_PSA_CAL_URL)

    ssh_cmd2 = ["ssh", esa_p_url.netloc, "ls nmd -1 | wc -l"]
    pipe = subprocess.Popen(ssh_cmd2,
                            shell=False,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output = pipe.communicate()[0]
    
    n_files = int(output.decode().strip()) - 2 #2 subdirs
    
    return n_files


