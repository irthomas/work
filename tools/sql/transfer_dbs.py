# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:21:22 2022

@author: iant
"""

import os
import posixpath
import paramiko

from tools.file.passwords import passwords

SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"
REMOTE_HDF5_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/"
REMOTE_EDDS_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/"
REMOTE_OBS_DB_FILE_PATH = "/bira-iasb/projects/NOMAD/Data/pfm_auxiliary_files/observation_type/obs_type.db"



def transfer_file_from_hera(local_path, remote_path):

    print("Connecting to hera")
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # This script doesn't work for me unless this line is added!
    p.connect("hera.oma.be", port=22, username="iant", password=passwords["hera"])
    
    sftp = p.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    p.close()



def transfer_cache_db(level):
    """copy cache db for chosen level from remote path to local computer"""

    local_path = os.path.join(SHARED_DIR_PATH, "db", level + ".db")
    
    if level in ["spacewire", "tc1553", "tm1553_84", "tm1553_244", "tm1553_372"]:
        remote_path = posixpath.join(REMOTE_EDDS_PATH, level, "cache.db")
        
    else:
        remote_path = posixpath.join(REMOTE_HDF5_PATH, level, "cache.db")
    
    print("Downloading cache.db for level %s" %level)
    transfer_file_from_hera(local_path, remote_path)
    


def transfer_obs_type_db():
    """copy obs type db from remote path to local computer"""

    local_path = os.path.join(SHARED_DIR_PATH, "db", "obs_type.db")
    
    print("Downloading obs_type.db")
    transfer_file_from_hera(local_path, REMOTE_OBS_DB_FILE_PATH)
