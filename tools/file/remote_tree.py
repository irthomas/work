# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:55:25 2022

@author: iant
"""

import os
import paramiko
import posixpath

SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"
REMOTE_HDF5_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/"
REMOTE_EDDS_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db/edds/"
REMOTE_OBS_DB_FILE_PATH = "/bira-iasb/projects/NOMAD/Data/pfm_auxiliary_files/observation_type/obs_type.db"

EXTENSIONS = {"spacewire":"EXM", "tc1553":"dat", "tm1553_84":"zip", "tm1553_244":"zip", "tm1553_372":"zip"}


def get_tree_filenames(user, password, level, channel="", host="hera.oma.be"):
    """Get a list of all h5 filenames present in HDF5 level subdirectories on a remote linux server. Search month by month""" 
    
    print("Connecting to %s" %host)
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect(host, port=22, username=user, password=password)
    
    filenames = []
    
    for year in range(2018, 2023):
        for month in range(1, 13):
            print("%02i/%04i" %(month, year))
    

            if level in EXTENSIONS.keys():
                remote_path = posixpath.join(REMOTE_EDDS_PATH, "%s/%04i/%02i" %(level, year, month))
                ext = EXTENSIONS[level]
                
            else:
                remote_path = posixpath.join(REMOTE_HDF5_PATH, "%s/%04i/%02i" %(level, year, month))
                ext = "h5"
            
            
            if channel == "":
                find_fmt = 'find %s/*/*.%s'
                find_cmd = find_fmt % (remote_path, ext)
            else:
                find_fmt = 'find %s/*/*%s*.%s'
                find_cmd = find_fmt % (remote_path, channel.upper(), ext)
                
                
            
            stdin, stdout, stderr = p.exec_command(find_cmd)
            
            output = stdout.read().decode().split("\n")
            
            for s in output:
                if s != "":
                    filenames.append(os.path.basename(s).replace(".h5",""))
        
    
    p.close()

    return filenames


