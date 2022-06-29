# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:17:19 2022

@author: iant

LINUX DATA DIRECTORY
"""

import posixpath


def get_h5_dir_linux(hdf5_filename):
    """get full file path from name"""
    
    linux_data_dir = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"

    file_level = "hdf5_level_%s" %hdf5_filename[16:20]
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]

    dir_path = posixpath.join(linux_data_dir, file_level, year, month, day) #choose a file
    
    return dir_path

