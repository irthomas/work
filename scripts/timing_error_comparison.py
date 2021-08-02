# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:27:09 2021

@author: iant
"""

import h5py
import os

ROOT_FOLDER = r"/bira-iasb/projects/work/NOMAD/Science/ian/timing_error_comparison/"


sub_folders = {
    "old_kernel_from_raw",
    "old_kernel_from_inserted",
    "new_kernel_from_raw",
    "new_kernel_from_inserted",
    }

filenames_fields = {
    "20210215_043242_0p1a_SO_1.h5":["DateTime"],
    "20210215_043242_0p1a_UVIS.h5":["DateTime"],
    }


for filename, fields in filenames_fields.items():
    
    
    for sub_folder in sub_folders:
        
        with h5py.File(os.path.join(ROOT_FOLDER, sub_folder, filename)) as f:
            for field in fields:
                d = f[field][...]
                
                print(sub_folder, filename, field)
                print(d[0])