# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:27:09 2021

@author: iant

VERSION 1: PRINT OUT FIRST ELEMENT OF SELECTED DATASETS FOR H5 FILES PRODUCED FROM VARIOUS LEVELS OF PIPELINE
"""

import h5py
import os

ROOT_FOLDER = r"/bira-iasb/projects/work/NOMAD/Science/ian/timing_error_comparison/"


sub_folders = {
    "old_kernel_from_inserted":"Conversion from incoming data using old kernel",
    "old_kernel_from_raw":"Conversion from raw files using old kernel",
    "new_kernel_from_inserted":"Conversion from incoming data using new kernel",
    "new_kernel_from_raw":"Conversion from raw files using new kernel",
    }

filenames_fields = {
    "20210215_043242_0p1a_SO_1.h5":["DateTime"],
    "20210215_043242_0p1a_UVIS.h5":["DateTime"],
    "20210215_043242_0p2a_UVIS_I.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210215_051321_0p1a_SO_1.h5":["DateTime"],
    "20210215_051321_0p1a_UVIS.h5":["DateTime"],
    "20210215_051321_0p2a_UVIS_E.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    }


for filename, fields in filenames_fields.items():
    print("##########")
    print(filename)
    
    for field in fields:
        print(field)
        for sub_folder, folder_text in sub_folders.items():
        
           with h5py.File(os.path.join(ROOT_FOLDER, sub_folder, filename)) as f:
                d = f[field][...]
                
                # print(folder_text, field)
                # print(d.flatten()[0].decode())
                print(d.flatten()[0])