#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:26:20 2021

@author: iant

VERSION 2: COMPARE FIRST ELEMENTS AND PLOT TRANSMITTANCE VARIATIONS IN L1 FILES
"""

import h5py
import os
import matplotlib.pyplot as plt
import re
import numpy as np

ROOT_FOLDER = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"


sub_folders = {
    "archive/hdf5/timing_error_210619":"Old (May 21)",
    "archive/hdf5/timing_error_210729":"After first reprocessing (June 21)",
    "hdf5":"New version (Aug 21)",
    # "new_kernel_from_inserted":"Conversion from incoming data using new kernel",
    # "new_kernel_from_raw":"Conversion from raw files using new kernel",
    }

filenames_fields = {
    "20210214_025902_0p2a_UVIS_I.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210116_011057_0p2a_UVIS_I.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210215_043242_0p2a_SO_1_I_134.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210214_025902_1p0a_UVIS_I.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210116_011057_1p0a_UVIS_I.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    "20210215_043242_1p0a_SO_A_I_134.h5":["Geometry/ObservationDateTime", "Geometry/Point0/TangentAlt"],
    }


for filename, fields in filenames_fields.items():
    print("##########")
    print(filename)
    year, month, day, level_short, channel = re.findall("(....)(..)(..)_......_(....)_(.*)_.*", filename)[0]

    if level_short == "1p0a":
        plt.figure()
        plt.title(filename)
        plt.xlabel("Point0/TangentAlt")
        plt.ylabel("Transmittance")
        

    
    
    for i, field in enumerate(fields):
        print(field)
        for sub_folder, folder_text in sub_folders.items():
            
            file_path = os.path.join(ROOT_FOLDER, sub_folder, "hdf5_level_%s" %level_short, year, month, day, filename)
            if os.path.exists(file_path):
                with h5py.File(file_path, "r") as f:
                    d = f[field][...]
                    
                    # print(folder_text, field)
                    # print(d.flatten()[0].decode())
                    print(d.flatten()[0], folder_text)
                    
                    if level_short == "1p0a":
                        if i == 0:
                            alts = f["Geometry/Point0/TangentAlt"][:, 0]
                            y = np.mean(f["Science/YMean"][...], axis=1)
                            plt.plot(alts, y, label=folder_text, alpha=0.5)
                        
    if level_short == "1p0a":
        plt.legend()
        plt.savefig("%s_tangent_alt.png" %filename)
