# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:50:18 2022

@author: iant

COMPARE NEW PIPELINE FILES
"""



import h5py
import os
import numpy as np


base_dir = "/bira-iasb/projects/work/NOMAD/test/iant"
base_test_dir = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"

# base_dir = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD"
# base_test_dir = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant"

short_level = "0p2b"
short_level_test = "0p3c"

# filename = "20181201_234801_xxxx_UVIS_I.h5"
filename = "20181201_070557_xxxx_UVIS_D.h5"



h5_path = os.path.join(base_dir, "hdf5", "hdf5_level_xxxx".replace("xxxx",short_level), 
                       filename[0:4], filename[4:6], filename[6:8], filename.replace("xxxx",short_level))
h5_test_path = os.path.join(base_test_dir, "hdf5", "hdf5_level_xxxx".replace("xxxx",short_level_test), 
                       filename[0:4], filename[4:6], filename[6:8], filename.replace("xxxx",short_level_test))

datasets = ["Science/X",
    "Science/Y",
    "Science/YNb",
    "Science/XUnitFlag",
    "Science/XNbBin"]


with h5py.File(h5_path, "r") as h5_f:
    with h5py.File(h5_test_path, "r") as h5_test_f:


        for dataset in datasets:
            print(dataset)
            data = h5_f[dataset][...]
            data_test = h5_test_f[dataset][...]
            
            if np.all(data == data_test):
                print("Datasets match")
            else:
                print(data.shape)
                print(data_test.shape)
                print(data[0], data_test[0], data[-1], data_test[-1])
                # print(data == data_test)
