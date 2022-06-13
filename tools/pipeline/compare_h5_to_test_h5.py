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


filename = "20180422_082630_0p3c_UVIS_D.h5"

h5_path = os.path.join(base_dir, "hdf5", "hdf5_level_0p2b", "2018", "04", "22", filename)
h5_test_path = os.path.join(base_test_dir, "hdf5", "hdf5_level_0p2b", "2018", "04", "22", filename)

datasets = ["Science/X",
    "Science/Y",
    "Science/YNb",
    "Science/XUnitFlag"]


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
                print(data[0], data_test[0])
                # print(data == data_test)
