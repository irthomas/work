# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:50:18 2022

@author: iant

COMPARE NEW PIPELINE FILES IN THE TEST DIRECTORY TO THOSE IN THE MAIN DATASTORE. CAN COMPARE DIFFERENT LEVELS
"""



import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import platform

if platform.system() == "Windows":
    base_dir = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD"
    base_test_dir = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant"
else:
    base_dir = "/bira-iasb/projects/work/NOMAD/test/iant"
    base_test_dir = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"


# short_level = "0p2b"
# short_level_test = "0p3c"

short_level = "1p0a"
short_level_test = short_level


def get_h5_filenames_from_dir(base_dir, short_level, year, month, day):
    """get list of filenames from one directory, usually in the test directory"""
    day_directory = os.path.join(base_dir, "hdf5", "hdf5_level_xxxx".replace("xxxx",short_level), "%02i" %year, "%02i" %month, "%02i" %day)
    h5s = [os.path.basename(s) for s in sorted(glob.glob(day_directory+r"/**/*.h5", recursive=True))]
    return h5s


filenames = get_h5_filenames_from_dir(base_test_dir, short_level_test, 2022, 12, 7)

# filenames = [
#     "20221007_130514_xxxx_LNO_1_DF_168.h5"
# ]




#old_dataset_name in datastore:new_dataset_name in test directory
datasets = {
    # "Science/X":"Science/X",
    # "Science/Y":"Science/Y",
    # "Science/YNb":"Science/YNb",
    # "Science/XUnitFlag":"Science/XUnitFlag",
    # "Science/XNbBin":"Science/XNbBin",
    "Science/YReflectanceFactor":"Science/YReflectanceFactorOld",
}



for filename in filenames:


    h5_path = os.path.join(base_dir, "hdf5", "hdf5_level_xxxx".replace("xxxx",short_level), 
                           filename[0:4], filename[4:6], filename[6:8], filename.replace("xxxx",short_level))
    h5_test_path = os.path.join(base_test_dir, "hdf5", "hdf5_level_xxxx".replace("xxxx",short_level_test), 
                           filename[0:4], filename[4:6], filename[6:8], filename.replace("xxxx",short_level_test))
    
    
    if not os.path.exists(h5_path):
        print("File %s does not exist; skipping" %h5_path)
        continue
    if not os.path.exists(h5_test_path):
        print("File %s does not exist; skipping" %h5_test_path)
        continue

    
    with h5py.File(h5_path, "r") as h5_f:
        with h5py.File(h5_test_path, "r") as h5_test_f:
    
    
            for dataset_name, dataset_name_new in datasets.items():
                print("Comparing %s to %s" %(dataset_name, dataset_name_new))
                data = h5_f[dataset_name][...]
                data_test = h5_test_f[dataset_name_new][...]
                
                if np.all(data == data_test):
                    print("Datasets match")
                else:
                    print(data.shape)
                    print(data_test.shape)
                    if data.ndim == 1:
                        print(data[0], data_test[0], data[-1], data_test[-1])
                    if data.ndim == 2:
                        print(data[0, 0], data_test[0, 0], data[-1, -1], data_test[-1, -1])
                        mid_ix = int(data.shape[0]/2)
                        plt.figure()
                        plt.title(os.path.basename(h5_path))
                        plt.plot(data[mid_ix, :])
                        plt.plot(data_test[mid_ix, :])
                        
