# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:05:28 2019

@author: iant

TEST SIGNAL LEVELS
"""

#import h5py
import os
import numpy as np

#DATA_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
DATA_DIRECTORY = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/")
#filename_base = ["20180328_150554_", "_LNO_1_D_190"]
#file_levels_base = ["0p1d", "0p1e", "0p2a", "0p3a"]
#
#y_all = []
#for file_level_base in file_levels_base:
#
#    filename_full = filename_base[0] + file_level_base + filename_base[1]
#    fileLevel = "hdf5_level_"+file_level_base
#    
#    
#    year = filename_full[0:4] #get the date from the filename to find the file
#    month = filename_full[4:6]
#    day = filename_full[6:8]
#    
#    with h5py.File(os.path.join(DATA_DIRECTORY, fileLevel, year, month, day, filename_full + ".h5"), "r") as f:
#        y_all.append(f["Science/Y"][...])
#        
#for y in y_all:
#    print(np.sum(y))
#    
#    

from hdf5_functions_v03 import makeFileList
#obspaths = ["*20180601*UVIS*_E"]
fileLevel = "hdf5_level_0p2a"
hdf5Files, hdf5Filenames, _ = makeFileList(["*201806*UVIS*_I"], fileLevel)
for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames): 
    print(hdf5Filename + ": "+str(hdf5File["Channel/AcquisitionMode"][0]))
    
hdf5Files, hdf5Filenames, _ = makeFileList(["*201806*UVIS*_E"], fileLevel)
for hdf5File, hdf5Filename in zip(hdf5Files, hdf5Filenames): 
    print(hdf5Filename + ": "+str(hdf5File["Channel/AcquisitionMode"][0]))
    
    
    