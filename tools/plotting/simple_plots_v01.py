# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:48:15 2017

@author: ithom

SIMPLE PLOTTING ROUTINES
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


from hdf5_functions_v02 import get_hdf5_attributes,get_dataset_contents
from pipeline_config_v03 import DATA_ROOT_DIRECTORY,figx,figy


filename = "20161123_223450_SO"
#filename = "20161123_225550_LNO"
file_level = "hdf5_level_0p1c"

year = filename[0:4]
month = filename[4:6]
day = filename[6:8]
filename=os.path.normcase(DATA_ROOT_DIRECTORY+os.sep+file_level+os.sep+year+os.sep+month+os.sep+day+os.sep+filename+".h5") #choose a file
hdf5_file = h5py.File(filename, "r") #open file

detector_data_all = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
binning = get_dataset_contents(hdf5_file,"Binning")[0]
window_top = get_dataset_contents(hdf5_file,"WindowTop")[0]

hdf5_file.close()

frame_indices = range(0,500,10)
column_index = 160
x = np.arange(24) * (binning[0]+1) + window_top[0]

plt.figure(figsize = (figx,figy))

for frame_index in frame_indices:

    data = detector_data_all[frame_index,:,column_index]
    
    plt.scatter(x,data)