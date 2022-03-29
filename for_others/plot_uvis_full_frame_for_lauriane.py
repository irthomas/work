# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:36:59 2022

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

#this is the index of the observation in the level 1.0a file (where first frame = 0)
level1_index = 20

#the index of a given frame in full frame files is 2 greater than index in level 1.0a files
i = level1_index + 2

#open file
with h5py.File(r"D:\DATA\hdf5\hdf5_level_0p3b\2021\06\03\20210603_010708_0p3b_UVIS_O.h5", "r") as hdf5_file:

    #get data from file
    y = hdf5_file["Science/Y"][...]
    y_mask = np.asfarray(hdf5_file["Science/YMask"][...])
            
    #choose mask: >1.1 = bad pixels only; >0.9 for hot and bad pixels
    # y_mask_tf = np.where(y_mask > 1.1) #bad pixels only
    y_mask_tf = np.where(y_mask > 0.9) #hot and bad pixels
            
    #make detector array with masked pixels set to NaN
    y_masked = np.copy(y)
    y_masked[y_mask_tf] = np.nan
            
            
    #plot unmasked detector frame
    plt.figure(figsize=(12, 4), constrained_layout=True)
    
    #constrain colour scheme to see pixel-to-pixel variations
    mean = np.nanmean(y[i, :, :])
    std = np.nanstd(y[i, :, :])
    plt.imshow(y[i, :, :], vmin=(mean - std * 0.5), vmax=(mean + std * 0.5))
    
    #plot masked detector frame
    plt.figure(figsize=(12, 4), constrained_layout=True)

    #constrain colour scheme to see pixel-to-pixel variations
    mean = np.nanmean(y_masked[i, :, :])
    std = np.nanstd(y_masked[i, :, :])
    plt.imshow(y_masked[i, :, :], vmin=(mean - std * 0.5), vmax=(mean + std * 0.5))
