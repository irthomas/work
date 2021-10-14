# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:57:37 2021

@author: iant

PLOT UVIS OFF-NADIR DATA
NOTE THAT LIMB SCANS 20161122_134403_0p2a_UVIS_C AND 20161126_200627_0p2a_UVIS_C DON'T WORK, AS VERTICALLY BINNED

OFF-NADIR DATA NOT USABLE - SLEW TOO FAST

"""

# import os
import numpy as np
# from datetime import datetime, timedelta
# import re

import matplotlib.pyplot as plt
from tools.file.hdf5_functions import open_hdf5_file



file_level = "hdf5_level_0p2a"
# file_level = "hdf5_level_1p0a"
# (20180504_150953|20180529_150145|20180625_191417|20180709_000545).*

hdf5_filenames = [
    "20180504_150953_0p2a_UVIS_D",
    "20180529_150145_0p2a_UVIS_D",
    "20180625_191417_0p2a_UVIS_D",
    "20180709_000545_0p2a_UVIS_D",
    ]

for hdf5_filename in hdf5_filenames:    
        
        
    #get file
    hdf5_file = open_hdf5_file(hdf5_filename)
    
    y_all = hdf5_file["Science/Y"][...]
    y_mean_px = np.mean(y_all[:, :, 500:1000], axis=2).T #mean of spectral pixels
    mask = np.all(np.isnan(y_mean_px) | np.equal(y_mean_px, 0), axis=1) #make nan row mask
    y_mean_px = y_mean_px[~mask].T #remove all rows with nans
    
    # y_mean_px_sub = y_mean_px - np.tile(y_mean_px[:, 0], (106,1)).T
   
    plt.figure()
    plt.imshow(y_mean_px.T)
    
