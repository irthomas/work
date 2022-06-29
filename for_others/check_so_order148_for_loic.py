# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:31:27 2022

@author: iant

CHECK SPECTRA IN ORDER 148 FILES FOR LOIC
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from tools.file.paths import paths
from tools.plotting.colours import get_colours

from tools.file.hdf5_functions import open_hdf5_file


filenames = [
    "20210217_192815_1p0a_SO_A_I_148",
    "20210325_233443_1p0a_SO_A_E_148",
    "20210417_052833_1p0a_SO_A_I_148",
    "20210522_220412_1p0a_SO_A_E_148",
    "20210525_185205_1p0a_SO_A_E_148",
    "20210528_193541_1p0a_SO_A_E_148",
    "20210610_152918_1p0a_SO_A_I_148",
    "20210616_051250_1p0a_SO_A_I_148",
    "20210625_093215_1p0a_SO_A_I_148",
    "20210703_085803_1p0a_SO_A_E_148",
    "20210704_055600_1p0a_SO_A_I_148",
    "20210710_211009_1p0a_SO_A_I_148",
    "20210712_023915_1p0a_SO_A_I_148",
    "20210803_091602_1p0a_SO_A_I_148",
    "20210912_101013_1p0a_SO_A_E_148",
    "20210926_042155_1p0a_SO_A_E_148",
    "20211129_015157_1p0a_SO_A_E_148",     
    ]


for filename in filenames[0:5]:
    h5 = open_hdf5_file(filename)
    
    
    bins = h5["Science/Bins"][:, 0]
    x = h5["Science/X"][0, :]
    y = h5["Science/Y"][...]
    alts = h5["Geometry/Point0/TangentAltAreoid"][:, 0]
    

    
    indices1 = np.where((alts > 65) & (alts < 190))[0]
    
    # unique_bins = list(set(bins))
    unique_bins = [124]
    
    for unique_bin in unique_bins:
        indices2 = np.where(bins == unique_bin)[0]
    
        indices = np.intersect1d(indices1, indices2)
        
        colours = get_colours(len(indices))
        
        
        plt.figure(figsize=(15,12))
        plt.title("%s: bin %i" %(filename, unique_bin))
        for i, index in enumerate(indices):
            plt.plot(y[index, :], color=colours[i], label=alts[index])
        
        plt.legend()