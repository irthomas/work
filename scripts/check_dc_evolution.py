# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:19:05 2023

@author: iant

CHECK NOMAD SO DC LEVEL
"""


import re
import numpy as np
import matplotlib.pyplot as plt
# import os




from tools.file.hdf5_functions import make_filelist, open_hdf5_file


# regex = re.compile("20200103_001452_0p3k_SO_A_E_134")
regex = re.compile(".*SO.*")



file_level = "hdf5_level_0p3k"



def get_so_data(regex, alt_cutoff, signal_cutoff):
    
    h5fs, h5s, _ = make_filelist(regex, file_level, path=r"C:\Users\iant\Documents\DATA\hdf5")
    
    
    d = {"alts":[], "h5s":[], "ys":[]}
    for file_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):
        
        alts = h5f["Geometry/Point0/TangentAltAreoid"][:, 0]
        
        
        good_ixs = np.where(alts < alt_cutoff)[0]
        alts = alts[good_ixs]
        
        y = h5f["Science/Y"][good_ixs, :]
        
        good_ixs2 = np.where(np.nanmean(y, axis=1) < signal_cutoff)[0]
        
        y = y[good_ixs2, :]
        alts = alts[good_ixs2]
        
        print(h5, np.nanmean(np.nanstd(y, axis=1)))
        
        d["alts"].extend(list(alts))
        d["h5s"].extend([h5 for i in range(len(alts))])
        d["ys"].extend(list(y))
    
    for key in ["alts", "ys"]:
        d[key] = np.asarray(d[key])
    
    return d

so_dict = get_so_data(regex, -5.0, 10000.0)

ys_mean = np.mean(so_dict["ys"], axis=1)
ys_std = np.std(so_dict["ys"], axis=1)

plt.plot(ys_mean)
plt.plot(ys_std)

