# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:58:51 2023

@author: iant

COMPARE THE CONTENTS OF TWO STANDALONE H5 FILES
CHECK ALL DATASETS EXIST
CHECK SIZE
CHECK 
"""


import numpy as np
import h5py

from tools.general.cprint import cprint


h51_filepath = r"C:/Users/iant/Dropbox/NOMAD/Python/tmp/20221125_082524_0p2a_LNO_1_D_189.h5"
h52_filepath = r"C:/Users/iant/Dropbox/NOMAD/Python/tmp/20221125_082524_0p2a_LNO_1_D_189_old.h5"


h51_f = h5py.File(h51_filepath, "r")
h52_f = h5py.File(h52_filepath, "r")


keys1a = h51_f.keys()
keys2a = h51_f.keys()


def compare_dset(dset1, dset2):
    dset1_shape = dset1[...].shape
    dset2_shape = dset2[...].shape
    
    if len(dset1_shape) == len(dset2_shape):
        if np.all(dset1_shape == dset2_shape):
            
    
            all_same = np.all(dset1[...] == dset2[...])
            
            if not all_same:
                any_same = np.any(dset1[...] == dset2[...])
                
                if any_same:
                    n_same = np.sum(dset1[...] == dset2[...])
                    n_not_same = np.sum(dset1[...] != dset2[...])
                    
                    mean_delta = np.sum(dset1[...] - dset2[...])
                    percent_change = mean_delta / np.sum(dset1[...]) * 100
                    
                    if percent_change < 1e-5:
                        text = "All are the same"
                        c = "g"
                    else:
                        text = "Some are the same (%i of %i, %0.4g %% difference)" %(n_same, n_same+n_not_same, percent_change)
                        c = "y"
                else:
                    mean_delta = np.sum(dset1[...] - dset2[...])
                    percent_change = np.abs(mean_delta / np.sum(dset1[...]) * 100)
                    
                    if percent_change > 0.1:
                        text = "None are the same (%0.4g %% difference)" %(percent_change)
                        c = "r"
                    else:
                        text = "None are the same but similar (%0.4g %% difference)" %(percent_change)
                        c = "y"
            else:
                text = "All are the same"
                c = "g"
        else:
            text = "Same dimensions, different shape"
            c = "r"
    else:
        text = "Different dimensions"
        c = "r"
        
    return text, c
    


for keya in keys1a:
    dset1 = h51_f[keya]
    dset2 = h52_f[keya]
    
    if type(dset1) == h5py._hl.dataset.Dataset:
        text, c = compare_dset(dset1, dset2)
        if c != "g":
            cprint((keya, text), c)
        
            
    else: #dataset is a group
        keys1b = h51_f[keya].keys()

        for keyb in keys1b:
            dset1 = h51_f[keya][keyb]
            dset2 = h52_f[keya][keyb]
                    
            if type(dset1) == h5py._hl.dataset.Dataset:
                text, c = compare_dset(dset1, dset2)
                if c != "g":
                    cprint((keya, keyb, text), c)


            else: #dataset is a group
                keys1c = h51_f[keya][keyb].keys()
        
                for keyc in keys1c:
                    dset1 = h51_f[keya][keyb][keyc]
                    dset2 = h52_f[keya][keyb][keyc]
                            
                    if type(dset1) == h5py._hl.dataset.Dataset:
                        text, c = compare_dset(dset1, dset2)
                        if c != "g":
                            cprint((keya, keyb, keyc, text), c)
