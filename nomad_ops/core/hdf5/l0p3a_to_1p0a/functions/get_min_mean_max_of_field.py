# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:08:20 2020

@author: iant
"""
import numpy as np

def get_min_mean_max_of_field(hdf5_file, field_name):
    field = hdf5_file[field_name][...]
    
    v_min = np.min(field)
    v_mean = np.mean(field)
    v_max = np.max(field)
    
    return v_min, v_mean, v_max
