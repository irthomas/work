# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:56:14 2021

@author: iant

load in web geo calc output

"""

import numpy as np


def read_webgeocalc(hdf5_filename, func):
    # hdf5_filename = "20201224_011635_0p1a_SO_1"
    
    path = r"C:\Users\iant\Documents\DATA\webgeolcalc\WGC_StateVector_%s.csv" %hdf5_filename
    
    
    # func = "spkpos"
    
    if func == "spkpos":
        regexp = "\"\s\s(\d*.\d*)\",(?:\d*.\d*),(?:\d*.\d*),(-?\d*.\d*),(-?\d*.\d*),(-?\d*.\d*)"
    
    data = np.fromregex(path, regexp, dtype=float)
    
    et = data[:, 0]
    out = data[:, 1:4]

    return et, out