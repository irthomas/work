# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:19:56 2023

@author: iant

NUMPY ARRAY MANIPULATIONS

"""

import numpy as np


#1d N rows to 2d repeated N rows x n columns each row the same
def arr1d_2d_same_rows(arr, nreps):
    
    return np.repeat(arr, nreps).reshape((-1, nreps))
