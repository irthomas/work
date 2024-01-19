# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:19:56 2023

@author: iant

NUMPY ARRAY MANIPULATIONS

"""

import numpy as np


# 1d N rows to 2d repeated N rows x nreps columns each row the same
# [0, 0, 0, 0]
# [1, 1, 1, 1]
def arr1d_2d_same_rows(arr, nreps):
    return np.repeat(arr, nreps).reshape((-1, nreps))


#1d N columns to 2d repeated N columns x nreps rows each column the same
# [0, 1, 2, 3]
# [0, 1, 2, 3]
def arr1d_2d_same_cols(arr, nreps):
    return np.tile(arr, (nreps, 1))

