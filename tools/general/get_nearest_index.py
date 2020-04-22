# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:00:06 2020

@author: iant

FIND INDICES


"""

def get_nearest_index(value, array):
    """get index of nearest value in a 1d array or list"""
    
    import numpy as np
    
    if type(array) == list:
        array = np.asarray(array)
    
    idx = np.abs(array - value).argmin()
    return idx