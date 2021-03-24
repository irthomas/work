# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:41:29 2020

@author: iant
"""

import numpy as np

def length(obj):
    
    if type(obj) == list:
        return len(obj)
    
    elif type(obj) == np.ndarray:
        shape = obj.shape
        
        if len(shape) == 0:
            return 1
        elif len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape
