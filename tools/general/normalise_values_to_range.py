# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:08:21 2022

@author: iant
"""

import numpy as np


def normalise_values_to_range(values, max_out, min_out):
    """normalise a range of values so that the max value in the range = max_out and the min value in the range = min_out"""
    
    min_value = np.min(values)
    max_value = np.max(values)
    
    return ((values - min_value)/(max_value - min_value)) * (max_out - min_out) + min_out


