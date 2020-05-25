# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:12:56 2020

@author: iant
"""

def get_nearest_datetime(datetime_list, search_datetime):
    """from a list of datetimes, get the closest index to the search_datetime"""
    import numpy as np
    
    time_diff = np.abs([date - search_datetime for date in datetime_list])
    return time_diff.argmin(0)
