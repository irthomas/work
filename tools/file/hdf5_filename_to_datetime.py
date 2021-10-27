# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:50:35 2021

@author: iant

CONVERT HDF5 FILENAME INTO A DATETIME
"""

from datetime import datetime

def hdf5_filename_to_datetime(hdf5_filename):
    """convert a hdf5 filename into a datetime object"""

    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]
    hour = hdf5_filename[9:11]
    minute = hdf5_filename[11:13]
    second = hdf5_filename[13:15]

    return datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
    
