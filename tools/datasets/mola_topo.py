# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:19:35 2024

@author: iant

MOLA ALBEDO MAPS
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def get_MOLA_topo_map(chosen_lat, chosen_lon):

# chosen_lat = 18.0
# chosen_lon = 226.0


    lat_name = 44
    lat_hem = "n"
    lon_name = 180
    
    lat_start = float(lat_name)
    lat_end = lat_start - 44.0
    
    lon_start = float(lon_name)
    if lon_start >= 180:
        lon_start -= 360.0
    
    lon_end = lon_start + 90.0
    
    dlat = 1.0 / 128.0
    dlon = 1.0 / 128.0
    
    
    
    mola_filename = "megt%02i%s%03ihb.img" %(lat_name, lat_hem, lon_name)
    
    
    path = os.path.join(r"C:\Users\iant\Downloads", mola_filename)
    
    with open(path, "rb") as f:
        a = np.fromfile(f, dtype='>i2')
        
    b = a.reshape((5632, 11520))
    
    extent = [lon_start, lon_end, lat_end, lat_start]
    
    return b, extent

