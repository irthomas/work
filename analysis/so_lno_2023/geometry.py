# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:43:12 2023

@author: iant
"""

import numpy as np

from analysis.so_lno_2023.functions.geometry import make_path_lengths



def get_geometry(h5_d, ix, alt_max, alt_delta):
    
    geom_d = {}
    
    geom_d["alt"] = h5_d["alts"][ix]
    geom_d["myear"] = h5_d["myear"]
    geom_d["ls"] = h5_d["ls"]
    geom_d["lat"] = h5_d["lats"][ix]
    geom_d["lon"] = h5_d["lons"][ix]
    geom_d["lst"] = h5_d["lst"][ix]
    
    geom_d["alt_grid"] = np.arange(geom_d["alt"], alt_max, alt_delta)
    path_lengths = make_path_lengths(geom_d["alt_grid"])
    geom_d["path_lengths_km"] = path_lengths
    
    return geom_d
     
    