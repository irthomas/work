# -*- coding: utf-8 -*-
"""
Created on Thu May 14 08:01:39 2020

@author: iant
"""

def output_filename(hdf5_basename, error):
    """change filename to pass (P) or fail (F) based on if error"""

    if error:
        hdf5_basename_split = hdf5_basename.split("_")
        hdf5_basename_split[5] = "DF"
    else:
        hdf5_basename_split = hdf5_basename.split("_")
        hdf5_basename_split[5] = "DP"
    hdf5_basename_new = "_".join(hdf5_basename_split)

    return hdf5_basename_new
