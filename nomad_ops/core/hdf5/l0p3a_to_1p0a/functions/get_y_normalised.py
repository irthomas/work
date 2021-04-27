# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:32:50 2020

@author: iant
"""

def get_y_normalised(hdf5_file, y_dataset_path):
    """get y dataset from file and normalise to counts per pixel per second"""
    import numpy as np
    
    y = hdf5_file[y_dataset_path][:, :]
    y[np.isnan(y)] = 0.0 #replace nans

    integration_time_raw = hdf5_file["Channel/IntegrationTime"][0]
    number_of_accumulations_raw = hdf5_file["Channel/NumberOfAccumulations"][0]
    integration_time = np.float(integration_time_raw) / 1.0e3 #microseconds to seconds
    number_of_accumulations = np.float(number_of_accumulations_raw)/2.0 #assume LNO nadir background subtraction is on
    bins_raw = hdf5_file["Science/Bins"][0, :]

    n_pixels = np.float(bins_raw[1] - bins_raw[0]) + 1.0
    n_seconds = integration_time * number_of_accumulations
    
    y = y / n_pixels / n_seconds
    return y
