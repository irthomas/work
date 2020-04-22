# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:47:27 2020

@author: iant
"""

def running_mean(detector_data, n_spectra_to_mean):
    """make a running mean of n data points. 
    detector data output has same length as input"""
    import numpy as np
    
    plus_minus = int((n_spectra_to_mean - 1) / 2)
    
    nSpectra = detector_data.shape[0]
    runningIndices = np.asarray([np.arange(runningIndexCentre-plus_minus, runningIndexCentre+(plus_minus+1), 1) for runningIndexCentre in np.arange(nSpectra)])
    runningIndices[runningIndices<0] = 0
    runningIndices[runningIndices>nSpectra-1] = nSpectra-1

    running_mean_data = np.zeros_like(detector_data)
    for rowIndex,indices in enumerate(runningIndices):
        running_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return running_mean_data
