# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:30:49 2020

@author: iant
"""
import numpy as np


def running_mean(detector_data, n_spectra_to_mean):
    """make a running mean of n data points. detector data output has same length as input"""
    nSpectra = detector_data.shape[0]
    running_mean_data = np.zeros_like(detector_data)#[0:(-1*(n_spectra_to_mean-1)), :])
    
    runningIndicesCentre = [np.asarray(range(startingIndex, startingIndex+n_spectra_to_mean)) for startingIndex in range(0, (nSpectra-n_spectra_to_mean)+1)]
    runningIndicesStart = [np.asarray([0] * index + list(range(0, 10 - index))) for index in range(5, 0, -1)]
    runningIndicesEnd = [np.asarray(list(range(nSpectra - n_spectra_to_mean + index, nSpectra)) + [nSpectra - 1] * index) for index in range(1,5)]
    runningIndices = runningIndicesStart + runningIndicesCentre + runningIndicesEnd
    for rowIndex,indices in enumerate(runningIndices):
        running_mean_data[rowIndex,:]=np.mean(detector_data[indices,:], axis=0)
    return running_mean_data

