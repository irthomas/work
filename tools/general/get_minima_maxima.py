# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:57:00 2020

@author: iant
"""
import numpy as np


def get_local_minima(values):
    """find indices of all local minima"""
    
    indices = (np.diff(np.sign(np.diff(values))) > 1).nonzero()[0] + 1
    return indices

def get_local_maxima(values):
    """find indices of all local minima"""
    
    indices = (np.diff(np.sign(np.diff(values))) < -1).nonzero()[0] + 1
    return indices




def get_local_minima_or_equals(values):
    """find indices of all local minima, including those with equal adjacent values """
    
    indices = (np.diff(np.sign(np.diff(values))) > 0).nonzero()[0] + 1
    return indices


def get_local_maxima_or_equals(values):
    """find indices of all local maxima, including those with equal adjacent values """
    
    indices = (np.diff(np.sign(np.diff(values))) < 0).nonzero()[0] + 1
    return indices
