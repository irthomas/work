# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:17:30 2020

@author: iant
"""

def smooth_hr(x, window_len=399, window='hanning'):
    """smooth a high res spectrum to a nomad-like resolution
    for SO, window_len ~ 399
    for LNO, window_len ~ 599"""
    import numpy as np
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is not 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y
