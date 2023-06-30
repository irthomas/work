# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:43:27 2023

@author: iant

DECONVOLVE HAPI TO LOWER RESOLUTION
"""


import numpy as np

def reduce_resolution(hapi_nus, hapi_trans, nu_step):
    
    hapi_nus_red = np.arange(hapi_nus[0], hapi_nus[-1], nu_step)
    
    ixs = np.digitize(hapi_nus, hapi_nus_red)
    
    hapi_trans_red = np.asarray([np.min(hapi_trans[ixs == i]) for i in range(1,len(hapi_nus_red)+1)])

    
    return hapi_nus_red, hapi_trans_red


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(hapi_nus, hapi_trans)

# hapi_nus_red, hapi_trans_red = reduce_resolution(hapi_nus, hapi_trans, 0.01)

# plt.figure()
# plt.plot(hapi_nus_red, hapi_trans_red)
