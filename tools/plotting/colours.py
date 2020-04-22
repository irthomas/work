# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:39:26 2020

@author: iant

PLOTTING COLOUR FUNCTIONS

"""

def get_colours(n_colours, cmap="Spectral"):
    """get a list of n_colours for the given colour map"""
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    cmap = plt.get_cmap(cmap)
    colours = [cmap(i) for i in np.arange(n_colours)/n_colours]
    return colours

