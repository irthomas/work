# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:39:26 2020

@author: iant

PLOTTING COLOUR FUNCTIONS

"""

def get_colours(n_colours, cmap="Spectral", colours=[]):
    """get a list of n_colours for the given colour map"""
    
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import numpy as np
    
    if len(colours) == 0:
        cmap = plt.get_cmap(cmap)
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colours)
    colours_ = [cmap(i) for i in np.arange(n_colours)/n_colours]
    return colours_
    
