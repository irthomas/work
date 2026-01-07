# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:19:39 2025

@author: iant

Make custom 3-colour colourmap with setpoints

Adapted from https://stackoverflow.com/questions/61235327/how-to-create-a-custom-colormap-with-three-colors
"""

import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def red_widegrey_blue():
    N = 256  # note, final colourmap maybe +-1 elements due to rounding

    tr = [int(N*0.45), int(N*0.1), int(N*0.45)]

    r = np.concatenate((np.linspace(0, 0.5, tr[0]), np.linspace(0.5, 0.5, tr[1]), np.linspace(0.5, 1, tr[2])), axis=None)
    g = np.concatenate((np.linspace(0, 0.5, tr[0]), np.linspace(0.5, 0.5, tr[1]), np.linspace(0.5, 0, tr[2])), axis=None)
    b = np.concatenate((np.linspace(1, 0.5, tr[0]), np.linspace(0.5, 0.5, tr[1]), np.linspace(0.5, 0, tr[2])), axis=None)

    new_cmap = ListedColormap(np.vstack((r, g, b)).T)

    # test
    # plt.imshow(np.outer(np.arange(0, 1, 0.01), np.ones(100)), aspect='auto', cmap=new_cmap)

    return new_cmap
