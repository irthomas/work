# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:39:12 2023

@author: iant
"""

import numpy as np


def make_path_lengths(alt_grid):
    radius = 3396. #km
    pl = []
    for alt in alt_grid:
        pl.append(2.0 * (np.sqrt((radius + alt)**2.0 - (radius + alt_grid[0])**2.0)) - np.sum(pl))
    pl.pop(0)
    pl.append(pl[-1])

    pl = np.asfarray(pl)
    # pl *= 1e5 #cm?
    return pl

