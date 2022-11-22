# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:28:19 2022

@author: iant
"""

import numpy as np

def make_perimeter_bounds(min_x, min_y, max_x, max_y, delta=1.0):
    """https://stackoverflow.com/questions/49398567/python-generate-all-points-in-a-rectangle"""
    x, y = min_x, min_y
    for dx, dy in (delta, 0), (0, delta), (-delta, 0), (0, -delta):
        while x in np.arange(min_x, max_x + delta, delta) and y in np.arange(min_y, max_y + delta, delta):
            yield (x, y)
            x += dx
            y += dy
        x -= dx
        y -= dy
        
       
        
# import matplotlib.pyplot as plt
# min_x = -2
# min_y = -2
# max_x = 2
# max_y = 2
# delta = 0.25

# bounds = list(make_perimeter_bounds(min_x, min_y, max_x, max_y, delta=delta))

# for bound in bounds:
#     plt.scatter(bound[0], bound[1])