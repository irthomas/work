# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:10:41 2022

@author: iant
"""

ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0

import numpy as np


def make_offset_vector(arcmin_offset):
    """make FOV vector given an offset in arcminutes e.g. 
    arcmin_offset = [1.0, 1.0] #in x and y directions, fov vector in z direction (0, 0, 1)"""

    radian_offset = np.asfarray(arcmin_offset) / ARCMINS_TO_RADIANS

    if arcmin_offset[0] == arcmin_offset[1] == 0:
        offset_vector = np.asfarray([0., 0., 1.])
    else:
        offset_vector = np.asfarray(
            [radian_offset[0], radian_offset[1], (np.sqrt(1.0 - radian_offset[0]**2.0 - radian_offset[1]**2.0))]
        )
    
    return offset_vector