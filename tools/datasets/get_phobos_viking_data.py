# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:29:38 2022

@author: iant
"""


import os
import numpy as np

from tools.file.paths import paths


def get_phobos_viking_data():
    
    path = os.path.join(paths["REFERENCE_DIRECTORY"], "Phobos_Viking_Mosaic_LowRes.txt")
    
    return np.loadtxt(path)