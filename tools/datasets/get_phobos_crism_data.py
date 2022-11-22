# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:45:53 2022

@author: iant
"""

import os
import numpy as np
from tools.file.paths import paths


def get_phobos_crism_data():
    """get radiance factor from CRISM Fraeman et al. digitisation
    https://doi.org/10.1016/j.icarus.2013.11.021"""
    
    crism_data = np.loadtxt(os.path.join(paths["REFERENCE_DIRECTORY"], "phobos_crism_fraeman14.csv"), skiprows=1, delimiter=",", unpack=True)
    
    return {"x":crism_data[0, :], "phobos_red":crism_data[1, :], "phobos_blue":crism_data[2, :]}