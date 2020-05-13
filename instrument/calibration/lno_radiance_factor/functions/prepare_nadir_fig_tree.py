# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:22:25 2020

@author: iant
"""

import os
import re

from instrument.calibration.lno_radiance_factor.config import ROOT_STORAGE_PATH


def prepare_nadir_fig_tree(figName):
    
    channel=figName.split('_')[3]
    
    # Move to config
    PATH_TRANS_LINREG_FIG = os.path.join(ROOT_STORAGE_PATH, "thumbnails_1p0a_radfac", channel)  
    
    m = re.match("(\d{4})(\d{2})(\d{2}).*", figName)
    year = m.group(1)
    month = m.group(2)
#    path_fig = os.path.join(PATH_TRANS_LINREG_FIG)
    path_fig = os.path.join(PATH_TRANS_LINREG_FIG, year, month)
    if not os.path.isdir(path_fig):
            os.makedirs(path_fig, exist_ok=True)
    return os.path.join(path_fig, figName)

