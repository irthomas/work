# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:22:25 2020

@author: iant
"""

import os
import re

from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import THUMBNAILS_DESTINATION


def prepare_nadir_fig_tree(hdf5_filename, diffraction_order):
    
    fig_name = "%s_ref_fac.png" %hdf5_filename
    fig_path = os.path.join(THUMBNAILS_DESTINATION, "lno_1p0a_reflectance_factor_order_%i" %diffraction_order)
    
    m = re.match("(\d{4})(\d{2})(\d{2}).*", fig_name)
    year = m.group(1)
    month = m.group(2)
    path_fig = os.path.join(fig_path, year, month) #note: not split by day
    if not os.path.isdir(path_fig):
            os.makedirs(path_fig, exist_ok=True)
    return os.path.join(path_fig, fig_name)

