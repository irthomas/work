# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:03:54 2024

@author: iant
"""

import re

from tools.file.hdf5_functions import make_filelist
from tools.general.cprint import cprint


regex = re.compile(".*_SO_._S")
# regex = re.compile("20230327_122435_.*_LNO_1_D._133")



file_level = "hdf5_level_0p1d"
    
h5_fs, h5s, _ = make_filelist(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")


for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):
    
    orders = h5_f["Channel/DiffractionOrder"][...]

    unique_orders = sorted(list(set(orders)))
    
    n_orders = len(unique_orders)
    
    text = "%s: %i" %(h5, n_orders)
    if n_orders < 28:
        cprint(text, "y")
    # else:
    #     print(text)
    
    
    