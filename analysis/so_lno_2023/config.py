# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:50:47 2023

@author: iant

CONFIG
"""

import os
import platform


CHANNEL = "lno"

if platform.system() == "Windows":
    
    ROOT_DIR = r"C:\Users\iant\Dropbox\NOMAD\Python\analysis\so_lno_2023"
    INPUT_DIR = os.path.join(ROOT_DIR, "inputs")
    ROOT_DATA_DIR = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a"
    
    