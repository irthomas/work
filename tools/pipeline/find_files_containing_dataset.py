# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:05:48 2022

@author: iant
"""

import re
import os
import numpy as np

from tools.file.hdf5_functions import make_filelist


regex = re.compile("20220..._......_...._SO_.*")
file_level="hdf5_level_1p0a"


field = "YErrorNorm"
group = "Science"

hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)


with open("output.txt", "w") as f:

    f.write("Filename,%s present?\n" %field)
    for i,(h5_f, h5) in enumerate(zip(hdf5_files, hdf5_filenames)):
        
        if np.mod(i, 100) == 0:
            print("%i/%i" %(i, len(hdf5_filenames)))
        
        h5_basename = os.path.basename(h5)
        
        if field in h5_f[group].keys():
            f.write("%s,Yes\n" %h5_basename)
        else:
            f.write("%s,No\n" %h5_basename)
        

        