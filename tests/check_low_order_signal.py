# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:10:15 2023

@author: iant


RATIO OF FULLSCAN ORDERS
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist


# regex = re.compile("20181029_203917_.*_UVIS_E")
# regex = re.compile("20180622_......_.*_UVIS_E")
regex = re.compile("20150..._......_.*_SO_.")

file_level = "hdf5_level_0p1a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)


min_aotf = 20000.
max_aotf = 20000.
for h5f, h5 in zip(hdf5_files, hdf5_filenames):
    aotfs = h5f["Channel/AOTFFrequency"][...]
    
    
    print(h5, max(aotfs), min(aotfs[aotfs>1.0]))
    

    if min(aotfs) < min_aotf and min(aotfs)>1.0:
        min_aotf = min(aotfs)

    if max(aotfs) > max_aotf:
        max_aotf = max(aotfs)
    
    
    if min(aotfs[aotfs>1.0]) < 15000.:
        ys = h5f["Science/Y"][...]
        
        y_mean = np.mean(ys[:, :, 160:240], axis=2)
        
        if np.max(y_mean) > 200.0:
            plt.figure()
            plt.title(h5)
            im = plt.imshow(y_mean.T, aspect="auto")
            plt.colorbar(im)
        
        # stop()
        