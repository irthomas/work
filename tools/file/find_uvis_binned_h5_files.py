# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:27:41 2022

@author: iant
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist



# regex = re.compile("20181029_203917_.*_UVIS_E")
# regex = re.compile("20180622_......_.*_UVIS_.*")
regex = re.compile("201812.._......_.*_UVIS_.*")

file_level = "hdf5_level_1p0a"



hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)



for file_index, (h5, h5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
    if np.mod(file_index, 100) == 0:
        print(file_index)

    binning = h5["Channel/HorizontalAndCombinedBinningSize"][0]
    mode = h5["Channel/AcquisitionMode"][0]
    y_shape = h5["Science/Y"][...].shape
    
   
    #if not full or horizontally binned frame, skip file
    # print(mode)
    if mode not in [0,2]:
        continue
    
    if y_shape[1] != 1024:
        print(h5_filename, y_shape, binning)
    

    h5.close()