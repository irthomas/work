# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:55:04 2022

@author: iant
"""


import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import spiceypy as sp
from datetime import datetime

from tools.file.hdf5_functions import make_filelist

from tools.spice.load_spice_kernels import load_spice_kernels





# regex = re.compile(".*_UVIS_P")
regex = re.compile("20210927_224950_.*_UVIS_P")
# regex = re.compile("20210921_132947_0p2a_UVIS_P")



h5_fs, h5s, _= make_filelist(regex, "hdf5_level_0p3b")

h5 = h5s[0]
h5_f = h5_fs[0]

y = h5_f["Science/Y"][...]    


plt.figure()
plt.plot(np.mean(y[15:30, :, 50:150], axis=(0,2)))

plt.yscale("log")
plt.grid()     
    
plt.figure()
plt.plot(np.mean(y[15:30, :, 50:150], axis=(0,2)))

# plt.yscale("log")
plt.grid()     
    
