#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:18:50 2022

@author: iant
"""


from tools.file.hdf5_functions import make_filelist
import matplotlib.pyplot as plt

from tools.file.write_log import write_log
from tools.file.hdf5_filename_to_datetime import hdf5_filename_to_datetime

import re
import numpy as np

regex = re.compile("20......_0[1-5]...._.*_SO_A_(I|E)_134")
# regex = re.compile("20180[45678].._.*_UVIS_E")


h5_fs, h5s, _= make_filelist(regex, "hdf5_level_0p3k")


occ_d = {"filename":[], "bin":[], "dt":[], "signal":[]}

# print("Filename, Integration time")
for h5, h5_f in zip(h5s, h5_fs):
    
    bins = h5_f["Science/Bins"][:, 0]
    sbsf = h5_f["Channel/BackgroundSubtraction"][0]
    
    if sbsf == 1:
        if "SO_A_I" in regex.pattern:
            y0 = h5_f["Science/Y"][1, :]
            bin0 = bins[1]
        else:
            y0 = h5_f["Science/Y"][-3, :]
            bin0 = bins[-3]
        
        # if bin0 in [120,121,122]:
        y0 = np.mean(y0[160:240])
        
        if y0 > 50000:
        
            dt = hdf5_filename_to_datetime(h5)
            
            occ_d["filename"].append(h5)
            occ_d["dt"].append(dt)
            occ_d["bin"].append(bin0)
            occ_d["signal"].append(y0)


plt.figure(figsize=(12, 5))
plt.scatter(occ_d["dt"], occ_d["signal"])
plt.ylabel("ADU counts when viewing Sun")
plt.xlabel("Observation time")
plt.title("NOMAD Solar Occultation channel: detector counts when viewing Sun")
