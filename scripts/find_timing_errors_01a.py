#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:57:11 2023

@author: iant

RUN THROUGH ALL 0.1A FILES TO FIND TC/TM TIMING ERRORS
"""

import re
from datetime import datetime
import numpy as np

# import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
# from tools.file.write_log import write_log
from tools.general.progress_bar import progress_bar



regex = re.compile("20(18|19|20|21|22|23)...._.*_SO_.*")


_, h5s, _= make_filelist(regex, "hdf5_level_0p1a", open_files=False, silent=True)

problem = {}

#loop on h5 filenames
for h5 in h5s: #progress_bar(h5s):
    
    h5_f = open_hdf5_file(h5)

    # bins = h5_f["Science/Bins"][:, 0]
    aotfs = h5_f["Channel/AOTFFrequency"][...]
    unique_aotfs = list(set(aotfs))
    
    #indices of first AOTF frequency
    ixs = np.where(aotfs == unique_aotfs[0])[0]
    
    obs_dt = [s.decode() for s in h5_f["Geometry/ObservationDateTime"][ixs, 0]]
    
    dts = [datetime.strptime(dt, "%Y %b %d %H:%M:%S.%f") for dt in obs_dt]
    
    #differences in milliseconds between consecutive frames
    ms_diff = np.asarray([td.total_seconds() * 1.0e6 for td in np.diff(dts)])
    
    #check that file is normal i.e. 1 second rhythm
    ms_mean = np.mean(ms_diff)
    
    good = True
    
    if np.abs(1.0e6 - ms_mean) < 10000.0:
        #check for consistency
        ms_min = np.min(ms_diff)
        ms_max = np.max(ms_diff)
        
        good = False
        
        if np.abs(1.0e6 - ms_min) < 10000.0:
            if np.abs(1.0e6 - ms_max) < 10000.0:
                good = True

    if not good:
        n_found = int(np.ceil(np.sum(np.abs(1.0e6 - ms_diff) > 10000.0) / 2.0))
        ixs_error = np.where(np.abs(1.0e6 - ms_diff) > 10000.0)[0]
        
        print("%s: %i errors found" %(h5, n_found), ms_min, ms_max)
