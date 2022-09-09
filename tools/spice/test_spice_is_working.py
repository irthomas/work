# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:20:56 2019

@author: iant

TEST IF SPICE IS WORKING

"""

import spiceypy as sp
import os
from tools.file.paths import paths

METAKERNEL_NAME = "em16_ops.tm"
#load spiceypy kernels
os.chdir(paths["KERNEL_DIRECTORY"])
sp.furnsh(paths["KERNEL_DIRECTORY"]+os.sep+METAKERNEL_NAME)
os.chdir(paths["BASE_DIRECTORY"])
print(sp.tkvrsn("toolkit"))




from datetime import datetime, timedelta

for days in range(10, 500, 1):
    
    dt_start = datetime.now() - timedelta(days=days)
    
    dt_str = datetime.strftime(dt_start, "%Y %b %d %H:%M:%S.%f")
    
    et = sp.str2et(dt_str)
    
    #go back in time to see if errors
    obs2SunVector = sp.spkpos("SUN", et, "TGO_NOMAD_SO", "None", "-143")
    
    print(dt_str)