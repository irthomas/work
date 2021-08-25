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


