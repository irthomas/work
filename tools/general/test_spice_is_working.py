# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:20:56 2019

@author: iant

TEST IF SPICE IS WORKING

"""

import spiceypy as sp
import os
KERNEL_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\kernels\mk"
METAKERNEL_NAME = "em16_ops.tm"
#load spiceypy kernels
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))

