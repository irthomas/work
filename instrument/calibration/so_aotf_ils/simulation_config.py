# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:43:09 2021

@author: iant

SIMULATION CONFIG
"""

import numpy as np

AOTF_OFFSET_SHAPE = "Gaussian"
# AOTF_OFFSET_SHAPE = "Constant"



# ORDER_RANGE = [192, 198]
# nu_range = [4250., 4500.]



ORDER_RANGE = [185, 205]
nu_range = [4150., 4650.]


pixels = np.arange(320)

# nu_range = [4309.7670539950705, 4444.765043408191]
D_NU = 0.005

abs_aotf_range = [26560, 26640]