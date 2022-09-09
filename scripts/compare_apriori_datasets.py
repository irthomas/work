# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:11:59 2022

@author: iant

COMPARE APRIORIS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from tools.datasets.get_mcd_data import get_mcd_data
from tools.datasets.get_gem_data import get_gem_data


myear=35
ls=181.0
lst=12.0
lat=0.0
lon=0.0
lst=12.0
mcd_dict = get_mcd_data(myear, ls, lat, lon, lst)
gem_dict = get_gem_data(myear, ls, lat, lon, lst)



fig, axes = plt.subplots(figsize=(18, 8), ncols=len(gem_dict.keys())-1, sharey=True, constrained_layout=True)

z_g = gem_dict["z"]
z_m = mcd_dict["z"]
for i, key in enumerate([k for k in gem_dict.keys() if k != "z"]):
    axes[i].plot(gem_dict[key], z_g, label="GEM %s" %key)
    axes[i].plot(mcd_dict[key], z_m, linestyle="--", label="MCD %s" %key)

    if key != "t":
        axes[i].set_xscale("log")
    axes[i].grid()
    axes[i].legend()