# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:08:03 2021

@author: iant
"""


import numpy as np
import os
import re
import json
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import scipy.signal as ss

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist



file_level = "hdf5_level_1p0a"

regex = re.compile("20210[234].*_UVIS_L")

d = {
"20210110_081723_1p0a_UVIS_L":"inertial",
"20210110_141130_1p0a_UVIS_L":"inertial",
"20210111_035759_1p0a_UVIS_L":"inertial",
"20210118_125831_1p0a_UVIS_L":"inertial",
"20210119_123407_1p0a_UVIS_L":"inertial",
"20210121_114505_1p0a_UVIS_L":"inertial",
"20210124_004129_1p0a_UVIS_L":"inertial",
"20210125_021446_1p0a_UVIS_L":"inertial",
"20210207_004839_1p0a_UVIS_L":"inertial",
"20210214_114124_1p0a_UVIS_L":"tracking",
"20210215_012700_1p0a_UVIS_L":"tracking",
"20210216_010227_1p0a_UVIS_L":"tracking",
"20210226_184044_1p0a_UVIS_L":"inertial",
"20210303_010335_1p0a_UVIS_L":"tracking",
"20210304_123009_1p0a_UVIS_L":"tracking",
"20210307_013127_1p0a_UVIS_L":"inertial",
"20210308_105830_1p0a_UVIS_L":"inertial",
"20210316_094345_1p0a_UVIS_L":"tracking",
"20210317_230452_1p0a_UVIS_L":"tracking",
"20210320_040919_1p0a_UVIS_L":"tracking",
    }

hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    print(hdf5_filename)

    tangent_alts = hdf5_file["Geometry/Point0/TangentAlt"][:,0] #get tangent altitudes for FOV centre
    
    tangent_alts[tangent_alts < -100.0] = np.nan #set invalid values to NaN
    
    tangent_alt_deltas = [t - s for s, t in zip(tangent_alts, tangent_alts[1:])] #get difference between consecutive tangent altitudes
    
    o_type = d[hdf5_filename]
    linestyle = {"inertial":"-", "tracking":":"}[o_type]
    colour = {"inertial":"r", "tracking":"g"}[o_type]
    
    ax1.plot(tangent_alts, linestyle=linestyle, label=hdf5_filename)
    # ax2.plot(tangent_alt_deltas, linestyle=linestyle, label=hdf5_filename)
    
    ax2.hist(tangent_alt_deltas, bins=100, alpha=0.1, color=colour, label="%s (%s)" %(hdf5_filename, o_type))
    
ax1.legend()

ax2.set_xlabel("Difference between consecutive tangent altitudes")
ax2.set_ylabel("Number of points within each histogram")
ax2.set_title("Histograms of the difference between consecutive altitudes")

ax2.legend()
