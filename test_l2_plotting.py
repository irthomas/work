# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:55:24 2026

@author: iant
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

l2_path = r"X:\projects\work\NOMAD\Science\so_occ_h2o_1\ord134\RESULTS\2023\09\20230930_090935_1p0a_SO_A_I_134_Level2.h5"

with h5py.File(l2_path, "r") as l2f:
    yfit = np.squeeze(l2f["Science/Y"][...])
    yobs = np.squeeze(l2f["Science/YObs"][...])

plt.figure()
plt.plot(yobs.T)
plt.plot(yfit.T)

# make ils shapes and copy ils + aotf order 134 to hdf5 and plot on website
