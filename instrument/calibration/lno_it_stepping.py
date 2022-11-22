# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:27:10 2022

@author: iant
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


with h5py.File("20161121_183450_0p1a_LNO_1.h5") as h5_f:
    it = h5_f["Channel/IntegrationTime"][...]
    
    y = h5_f["Science/Y"][...]
    

# plt.plot(y[:, 20, :].T)
plt.title("LNO solar observations: integration time stepping")
plt.scatter(it*1e3, y[:, 19, 200])
plt.scatter(it*1e3, y[:, 20, 200])
plt.scatter(it*1e3, y[:, 21, 200])
plt.xlabel("Integration time (microseconds)")
plt.ylabel("Counts")
plt.grid()