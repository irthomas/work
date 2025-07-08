# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:46:58 2024

@author: iant

DONE:
    PLOT NEW BINNED DATASET

TO DO:
    PLOT UNBINNED
    2D GROUNDTRACK PLOT WITH FOVS


"""

from tools.file.hdf5_functions import open_hdf5_file
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

# h5s = [
#     "20250407_190226_1p0a_LNO_1_DP_168",
#     "20250407_190226_1p0a_LNO_1_DP_189",
#     "20250407_190226_1p0a_LNO_1_DP_190",
# ]
h5s = [
    "20250407_190226_1p0a_LNO_1_DP_168",
    "20250407_190226_1p0a_LNO_1_DP_189",
    "20250407_190226_1p0a_LNO_1_DP_190",
]


d = {}
for h5 in h5s:
    h5f = open_hdf5_file(h5, path=DATA_PATH)

    # print(h5f["Science"].keys())
    order = h5.split("_")[-1]

    ypeaks_b = h5f["Science/YPeakReflectanceFactor"][:]
    ypeaks_b_err = h5f["Science/YPeakReflectanceFactorError"][:]
    lats_b = h5f["Geometry/Point0/Lat"][:, 0]
    lons_b = h5f["Geometry/Point0/Lon"][:, 0]
    szas_b = h5f["Geometry/Point0/SunSZA"][:, 0]
    # ypeak_unbinned = h5f["Science/YPeakReflectanceFactor"][...]

    # plt.scatter(lons_b[szas_b < 70.], lats_b[szas_b < 70.], c=ypeaks_b[szas_b < 70.])
    # plt.scatter(lons_b, lats_b, c=szas_b)

    # plt.plot(lats_b[szas_b < 80.], ypeaks_b[szas_b < 80.])
    plt.errorbar(lats_b[szas_b < 80.], ypeaks_b[szas_b < 80.], yerr=ypeaks_b_err[szas_b < 80.], label="Order %s" % order)

plt.title("%s: SZA Range %i - %i degrees" % (h5[0:29], min(szas_b), 80))
plt.xlabel("Latitude")
plt.ylabel("Reflectance Factor")
plt.legend()
