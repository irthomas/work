# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:45:48 2024

@author: iant
"""


import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist2


regex = re.compile("2024...._......_0p3k_UVIS_I")
file_level = "hdf5_level_0p3k"


h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
spectra_dict = {}


d = {}
for i, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

    alts_all = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
    ixs = np.where((alts_all > 150) & (alts_all < 200))[0]

    y = h5_f["Science/Y"][...]
    x = h5_f["Science/X"][0, :]

    if y.shape[1] > 1000:
        d[h5] = {"y": y[ixs, :], "x": x, "alts": alts_all[ixs]}

    # ax1.plot(y[ixs, :])

    # ax1.set_ylabel("Signal")
    # ax2.set_xlabel("Tangent Altitude (km)")
    # ax1.grid()
    # ax2.grid()
    # ax1.legend()
    # ax2.legend()

    # fig1.savefig("%s_pointing_deviation.png" %h5)

mean_spectrum = np.mean(np.concatenate([d[h5]["y"] for h5 in d.keys()]), axis=0)

fig1, ax1 = plt.subplots()
for h5 in d.keys():
    for y in d[h5]["y"]:
        plt.plot(d[h5]["x"], y / mean_spectrum)

plt.xlabel("Wavelength (nm")
plt.ylabel("Raw signal")
plt.grid()
