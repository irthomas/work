# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:38:07 2025

@author: iant


CHECK NOMAD SLOW FULLSCAN POINTING BEFORE AND AFTER 2024 GYROLESS UPDATE
"""


import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist2
from tools.plotting.colours import get_colours

regex = re.compile("20241..._......_0p3a_SO_._S")
file_level = "hdf5_level_0p3a"


def get_fullscan_data(regex, file_level):
    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for i, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        # get bins, check if slow fullscan
        first_bin = h5_f["Science"]["Bins"][0, :]
        if first_bin[1] == first_bin[0]:

            alts_all = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            ixs = np.where(alts_all > 200)[0]

            y = h5_f["Science/Y"][...]
            bins = h5_f["Science"]["Bins"][:, 0]
            order = h5_f["Channel"]["DiffractionOrder"][:]

            d[h5] = {"ys": y[ixs, :], "orders": order[ixs], "bins": bins[ixs], "alts": alts_all[ixs]}

    return d


d = get_fullscan_data(regex, file_level)


fig1, ax1 = plt.subplots()

colours = get_colours(len(d.keys()), cmap="brg")

for h5_ix, h5 in enumerate(d.keys()):

    bins = d[h5]["bins"]
    frame_start_ixs = np.where(bins == np.min(bins))[0]
    # if ingress, plot first detector frames
    if d[h5]["alts"][0] > 201:
        # print(h5, bins[0:50])
        i = 10
    else:
        a = d[h5]["alts"]
        # stop()
        i = 50

    frame_ixs = [frame_start_ixs[i], frame_start_ixs[i+1]]

    bins = d[h5]["bins"][frame_ixs[0]:frame_ixs[1]]
    order = d[h5]["orders"][frame_ixs[0]:frame_ixs[1]]
    y_raw = d[h5]["ys"][frame_ixs[0]:frame_ixs[1], 200]
    print(order)

    plt.plot(y_raw/np.max(y_raw), bins, label=h5, color=colours[h5_ix])

plt.xlabel("Detector row")
plt.ylabel("Normalised signal")
plt.grid()
plt.legend()
