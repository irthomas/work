# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:16:44 2026

@author: 

READ SOIR MINISCAN PDS3 DATA
"""


import os
import numpy as np
import glob
import matplotlib.pyplot as plt

# DATA_DIR = r"C:\Users\iant\Documents\DATA\soir\VEX-Y-V-SPICAV-3-SOIR-V5.0\DATA\20060910_M01"
DATA_DIR = r"C:\Users\iant\Documents\DATA\soir\VEX-Y-V-SPICAV-3-SOIR-V5.0\DATA\20060910_M02"
DATA_DIR = r"C:\Users\iant\Documents\DATA\soir\VEX-Y-V-SPICAV-2-SOIR-V1.0\DATA\20060910_M05_S02"
BEST_BIN = 2.0

# TODO: check this data
r"X:\projects\Venus_Express\Soir_Science\VEX_SOIR\Spectra\0.1\20060910_M01"

tab_paths = list(sorted(glob.glob(DATA_DIR + os.sep + "*.TAB")))

if "SOIR-V5.0" in DATA_DIR:
    data_d = {}
    for tab_path in tab_paths:
        with open(tab_path, "r") as f:
            lines = f.readlines()

            bins = []
            counts = []
            for line_ix, line in enumerate(lines):
                lines_split = line.split(",")

                if line_ix == 0:
                    aotf = float(lines_split[15])  # always same freq in one file

                bins.append(float(lines_split[1]))
                counts.append([float(i) for i in lines_split[23:343]])

            bins = np.asarray(bins)
            counts = np.asarray(counts)

            best_bin_ixs = np.where(bins == BEST_BIN)[0]
            best_bin_counts = np.mean(counts[best_bin_ixs, :], axis=0)

            data_d[aotf] = {"bins": bins, "counts": counts, "best_counts": best_bin_counts}

if "SOIR-V1.0" in DATA_DIR:

    tab_path = tab_paths[0]  # obs data is first file
    data_d = {}

    with open(tab_path, "r") as f:
        lines = f.readlines()

        for line_ix, line in enumerate(lines):
            lines_split = line.replace("\"", "").split(",")

            bins = []
            counts = []
            if lines_split[4].strip() == "O":
                for j in np.arange(5, 2564, 320):
                    counts.append([float(i) for i in lines_split[j:j + 320]])
                counts = np.asarray(counts)
                best_bin_ixs = [2, 3]
                best_bin_counts = np.mean(counts[best_bin_ixs, :], axis=0)

                data_d[line_ix] = {"counts": counts, "best_counts": best_bin_counts}


aotf_array = []
aotfs = list(data_d.keys())
for aotf in aotfs:
    aotf_array.append(data_d[aotf]["best_counts"] / np.max(data_d[aotf]["best_counts"]))
    # aotf_array.append(data_d[aotf]["best_counts"])

# aotf = 13642.0

aotf_array = np.asarray(aotf_array)

plt.figure()
plt.imshow(aotf_array.T)
