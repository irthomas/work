# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:41:37 2024

@author: iant

AOTF PEAK WAVELENGTH VS TEMPERATURE

TODO:
    IMPROVE FITTING RESIDUALS
    
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
# from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist2
from instrument.calibration.so_lno_2023.fit_blaze_shape import fit_blaze

file_level = "hdf5_level_1p0a"


regex = re.compile("20....01_......_.*_LNO_1_D._168")


max_sza = 20.0


def get_lno_data(regex, file_level, max_sza):
    h5fs, h5s, h5_paths = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for h5f, h5 in zip(h5fs, h5s):

        szas = h5f["Geometry/Point0/SunSZA"][:, 0]
        good_ixs = np.where(szas < max_sza)[0]

        # skip if more than 3 subdomains
        nsubs = int(h5f.attrs["NSubdomains"])

        # print(len(good_ixs), nsubs)
        if len(good_ixs) > 3 and nsubs < 4:
            y_raw = h5f["Science/YUnmodified"][good_ixs, :]
            ts = h5f["Channel/InterpolatedTemperature"][good_ixs]
            d[h5] = {"y_raw": y_raw, "szas": szas[good_ixs], "ts": ts}

    return d


if "d" not in globals():
    d = get_lno_data(regex, file_level, max_sza)

ts = []
peaks = []

for h5 in d.keys():
    y_raw = d[h5]["y_raw"]

    for spec_ix, y in enumerate(y_raw):

        blaze = fit_blaze(y)
        # plt.plot(blaze)

        ix = np.where(blaze == np.max(blaze))[0]
        t = d[h5]["ts"][spec_ix]

        if ix > 150 and ix < 250:
            ts.append(t)
            peaks.append(ix)

    # plt.plot(y_raw[::10, :].T)

plt.title("Raw spectrum peak vs temperature")
plt.scatter(ts, peaks, alpha=0.3)
plt.xlabel("Instrument temperature")
plt.ylabel("Peak pixel column")

best_fit = np.polyfit(ts, peaks, 1)
best_vals = np.polyval(best_fit, ts)

plt.plot(ts, best_vals, "k--")
plt.text(ts[100], best_vals[100], "y = %0.5f x + %0.5f" % (best_fit[0], best_fit[1]))

plt.grid()
plt.savefig("AOTF_peak_vs_temperature.png")
