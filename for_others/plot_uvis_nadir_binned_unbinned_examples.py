# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:56:44 2025

@author: iant

PLOT UVIS BINNED VS UNBINNED
"""

import numpy as np
import matplotlib.pyplot as plt
from tools.file.hdf5_functions import open_hdf5_file


h5s = ["20181107_190651_1p0a_UVIS_D", "20181102_211500_1p0a_UVIS_D"]
# h5s = ["20181102_211500_1p0a_UVIS_D"]


data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
# data_path = r"C:\Users\iant\Documents\DATA\hdf5"


for h5 in h5s:

    h5f = open_hdf5_file(h5, path=data_path)

    ys = h5f["Science/Y"][...]
    yerrors = h5f["Science/YError"][...]
    snrs = ys/yerrors
    xs = h5f["Science/X"][...]

    binning = h5f["Channel/HorizontalAndCombinedBinningSize"][0]

    szas = np.mean(h5f["Geometry/Point0/SunSZA"][...], axis=1)

    ix_min = np.argmin(szas) + 10

    sza = szas[ix_min]

    # plt.plot(xs.T, ys.T)

    print(snrs[ix_min, :])

    plt.plot(xs[ix_min, :], ys[ix_min, :], label="%s: solar zenith angle %0.2f degrees" % (h5, sza))

plt.legend()
plt.grid()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance")
plt.title("UVIS nadir binning vs no binning")
