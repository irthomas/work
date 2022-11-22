# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:17:03 2022

@author: iant
"""


import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import open_hdf5_file


h5 = "20220301_063212_0p3k_SO_A_E_185"


h5_f = open_hdf5_file(h5)


y = h5_f["Science/Y"][...]
alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]

bins = h5_f["Science/Bins"][:, 0]

unique_bins = sorted(list(set(bins)))
unique_bin = unique_bins[-1]

y_bin = y[bins == unique_bin, :]

#find indices of spectra where 0.1 < median transmittance < 0.95
y_median = np.median(y_bin, axis=1)
# indices = list(np.where((y_median > 0.1) & (y_median < 0.95))[0])
indices = list(np.where((y_median/np.max(y_median) > 0.3) & (y_median/np.max(y_median) < 1.9))[0])


y_sun_spectra = y_bin[500:600, :]
y_sun = np.mean(y_sun_spectra, axis=0)

plt.figure()
for index in indices:
    plt.plot(y_bin[index, :].T/y_sun)

# plt.plot(y_bin[500:600, :].T/y_sun)

plt.figure()

# for pixel in [100, 160, 200, 240]:
for pixel in [199, 200, 201, 202]:
    plt.plot(y_bin[:, pixel]/np.mean(y_bin[:, 199:203], axis=1), label=pixel)
plt.legend()