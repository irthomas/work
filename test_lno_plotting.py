# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:46:49 2020

@author: iant
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np

MAX_SZA = 12.5



with h5py.File("20190101_075935_1p0a_LNO_1_DP_189.h5", "r") as hdf5_file:
    y_reff = hdf5_file["Science/YReflectanceFactor"][...]
    x = hdf5_file["Science/X"][:]
    y_flat = hdf5_file["Science/YReflectanceFactorFlat"][...]
    mean_curve = hdf5_file["Science/MeanCurveShifted"][...]

    sza = np.mean(hdf5_file["Geometry/Point0/IncidenceAngle"][...], axis=1)
    temperature = float(hdf5_file["Channel/MeasurementTemperature"][0][0])

    valid_ys = np.where(sza < MAX_SZA)[0]
    # if len(valid_ys) == 0:
        # continue
    

plt.figure()
plt.ylim((0,1))
plt.plot(x, mean_curve)
for index in valid_ys:
    plt.plot(x, y_reff[index, :])
    # plt.plot(x, y_flat[index, :])