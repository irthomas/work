# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:23:53 2020

@author: iant

PLOT ACE SOLSPEC AND PFS SOLSPEC SPECTRA. ACE CUTS OFF AT 4430CM-1 -> NEED UP TO 4515CM-1
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



from tools.file.paths import paths



solspec_filepath = os.path.join(paths["REFERENCE_DIRECTORY"], "nomad_solar_spectrum_solspec.txt")
pfs_filepath = os.path.join(paths["REFERENCE_DIRECTORY"], "pfsolspec_hr.dat")


nu_solar1, I0_solar1 = np.loadtxt(solspec_filepath, unpack=True)
nu_solar2, I0_solar2 = np.loadtxt(pfs_filepath, unpack=True)

i1 = np.where((nu_solar1 > 4300) & (nu_solar1 < 4430))[0]
i2 = np.where((nu_solar2 > 4430) & (nu_solar2 < 4515))[0]

scalar = I0_solar1[i1[-1]] / I0_solar2[i2[0]]

plt.plot(nu_solar1[i1], I0_solar1[i1])
plt.plot(nu_solar2[i2], I0_solar2[i2] * scalar)