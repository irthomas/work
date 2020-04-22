# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:45:39 2019

@author: iant

SIMULATE SOLAR LINE SHIFTS IN SO

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tools.spectra.solar_spectrum_so import so_solar_line_temperature_shift
from instrument.nomad_so_instrument import nu_mp


diffractionOrder = 134
instrumentTemperatures = [-5.0,-4.0]
solspecFile = os.path.join("reference_files", "nomad_solar_spectrum_solspec.txt")

lineShift = so_solar_line_temperature_shift(diffractionOrder, instrumentTemperatures, solspecFile, adj_orders=2, cutoff=0.999)
nu = nu_mp(diffractionOrder, np.arange(320), instrumentTemperatures[0])

plt.plot(nu, lineShift)