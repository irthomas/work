# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:03:37 2019

@author: iant

SIMULATE SOLAR LINE SHIFTS IN lNO
"""


import os
import numpy as np
import matplotlib.pyplot as plt

from tools.spectra.solar_spectrum_lno import lno_solar_line_temperature_shift
from instrument.nomad_lno_instrument import nu_mp


diffractionOrder = 134
instrumentTemperatures = [-5.0,-4.0]
solspecFile = os.path.join("reference_files", "nomad_solar_spectrum_solspec.txt")

lineShift = lno_solar_line_temperature_shift(diffractionOrder, instrumentTemperatures, solspecFile, adj_orders=2, cutoff=0.999)
nu = nu_mp(diffractionOrder, np.arange(320), instrumentTemperatures[0])

plt.plot(nu, lineShift)

    