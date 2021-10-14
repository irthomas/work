# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:23:12 2021

@author: iant

DIFFERENCE BETWEEN MINISCAN FRAMES
"""

import re
import numpy as np
# import lmfit


import matplotlib.pyplot as plt

# from instrument.nomad_so_instrument import nu_mp
# from tools.asimut.ils_params import get_ils_params
from tools.plotting.colours import get_colours

from instrument.calibration.so_aotf_ils.simulation_functions_v02 import (get_cal_params, blaze_conv, get_ils_params, get_file, get_data_from_file, 
     select_data, fit_temperature, make_param_dict, calc_spectrum, fit_spectrum, get_solar_spectrum, get_all_x, find_absorption_minimum)

from instrument.calibration.so_aotf_ils.simulation_config import sim_parameters, AOTF_OFFSET_SHAPE, BLAZE_WIDTH_FIT, AOTF_FROM_FILE


line = 4383.5
# line = 4276.1
# line = 3787.9


if line == 4383.5:
    regex = re.compile("20190416_020948_0p2a_SO_1_C")
    index = 0
    # index = 80

if line == 4276.1:
    regex = re.compile("20180716_000706_0p2a_SO_1_C")
    index = 72




"""spectral grid and blaze functions of all orders"""

#get data, fit temperature and get starting function parameters
hdf5_file, hdf5_filename = get_file(regex)

# solar_spectrum = "ACE"
# solar_spectrum = "PFS"
d = {}
d["line"] = line
d = get_data_from_file(hdf5_file, hdf5_filename, d)
d = get_all_x(hdf5_file, d)
y = d["spectra"]

# for i in range(255):
    
diff = y[1:257,:] - y[0, :]

colours = get_colours(50)
for i in range(50):
    plt.plot(d["x_all"][i, :], diff[i, :], label=i, color=colours[i])
plt.legend()