# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:04:12 2024

@author: iant

LOAD MINISCAN PKLS
"""
import os

from tools.plotting.save_load_figs import load_ifig

# from instrument.calibration.so_lno_2023.fit_absorption_miniscan_array import trap_absorption
from instrument.calibration.so_lno_2023.solar_line_dict import solar_line_dict

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

for h5_prefix, solar_line_data_all in solar_line_dict.items():  # loop through files
    channel = h5_prefix.split("-")[0].lower()

    # get data from miniscan file
    path = os.path.join(MINISCAN_PATH, channel, "%s_ifig.pkl" % h5_prefix)

    load_ifig(path)
