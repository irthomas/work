# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:07:46 2023

@author: iant

COMPARE TIMESTAMPS BEFORE AND AFTER PACKET ERROR CORRECTION
"""


import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt




ROOT_DATA_DIR = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p1a"
ROOT_TEST_DATA_DIR = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5\hdf5_level_0p1a"

#get list of affected files

error_filepaths = glob.glob(r"X:\projects\NOMAD\Instrument\SOFTWARE-FIRMWARE\nomad_ops\*_packet_timestamp_correction.png")

for error_filepath in error_filepaths:
    
    basename = os.path.basename(error_filepath)
    h5 = basename[:15] + "_0p1a_SO_1"
    
    year = h5[0:4]
    month = h5[4:6]
    day = h5[6:8]
    
    h5_filepath1 = os.path.join(ROOT_DATA_DIR, year, month, day, "%s.h5" %h5)
    h5_filepath2 = os.path.join(ROOT_TEST_DATA_DIR, year, month, day, "%s.h5" %h5)
    
    
    with h5py.File(h5_filepath1, "r") as h5f:
        ts1 = h5f["Timestamp"][...]

    with h5py.File(h5_filepath2, "r") as h5f:
        ts2 = h5f["Timestamp"][...]
    
    fig1, (ax1, ax2) = plt.subplots(figsize=(10, 8), nrows=2, sharex=True)
    fig1.suptitle(h5)
    
    ax1.plot(ts1, label="Old")
    ax1.plot(ts2, label="New")
    ax1.legend()
    # ax2.plot(ts1 - ts2)
    ax2.plot(np.diff(ts1))
    ax2.plot(np.diff(ts2))
    
    stop()
