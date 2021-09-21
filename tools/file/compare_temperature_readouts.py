# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:39:35 2021

@author: iant

COMPARE TEMPERATURES IN FILES
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours
from tools.file.get_hdf5_temperatures import get_interpolated_temperatures

# regex = re.compile("20210201_0.*_1p0a_SO_A_E_134")
regex = re.compile("20......_(01|02)...._1p0a_SO_A_[IEG]_134")
# regex = re.compile("20180528_012146_1p0a_SO_A_E_134")
file_level = "hdf5_level_1p0a"

# regex = re.compile("20......_......_0p2a_SO_._C")
# file_level = "hdf5_level_0p2a"

channel = "SO"
precooling = True

chosen_bin = 3

hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
colours = get_colours(len(hdf5_filenames))


sensors = ["Temperature/NominalSO", "Temperature/RedundantSO", "Housekeeping/AOTF_TEMP_SO", "Housekeeping/SENSOR_1_TEMPERATURE_SO", ]

with PdfPages("%s_temperature_comparison_precooling.pdf" 
              %(regex.pattern).replace(".","").replace("*","").replace("(","").replace(")","").replace("|","")) as pdf: #open pdf

    for i, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):

        if np.mod(i, 10) == 0:
            print("%i/%i" %(i, len(hdf5_filenames)))

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(4, 1)
        ax1a = fig.add_subplot(gs[0:2, 0])
        ax1b = fig.add_subplot(gs[2, 0], sharex=ax1a)
        ax1c = fig.add_subplot(gs[3, 0], sharex=ax1a)
        
        if file_level == "hdf5_level_1p0a":
            bins = hdf5_file["Science/Bins"][:, 0]
            unique_bins = sorted(list(set(bins)))
            bin_indices = np.where(bins == unique_bins[chosen_bin])[0]
        if file_level == "hdf5_level_0p2a":
            bin_indices = np.arange(len(hdf5_file["Geometry/ObservationDateTime"][:, 0]))
        
        if precooling:
            bin_indices = np.arange(600)
        
        d = {}
        for sensor in sensors:
            if "Housekeeping" in sensor:
                suffix = "Median filter"
                d[sensor] = get_interpolated_temperatures(hdf5_file, channel, plot=False, sensor=sensor, t_filter="median", precooling=precooling)
            else:
                suffix = "Median and quadratic fit"
                d[sensor] = get_interpolated_temperatures(hdf5_file, channel, plot=False, sensor=sensor, t_filter="m+q", precooling=precooling)
        
            ax1a.plot(d[sensor][bin_indices], label="%s (%s)" %(sensor, suffix))
    
        fig.suptitle("%s: Temperature Sensor Comparison" %hdf5_filename)
        ax1b.plot(d["Temperature/NominalSO"][bin_indices] - d["Housekeeping/SENSOR_1_TEMPERATURE_SO"][bin_indices], color="C3")
        ax1b.set_title("NominalSO - SENSOR_1_TEMPERATURE_SO")
        ax1c.plot(d["Temperature/NominalSO"][bin_indices] - d["Housekeeping/AOTF_TEMP_SO"][bin_indices], color="C2")
        ax1c.set_title("NominalSO - AOTF_TEMP_SO")
        ax1a.set_ylabel("NOMAD Temperature (C)")
        ax1c.set_xlabel("Frame index (bin %i)" %chosen_bin)
        ax1a.grid()
        ax1b.grid()
        ax1c.grid()
        ax1a.legend(loc="lower right")
        
        pdf.savefig()
        plt.close()
