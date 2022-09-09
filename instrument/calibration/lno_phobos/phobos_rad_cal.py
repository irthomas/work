# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:29:14 2022

@author: iant

CALIBRATE LNO PHOBOS OBSERVATIONS
"""


import re
import numpy as np
from matplotlib import pyplot as plt

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours

from instrument.calibration.lno_phobos.bb_ground_cal import rad_cal_order



file_level = "hdf5_level_0p3a"


fig2, ax2a = plt.subplots(figsize=(15, 10), constrained_layout=True)

fig2.suptitle("Phobos Radiance Calibration")

for regex_str in ["20220710.*_LNO_._P", "20220713.*_LNO_._P", "20220714.*_LNO_._P", "20220719.*_LNO_._P"]:
    regex = re.compile(regex_str)

    h5_files, h5_filenames, _ = make_filelist(regex, file_level)
    
    
    
    cal_h5 = "20150426_054602_0p1a_LNO_1"
    
    
    
    # fig1, (ax1a, ax1b) = plt.subplots(figsize=(15, 10), nrows=2, sharex=True, constrained_layout=True)
    
    rad_d = {"wavenumber":[], "bin":[], "radiance":[], "radiance_error":[]}
    
    for file_ix, (h5_f, h5) in enumerate(zip(h5_files, h5_filenames)):
    
        # observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
        bins = h5_f["Science/Bins"][...]
        
        x = h5_f["Science/X"][0, :]
        y = h5_f["Science/Y"][...]
        
        order = h5_f["Channel/DiffractionOrder"][0]
        
        cal_d = rad_cal_order(cal_h5, order)
        counts_per_rad = cal_d["counts_per_rad"]
        
        
        # ydimensions = y.shape
        # nSpectra = ydimensions[0]
    
        #on first run find unique bins and colours
        if file_ix == 0:
            unique_bins = sorted(list(set(bins[:, 0])))
            colours = get_colours(len(unique_bins))
    
    
        
        for bin_ix, unique_bin in enumerate(unique_bins):
            indices = np.where(bins[:, 0] == unique_bin)[0]
            
            y_mean = np.mean(y[indices, :], axis=0)
    
            if file_ix == 0:
                label = unique_bin
            else:
                label = ""
    
            y_rad = y_mean / counts_per_rad        
    
            # ax1a.plot(x, y_mean, label=label, color=colours[bin_ix])
            # ax1b.plot(x, y_rad, label=label, color=colours[bin_ix])
            
            x_mean = np.mean(x)
            y_rad_mean = np.mean(y_rad[160:240])
            y_rad_std = np.std(y_rad[160:240])

            # y_rad_mean = np.mean(y_rad)
            # y_rad_std = np.std(y_rad)
            
            rad_d["wavenumber"].append(x_mean)
            rad_d["bin"].append(unique_bin)
            rad_d["radiance"].append(y_rad_mean)
            rad_d["radiance_error"].append(y_rad_std)
            
    
    # ax1a.legend()
    # ax1a.grid()
    # ax1b.legend()
    # ax1b.grid()
    
    
    #convert to np arrays
    for key in rad_d.keys():
        rad_d[key] = np.array(rad_d[key])
    
    
    
    
    for bin_ix, unique_bin in enumerate(unique_bins):
        
        bin_colours = {146:"r", 149:"g", 152:"b", 155:"k"}
        colour = bin_colours[unique_bin]
        
        indices = np.where(rad_d["bin"] == unique_bin)[0]
        ax2a.scatter(rad_d["wavenumber"][indices], rad_d["radiance"][indices], color=colour, label=unique_bin)
        ax2a.errorbar(rad_d["wavenumber"][indices], rad_d["radiance"][indices], yerr=rad_d["radiance_error"][indices], \
                      color=colour, capsize=4, label=unique_bin)
    
ax2a.legend()
ax2a.grid()
ax2a.set_xlabel("Diffraction order central wavenumber")
ax2a.set_ylabel("Mean radiance of diffraction order centre W/cm2/sr/cm-1")