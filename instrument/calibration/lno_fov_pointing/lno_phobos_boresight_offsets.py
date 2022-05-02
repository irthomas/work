# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:50:10 2022

@author: iant

READ LNO BORESIGHT OFFSET PHOBOS DATA AFTER FEB 2022 PATCH
"""

import re
import matplotlib.pyplot as plt
import numpy as np


from tools.file.hdf5_functions import make_filelist

diffraction_order = 169

# regex = re.compile("2022...._......_.*_LNO_._P_%i" %diffraction_order)
regex = re.compile("20220326_......_.*_LNO_._P_%i" %diffraction_order)

file_level = "hdf5_level_0p3a"
# file_level = "hdf5_level_0p1d"


#Times:
#2022-Mar-04 14:50:33.001 #Phase 34-54; no offset; mid-poor signal
#2022-Mar-26 21:20:35.001 #Phase 44-36; Z offset -4; best signal
#2022-Mar-29 10:15:33.001 #Phase 62-52; Z offset -2; mid signal
#2022-Mar-29 18:07:33.001 #Phase 50-41; Z offset +2; poor signal
#2022-Apr-01 14:54:33.001 #Phase 54-46; Z offset +4; no signal


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
# stop()


bad_pixels = {
    144:[12, 102]
    }

good_bins = {
    "20220326_211035_0p3a_LNO_1_P_169":[146, 148, 150, 152]
    }


for h5, h5_filename in zip(hdf5_files, hdf5_filenames):
# h5 = hdf5_files[0]
# h5_filename = hdf5_filenames[0]

    dt_str = h5["Geometry/ObservationDateTime"][0, 0].decode()
    start_time = dt_str[0:4] + "-" + dt_str[5:8] + "-" + dt_str[9:-3]
    print(start_time)
    print(h5["Geometry/ObservationDateTime"][-1, -1])

    y_all = np.asfarray(h5["Science/Y"][...])
    bins = h5["Science/Bins"][:, 0]
    
    h5.close()
    
    
    
    
    unique_bins = sorted(list(set(bins)))
    
    bin_d = {}
    for unique_bin in unique_bins:
        idx = np.where(bins == unique_bin)[0]

        y = y_all[idx, :]
        if unique_bin in bad_pixels.keys():
            for px in bad_pixels[unique_bin]:
                interpolated_values = (y[:, px-1] + y[:, px+1])/2.0
                y[:, px] = interpolated_values
        
        bin_d[unique_bin] = {}
        bin_d[unique_bin]["raw"] = y
        
            
        
        bin_d[unique_bin]["mean_px"] = np.mean(y_all[idx, :], axis=1)
        # bin_d[unique_bin]["mean_px"] = np.mean(y_all[idx, 160:240], axis=1)
        bin_d[unique_bin]["mean_all"] = np.mean(y_all[idx, :])
        
        # plt.figure()
        # plt.title(unique_bin)
        # plt.plot(bin_d[unique_bin]["raw"].T)
        # plt.plot(np.mean(bin_d[unique_bin]["raw"], axis=0), "k--")
    # stop()
    
    
    # fig = plt.figure(figsize=(16,10), constrained_layout=True)
    # im = plt.imshow(y.T, aspect="auto")
    # plt.colorbar(im)
    
    
    
    # plt.figure()
    # plt.title("LNO phobos observation after patching: %s" %h5_filename)
    # plt.plot(bin_d.keys(), [bin_d[i]["mean_all"] for i in bin_d.keys()])
    
    grid_2d = np.asfarray([bin_d[i]["mean_px"] for i in bin_d.keys()])

    # plt.figure()
    # for i in bin_d.keys():
    #     mean_px = bin_d[i]["mean_px"]
    #     rolling_mean_px = moving_average(mean_px, 20)
    #     rolling_mean_x = moving_average(np.arange(len(mean_px)), 20)
    #     plot = plt.plot(mean_px, label=i, alpha=0.5)
    #     plt.plot(rolling_mean_x, rolling_mean_px, label=i, alpha=1, color=plot[0].get_color())
    # plt.legend()
    # plt.axhline(y=0, color="k")
    # stop()
    
    plt.figure(figsize=(10,5), constrained_layout=True)
    plt.title("LNO phobos observation after patching: %s" %h5_filename)
    im = plt.imshow(grid_2d, extent=(0, grid_2d.shape[1], unique_bins[0], unique_bins[-1]+2), aspect="auto", origin="lower")
    cbar = plt.colorbar(im)
    cbar.set_label("Mean signal on each detector bin", rotation=270, labelpad=10)
    
    plt.xlabel("Frame Number")
    plt.ylabel("Detector row")
    plt.savefig("%s_phobos_means.png" %h5_filename)
    
    ill_bins = [144, 146, 148, 150, 152, 154, 156]

    plt.figure(figsize=(9,7), constrained_layout=True)
    plt.title("LNO phobos observation after patching: %s" %h5_filename)
    
    for ill_bin in ill_bins:
        spectrum = np.mean(bin_d[ill_bin]["raw"], axis=0)
    
        plt.plot(spectrum, label="Detector bin %i" %ill_bin)
    plt.legend()
    plt.savefig("%s_phobos_spectra.png" %h5_filename)
    