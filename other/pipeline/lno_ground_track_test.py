# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:04:22 2023

@author: iant

PLOT LNO GROUNDTRACKS TO CHECK NEW PIPELINE GEOMETRY MODES (UNBINNED, 5 POINTS ETC.)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

new_hdf5_filepath = "C:\\Users\\iant\\Dropbox\\NOMAD\\Python\\tmp\\20221125_082524_0p2a_LNO_1_D_189.h5"
new_hdf5_filepath = "C:\\Users\\iant\\Dropbox\\NOMAD\\Python\\tmp\\20221125_082524_0p2a_LNO_1_D_189_old.h5"


with h5py.File(new_hdf5_filepath, "r") as h5_f:
    plt.figure(figsize=(10, 8))
    n_points = int([s for s in list(h5_f["Geometry"].keys()) if "Point" in s][-1].replace("Point", ""))
    lats = [h5_f["Geometry/Point%i/Lat" %i][...] for i in range(n_points+1)]
    lons = [h5_f["Geometry/Point%i/Lon" %i][...] for i in range(n_points+1)]
    bins = h5_f["Science/Bins"][:, 0]
    unique_bins = list(set(bins))
    
    # for lat_arr, lon_arr in zip(lats[0:1], lons[0:1]): #loop through PointX
    # for lat_arr, lon_arr in zip(lats, lons): #loop through PointX, giving a 2d array of start and end lat/lons
    for k, unique_bin in enumerate(unique_bins):
        ixs = np.where(unique_bin == bins)[0] #loop through bins

        for i in ixs[0:2]:
            for j in range(len(lons[0].shape[1])):
                rectangle = np.asarray([
                    [lons[1][i, j], lats[1][i, j]], \
                    [lons[2][i, j], lats[2][i, j]], \
                    [lons[3][i, j], lats[3][i, j]], \
                    [lons[4][i, j], lats[4][i, j]], \
                    [lons[1][i, j], lats[1][i, j]], \
                ])
                    
                if j == 0 and i == ixs[0]:
                    label = "Bin %i edges" %k
                else:
                    label = ""
                    
                plt.plot(rectangle[:, 0], rectangle[:, 1], "C%i" %k, label=label)
                plt.scatter(lons[0][i, j], lats[0][i, j], c="C%i" %k, label=label.replace("edges", "centre"))
        
             
        
            # plt.scatter(lon_arr, lat_arr, c=np.repeat(bins, 2).reshape((-1, 2)))
    # plt.scatter(lon_arr[0, 0], lat_arr[0, 0], label=lon_arr[0, 0])
    # plt.scatter(lon_arr[1, 0], lat_arr[1, 0])
    plt.legend(loc="right")
    plt.grid()
    plt.title("%s unbinned geometry" %os.path.basename(new_hdf5_filepath))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    

        