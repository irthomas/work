# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:28:37 2021

@author: iant

PLOT UVIS TYPE 0 LIMBS - CHECK AURORAL EMISSION
"""
import re
import numpy as np

import matplotlib.pyplot as plt
# import os
# import h5py
from scipy.signal import savgol_filter

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours

#select file, level, index of observation (note index will change between 0.3B and later levels) and if data is full frame
input_dicts = {
    # "20210603_010708_0p2a_UVIS_O":{"file_level":"hdf5_level_0p2a", "chosen_idx":43, "ff":True},
    # "20210603_010708_0p3b_UVIS_O":{"file_level":"hdf5_level_0p3b", "chosen_idx":43, "ff":True},
    # "20210603_010708_0p3c_UVIS_O":{"file_level":"hdf5_level_0p3c", "chosen_idx":41, "ff":False},
    # "20210603_010708_1p0a_UVIS_O":{"file_level":"hdf5_level_1p0a", "chosen_idx":41, "ff":False},

    # "20210724_013356_0p2a_UVIS_O":{"file_level":"hdf5_level_0p2a", "chosen_idx":88, "ff":True},
    # "20210724_013356_0p3b_UVIS_O":{"file_level":"hdf5_level_0p3b", "chosen_idx":88, "ff":True},
    # "20210724_013356_0p3c_UVIS_O":{"file_level":"hdf5_level_0p3c", "chosen_idx":86, "ff":False},
    # "20210724_013356_1p0a_UVIS_O":{"file_level":"hdf5_level_1p0a", "chosen_idx":86, "ff":False},


    # "20210725_050456_0p2a_UVIS_O":{"file_level":"hdf5_level_0p2a", "chosen_idx":86, "ff":True},
    "20210725_050456_0p3b_UVIS_O":{"file_level":"hdf5_level_0p3b", "chosen_idx":86, "ff":True},
    # "20210725_050456_0p3c_UVIS_O":{"file_level":"hdf5_level_0p3c", "chosen_idx":84, "ff":False},
    # "20210725_050456_1p0a_UVIS_O":{"file_level":"hdf5_level_1p0a", "chosen_idx":84, "ff":False},


}

#loop through chosen files
for regex_txt, input_dict in input_dicts.items():
    
    file_level = input_dict["file_level"]
    i = input_dict["chosen_idx"]
    ff = input_dict["ff"]
    
    #search for file and get it if found
    regex = re.compile(regex_txt)
    hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)
    hdf5_file = hdf5_files[0]
    hdf5_filename = hdf5_filenames[0]

    #get data from file
    y = hdf5_file["Science/Y"][...]
    y_mask = np.asfarray(hdf5_file["Science/YMask"][...])
    
    #choose mask: >1.1 = bad pixels only; >0.9 for hot and bad pixels
    # y_mask2 = np.where(y_mask > 1.1)
    y_mask_tf = np.where(y_mask > 0.9)
    
    #make detector array with masked pixels set to NaN
    y_masked = np.copy(y)
    y_masked[y_mask_tf] = np.nan
    
    
    n_spectra = y.shape[0]
    x = np.arange(y.shape[1])
    
    #get aux info from file
    alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][...]

    lats = hdf5_file["Geometry/Point0/Lat"][...]
    lons = hdf5_file["Geometry/Point0/Lon"][...]
    dates = hdf5_file["Geometry/ObservationDateTime"][...]
        
            
    
    if not ff:
    
        #if not full frame, just plot the spectrum e.g. 0.3C or 1.0A data
        plt.figure(figsize=(12, 3), constrained_layout=True)
        plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
        plt.plot(x, y[i, :], label = "Science/Y")
        
        #apply sav gol smoothing filter, plot smoothed line on figure
        sav_gol = savgol_filter(y[i, :], 39, 2)
        plt.plot(x, sav_gol, label="Smoothed")
        plt.ylabel("Counts")
        plt.xlabel("Pixel Number")
        plt.savefig("%s.png" %hdf5_filename)
        
    else:
        #plot detector frame
        # plt.figure(figsize=(12, 4), constrained_layout=True)
        # plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
        # plt.imshow(y[i, :, :])
        
        #plot binned spectrum
        x_range = np.arange((130-57), (210-57)) #illuminated region
        plt.figure()
        plt.title("Binned detector rows with masking")
        plt.xlabel("Pixel number")
        plt.ylabel("Detector counts")
        plt.plot(np.nansum(y_masked[i, x_range, :], axis=0))

        #plot individual detector columns
        columns_to_plot = [789, 790, 791, 792, 793, 794, 795] #detector columns to plot
        colours = get_colours(len(columns_to_plot))

        plt.figure(figsize=(12, 3.5), constrained_layout=True)
        plt.title("%s: Detector columns before masking" %(dates[i, 0].decode()))
        plt.xlabel("Detector row")
        plt.ylabel("Detector counts")
        for j, colour in zip(columns_to_plot, colours):
            plt.plot(x_range, y[i, x_range, j], label="Detector column %i" %j, color=colour)
            # plt.scatter(x_range, y_mask[i, x_range, j] * 100.0, color=colour)
            plt.axhline(y=np.nanmean(y[i, x_range, j]), linestyle="--", color=colour)
        plt.legend()
        plt.savefig("%s_unmasked.png" %hdf5_filename)
        
        plt.figure(figsize=(12, 3.5), constrained_layout=True)
        plt.title("%s: Detector columns after masking" %(dates[i, 0].decode()))
        plt.xlabel("Detector row")
        plt.ylabel("Detector counts")
        for j, colour in zip(columns_to_plot, colours):
            plt.plot(x_range, y_masked[i, x_range, j], label="Detector column %i" %j, color=colour)
            # plt.scatter(x_range, y_mask[i, x_range, j] * 100.0, color=colour)
            plt.axhline(y=np.nanmean(y_masked[i, x_range, j]), linestyle="--", color=colour)
        plt.legend()
        plt.savefig("%s_masked.png" %hdf5_filename)
        
        
        # plt.imshow(y[i, 100:240, :], aspect=5)
        # plt.xlabel("Pixel Number")
        # plt.ylabel("Detector Row (offset by 100)")
        
        # plt.scatter(np.arange(100,240), y[i, 100:240, 800])
        # plt.plot(np.arange(100,240), y[i, 100:240, 800])
        # plt.xlabel("Detector row (no offset)")
        # plt.ylabel("Counts")
        # plt.title("Typical UVIS nadir spectrum")
