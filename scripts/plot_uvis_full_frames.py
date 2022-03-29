# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:28:37 2021

@author: iant

PLOT UVIS TYPE 0 LIMBS - CHECK FOR SPECTRAL OSCILLATIONS
"""
import re
import numpy as np

import matplotlib.pyplot as plt
# import os
# import h5py
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours


obs_name = "20210725_050456"; indices = [12, 14]
# obs_name = "20210724_013356"; indices = [14, 21, 57, 70, 75, 77, 83]
# obs_name = "20210603_010708"; indices = []


regs = [
    re.compile("%s_0p3b_UVIS_O" %obs_name),
    # re.compile("%s_1p0a_UVIS_O" %obs_name)
    ]


h5s = []
h5_names = []
for regex in regs:   
#search for file and get it if found
    file_level = "hdf5_level_%s" %regex.pattern[16:20]
    
    hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)
    h5s.append(hdf5_files[0])
    h5_names.append(hdf5_filenames[0])


for hdf5_file, hdf5_filename in zip(h5s, h5_names):

    ff = hdf5_filename[16:20] in ["0p2a", "0p2b", "0p3b"]

    for ix in indices:
        if ff:
            i = ix + 2
        else:
            i = ix
    
        
        
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
            plt.figure(figsize=(12, 4), constrained_layout=True)
            plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
            
            frame = y[i, :, :]
            mean = np.nanmean(frame)
            std = np.nanstd(frame)
            plt.imshow(frame, vmin=(mean - std * 0.5), vmax=(mean + std * 0.5))
            plt.colorbar()
            
            
            
            
            #plot binned spectrum
            # x_range = np.arange((130-57), (210-57)) #illuminated region
            # plt.figure()
            # plt.title("Binned detector rows with masking")
            # plt.xlabel("Pixel number")
            # plt.ylabel("Detector counts")
            # plt.plot(np.nansum(y_masked[i, x_range, :], axis=0))

            # plt.figure(figsize=(12, 4), constrained_layout=True)
            # plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
            # line = frame[48, :]
            # mean = np.nanmean(line)
            # std = np.nanstd(line)
            # # line[line > (mean+std*3.)] = mean
            # # line[line < (mean-std*3.)] = mean
            # plt.plot(line)
            # sav_gol = savgol_filter(line, 39, 2)
            # plt.plot(sav_gol, label="Smoothed")

            # line = frame[49, :]
            # mean = np.nanmean(line)
            # std = np.nanstd(line)
            # # line[line > (mean+std*3.)] = mean
            # # line[line < (mean-std*3.)] = mean
            # plt.plot(line)
            # sav_gol = savgol_filter(line, 39, 2)
            # plt.plot(sav_gol, label="Smoothed")

            # line = frame[50, :]
            # mean = np.nanmean(line)
            # std = np.nanstd(line)
            # # line[line > (mean+std*3.)] = mean
            # # line[line < (mean-std*3.)] = mean
            # plt.plot(line)
            # sav_gol = savgol_filter(line, 39, 2)
            # plt.plot(sav_gol, label="Smoothed")

            plt.figure(figsize=(12, 4), constrained_layout=True)
            mean = np.nanmean(frame)
            std = np.nanstd(frame)
            frame[frame > (mean+std*1.)] = mean
            frame[frame < (mean-std*1.)] = mean
            
            frame = gaussian_filter(frame, 1)
            plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
            plt.imshow(frame)
            # plt.colorbar()

            plt.figure(figsize=(12, 4), constrained_layout=True)
            plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" %(dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i,0], lons[i,0]))
            for r in [48,49,50]:
                line = frame[r, :]
                mean = np.nanmean(line)
                std = np.nanstd(line)
                # line[line > (mean+std*3.)] = mean
                # line[line < (mean-std*3.)] = mean
                plt.plot(line)
                sav_gol = savgol_filter(line, 39, 2)
                plt.plot(sav_gol, label="Smoothed")
