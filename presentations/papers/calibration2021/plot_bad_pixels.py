# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:13:01 2021

@author: iant

PLOT BAD PIXEL MAPS
"""

import numpy as np
import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares

import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths, FIG_X, FIG_Y


file_level = "hdf5_level_0p1a"
channel = "so"

if channel == "lno":
    regex = re.compile("20161125_(0[6-9]|1[0-3]).*")
    nIntTimes = 480

    min_gradient = 29.
    max_gradient = 40.
    min_chisq = 140000.
    max_chisq = 200000.
    aspect = 1
    extent = [0,256]

elif channel == "so":
    regex = re.compile("(20161125_020250|20161125_032550)_0p1a_SO_1")
    nIntTimes = 960

    min_gradient = 56.
    max_gradient = 60.
    min_chisq = 200000.
    max_chisq = 300000.
    aspect = 2
    extent = [128-50, 128+50]


#get files
hdf5_files, hdf5_filenames, titles = make_filelist(regex, file_level)



full_frame = np.zeros((nIntTimes,256,320)) * np.nan
# plt.figure(figsize=(FIG_X, FIG_Y))



for file_index, hdf5_file in enumerate(hdf5_files):
    y = hdf5_file["Science/Y"][...]
    
    integration_time = hdf5_file["Channel/IntegrationTime"][0]
    window_top = hdf5_file["Channel/WindowTop"][0]
    # frames_to_compare = range(80,100)
    
    if file_index==0:
        
        temperature = hdf5_file["Housekeeping/AOTF_TEMP_%s" %channel.upper()]
        nomad_temperature = np.mean(temperature[1:10])
    
    frame_to_plot=250
    
    window_bottom = window_top+24
    print(window_top, window_bottom)

    if file_index > 0:
        ratio_region_old = full_frame[frame_to_plot, window_top-1, 180:220]
        ratio_region_new = y[frame_to_plot, 0, 180:220]
        average_old_new_ratio = np.nanmean(ratio_region_old) / np.nanmean(ratio_region_new)

    else:
        average_old_new_ratio = 1.0
    corrected_detector_data = y[:,:,:] * average_old_new_ratio
    full_frame[:,window_top:window_bottom,:]=corrected_detector_data[:,:,:]
    
full_frame = full_frame - full_frame[0, :, :]
# full_frame_corrected = full_frame[frame_to_plot, :, :] / full_frame[frame_to_plot, 152, :]

plt.figure(figsize=(FIG_X, FIG_Y))
plt.title("MCO Integration Time Stepping")
plt.xlabel("Detector Column (Spectral Dimension)")
plt.ylabel("Detector Row (Spatial Dimension)")
# plt.imshow(np.log(full_frame[frame_to_plot,:,:]-0.90*np.nanmean(full_frame[frame_to_plot,:,:])),interpolation='none',cmap=plt.cm.gray)
plt.imshow(full_frame[frame_to_plot, :, :], interpolation='none', cmap=plt.cm.gray, origin="upper")
plt.colorbar()

plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(full_frame[frame_to_plot, 152, :])
plt.plot(full_frame[frame_to_plot, 151, :])
plt.plot(full_frame[frame_to_plot, :, 200])
# plt.plot(full_frame[:, 152, 200])


frame_gradient=np.zeros((256,320)) * np.nan
frame_chisq=np.zeros((256,320))


for i in range(full_frame.shape[1]):
    for j in range(full_frame.shape[2]):
        if ~np.isnan(full_frame[0, i , j]):
            gradient = np.polyfit(np.arange(240), full_frame[0:240, i, j], 1)[0]
            if gradient < min_gradient:
                frame_gradient[i, j] = max_gradient
                full_frame[100:110, i, j] = 10.0e9 #if abnormal gradient, make data really bad for chisq fit
            else:
                frame_gradient[i, j] = gradient
           
           
plt.figure(figsize=(FIG_X, FIG_Y))
plt.title("MCO Integration Time Stepping Gradient")
plt.xlabel("Detector Column (Spectral Dimension)")
plt.ylabel("Detector Row (Spatial Dimension)")
plt.imshow(frame_gradient, interpolation='none', cmap=plt.cm.gray, vmin=min_gradient, vmax=max_gradient, origin="upper")
plt.colorbar()

from scipy.stats import chisquare
frame_chisq=chisquare(full_frame[40:240, :, :], axis=0)[0]

masked_chisq = np.ma.masked_where(frame_chisq < max_chisq, frame_chisq) #masked array of very bad pixels
masked_chisq[121, 256] = 1.9e12

plt.figure(figsize=(FIG_X, FIG_Y))
plt.title("Integration Time Stepping: Chi Squared Deviation from Linear Fit")
plt.xlabel("Detector Column (Spectral Dimension)")
plt.ylabel("Detector Row (Spatial Dimension)")
plt.imshow(frame_chisq[extent[0]:extent[1],:], interpolation='none', cmap=plt.cm.gray, vmin=min_chisq, vmax=max_chisq, aspect=aspect, extent=(0,320,extent[1],extent[0]), origin="upper")
plt.colorbar()
plt.imshow(masked_chisq[extent[0]:extent[1],:], interpolation='none', cmap=plt.cm.autumn, vmin=1.0e13, vmax=1.1e13, aspect=aspect, extent=(0,320,extent[1],extent[0]), origin="upper") #make really bad pixels red
