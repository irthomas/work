# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:24:33 2021

@author: iant

CHECK PHOBOS / DEIMOS OBSERVATIONS
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist


#set up subplots
fig = plt.figure(figsize=(16,10), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[1, 0])
ax1c = fig.add_subplot(gs[0, 1], sharey=ax1a)
ax1d = fig.add_subplot(gs[1, 1], sharey=ax1b)

fig.suptitle("Phobos observations by LNO and UVIS")


ax = [[ax1a, ax1c], [ax1b, ax1d]]



file_info = {
    # "20210921_132947_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3.},
    # "20210921_132947_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3},
    
    # "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3.},
    # "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3},
    
    # "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[9, 35], "aspect":3.},
    # "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[6, 24], "aspect":0.3},
    
    "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[70, -1], "aspect":3.},
    "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[45, 80], "aspect":0.3},
    }
    
file_level = "hdf5_level_0p2a"

for i, filename in enumerate(file_info.keys()):
    regex = re.compile(filename) #(approx. orders 188-202) in steps of 8kHz
    
    hdf5_files, _, _ = make_filelist(regex, file_level) #open file, get matching filename
    
    hdf5_file = hdf5_files[0] #take first found file only
    
    frame_range = file_info[filename]["frame_range"] #define which frames to plot
    px_range = file_info[filename]["px_range"] #define which pixels to average together
    
    y_all = hdf5_file["Science/Y"][...]
    obs_times = [hdf5_file["Geometry/ObservationDateTime"][i, 0].decode()[:-7] for i in frame_range]
    
    
    y_mean_px = np.mean(y_all[frame_range[0]:frame_range[1], :, px_range], axis=2).T #mean of spectral pixels
    mask = np.all(np.isnan(y_mean_px) | np.equal(y_mean_px, 0), axis=1) #make nan row mask
    y_mean_px = y_mean_px[~mask].T #remove all rows with nans
    # y_mean_px = y_mean_px.T #no masking
    
    y_mean_frames = np.mean(y_mean_px, axis=0) #mean of chosen frames and pixels

    aspect = file_info[filename]["aspect"] #plotting aspect ratio
    
    
    ax[i][0].set_title("%s: spectrally averaged counts (columns %i-%i)\n%s - %s" 
                       %(filename, min(px_range), max(px_range), obs_times[0], obs_times[1]))
    ax[i][1].set_title("%s: spectrally averaged counts (columns %i-%i)\n%s - %s" 
                       %(filename, min(px_range), max(px_range), obs_times[0], obs_times[1]))
    ax[i][0].imshow(y_mean_px[:, :].T, aspect=aspect)
    
    ax[i][1].plot(y_mean_frames, np.arange(len(y_mean_frames)))
    
    ax[i][0].set_ylabel("Detector row (offset)")
    ax[i][1].set_ylabel("Detector row (offset)")
    ax[i][0].set_xlabel("Detector frame number")
    ax[i][1].set_xlabel("Mean counts of all frames")
    
    ax[i][0].grid()
    ax[i][1].grid()
    
    # if "LNO" in filename:
    #     plt.figure()
    #     plt.plot(np.mean(y_all[frame_range[0]:frame_range[1], 11, :], axis=0), "k--")
    #     plt.plot(np.mean(y_all[frame_range[0]:frame_range[1], 10, :], axis=0), "k.")
    #     plt.plot(np.mean(y_all[frame_range[0]:frame_range[1], 12, :], axis=0), "k.")
    
    
plt.savefig("%s_Phobos_Observation_%i-%i.png" %(filename[0:15], frame_range[0], frame_range[1]))