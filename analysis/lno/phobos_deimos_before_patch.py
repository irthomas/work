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


SAVE_FIGS = True
# SAVE_FIGS = False

#set up subplots
fig = plt.figure(figsize=(16,10), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[1, 0])
ax1c = fig.add_subplot(gs[0, 1], sharey=ax1a)
ax1d = fig.add_subplot(gs[1, 1], sharey=ax1b)



ax = [[ax1a, ax1c], [ax1b, ax1d]]



file_info = {
    # "20210921_132947_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3.},
    # "20210921_132947_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3},
    
    # "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3.},
    # "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3},
    
    # "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[9, 35], "aspect":3.},
    # "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[6, 24], "aspect":0.3},
    
    # "20210927_224950_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[70, -1], "aspect":3.},
    # "20210927_224950_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[45, 80], "aspect":0.3},

    # "20211110_012749_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3., "boresight":"TGO -Y"},
    # "20211110_012749_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3, "boresight":"TGO -Y"},

    # "20211112_221419_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3., "boresight":"UVIS"},
    # "20211112_221419_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3, "boresight":"UVIS"},

    # "20211123_173753_0p2a_LNO_1_P":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3., "boresight":"LNO"},
    # "20211123_173753_0p2a_UVIS_P":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3, "boresight":"LNO"},

    # "20211229_072837_0p2a_LNO_1_Q":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3., "boresight":"LNO"},
    # "20211229_072837_0p2a_UVIS_Q":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3, "boresight":"LNO"},

    "20220102_015237_0p2a_LNO_1_Q":{"px_range":range(120, 280), "frame_range":[0, -1], "aspect":3., "boresight":"LNO"},
    "20220102_015237_0p2a_UVIS_Q":{"px_range":range(500, 1000), "frame_range":[0, -1], "aspect":0.3, "boresight":"LNO"},

    }
    
file_level = "hdf5_level_0p2a"

for i, filename in enumerate(file_info.keys()):
    regex = re.compile(filename) #(approx. orders 188-202) in steps of 8kHz
    
    hdf5_files, _, _ = make_filelist(regex, file_level) #open file, get matching filename
    
    hdf5_file = hdf5_files[0] #take first found file only
    
    frame_range = file_info[filename]["frame_range"] #define which frames to plot
    px_range = file_info[filename]["px_range"] #define which pixels to average together
    boresight = file_info[filename]["boresight"] #define which pixels to average together
    
    y_all = hdf5_file["Science/Y"][...]
    obs_times = [hdf5_file["Geometry/ObservationDateTime"][i, 0].decode()[:-7] for i in frame_range]
    
    
    y_mean_px = np.mean(y_all[frame_range[0]:frame_range[1], :, px_range], axis=2).T #mean of spectral pixels
    mask = np.all(np.isnan(y_mean_px) | np.equal(y_mean_px, 0), axis=1) #make nan row mask
    y_mean_px = y_mean_px[~mask].T #remove all rows with nans
    # y_mean_px = y_mean_px.T #no masking
    
    y_mean_frames = np.mean(y_mean_px, axis=0) #mean of chosen frames and pixels

    aspect = file_info[filename]["aspect"] #plotting aspect ratio
    
    fig.suptitle("Phobos observations by LNO and UVIS (%s boresight)" %boresight)

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
    
    if "UVIS" in filename:
        plt.figure(figsize=(10, 5), constrained_layout=True)
        
        #find row with highest signal
        mean = np.mean(y_all[frame_range[0]:frame_range[1], :, px_range], axis=(0,2))
        mean[np.isnan(mean)] = 0.
        max_row = np.argmax(mean)
        
        
        
        det_region = y_all[frame_range[0]:frame_range[1], (max_row-15):(max_row+15), 8:1032]
        spectrum = np.mean(det_region, axis=(0,1))
        plt.title("UVIS: Phobos uncalibrated spectrum, mean of detector rows %i-%i" %(max_row-15, max_row+15))
        plt.xlabel("Wavelength nm")
        plt.ylabel("ADU counts")
        plt.grid()
        
        #load any file to get wavenumbers
        import h5py
        with h5py.File(r"D:\DATA\hdf5\hdf5_level_1p0a\2019\02\02\20190202_013117_1p0a_UVIS_D.h5", "r") as h5:
            x = h5["Science/X"][...]
            plt.plot(x[0, :], spectrum)
        
        if SAVE_FIGS:
            plt.savefig("%s_UVIS_Phobos_uncalibrated_spectrum.png" %(filename[0:15]))

    if "LNO" in filename:
        plt.figure(figsize=(10, 5), constrained_layout=True)
        
        #find row with highest signal
        mean = np.mean(y_all[frame_range[0]:frame_range[1], :, px_range], axis=(0,2))
        mean[np.isnan(mean)] = 0.
        max_row = np.argmax(mean)
        
        
        #illuminated row
        det_region = y_all[frame_range[0]:frame_range[1], max_row, :]
        spectrum = np.mean(det_region, axis=0)
        
        #dark row
        dark_stds = []
        for row in range(y_all.shape[1]):
            if row not in [max_row-1, max_row, max_row+1]:
                det_region = y_all[frame_range[0]:frame_range[1], row, :]
                dark = np.mean(det_region, axis=0)
                
                if row == 9:
                    dark_to_plot = dark
        
                dark_mean = np.mean(dark)
                dark_stds.append(np.std(dark))
        
        dark_std = np.median(dark_stds)

        plt.title("LNO: Phobos uncalibrated spectrum, mean of detector row %i" %(max_row))
        plt.xlabel("Wavenumbers cm-1")
        plt.ylabel("ADU counts")
        plt.grid()
        
        #load any file to get wavenumbers
        import h5py
        with h5py.File(r"D:\DATA\hdf5\hdf5_level_1p0a\2018\05\02\20180502_041239_1p0a_LNO_1_DP_169.h5", "r") as h5:
            x = h5["Science/X"][...]
        plt.plot(x, spectrum, label="Phobos")
        plt.plot(x, dark_to_plot, label="Dark")
        plt.axhline(y=(dark_mean+dark_std), color="k", linestyle="--", label="Dark stdev = %0.2f" %dark_std)
        plt.axhline(y=(dark_mean-dark_std), color="k", linestyle="--")
        
        #bin dark and phobos spectra with 2cm res
        bin_size = 1 #cm-1
        bins = np.arange(x[0], x[-1], bin_size)
        x_means = np.mean([bins[1::], bins[:-1:]], axis=0)
        dig = np.digitize(x, bins)
        
        dark_values = [dark[np.where(dig == i)[0]] for i in range(1, len(bins))]
        dark_means = np.array([np.mean(a) for a in dark_values])
        dark_bin_std = np.std(dark_means)

        spectrum_values = [spectrum[np.where(dig == i)[0]] for i in range(1, len(bins))]
        spectrum_means = np.array([np.mean(a) for a in spectrum_values])


        plt.scatter(x_means, dark_means, c="k", marker="o", label="Dark (%icm-1 binned) stdev = %0.2f" %(bin_size, dark_bin_std))
        # plt.scatter(x_means, spectrum_means, c="k", marker="o")
        
        plt.legend()
        if SAVE_FIGS:
            plt.savefig("%s_LNO_Phobos_uncalibrated_spectrum.png" %(filename[0:15]))



if SAVE_FIGS:
    if frame_range == [0, -1]:
        fig.savefig("%s_phobos_observation_all.png" %(filename[0:15]))    
    else:  
         fig.savefig("%s_phobos_observation_%i-%i.png" %(filename[0:15], frame_range[0], frame_range[1]))
         
 
