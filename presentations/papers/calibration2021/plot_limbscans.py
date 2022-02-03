# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:11:13 2021

@author: iant

PLOT LNO LIMB SCAN
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file, get_files_from_datastore


limbscan_dict = {
    "LNO limb scan: scan direction parallel to long edge of FOV":["20170305_180300_0p1a_LNO_1"],
    "LNO limb scan: scan direction perpendicular to long edge of FOV":["20170306_180300_0p1a_LNO_1"],
}


bins_to_use = {
    1:"tab:blue",
    4:"tab:orange",
    7:"tab:green",
    10:"tab:red"
}


ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

frame_indices = np.arange(120, 410)

for limbscan_index, (title, hdf5_filenames) in enumerate(limbscan_dict.items()):
    
    
    for scan_index, hdf5_filename in enumerate(hdf5_filenames):

        
        try:
            hdf5_file = open_hdf5_file(hdf5_filename)
        except OSError:
            get_files_from_datastore([hdf5_filename])
            hdf5_file = open_hdf5_file(hdf5_filename)


        detector_data_all = hdf5_file["Science/Y"][...]
        # detector_data_bins = hdf5_file["Science/Bins"][...]
        datetime_all = hdf5_file["Geometry/ObservationDateTime"][...]
        window_top_all = hdf5_file["Channel/WindowTop"][...]
        window_height = hdf5_file["Channel/WindowHeight"][0]+1
        binning = hdf5_file["Channel/Binning"][0]+1
        sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
        aotf_freq_all = hdf5_file["Channel/AOTFFrequency"][...]
        

        print(window_top_all[0], window_height, binning, sbsf)
        
        # x = np.arange(detector_data_all.shape[0])
        n_bins = detector_data_all.shape[1]
        
        scaler=4.0 #conversion factor between the two diffraction orders
        # scaler=3.8 #conversion factor between the two diffraction orders
        
        
        aotf_freq_subd=aotf_freq_all[0]
        detector_data_bins = np.asfarray([frame*scaler if aotf_freq_all[index]==aotf_freq_subd else frame for index,frame in enumerate(list(detector_data_all))])
        
    
        chosen_range = [160,240]
        zero_indices = [*range(0,20), *range(300,320)] #for scaling all spectra to a common zero level on first and last pixels
        
        mean_offsets = np.zeros_like(detector_data_bins)
        
        mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
        for column_index in range(320):
            mean_offsets[:,:,column_index] = mean_offset
        
        offset_data_bins = detector_data_bins - mean_offsets
        
        spec_summed_data = np.sum(detector_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
        offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
        frame_range = np.arange(len(offset_spec_summed_data))
        
        plt.figure(figsize = (8.5, 4), constrained_layout=True)
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Sum of detector bin signal")
        
        
        for row_index, colour in bins_to_use.items(): #range(len(offset_spec_summed_data[0,:])): #plot each row separately
            summed_row = offset_spec_summed_data[:,row_index]
            plt.scatter((frame_indices - min(frame_indices))*7.5, summed_row[frame_indices], linewidth=0, color=colour, label="Detector %s bin" %ordinal(row_index+1))
            # plt.plot(frame_indices - min(frame_indices), summed_row[frame_indices])
        
        
            if hdf5_filename == "20170305_180300_0p1a_LNO_1":
                line_indices = {
                    1:[63, 168, 240],
                    4:[71, 160, 248],
                    7:[79, 152, 256],
                    10:[87, 144, 264],
                    }
            elif hdf5_filename == "20170306_180300_0p1a_LNO_1":
                line_indices = {
                    1:[70.5, 161., 245.],
                    4:[70.5, 161., 245.],
                    7:[70.5, 161., 245.],
                    10:[70.5, 161., 245.],
                }
            
            if row_index in line_indices:
                for vertical_line in line_indices[row_index]:
                    plt.axvline(x=vertical_line*7.5, color=colour, linestyle="--")
            
        plt.title(title)
        plt.legend()
        plt.grid()
        
        plt.savefig("lno_limb_scan_%s.png" %hdf5_filename, dpi=300)
