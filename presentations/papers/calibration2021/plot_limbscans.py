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
    "LNO limb scan: direction parallel to long edge of FOV":["20170305_180300_0p1a_LNO_1"],
    "LNO limb scan: direction perpendicular to long edge of FOV":["20170306_180300_0p1a_LNO_1"],
}


bins_to_use = [1,4,7,10]

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
        
        x = np.arange(detector_data_all.shape[0])
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
        
        plt.figure(figsize = (6, 5), constrained_layout=True)
        plt.xlabel("Frame Number")
        plt.ylabel("Sum of detector bin signal")
        
        
        for row_index in bins_to_use: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
            summed_row = offset_spec_summed_data[:,row_index]
            plt.scatter(x ,summed_row, linewidth=0, label="Detector bin %i" %row_index)
        
        plt.title(title)
        plt.legend()
        plt.grid()
        
        plt.savefig("lno_limb_scan_%s.png" %hdf5_filename, dpi=300)
