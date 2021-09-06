# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:22:04 2021

@author: iant

PLOT SPECTRA IN A FILE
"""
import re
# import numpy as np

import matplotlib.pyplot as plt
# import os
# import h5py
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist



#read in h5 file
regex = re.compile("20210715_210153_1p0a_UVIS_O") #(approx. orders 188-202) in steps of 8kHz

file_level="hdf5_level_1p0a"

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)

for hdf5_file, hdf5_filename in zip(hdf5_files, hdf5_filenames):

    observation_type = "O"
    ielo = observation_type in ["I", "E", "L", "O"]
    
    y = hdf5_file["Science/Y"][...]
    x = hdf5_file["Science/X"][0, :]
    
    n_spectra = y.shape[0]
    
    if ielo:
        
        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][...]
        # alts_mean = np.mean(alts, axis=1)
        
    with PdfPages("%s_spectra.pdf" %(hdf5_filename)) as pdf: #open pdf
            
        for i in range(n_spectra):
            plt.figure(figsize=(12, 4), constrained_layout=True)
            
            if ielo:
                plt.title("%s: i=%i, %0.1f-%0.1fkm" %(hdf5_filename, i, alts[i, 0], alts[i, 1]))
            else:
                plt.title("%s: i=%i" %(hdf5_filename, i))
                
            plt.plot(x, y[i, :], label = "Science/Y")
            
            sav_gol = savgol_filter(y[i, :], 39, 2)
            plt.plot(x, sav_gol, label="Smoothed")
            
            plt.grid()
            plt.legend(loc="lower right")
            
            pdf.savefig()
            plt.close()
        