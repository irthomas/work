# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:25:03 2022

@author: iant

CALCULATE RATIO BETWEEN IAA COVARIANCE ERROR AND YERROR (WAS YERRORNORM)

"""

# import re
import numpy as np
import os
import glob
import h5py

import matplotlib.pyplot as plt
from tools.plotting.colours import get_colours

def progress(iterable, length=50):
    total = len(iterable)

    def printProgressBar (iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = "*" * filledLength + '-' * (length - filledLength)
        print(f'\r |{bar}| {percent}% ', end = "\r")

    printProgressBar(0)

    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()


# from tools.file.hdf5_functions import make_filelist, open_hdf5_file


# file_level = "hdf5_level_1p0a"
# regex = re.compile("202201.*_.*_1p0a_SO_A_E_189")

HDF5_DIRECTORY = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/test/iant/hdf5/hdf5_level_1p0a"
# HDF5_DIRECTORY = r"E:\DATA\hdf5\hdf5_level_1p0a"
#make filelists
hdf5_filepath_list = sorted(glob.glob(HDF5_DIRECTORY+r"/**/*.h5", recursive=True))

chosen_bin = 3

d = {}
for h5 in progress(hdf5_filepath_list):
    
    if "_1p0a_SO_A_E_190" in h5:

        h5_f = h5py.File(h5)
        h5_prefix = os.path.basename(h5)[0:15]
    
        if "YErrorCovariance" in h5_f["Science"].keys():
       
            # y = h5_f["Science/Y"][...]
            # y = h5_f["Science/YCovariance"][...]
            bins = h5_f["Science/Bins"][:, 0]
            unique_bins = sorted(list(set(bins)))
            bin_ixs = np.where(bins == unique_bins[chosen_bin])
            
            
            y_err = h5_f["Science/YError"][bin_ixs, 160:240]
            y_err_cov = h5_f["Science/YErrorCovariance"][bin_ixs, 160:240]
        
            y_err_centre_mean = np.mean(y_err, axis=1)
            y_err_cov_centre_mean = np.mean(y_err_cov, axis=1)
            
            ratio = y_err_cov_centre_mean / y_err_centre_mean
            
            d[h5_prefix] = {
                "y_err_centre":y_err_centre_mean, 
                "y_err_cov_centre":y_err_cov_centre_mean,
                "ratio":ratio}

colours = get_colours(len(d.keys()))
for h5_ix, h5_prefix in enumerate(d.keys()):
    plt.plot(d[h5_prefix]["ratio"], color=colours[h5_ix])