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
from datetime import datetime

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


def get_data(hdf5_filepath_list, chosen_bin):
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
                bin_ixs = np.where(bins == unique_bins[chosen_bin])[0]
                
                
                y_err = h5_f["Science/YError"][:, 160:240]
                y_err_cov = h5_f["Science/YErrorCovariance"][:, 160:240]
            
                y_err_centre_mean = np.mean(y_err[bin_ixs, :], axis=1)
                y_err_cov_centre_mean = np.mean(y_err_cov[bin_ixs, :], axis=1)
                
                ratio = y_err_cov_centre_mean / y_err_centre_mean
                
                
                #remove very bad points
                if y_err_cov_centre_mean[-1] < 0.01:
                
                    d[h5_prefix] = {
                        "y_err_centre":y_err_centre_mean, 
                        "y_err_cov_centre":y_err_cov_centre_mean,
                        "ratio":ratio,
                        "n_frames":len(ratio)}
    return d


# d = get_data(hdf5_filepath_list, chosen_bin)


# plt.figure()
# colours = get_colours(len(d.keys()))
# for h5_ix, h5_prefix in enumerate(d.keys()):
#     plt.plot(d[h5_prefix]["ratio"], color=colours[h5_ix])
    
# plt.title("Covariance error vs YError for all SO egress order 190 observations")
# plt.grid()
# plt.xlabel("Frame index")
# plt.ylabel("Covariance error vs YError")



ratio_toa = [d[h5_prefix]["ratio"][-1] for h5_prefix in d.keys()]
y_err_toa = [d[h5_prefix]["y_err_centre"][-1] for h5_prefix in d.keys()]
y_cov_err_toa = [d[h5_prefix]["y_err_cov_centre"][-1] for h5_prefix in d.keys()]
dt = [datetime.strptime(h5_prefix, "%Y%m%d_%H%M%S") for h5_prefix in d.keys()]
colours = [colours[i] for i in range(len(d.keys()))]
n_frames = np.array([d[h5_prefix]["n_frames"] for h5_prefix in d.keys()])

#to plot colours better
n_frames_mod = np.copy(n_frames)
n_frames_mod[n_frames_mod > 400] = 400


plt.figure()
plt.grid()
plt.title("Covariance error vs YError for all SO egress order 190 observations")
plt.xlabel("Observation datetime")
plt.ylabel("Covariance error vs YError at top of atmosphere")
plt.scatter(dt, ratio_toa, c=np.log(n_frames_mod/max(n_frames_mod)))



plt.figure()
plt.grid()
plt.xlabel("YError")
plt.ylabel("Covariance error")
plt.scatter(y_err_toa, y_cov_err_toa, color=colours)

plt.figure()
plt.grid()
plt.title("YError")
plt.scatter(dt, y_err_toa, c=np.log(n_frames_mod/max(n_frames_mod)))


plt.figure()
plt.grid()
plt.title("Covariance error for order 190 egress observations")
plt.xlabel("Observation datetime")
plt.ylabel("Covariance error at top of atmosphere")
plt.scatter(dt, y_cov_err_toa, c=np.log(n_frames_mod/max(n_frames_mod)))
