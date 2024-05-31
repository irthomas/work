# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:04:49 2024

@author: iant

COUNT NUMBER OF SOLAR OCCULTATIONS AND NUMBER OF SPECTRA
"""

import re
import h5py 
import glob
import os
import platform
import numpy as np

# from tools.file.hdf5_functions import make_filelist
# from tools.general.cprint import cprint



#search 1.0a for normal occultations
# regex = re.compile("20240101_.*_SO_._._\d*")

regex = re.compile("20......_.*_SO_._._\d*")
file_level = "hdf5_level_1p0a"

if platform.system() == "Windows":
    path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5" + os.sep + file_level# + r"\2018"
else:
    path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5" + os.sep + file_level# + r"\2018"


# def get_fullscan_h5_prefixes(path, file_level, regex):

    
def get_all_h5_paths(path):
    print("Getting file list")
    h5_paths = sorted(glob.glob(path + os.sep + "**" + os.sep + "*.h5", recursive=True))
    print("%i files found" %len(h5_paths))
    
    return h5_paths

if "h5_paths" not in globals():
    h5_paths = get_all_h5_paths(path)


h5_basenames = [os.path.basename(h5) for h5 in h5_paths]
match_ixs = [i for i,s in enumerate(h5_basenames) if re.match(regex, s)]
print("%i matching files found" %len(match_ixs))


print("Checking the number of spectra in each file")
h5_dict = {}
for i, match_ix in enumerate(match_ixs):
    
    if np.mod(i, 1000) == 0:
        print("%i/%i" %(i, len(match_ixs)))

        
    h5_path = h5_paths[match_ix]
    h5_basename = h5_basenames[match_ix]
        
    with h5py.File(h5_path) as h5_f:
        
        n_spec = h5_f.attrs["NSpec"]
        h5_dict[h5_basename] = int(n_spec)
        
        # aotfs = h5_f["Channel/AOTFFrequency"][...]
        # h5_dict[h5_basename] = len(aotfs)



#count number of spectra
print("Counting total spectra")
n_spectra = 0

for i, h5 in enumerate(h5_dict.keys()):

    if np.mod(i, 1000) == 0:
        print("%i/%i" %(i, len(match_ixs)))


    n_spectra += h5_dict[h5]
    
print("%i spectra in files" %n_spectra)