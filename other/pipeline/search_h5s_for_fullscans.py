# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:03:54 2024

@author: iant

FIND NUMBER OF ORDERS IN A HDF5 FILE (E.G. FULLSCANS), OUTPUT FILENAME AND NUMBER OF ORDERS
"""

import re
import h5py 
import glob
import os
import platform

# from tools.file.hdf5_functions import make_filelist
from tools.general.cprint import cprint


#search 0.1d for _S files
# regex = re.compile(".*_SO_._S")
# # regex = re.compile("20230327_122435_.*_LNO_1_D._133")
# file_level = "hdf5_level_0p1d"

#search 0.1a for fullscans
regex = re.compile(".*_SO_.*")
file_level = "hdf5_level_0p1a"
file_level_short = file_level[-4:]

if platform.system() == "Windows":
    path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5" + os.sep + file_level# + r"\2018"
else:
    path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5" + os.sep + file_level# + r"\2018"


def get_fullscan_h5_prefixes(path, file_level, regex):

    print("Getting file list")
    h5_paths = sorted(glob.glob(path + os.sep + "**" + os.sep + "*.h5", recursive=True))
    print("%i files found" %len(h5_paths))
    
    h5_basenames = [os.path.basename(h5) for h5 in h5_paths]
    match_ixs = [i for i,s in enumerate(h5_basenames) if re.match(regex, s)]
    
    h5_prefixes = []
    n_freqs = []
    for match_ix in match_ixs:
        
        h5_path = h5_paths[match_ix]
        h5_basename = h5_basenames[match_ix]
        
        with h5py.File(h5_path) as h5_f:
            
            if "0p1a" in file_level:
                aotfs = h5_f["Channel/AOTFFrequency"][...]
                unique_orders = sorted(list(set(aotfs)))
            else:
                orders = h5_f["Channel/DiffractionOrder"][...]
                unique_orders = sorted(list(set(orders)))
                
    
        
        n_orders = len(unique_orders)
        
        text = "%s: unique AOTF frequencies = %i" %(h5_basename[0:15], n_orders)
        if n_orders > 6 and n_orders < 12:
            cprint(text, "y")
            h5_prefixes.append(h5_basename[0:15])
            n_freqs.append(n_orders)
    
        elif n_orders > 11 and n_orders < 28:
            cprint(text, "g")
            h5_prefixes.append(h5_basename[0:15])
            n_freqs.append(n_orders)
    
        elif n_orders > 27 and n_orders < 250:
            cprint(text, "r")
            h5_prefixes.append(h5_basename[0:15])
            n_freqs.append(n_orders)

        elif n_orders > 249:
            cprint(text, "b")
            h5_prefixes.append(h5_basename[0:15])
            n_freqs.append(n_orders)
    
        # else:
        #     print(text)
        
    return h5_prefixes, n_freqs
        
    
if "h5_prefixes" not in globals():
    h5_prefixes, n_freqs = get_fullscan_h5_prefixes(path, file_level, regex)
    

#change path to find matching file in another level

level_short_new = "0p1d"


path_new = path.replace(file_level_short, level_short_new)

print("Searching for matching files in another level")
for h5_prefix, n_freq in zip(h5_prefixes, n_freqs):
    year = h5_prefix[0:4]
    month = h5_prefix[4:6]
    day = h5_prefix[6:8]
    
    day_path_new = os.path.join(path_new, year, month, day) #note: may need to add/remove year month day depending on search path
    h5_paths = sorted(glob.glob(day_path_new + os.sep + "*.h5"))
    
    h5_basenames = [os.path.basename(h5) for h5 in h5_paths]
    
    matching_ixs = [i for i,s in enumerate(h5_paths) if h5_prefix in s]
    
    #get number of matching files in another level
    n_matches = len(matching_ixs)
    
    # text = "%s: %s" %(h5_prefix, [h5_basenames[i] for i in matching_ixs])
    #(number of freqs, number of files in level %s)
    text = "%s, %i, %i " %(h5_prefix, n_freq, n_matches)
    if ((n_freq > 27) and (n_matches > 27)):
        cprint(text, "r")
    elif ((n_freq == 12) and (n_matches != 12)):
        cprint(text, "r")
    elif ((n_freq == 16) and (n_matches != 16)):
        cprint(text, "r")
    elif ((n_freq == 11) and (n_matches != 11)):
        cprint(text, "r")

    
    
    if ((n_freq > 27) and (n_matches > 27)) or ((n_freq == 12) and (n_matches != 12)) or ((n_freq == 16) and (n_matches != 16)) or ((n_freq == 11) and (n_matches != 11)):
        with open("fullscans.txt", "a") as f:
            f.write(text+"\n")
    