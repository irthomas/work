# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:22:09 2021

@author: iant

PLOT ASIMUT OUTPUT WITH ILS AND AOTF FIT

"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import configparser
import h5py

from tools.file.paths import paths
from tools.plotting.colours import get_colours


sub_dirs = ["so_aotf_ils", "CO"]

hdf5_filename = "20180930_113957_1p0a_SO_A_I_189"


#read in 


asi_info = {
    "asi_dir":os.path.join(paths["RETRIEVAL_DIRECTORY"], *sub_dirs, "Retrievals", "Input")
}
asi_info["asi"] = os.path.join(asi_info["asi_dir"], "%s.asi" %hdf5_filename)

asi_config = configparser.ConfigParser(allow_no_value=True)
asi_config.read(asi_info["asi"])
asi_config_dict = asi_config._sections



for asi_name, dict_name in {"dirinput":"input_dir", "dirresults":"out_dir", "dirsave":"save_dir", "dirspectra":"hdf5_dir"}.items():
    dir_linux = asi_config_dict["Directories"][asi_name]
    sub_dir_ix = dir_linux.find(sub_dirs[0])
    if sub_dir_ix > 0:
        dir_win = os.path.join(paths["RETRIEVAL_DIRECTORY"], *dir_linux[sub_dir_ix:].split("/"))
        asi_info[dict_name] = dir_win                     

asi_info["inp"] = []
for inp_name in asi_config_dict["List"].keys():
    asi_info["inp"].append(os.path.join(asi_info["input_dir"], inp_name))

if len(asi_info["inp"]) == 1: 
    asi_info["inp"] = asi_info["inp"][0]

asi_info["hdf5"] = os.path.join(asi_info["hdf5_dir"], hdf5_filename+"_corr.h5")



inp_config = configparser.ConfigParser(allow_no_value=True)
inp_config.read(asi_info["inp"])
inp_config_dict = inp_config._sections


for asi_name, dict_name in {"fileaotffilter":"aotf", "fileilsparam":"ils"}.items():
    dir_linux = inp_config_dict["SP1"][asi_name]
    sub_dir_ix = dir_linux.find(sub_dirs[0])
    if sub_dir_ix > 0:
        dir_win = os.path.join(paths["RETRIEVAL_DIRECTORY"], *dir_linux[sub_dir_ix:].split("/"))
        asi_info[dict_name] = dir_win                     

asi_info["A_nu0"] = float(inp_config_dict["SP1"]["aotfcentralwnb"])

vals = inp_config_dict["SP1"]["spectraid_list"]
vals = vals.replace("val[ ","").replace("]","").split(" ")
asi_info["indices"] = [int(v) for v in vals]

asi_info["out"] = os.path.join(asi_info["out_dir"], "%s_out.h5" %hdf5_filename)
# plt.figure()
#get data from output file
with h5py.File(asi_info["out"], "r") as f:

    y_all = np.zeros((len(f.keys()), 320))
    yr_all = np.zeros((len(f.keys()), 320))
    for key in f.keys():
        ix = int(key.replace("Pass_","")) - 1
        y = f[key]["Fit_0"]["Y"][...]
        yr = f[key]["Fit_0"]["YCALC"][...]
        
        y_all[ix, :] = y
        yr_all[ix, :] = yr


# with h5py.File(asi_info["hdf5"], "r") as f2:
    
#     y_error = 

# fig, ax = plt.subplots(figsize=(10,5))

# fig = plt.figure(constrained_layout=True)
fig = plt.figure(figsize=(10,5))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[0:2, 0])
ax2 = fig.add_subplot(gs[2, 0])


i = 109
ax1.plot(y_all[i, :], label="Y")
ax1.plot(yr_all[i, :], label="Y retrieved")

residual = y_all[i, :] - yr_all[i, :]
ax2.plot(residual, label="Residual")

ax2.set_xlabel("Pixel")
ax1.set_ylabel("Transmittance")
ax2.set_ylabel("Residual")
ax1.set_title(hdf5_filename)
fig.savefig("%s_retrieval.png" %hdf5_filename)