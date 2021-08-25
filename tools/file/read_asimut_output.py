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
asi_info["YError"] = inp_config_dict["SP1"]["datayerrorselect"]

vals = inp_config_dict["SP1"]["spectraid_list"]
vals = vals.replace("val[ ","").replace("]","").split(" ")
asi_info["indices"] = [int(v) for v in vals]

asi_info["out"] = os.path.join(asi_info["out_dir"], "%s_out.h5" %hdf5_filename)
# plt.figure()
#get data from output file
with h5py.File(asi_info["out"], "r") as f:
    
    n = len(f.keys())

    y_all = np.zeros((n, 320))
    yr_all = np.zeros((n, 320))
    
    # passes = sorted(list(f.keys()))
    # pass_nos = sorted([int(s.strip("Pass_")) for s in passes])
    # for pass_no in pass_nos:
        
    #     pass_name = "Pass_%i" %pass_no

    for key in f.keys():
        ix = int(key.replace("Pass_",""))
        y = f[key]["Fit_0"]["Y"][...]
        yr = f[key]["Fit_0"]["YCALC"][...]
        y_all[n - ix, :] = y
        yr_all[n - ix, :] = yr


with h5py.File(asi_info["hdf5"], "r") as f2:
    
    x = f2["Science/X"][0,:]
    y_error_all = f2["Science/%s" %asi_info["YError"]][...]
    alts = np.mean(f2["Geometry/Point0/TangentAltAreoid"][...], axis=1)

# fig, ax = plt.subplots(figsize=(10,5))

# fig = plt.figure(constrained_layout=True)
fig = plt.figure(figsize=(10,8), constrained_layout=True)
gs = fig.add_gridspec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

ax1.grid()
ax2.grid()


i = 100
i_l1 = asi_info["indices"][i]

ax1.plot(x, y_all[i, :], label="Y")
ax1.plot(x, yr_all[i, :], label="Y retrieved")

residual = y_all[i, :] - yr_all[i, :]
ax2.plot(x, residual, color="k", label="Residual")
ax2.fill_between(x, y_error_all[i_l1, :], -y_error_all[i_l1, :], color="g", label=asi_info["YError"], alpha=0.3)

ax2.set_xlabel("Pixel")
ax1.set_ylabel("Transmittance")
ax2.set_ylabel("Residual")
ax1.set_title("%s: i=%i, %0.1fkm" %(hdf5_filename, i, alts[i_l1]))

ax1.set_ylim((min(y_all[i, 20:])-0.01, max(y_all[i, 20:])+0.01))
ax2.set_ylim((min(residual[20:])-0.003, max(residual[20:])+0.003))
# ax2.set_ylim((-y_error_all[i_l1, 50], y_error_all[i_l1, 50]))

ax1.legend()
ax2.legend()
fig.savefig("%s_retrieval.png" %hdf5_filename)