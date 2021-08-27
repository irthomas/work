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

from matplotlib.backends.backend_pdf import PdfPages

from tools.file.paths import paths
from tools.plotting.colours import get_colours


sub_dirs = ["so_aotf_ils", "CO"]

hdf5_filename = "20180930_113957_1p0a_SO_A_I_189"

spectral_lines_dict = {
    188:[4227.354, 4231.685, 4235.947, 4240.140, 4244.264, 4248.318, 4252.302, 4256.217],
    189:[4248.318, 4252.302, 4256.217, 4263.837, 4267.542, 4271.177, 4274.741, 4278.235],
    190:[4271.176, 4274.741, 4278.234, 4281.657, 4285.009, 4288.290, 4291.499, 4294.638, 4297.705, 4300.700, 4303.623],
}


order = int(hdf5_filename[-3:])

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


order = int(hdf5_filename[-3:])
spectral_lines_nu = spectral_lines_dict[order]


with PdfPages("%s_retrieval.pdf" %hdf5_filename) as pdf: #open pdf
    for i in range(n):
        
        if np.mod(i, 10) == 0:
            print("%i/%i" %(i, n))
        fig = plt.figure(figsize=(10,8), constrained_layout=True)
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        
        ax1.grid()
        ax2.grid()
        
        for spectral_line_nu in spectral_lines_nu:
            ax1.axvline(spectral_line_nu, c="k", linestyle="--", alpha=0.5)
            ax2.axvline(spectral_line_nu, c="k", linestyle="--", alpha=0.5)


        i_l1 = asi_info["indices"][i]
        
        ax1.plot(x, y_all[i, :], label="Y")
        ax1.plot(x, yr_all[i, :], label="Y retrieved")
        
        residual = y_all[i, :] - yr_all[i, :]
        if len(residual) == 320: #if all pixels
            chi_squared = np.sum(((yr_all[i, 50:300] - y_all[i, 50:300])**2) / y_all[i, 50:300])
        
        ax2.plot(x, residual, color="k", label="Residual")
        ax2.fill_between(x, y_error_all[i_l1, :], -y_error_all[i_l1, :], color="g", label=asi_info["YError"], alpha=0.3)
        
        ax2.set_xlabel("Pixel")
        ax1.set_ylabel("Transmittance")
        ax2.set_ylabel("Residual")
        ax1.set_title("%s: i=%i, %0.1fkm" %(hdf5_filename, i_l1, alts[i_l1]))
        
        ax1.set_ylim((min(y_all[i, 20:])-0.01, max(y_all[i, 20:])+0.01))
        # ax2.set_ylim((min(residual[20:])-0.003, max(residual[20:])+0.003))
        ax2.set_ylim((-0.015, 0.015))
        if len(residual) == 320: #if all pixels
            ax2.text(0.65, 0.028, "Chi-squared (pixels 50-300): %0.2e" %chi_squared, transform=ax2.transAxes, bbox=dict(ec='None', fc='white', alpha=0.8))
        
        ax1.legend()
        ax2.legend()
        pdf.savefig()
        plt.close()

