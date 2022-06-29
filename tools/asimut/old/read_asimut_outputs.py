# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:12:39 2021

@author: iant


PLOT ASIMUT OUTPUT COMPARING DIFFERENT FILES

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

suffixes = ["_0", "_1", "_2", "_3"]
n_plots = len(suffixes)

variable = "FileAotfFilter"

spectral_lines_dict = {
    188:[4227.354, 4231.685, 4235.947, 4240.140, 4244.264, 4248.318, 4252.302, 4256.217],
    189:[4248.318, 4252.302, 4256.217, 4263.837, 4267.542, 4271.177, 4274.741, 4278.235],
    190:[4271.176, 4274.741, 4278.234, 4281.657, 4285.009, 4288.290, 4291.499, 4294.638, 4297.705, 4300.700, 4303.623],
}

order = int(hdf5_filename[-3:])
spectral_lines_nu = spectral_lines_dict[order]

asi_info_dicts = {}
asi_data_dicts = {}

for suffix in suffixes:
    
    asi_info = asi_info_dicts[suffix] = {}
    asi_data = asi_data_dicts[suffix] = {}
    
    asi_info["asi_dir"] = os.path.join(paths["RETRIEVAL_DIRECTORY"], *sub_dirs, "Retrievals", "Input")
    asi_info["asi"] = os.path.join(asi_info["asi_dir"], "%s%s.asi" %(hdf5_filename, suffix))
    
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
    
    asi_info["hdf5"] = os.path.join(asi_info["hdf5_dir"], "%s_corr.h5" %(hdf5_filename))
    
    
    
    inp_config = configparser.ConfigParser(allow_no_value=True)
    inp_config.read(asi_info["inp"])
    inp_config_dict = inp_config._sections
    
    asi_info["FileAotfFilter"] = os.path.basename(inp_config_dict["SP1"]["FileAotfFilter".lower()])
    
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
    
    asi_info["out"] = os.path.join(asi_info["out_dir"], "%s%s_out.h5" %(hdf5_filename, suffix))
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
    
    asi_data["y_all"] = y_all
    asi_data["yr_all"] = yr_all
    
    
    with h5py.File(asi_info["hdf5"], "r") as f2:
        
        x = f2["Science/X"][0,:]
        y_error_all = f2["Science/%s" %asi_info["YError"]][...]
        alts = np.mean(f2["Geometry/Point0/TangentAltAreoid"][...], axis=1)
    
    asi_data["x"] = x
    asi_data["y_error_all"] = y_error_all
    asi_data["alts"] = alts
    
    


    
with PdfPages("%s%s_retrieval.pdf" %(hdf5_filename, "")) as pdf: #open pdf
    for i in range(n):
        
        if np.mod(i, 10) == 0:
            print("%i/%i" %(i, n))

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(4, 2)
        ax1a = fig.add_subplot(gs[0, 0])
        ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)
        ax2a = fig.add_subplot(gs[0, 1])
        ax2b = fig.add_subplot(gs[1, 1], sharex=ax1a)
        ax3a = fig.add_subplot(gs[2, 0])
        ax3b = fig.add_subplot(gs[3, 0], sharex=ax1a)
        ax4a = fig.add_subplot(gs[2, 1])
        ax4b = fig.add_subplot(gs[3, 1], sharex=ax1a)
        
        axes = [[ax1a, ax1b], [ax2a, ax2b], [ax3a, ax3b], [ax4a, ax4b]]
        ax4b.set_xlabel("Wavenumber cm-1")
        ax2b.set_xlabel("Wavenumber cm-1")
        
        i_l1 = asi_info["indices"][i] #get index in l1.0a file
        
        for j, (suffix, ax_grp) in enumerate(zip(suffixes, axes)):
            for ax in ax_grp:
                ax.grid()
                for spectral_line_nu in spectral_lines_nu:
                    ax.axvline(spectral_line_nu, c="k", linestyle="--", alpha=0.5)
        
            asi_info = asi_info_dicts[suffix]
            asi_data = asi_data_dicts[suffix]

            if j == 0:
                fig.suptitle("%s%s: i=%i, %0.1fkm" %(hdf5_filename, suffix, i_l1, asi_data["alts"][i_l1]))
                
        
        
            ax_grp[0].plot(x, asi_data["y_all"][i, :], label="Y")
            ax_grp[0].plot(x, asi_data["yr_all"][i, :], label="Y retrieved")
            
            residual = asi_data["y_all"][i, :] - asi_data["yr_all"][i, :]
            if len(residual) == 320: #if all pixels
                chi_squared = np.sum(((asi_data["yr_all"][i, 50:300] - asi_data["y_all"][i, 50:300])**2) / asi_data["y_all"][i, 50:300])
            
            ax_grp[1].plot(x, residual, color="k", label="Residual")
            ax_grp[1].fill_between(x, asi_data["y_error_all"][i_l1, :], -asi_data["y_error_all"][i_l1, :], color="g", label=asi_info["YError"], alpha=0.3)
            
            ax_grp[0].set_title("%s" %asi_info["FileAotfFilter"])
            ax_grp[0].set_ylabel("Transmittance")
            ax_grp[1].set_ylabel("Residual")
            
            ax_grp[0].set_ylim((min(asi_data["y_all"][i, 20:])-0.01, max(asi_data["y_all"][i, 20:])+0.01))
            ax_grp[1].set_ylim((-0.015, 0.015))
            if len(residual) == 320: #if all pixels
                ax_grp[1].text(0.5, 0.05, "Chi-squared (pixels 50-300): %0.2e" %chi_squared, transform=ax_grp[1].transAxes, bbox=dict(ec='None', fc='white', alpha=0.8))
            
            ax_grp[0].legend(loc="upper right")
            ax_grp[1].legend(loc="upper right")
        pdf.savefig()
        plt.close()

