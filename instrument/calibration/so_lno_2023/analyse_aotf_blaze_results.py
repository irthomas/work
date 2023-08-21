# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:46:01 2023

@author: iant

PLOT AND ANALYSE MINISCAN AOTF AND BLAZES

"""


import os
import numpy as np
import matplotlib.pyplot as plt


from tools.plotting.colours import get_colours


channel = "so"
# channel = "lno"

AOTF_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_aotfs.txt" %channel)
BLAZE_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_blazes.txt" %channel)




#read in AOTFs
# aotfs = {"nu":[], "aotf":[]}
# with open(AOTF_OUTPUT_PATH, "r") as f:
#     lines = f.readlines()
    
# for line in lines:
#     line_split = line.split("\t")
#     n_nus = int(len(line_split)/2)
#     aotfs["nu"].append(np.asfarray(line_split[0:n_nus]))
#     aotfs["aotf"].append(np.asfarray(line_split[n_nus:]))
    
# aotfs["peak_ix"] = [np.argmax(aotf) for aotf in aotfs["aotf"]]
# aotfs["peak_nu"] = [nu[ix] for ix,nu in zip(aotfs["peak_ix"], aotfs["nu"])]
    
# plt.figure()
# plt.title("AOTF functions")
# plt.xlabel("Wavenumber")
# plt.ylabel("AOTF transmittance")
# plt.grid()
# for nu, aotf in zip(aotfs["nu"], aotfs["aotf"]):
#     plt.plot(nu, aotf)

# aotf_min = np.min(aotfs["peak_nu"])
# aotf_max = np.max(aotfs["peak_nu"])
# aotf_range = np.linspace(np.min(aotfs["peak_nu"]), np.max(aotfs["peak_nu"]), num=20)
# aotf_colours = get_colours(len(aotf_range), cmap="rainbow")

# plt.figure()
# plt.title("AOTF functions")
# plt.xlabel("Wavenumber")
# plt.ylabel("AOTF transmittance")
# plt.grid()
# for i, (nu, aotf) in enumerate(zip(aotfs["nu"], aotfs["aotf"])):
    
#     peak_nu = aotfs["peak_nu"][i]
#     colour = aotf_colours[np.searchsorted(aotf_range, peak_nu)]
#     aotf_peak_ixs = np.argmax(aotf)
#     plt.plot(nu - nu[aotf_peak_ixs], aotf, color=colour, alpha=0.1, label=peak_nu)
# plt.legend()




#read in blazes
blazes = {"aotf":[], "blaze":[]}
with open(BLAZE_OUTPUT_PATH, "r") as f:
    lines = f.readlines()
    
for line in lines:
    line_split = line.split("\t")
    blazes["aotf"].append(float(line_split[0]))
    blazes["blaze"].append(np.asfarray(line_split[1:]))
    
blaze_min = np.min(blazes["aotf"])
blaze_max = np.max(blazes["aotf"])
blaze_range = np.linspace(blaze_min, blaze_max, num=20)
blaze_colours = get_colours(len(blaze_range), cmap="rainbow")

mean_blaze = np.mean(np.asfarray(blazes["blaze"]), axis=0)

plt.figure()
plt.title("Blaze functions")
plt.xlabel("Pixel number")
plt.ylabel("Blaze function")
plt.grid()
for aotf_freq, blaze in zip(blazes["aotf"], blazes["blaze"]):
    colour = blaze_colours[np.searchsorted(blaze_range, aotf_freq)]
    plt.plot(np.linspace(0., 320., num=len(blaze)), blaze, color=colour, alpha=0.1, label=aotf_freq)
    plt.plot(np.linspace(0., 320., num=len(blaze)), mean_blaze, color="k", alpha=0.7)

plt.legend()

plt.figure()
colours = get_colours(20)
for i, px_ix in enumerate(np.linspace(0, len(blaze)-1, num=20)):
    points = np.asfarray(blazes["blaze"])[:, int(px_ix)]
    plt.scatter(blazes["aotf"], points / max(points), color=colours[i], label=px_ix)
plt.legend()