# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:41:36 2021

@author: iant

PLOT FULLSCAN FULL SPECTRAL RANGE FOR PAPER
"""


import os
import numpy as np
import re

from tools.spectra.baseline_als import baseline_als
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from tools.file.hdf5_functions import make_filelist



# SAVE_FIGS = False
SAVE_FIGS = True


regex = re.compile("20190117_061101.*_SO_.*_S")
fileLevel = "hdf5_level_1p0a"


"""plot miniscans"""
hdf5Files, hdf5Filenames, _ = make_filelist(regex, fileLevel, silent=True)

hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]

plt.figure(figsize=(16,5), constrained_layout=True)
alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
half_length = len(alts)

isMerged = False
if alts[0] > 250.0 and alts[-1] > 250.0:
    isMerged = True
    half_length = int(len(alts)/2)

if isMerged:
    alts = alts[0:half_length]
    
y = hdf5_file["Science/Y"][0:half_length, 50:]
x = hdf5_file["Science/X"][0:half_length, 50:]
diffraction_orders = hdf5_file["Channel/DiffractionOrder"][0:half_length]
bins = hdf5_file["Science/Bins"][0:half_length, 0]

y_mean = np.mean(y, axis=1)

chosen_bin = 124

unique_orders = sorted(list(set(diffraction_orders)))[5:95]
for chosen_order in unique_orders:

    if chosen_order in [119, 120, 121]:
        colour = "C0"
        label = "HDO lines"
    elif chosen_order in [127, 128, 129, 130]:
        colour = "C1"
        label = "HCl lines"
    elif chosen_order in [133, 134, 135, 136]:
        colour = "C2"
        label = "H$\mathregular{_2}$O lines"
    elif chosen_order in [147, 148, 149]:
        colour = "C3"
        label = "CO$\mathregular{_2}$ lines"
    elif chosen_order in list(range(157, 167)):
        colour = "C4"
        label = "CO$\mathregular{_2}$ lines"
    elif chosen_order in [167, 168, 169, 170, 171]:
        colour = "C5"
        label = "H$\mathregular{_2}$O lines"
    elif chosen_order in [186, 192, 193]:
        colour = "C6"
        label = "CO lines"
    elif chosen_order in [187, 188, 189, 190, 191]:
        colour = "C8"
        label = "CO lines"
    else:
        colour = "grey"
        
    if np.mod(chosen_order, 2) == 0:
        r_colour = "blue"
        offset = 0.52
    else:
        r_colour = "red"
        offset = 0.55
        

    ix_order = np.where((chosen_order == diffraction_orders) & (chosen_bin == bins))[0]
    ix_signal = np.where((y_mean > 0.5) & (y_mean < 0.9))
    ix = np.intersect1d(ix_order, ix_signal)
    
    i = ix[0]

    plt.gca().add_patch(Rectangle((x[i, 0], 0.4), x[i, -1] - x[i, 0], 0.7, color=r_colour, alpha=0.1))
    plt.text(x[i, 0], offset, chosen_order, color=r_colour)
        
    
    y_cont = baseline_als(y[i, :], lam=250.0, p=0.99)
    y_cr = y[i, :] / y_cont

    if colour == "grey" or chosen_order not in [119, 127, 133, 147, 157, 167, 186, 187]:
        plt.plot(x[i, :], y_cr/np.max(y_cr), color=colour)
    else:
        plt.plot(x[i, :], y_cr/np.max(y_cr), color=colour, label=label)


plt.xlim((2580, 4625))
plt.ylim((0.5, 1.01))
plt.ylabel("Normalised transmittance")
plt.xlabel("Wavenumber (cm$\mathregular{^{-1}}$)")
plt.title("SO channel: transmittance of all diffraction orders")
plt.grid()
plt.legend(loc="center right")

if SAVE_FIGS:
    plt.savefig(os.path.join("so_fullscan_%s" %hdf5_filename), dpi=300)

