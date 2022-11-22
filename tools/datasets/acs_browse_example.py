# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:58:03 2022

@author: iant
"""


import matplotlib.pyplot as plt
import numpy as np


transmittance = np.arange(150) / 150
transmittance_array = np.reshape(np.repeat(transmittance, 640), (150, 640))

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(13, 7))

for i in range(5): #loop through 5 columns
    for j in range(2): #loop through 2 rows
        im = axes[j, i].imshow(transmittance_array, origin="lower", aspect="auto")
        axes[j, i].set_title("Order %i" %((i+1)*(j+1)))
        
        if i == 0:
            axes[j,i].set_ylabel("Frame index")
            
        if i > 0:
            axes[j, i].set_yticklabels([])

        if j == 0:
            axes[j, i].set_xticklabels([])

        if j == 1:
            axes[j, i].set_xticks([0, 319, 639])
            axes[j, i].set_xticklabels([1, 320, 640])
            axes[1, i].set_xlabel("Pixel number")

        
        


fig.suptitle("acs_cal_sc_nir_20190301T201042-20190301T201643-5666-1-2")

#add colourbar
cbar_axis = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cbar_axis)
cbar.set_label("Transmittance", rotation=270, labelpad=20)
cbar.set_ticks(np.arange(10.0)/10.0)
