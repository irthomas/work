# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:05:29 2022

@author: iant
"""

import re
import numpy as np
from matplotlib import pyplot as plt


from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours


from instrument.nomad_so_instrument_v01 import m_aotf

file_level = "hdf5_level_0p1a"

# regex = re.compile("20150404_(07|08|09|10|11|12)...._.*") #all observations
regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)
h5_files, h5_filenames, _ = make_filelist(regex, file_level)

for h5_f, h5 in zip(h5_files, h5_filenames):
    
    y = h5_f["Science/Y"][...]
    aotf_f = h5_f["Channel/AOTFFrequency"][...]
    unique_freqs = sorted(list(set(aotf_f)))
    orders = np.array([m_aotf(i) for i in aotf_f])
    unique_orders = sorted(list(set(orders)))

    
    bin_ = 12
    
    y_bin_mean = np.mean(y, axis=1)
    
    
    # colours = get_colours(len(unique_orders))
    
    plt.figure(figsize=(15,10))
    plt.title("%s: CH4 gas cell and atmospheric lines" %h5)
    
    measured_orders = []
    for i, y_spectrum in enumerate(y_bin_mean):
        
        order = orders[i]
        order_ix = np.where(order == unique_orders)[0][0]
        
        if order not in measured_orders:
            measured_orders.append(order)
            label = "Order %i" %order
        else:
            label = ""
        
        plt.plot(y_spectrum, color="C%i" %order_ix, label=label, alpha=0.3)
    plt.legend()
    plt.grid()
    plt.xlabel("Pixel number")
    plt.ylabel("Signal (counts)")
    
