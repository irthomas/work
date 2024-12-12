# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:54:23 2024

@author: iant

Check diagonally corrected data, to find issues and improve the correction if needed
"""

import os
from astropy.io import fits
# import numpy as np
import matplotlib.pyplot as plt

# channel = "so"
channel = "lno"

# search for miniscan files with the following characteristics
# aotf step in kHz
aotf_steppings = [4]
# diffraction order of first spectrum in file
starting_orders = list(range(163, 210))
# starting_orders = [164]


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")


# check for data available in miniscan dir
filenames = os.listdir(os.path.join(MINISCAN_PATH, channel))
# list all fits files
h5_prefixes = [s.replace(".fits", "") for s in filenames if ".fits" in s and s]
# list those with chosen AOTF stepping (in KHz)
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-1]) in aotf_steppings]
# list those with chosen aotf diffraction orders
h5_prefixes = [s for s in h5_prefixes if int(s.split("-")[-2]) in starting_orders]
print("%i files found matching the desired stepping and diffraction order start" % len(h5_prefixes))


for file_ix, h5_prefix in enumerate(h5_prefixes[0:5]):  # loop through files
    print("%i/%i: %s" % (file_ix+1, len(h5_prefixes), h5_prefix))
    channel = h5_prefix.split("-")[0].lower()

    # get data from miniscan file
    with fits.open(os.path.join(MINISCAN_PATH, channel, "%s.fits" % h5_prefix), lazy_load_hdus=True) as hdul:
        keys = [i.name for i in hdul if i.name != "PRIMARY"]
        n_reps = len([i for i, key in enumerate(keys) if "ARRAY" in key])

        arrs = []
        aotfs = []
        ts = []
        for i in range(n_reps):
            arrs.append(hdul["ARRAY%02i" % i].data)
            aotfs.append(hdul["AOTF%02i" % i].data)
            ts.append(hdul["T%02i" % i].data)

    plt.figure()
    plt.title(h5_prefix)
    for arr_ix, arr in enumerate(arrs):
        plt.plot(arr[:, 800], label="Row %i repeat %i" % (800, arr_ix))
        plt.plot(arr[500, :], label="Column %i repeat %i" % (500, arr_ix))

    plt.legend()
