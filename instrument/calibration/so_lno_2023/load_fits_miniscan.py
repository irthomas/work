# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:34:07 2024

@author: iant

LOAD FITS FILE
"""

import os
from astropy.io import fits


def load_fits_miniscan(h5_prefix, miniscan_path):

    channel = h5_prefix.split("-")[0].lower()
    # get data from miniscan file
    with fits.open(os.path.join(miniscan_path, channel, "%s.fits" % h5_prefix)) as hdul:
        keys = [i.name for i in hdul if i.name != "PRIMARY"]
        n_reps = len([i for i, key in enumerate(keys) if "ARRAY" in key])

        arrs = []
        aotfs = []
        ts = []
        for i in range(n_reps):
            arrs.append(hdul["ARRAY%02i" % i].data)
            aotfs.append(hdul["AOTF%02i" % i].data)
            ts.append(hdul["T%02i" % i].data)
            print("Loading %s rep %i: array sizes:" % (h5_prefix, i), arrs[-1].shape, aotfs[-1].shape, ts[-1].shape)

    return arrs, aotfs, ts
