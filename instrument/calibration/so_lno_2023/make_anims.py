# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:16:04 2024

@author: iant

MINISCAN ANIMATION 

"""
import os
import numpy as np

from instrument.calibration.so_lno_2023.load_fits_miniscan import load_fits_miniscan

from tools.plotting.anim import make_line_anim


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

h5_prefix = "LNO-20220619-140101-164-4"
# h5_prefix = "LNO-20181106-195839-170-4"

arrs, aotfs, ts = load_fits_miniscan(h5_prefix, MINISCAN_PATH)

px_ixs = np.arange(arrs[0].shape[1])

# normalise arrays
array_norms = []
for arr_ix in range(len(arrs)):
    array_norm = np.zeros_like(arrs[arr_ix])
    for i, spectrum in enumerate(arrs[arr_ix]):
        array_norm[i, :] = spectrum / np.max(spectrum)
    array_norms.append(array_norm)

# find relative stds to find pixels without absorption lines
array_stds = [np.std(array_norm, axis=0)/np.mean(array_norm, axis=0) for array_norm in array_norms]


anim_d = {}
anim_d["x"] = {"y": px_ixs}
# anim_d["y"] = {"y": array}
anim_d["y"] = {"y": array_norm}
anim_d["xlabel"] = "Pixel number"
anim_d["ylabel"] = "Diagonally corrected"
anim_d["interval"] = 10

anim_d["x_params"] = {"1d": True}

anim_d["text"] = ["%i" % i for i in range(arrs[0].shape[0])]
anim_d["text_position"] = [100, 0.2]
anim_d["format"] = "html"

anim_d["save"] = False
anim_d["filename"] = "test"

anim = make_line_anim(anim_d)


# d["x"] = {"y": px_ixs}
# d["y"] = {"y": y_corrected}
# d["xlabel"] = "Pixel number"
# d["ylabel"] = "Raw signal"

# d["x_params"] = {"1d": True}

# d["text"] = ["%i" % i for i in range(y_corrected.shape[0])]
# d["text_position"] = [0, 0]
# d["format"] = "html"

# d["save"] = False
# d["filename"] = "test"


# anim = make_line_anim(d)