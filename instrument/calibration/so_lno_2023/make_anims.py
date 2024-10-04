# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:16:04 2024

@author: iant

MINISCAN ANIMATION 

"""
import numpy as np

from tools.plotting.anim import make_line_anim

# after running s01_correct_miniscan_diagonals
array_name = "array_diag0_hr"
# miniscan_name = "LNO-20200628-135310-164-8"
miniscan_name = "LNO-20220619-140101-164-4"


array = d2[miniscan_name][array_name]
px_ixs = np.arange(array.shape[1])

array_norm = np.zeros_like(array)
for i, spectrum in enumerate(array):
    array_norm[i, :] = spectrum / np.max(spectrum)


anim_d = {}
anim_d["x"] = {"y": px_ixs}
anim_d["y"] = {"y": array}
# d["y"] = {"y": array_norm}
anim_d["xlabel"] = "Pixel number"
anim_d["ylabel"] = "Diagonally corrected"

anim_d["x_params"] = {"1d": True}

anim_d["text"] = ["%i" % i for i in range(array.shape[0])]
anim_d["text_position"] = [0, 0]
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
