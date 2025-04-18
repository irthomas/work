# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:39:10 2020

@author: iant

SPICE FUNCTIONS


"""


import spiceypy as sp
import os


from tools.file.paths import paths


def load_spice_kernels(planning=False):

    os.chdir(paths["KERNEL_DIRECTORY"])

    if planning:
        sp.furnsh(os.path.join(paths["KERNEL_DIRECTORY"], "em16_plan.tm"))
    else:
        sp.furnsh(os.path.join(paths["KERNEL_DIRECTORY_OPS"], "em16_ops.tm"))
    print(sp.tkvrsn("toolkit"))
    os.chdir(paths["BASE_DIRECTORY"])
