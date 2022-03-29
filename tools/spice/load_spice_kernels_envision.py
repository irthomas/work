# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:39:10 2020

@author: iant

SPICE FUNCTIONS


"""


import spiceypy as sp
import os


from tools.file.paths import paths


ENVISION_KERNEL_DIR = os.path.normcase(r"C:\Users\iant\Documents\DATA\envision_kernels\envision\kernels\mk")

def load_spice_kernels_envision(kernel_name):
    
    os.chdir(ENVISION_KERNEL_DIR)
    
	sp.furnsh(ENVISION_KERNEL_DIR, kernel_name))
    print(sp.tkvrsn("toolkit"))
    os.chdir(paths["BASE_DIRECTORY"])

