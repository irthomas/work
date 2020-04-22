# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:14:49 2020

@author: iant

LOAD LOCAL SPICE KERNELS

"""
import os


"""where to find the SPICE metakernel?"""
BASE_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Dropbox", "NOMAD", "Python", "nomad_obs")
KERNEL_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Documents", "DATA", "local_spice_kernels", "kernels", "mk")



"""which SPICE metakernel to use?"""
METAKERNEL_NAME = "em16_ops.tm" #don't use for planning!!



#load spiceypy kernels
import spiceypy as sp
print("KERNEL_DIRECTORY=%s, METAKERNEL_NAME=%s" %(KERNEL_DIRECTORY, METAKERNEL_NAME))
os.chdir(KERNEL_DIRECTORY)
sp.furnsh(METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(BASE_DIRECTORY)


