# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:02:23 2022

@author: iant

RUN LEVEL 0.3K TO 1.0A OF NOMAD DATA PIPELINE ON A FILE
"""



import os

import level1.Level_1_functions as l1f



NOMAD_TMP_DIR = "."

hdf5file_path = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p3k\2018\09\30\20180930_113957_0p3k_SO_A_I_189.h5"


hdf5_basename = os.path.basename(hdf5file_path)
h5file_out_path="{0}/{1}".format(NOMAD_TMP_DIR,hdf5_basename)
out = l1f.TransmittancesAlgo(hdf5file_path, h5file_out_path)



os.rename(out[0], out[0].replace("0p3k", "1p0a"))

print("Done")