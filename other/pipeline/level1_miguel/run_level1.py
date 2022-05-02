# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:02:23 2022

@author: iant

RUN LEVEL 0.3K TO 1.0A OF NOMAD DATA PIPELINE ON A FILE
"""



import os

import Level_1_functions as l1f #change path if necessary




# hdf5_filepath = "20180502_133902_0p3k_SO_A_E_121.h5" #path to file
# hdf5_filepath = "20180502_133902_0p3k_SO_A_E_134.h5" #path to file
# hdf5_filepath = "20180502_133902_0p3k_SO_A_E_149.h5" #path to file
# hdf5_filepath = "20180502_133902_0p3k_SO_A_E_165.h5" #path to file
# hdf5_filepath = "20180502_133902_0p3k_SO_A_E_167.h5" #path to file
hdf5_filepath = "20180502_133902_0p3k_SO_A_E_190.h5" #path to file


#make output file path
hdf5_basename = os.path.basename(hdf5_filepath)
new_hdf5_basename = hdf5_basename.replace("0p3k", "1p0a")
new_hdf5_filepath = hdf5_filepath.replace(hdf5_basename, new_hdf5_basename)

#set make_plots=False to turn off plots to speed up conversion (optional)
out = l1f.TransmittancesAlgo(hdf5_filepath, new_hdf5_filepath, make_plots=True)


print("Done: %s converted to %s" %(hdf5_basename, new_hdf5_basename))