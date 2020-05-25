# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:47:54 2020

@author: iant

find all grazing occultations

"""

#import os
import re



from tools.file.hdf5_functions import make_filelist


regex = re.compile(".*_SO_._G_.*")

file_level = "hdf5_level_0p3k"

_, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False)

hdf5_filenames_sorted = sorted(hdf5_filenames)

for hdf5_filename_sorted in hdf5_filenames_sorted:
    print(hdf5_filename_sorted)