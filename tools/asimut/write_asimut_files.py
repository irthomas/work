# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:25:26 2021

@author: iant

WRITE ASIMUT INP AND ASI FILES FROM TEMPLATES

"""

import re
# import numpy as np
# import matplotlib.pyplot as plt
import os
# import sys
import configparser
# import h5py
import posixpath

# from matplotlib.backends.backend_pdf import PdfPages

from tools.file.paths import paths
from tools.file.hdf5_functions import make_filelist
# from tools.plotting.colours import get_colours


sub_dirs = ["so_aotf_ils", "CO"]

hdf5_filename = "20180930_113957_1p0a_SO_A_I_189"
gem_version = "gem-mars-a758"
suffix = "_corr"
order = int(hdf5_filename[-3:])

A_nu0 = 4267.0858

base_dir = "/bira-iasb/projects/work/NOMAD/Science/ian/so_aotf_ils/CO/"

aotf_filenames = [
    "AOTF_from_fitting_4276cm-1_solar_line_fitted.txt",
]
#load hdf5
#get indices of bins
regex = re.compile(hdf5_filename) #(approx. orders 188-202) in steps of 8kHz
chosen_bin = 3

file_level="hdf5_level_1p0a"

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)


for hdf5_file, hdf5_filename, hdf5_path in zip(hdf5_files, hdf5_filenames, hdf5_paths):
    bins = hdf5_file["Science/Bins"][...]


template_dir = os.path.join(paths["BASE_DIRECTORY"], "tools", "asimut", "templates")


asi_template = configparser.ConfigParser(allow_no_value=True)
asi_template.read(os.path.join(template_dir, "asimut_template.asi"))
inp_template = configparser.ConfigParser(allow_no_value=True)
inp_template.read(os.path.join(template_dir, "asimut_template.inp"))

for aotf_filename in aotf_filenames:

    asi_template_dict = asi_template._sections
    inp_template_dict = inp_template._sections
    
    template_inputs = {
        "hdf5_dir":posixpath.join(base_dir, "hdf5"),
        "atm_dir":posixpath.join(base_dir, "Atmosphere", gem_version),
        "input_dir":posixpath.join(base_dir, "Retrievals", "Input"),
        "save_dir":posixpath.join(base_dir, "Retrievals", "Save"),
        "out_dir":posixpath.join(base_dir, "Retrievals", "Results"),
        "inp":hdf5_filename+".inp",
    
        "hdf5":hdf5_filename+suffix+".h5",
        "vals":"",
        "aotf_filepath":posixpath.join(base_dir, "Instrument", "AOTF", aotf_filename),
        "A_nu0":A_nu0,
        "ils_filepath":posixpath.join(base_dir, "Instrument", "ILS", hdf5_filename+"_ils.dat"),
        "atm_list":"%s/%s_%s_list.dat" %(hdf5_filename, gem_version, hdf5_filename)
        }
    
    


