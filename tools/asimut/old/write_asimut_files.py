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

local_base_dir = r"C:\Users\iant\Documents\DATA\retrievals\so_aotf_ils\CO"

aotf_filenames = [
    "AOTF_from_fitting_4276cm-1_solar_line_fitted.txt",
    "AOTF_from_fitting_4276cm-1_solar_line_smoothed.txt",

    "AOTF_from_fitting_4384cm-1_solar_line_fitted.txt",
    "AOTF_from_fitting_4384cm-1_solar_line_smoothed.txt",
]
#load hdf5
#get indices of bins
regex = re.compile(hdf5_filename) #(approx. orders 188-202) in steps of 8kHz
chosen_bin = 3

file_level="hdf5_level_1p0a"

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)



def make_asi_dict(dict_in, template_inputs):
    
    dict_new = {}

    for k1, v1 in dict_in.items():
        if type(v1) == dict:
            dict_new[k1] = {}
            for k2, v2 in v1.items():
                found = False
                if k2 is not None:
                    if k2[0] == "%":
                        key = k2[1:]
                        if key in template_inputs.keys():
                            # print("Replacing key", key)
                            dict_new[k1][template_inputs[key]] = None
                            found = True
                        else:
                            print("Error: %s not replaced" %key)
                    else:
                        dict_new[k1][k2] = v2
                else:
                    dict_new[k1][k2] = v2


                if v2 is not None:
                    if v2[0] == "%":
                        value = v2[1:]
                        if value in template_inputs.keys():
                            # print("Replacing value", value)
                            dict_new[k1][k2] = template_inputs[value]
                        else:
                            print("Error: %s not replaced" %v2)
                    else:
                        dict_new[k1][k2] = v2
                else:
                    if not found:
                        print(k2)
                        dict_new[k1][k2] = v2

    return dict_new




for hdf5_file, hdf5_filename, hdf5_path in zip(hdf5_files, hdf5_filenames, hdf5_paths):
    bins = hdf5_file["Science/Bins"][...]


template_dir = os.path.join(paths["BASE_DIRECTORY"], "tools", "asimut", "templates")



asi_template = configparser.ConfigParser(allow_no_value=True)
asi_template.optionxform=str
asi_template.read(os.path.join(template_dir, "asimut_template.asi"))
inp_template = configparser.ConfigParser(allow_no_value=True)
inp_template.optionxform=str
inp_template.read(os.path.join(template_dir, "asimut_template.inp"))

file_no = -1




sh = "#!/bin/bash\n"
sh += "cd /home/iant/linux/ASIMUT/trunk\n"


for aotf_filename in aotf_filenames:
    
    file_no += 1

    asi_template_dict = asi_template._sections
    inp_template_dict = inp_template._sections
    
    asi = "%s_%i.asi" %(hdf5_filename, file_no)
    inp = "%s_%i.inp" %(hdf5_filename, file_no)
    
    vals_string = "val[ 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 95 99 103 107 111 115 119 123 127 131 135 139 143 147 151 155 159 163 167 171 175 179 183 187 191 195 199 203 207 211 215 219 223 227 231 235 239 243 247 251 255 259 263 267 271 275 279 283 287 291 295 299 303 307 311 315 319 323 327 331 335 339 343 347 351 355 359 363 367 371 375 379 383 387 391 395 399 403 407 411 415 419 423 427 431 435 439 443 447 451 455 459 463 467 471 475 479 483 487 491]"
    
    template_inputs = {
        "hdf5_dir":posixpath.join(base_dir, "hdf5"),
        "atm_dir":posixpath.join(base_dir, "Atmosphere", gem_version),
        "input_dir":posixpath.join(base_dir, "Retrievals", "Input"),
        "save_dir":posixpath.join(base_dir, "Retrievals", "Save"),
        "out_dir":posixpath.join(base_dir, "Retrievals", "Results"),
        "inp":inp,
    
        "hdf5":hdf5_filename+suffix+".h5",
        "id_list":vals_string,
        "aotf_filepath":posixpath.join(base_dir, "Instrument", "AOTF", aotf_filename),
        "A_nu0":"%0.4f" %A_nu0,
        "ils_filepath":posixpath.join(base_dir, "Instrument", "ILS", hdf5_filename+"_ils.txt"),
        "atm_list":"%s/%s_%s_list.dat" %(hdf5_filename, gem_version, hdf5_filename)
        }


    
    asi_template_dict_new = make_asi_dict(asi_template_dict, template_inputs)
    inp_template_dict_new = make_asi_dict(inp_template_dict, template_inputs)


    
    local_asi_path = os.path.join(local_base_dir, "Retrievals", "Input", asi)
    local_inp_path = os.path.join(local_base_dir, "Retrievals", "Input", inp)
    
    error = False
    if not os.path.exists(local_asi_path):
        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform=str
        config.read_dict(asi_template_dict_new)
        with open(local_asi_path, 'w') as configfile:
            config.write(configfile)
    else:
        error = True

    if not os.path.exists(local_inp_path):
        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform=str
        config.read_dict(inp_template_dict_new)
        with open(local_inp_path, 'w') as configfile:
            config.write(configfile)
    else:
        error = True


    sh += "./asimut %s/%s\n" %(template_inputs["input_dir"], asi) 
    #"/bira-iasb/projects/work/NOMAD/Science/ian/so_aotf_ils/CO/Retrievals/Input/20180930_113957_1p0a_SO_A_I_189.asi"

with open("asimut.sh", "w") as f:
    f.write(sh)    


