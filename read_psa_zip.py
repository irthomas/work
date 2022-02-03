# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:25:02 2021

@author: iant

READ PSA FILES FROM ZIP FILES
"""


import zipfile
# import os
import numpy as np
import matplotlib.pyplot as plt

# zip_file_path = r"D:\DATA\psa\data_calibrated\2018\04\21\nmd_cal_sc_lno_20180421-18312100-d-168_3.0.zip"
zip_file_path = r"C:\Users\iant\Dropbox\NOMAD\Python\2018\04\22\nmd_cal_sc_lno_20180422-00345600-d-169_3.0.zip"
# zip_file_path = r"D:\DATA\psa\data_calibrated\2018\04\21\nmd_cal_sc_so_20180421-20211100-a-e-190_3.0.zip"
# zip_file_path = r"D:\DATA\psa\data_calibrated\2018\04\21\nmd_cal_sc_uvis_20180421-18312100-d_3.0.zip"
# zip_file_path = r"D:\DATA\psa\data_calibrated\2018\04\22\nmd_cal_sc_uvis_20180422-00165000-e_3.0.zip"

arch = zipfile.ZipFile(zip_file_path, 'r')
arch_list = arch.namelist()

tab_filename = [i for i in arch_list if ".tab" in i][0]
tab_data = arch.read(tab_filename).decode().split("\n")

if tab_data[-1] == "":
    tab_data.pop(-1)

if "lno" in tab_filename:
    n_lines = 1071
    wvn_start = 111
    n_pixels = 320
    v = [0.0, 0.6]

if "so" in tab_filename:
    n_lines = 1063
    wvn_start = 103
    n_pixels = 320
    v = [0.0, 1.1]

if "uvis" in tab_filename and "-d_" in tab_filename:
    n_lines = 4272
    wvn_start = 176
    n_pixels = 1024
    v = [0.0, 0.04]

if "uvis" in tab_filename and ("-i_" in tab_filename or "-e_" in tab_filename):
    n_lines = 4256
    wvn_start = 160
    n_pixels = 1024
    v = [0.0, 1.1]

tab_array = np.zeros((len(tab_data), n_lines))

for i, line in enumerate(tab_data):
    tab_array[i, :] = line[55:].split()

wavenumbers = tab_array[:, wvn_start:(wvn_start+n_pixels)]
ref_fac = tab_array[:, (wvn_start+n_pixels):(wvn_start+n_pixels*2)]
ref_fac_bl = tab_array[:, (wvn_start+n_pixels*2):(wvn_start+n_pixels*3)]

plt.title(tab_filename)
plt.imshow(ref_fac, vmin=v[0], vmax=v[1])
