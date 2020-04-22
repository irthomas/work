# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:57:10 2020

@author: iant

CORRECT HDF5 FILES SCIENCE/Y USING PIXEL CORRECTION
"""



import os
import matplotlib.pyplot as plt
import numpy as np
import re
import h5py

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions_v04 import makeFileList
from tools.spectra.fit_polynomial import fit_polynomial
from tools.general.get_nearest_index import get_nearest_index
from tools.spectra.so_non_linearity_correction import make_so_correction_dict, correct_so_observation


#select obs for deriving correction
#be careful of which detector rows are used. Nom pointing only after 2018 Aug 11

### order 129
regex = re.compile("20180(813|816|818|820|823|827)_.*_SO_A_I_129") #row120 used 
regex_all = re.compile("20.*_SO_A_[IE]_129")
file_index = 63 #best 129 spectrum

### order 130
#regex = re.compile("20(180828|180830|180901|181125|181201|181207|190203|190311|190504|191211)_.*_SO_A_[IE]_130") #row120 used 
#regex_all = re.compile("20.*_SO_A_[IE]_130")
#file_index = 65 #best 130 spectrum

file_level = "hdf5_level_1p0a"
toa_alt = 100.0






  
correction_dict = make_so_correction_dict(regex, file_level, toa_alt)

hdf5_files_all, hdf5_filenames_all, _ = makeFileList(regex_all, file_level, silent=True)

diffraction_order = int(hdf5_filenames_all[0].split("_")[-1])
if diffraction_order == 130:
    indices_without_absorptions = list(range(64))+list(range(74, 121))+list(range(131,164))+\
    list(range(181,191))+list(range(200,215))+list(range(226, 320))

elif diffraction_order == 129:
    indices_without_absorptions = list(range(100))+list(range(120, 320))

else:
    print("Error: define continuum pixels")


for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files_all, hdf5_filenames_all)):

    print("%i/%i:" %(file_index, len(hdf5_filenames_all)), hdf5_filename)
    output_filepath = os.path.join(paths["DATA_DIRECTORY"], hdf5_filename+"_px_correction")
    input_filepath = os.path.join(paths["DATA_DIRECTORY"], file_level, hdf5_filename[0:4], hdf5_filename[4:6], hdf5_filename[6:8], hdf5_filename)

    correct_so_observation(input_filepath, output_filepath, correction_dict, indices_without_absorptions)




hdf5_filename = hdf5_filenames_all[file_index]
hdf5_file = hdf5_files_all[file_index]
output_filepath = os.path.join(paths["DATA_DIRECTORY"], hdf5_filename+"_px_correction")

print(hdf5_filename)
print(output_filepath)

diffraction_order = int(hdf5_filename.split("_")[-1])
if diffraction_order == 130:

    indices_no_strong_abs = list(range(64))+list(range(74, 121))+list(range(131,164))+\
    list(range(181,191))+list(range(200,215))+list(range(226, 320))

elif diffraction_order == 129:
    indices_no_strong_abs = list(range(100))+list(range(120, 320))


y_old = hdf5_file["Science/Y"][...]

with h5py.File(output_filepath+".h5", "r") as f:
    y_new = f["Science/Y"][...]

plot_centre_index = get_nearest_index(0.4, y_old[:, 200])

fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
pixels = np.arange(320.0)

for index, plot_index in enumerate(range(plot_centre_index-3, plot_centre_index+4)):
    y1 = y_old[plot_index, :]
    y1_continuum = fit_polynomial(pixels, y1, 5, indices=indices_no_strong_abs)
    y1_corr = y1 / y1_continuum
    
    y2 = y_new[plot_index, :]
    y2_continuum = fit_polynomial(pixels, y2, 5, indices=indices_no_strong_abs)
    y2_corr = y2 / y2_continuum
    
    alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
    
    ax1.plot(y1_corr - index/100.0, label="%0.1fkm before correction, T=%0.3f" %(alts[plot_index], y1_continuum[200]), linestyle="--")
    ax1.plot(y2_corr - index/100.0, label="%0.1fkm after correction, T=%0.3f" %(alts[plot_index], y1_continuum[200]))
    ax1.legend()
