# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:25:54 2022

@author: iant

RUN ASIMUT RETRIEVALS ON ADA:
    1. MAKE DIRECTORY LOCALLY FOR EACH HDF5
    2. ADD AOTF, ILS, APRIORI, INP AND ASI TO THAT DIRECTORY
    3. MAKE SAVE AND RESULTS SUB DIR
    4. MAKE SH SCRIPT
    5. TRANSFER TO HERA
    6. RUN ASIMUT
    7. TRANSFER BACK TO LOCAL DIR
    8. CLEAN UP AND PLOT THE RESULTS
    
TO DO:
    MAKE LEVEL2 HDF5 CONTAINING ALL LEVEL1.0A DATA
    PARALLELISE: RUN MULTIPLE FILES SIMULTANEOUSLY ON ONE SERVER AND ACROSS SERVERS
    PRE-ANALYSE THE 1.0A FILE (CORRECT AOTF CENTRE, SPECTRAL CALIBRATION, ETC)
    PREPARE INPUTS FOR OTHER ORDERS
"""

import os
import shutil
import posixpath
import time

import numpy as np

from instrument.nomad_so_instrument_v03 import aotf_peak_nu, aotf_nu_to_order, lt22_waven

from tools.asimut.asimut_ils import asimut_ils
from tools.asimut.asimut_aotf import asimut_aotf
from tools.asimut.asimut_wavenb import asimut_wavenb
from tools.asimut.asimut_atmosphere import asimut_atmosphere
from tools.asimut.asimut_inp import asimut_inp
from tools.asimut.asimut_asi import asimut_asi
from tools.asimut.asimut_sh import asimut_sh
from tools.asimut.copy_inputs_to_hera import copy_inputs_to_hera
from tools.asimut.get_asimut_output import get_asimut_output
from tools.asimut.make_output_dict import make_output_dict
from tools.asimut.plot_asimut_dict_pdf import plot_asimut_dict_pdf
from tools.asimut.execute_asimut import execute_asimut, run_remote_command

from tools.datasets.get_gem_data import get_gem_data_from_h5
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.get_h5_dir_linux import get_h5_dir_linux


"""user variables"""
#name of file to convert
# h5 = "20220404_022239_1p0a_SO_A_E_134" #no water
h5 = "20220101_005247_1p0a_SO_A_E_136" #high water
# h5 = "20220304_222120_1p0a_SO_A_I_148" #high water
#suffix of files to be created (can choose any)
title = "test"
#choose which ada machine to run on
ada = 6

#select local and remote base directories. A subfolder will be made in each with the name of the h5 file
base_dir_linux = "/bira-iasb/projects/work/NOMAD/Science/ian/CH4_v01/"
tmp_base_dir = r"C:\Users\iant\Dropbox\NOMAD\Python\tmp"






h5_f = open_hdf5_file(h5) #open file


#find indices of spectra where 0.1 < median transmittance < 0.95
y_median = np.median(h5_f["Science/Y"][...], axis=1)
# indices = list(np.where((y_median > 0.1) & (y_median < 0.95))[0])
indices = list(np.where((y_median > 0.3) & (y_median < 0.7))[0])

print("%i spectra will be analysed" %len(indices))

aotf_freq = h5_f["Channel/AOTFFrequency"][0]
t = np.mean(h5_f["Channel/InterpolatedTemperature"][:])


#make filenames and directory names and paths
aotf_filename = "%s_aotf.txt" %title
ils_filename = "%s_ils.txt" %title
wavenb_filename = "%s_wavenb.txt" %title
atmos_filename = "%s_atmosphere.txt" %title
inp_filename = "%s.inp" %title
asi_filename = "%s.asi" %title
sh_filename = "asimut.sh"
out_filename = "%s_out.h5" %title



#make dirs
tmp_dir = os.path.join(tmp_base_dir, h5)

save_dir = os.path.join(tmp_dir, "SAVE")
results_dir = os.path.join(tmp_dir, "RESULTS")

#delete local directory if it exists
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)
os.makedirs(save_dir)
os.makedirs(results_dir)



aotf_filepath = os.path.join(tmp_dir, aotf_filename)
ils_filepath = os.path.join(tmp_dir, ils_filename)
wavenb_filepath = os.path.join(tmp_dir, wavenb_filename)
atmos_filepath = os.path.join(tmp_dir, atmos_filename)
inp_filepath = os.path.join(tmp_dir, inp_filename)
asi_filepath = os.path.join(tmp_dir, asi_filename)
sh_filepath = os.path.join(tmp_dir, sh_filename)
out_filepath = os.path.join(results_dir, out_filename)

pdf_filepath = "%s_%s.pdf" %(h5, title)



h5_dir_linux = get_h5_dir_linux(h5) #get linux path of 1.0a hdf5 file in datastore

dir_linux = posixpath.join(base_dir_linux, h5)
atmos_dir_linux = dir_linux
input_dir_linux = dir_linux
save_dir_linux = posixpath.join(dir_linux, "SAVE")
results_dir_linux = posixpath.join(dir_linux, "RESULTS")

aotf_filepath_linux =posixpath.join(dir_linux, aotf_filename)
ils_filepath_linux = posixpath.join(dir_linux, ils_filename)
wavenb_filepath_linux = posixpath.join(dir_linux, wavenb_filename)
atmos_filepath_linux = posixpath.join(dir_linux, atmos_filename)
inp_filepath_linux = posixpath.join(dir_linux, inp_filename)
asi_filepath_linux = posixpath.join(dir_linux, asi_filename)
sh_filepath_linux = posixpath.join(dir_linux, sh_filename)
out_filepath_linux = posixpath.join(results_dir_linux, out_filename)


#delete directory on hera if it exists
command = "rm -rf %s" %dir_linux
run_remote_command(command)




"""start the analysis"""

#get aotf peak wavenumber
aotf_nu = aotf_peak_nu(aotf_freq, t)

#get diffraction order
order = aotf_nu_to_order(aotf_nu)

#pixel wavenumbers
px_nu = lt22_waven(order, t)


#TODO: correct spectral calibration from data



#write spectral coefficient file
asimut_wavenb(order, t, wavenb_filepath)


#write ILS file
asimut_ils(aotf_nu, px_nu, ils_filepath)

#write AOTF file
asimut_aotf(aotf_nu, aotf_filepath)


#get atmosphere data from GEM
atmos_dict = get_gem_data_from_h5(h5_f, reference_altitude=50.0, index=None, plot=False)

#write apriori atmosphere
asimut_atmosphere(atmos_dict, atmos_filepath)


#write inp file
asimut_inp(h5, indices, aotf_nu, inp_filepath, aotf_filepath_linux, ils_filepath_linux, wavenb_filepath_linux, atmos_filename)


#write asi file
asimut_asi(asi_filepath, inp_filename, h5_dir_linux, atmos_dir_linux, input_dir_linux, save_dir_linux, results_dir_linux)


#write shell script to run on chosen machine
asimut_sh(ada, sh_filepath, asi_filepath_linux)


#when made, transfer all files to hera
remote_dirpaths = [dir_linux, save_dir_linux, results_dir_linux]
local_filepaths = [aotf_filepath, ils_filepath, wavenb_filepath, atmos_filepath, inp_filepath, asi_filepath, sh_filepath]
remote_filepaths = [aotf_filepath_linux, ils_filepath_linux, wavenb_filepath_linux, atmos_filepath_linux, inp_filepath_linux, asi_filepath_linux, sh_filepath_linux]
copy_inputs_to_hera(remote_dirpaths, local_filepaths, remote_filepaths)


#run asimut and time it
timer_start = time.time()
execute_asimut(sh_filepath_linux)
print("Asimut ran in %0.1f seconds" %(time.time() - timer_start))

#transfer output back
get_asimut_output(out_filepath, out_filepath_linux)


#read in retrieval data to dictionary
d = make_output_dict(inp_filepath, out_filepath, h5_f)


#plot results in a pdf
plot_asimut_dict_pdf(h5, d, pdf_filepath)

#close hdf5 file
h5_f.close()