# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:23:42 2019

@author: iant
"""
import h5py
import numpy as np
#
hdf5file_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p3a\2018\07\25\20180725_011449_0p3a_LNO_1_D_190.h5"
hdf5file = h5py.File(hdf5file_path, "r")

plt.figure()
frame_index = 133
y = hdf5file["Science/Y"][frame_index, :]
x = hdf5file["Science/X"][frame_index, :]
plt.plot(x, y, label="Order 190 Frame %i" %frame_index, alpha=0.5)

y = np.mean(hdf5file["Science/Y"][(frame_index-1):(frame_index+2), :], axis=0)
plt.plot(x, y, label="Order 190 Mean frames %i:%i" %((frame_index-1),(frame_index+2)))
plt.legend()

title = hdf5file_path.split(os.sep)[-1]

hdf5file.close()


hdf5file_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p3a\2018\07\25\20180725_011449_0p3a_LNO_1_D_196.h5"
hdf5file = h5py.File(hdf5file_path, "r")


y = hdf5file["Science/Y"][frame_index, :]
x = hdf5file["Science/X"][frame_index, :]
plt.plot(x, y, label="Order 196 Frame %i" %frame_index, alpha=0.5)

y = np.mean(hdf5file["Science/Y"][(frame_index-1):(frame_index+2), :], axis=0)
plt.plot(x, y, label="Order 196 Mean frames %i:%i" %((frame_index-1),(frame_index+2)))
plt.legend()

title += " & "
title += hdf5file_path.split(os.sep)[-1]

plt.title(title)
hdf5file.close()

#
#inputfile_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\.tmp\20180421_202111_1p0b_SO_A_E_134\INPUT\Order134.inp"
#
##def squeezeDatasets(hdf5file):
#"""remove unneccessary dimensions from asimut-generated h5 files"""
#
#for key in list(hdf5file.keys()):
#    is_dataset = isinstance(hdf5file[key], h5py.Dataset)
#    if is_dataset:
#        print(key)
#        print(hdf5file[key].shape)
#        print(np.squeeze(hdf5file[key]).shape)
#        dset_copy = np.squeeze(hdf5file[key])
#        del hdf5file[key]
#        hdf5file[key] = dset_copy
#    else:
#        for subkey in list(hdf5file[key].keys()):
#            is_dataset = isinstance(hdf5file[key][subkey], h5py.Dataset)
#            if is_dataset:
#                print(subkey)
#                print(hdf5file[key][subkey].shape)
#                print(np.squeeze(hdf5file[key][subkey]).shape)
#                dset_copy = np.squeeze(hdf5file[key][subkey])
#                del hdf5file[key][subkey]
#                hdf5file[key][subkey] = dset_copy
#            else:
#                for subsubkey in list(hdf5file[key][subkey].keys()):
#                    is_dataset = isinstance(hdf5file[key][subkey][subsubkey], h5py.Dataset)
#                    if is_dataset:
#                        print(subsubkey)
#                        print(hdf5file[key][subkey][subsubkey].shape)
#                        print(np.squeeze(hdf5file[key][subkey][subsubkey]).shape)
#                        dset_copy = np.squeeze(hdf5file[key][subkey][subsubkey])
#                        del hdf5file[key][subkey][subsubkey]
#                        hdf5file[key][subkey][subsubkey] = dset_copy
#
#
#with open(inputfile_path, "r") as f:
#    inputfile_contents = f.readlines()
#
#
#    for line in inputfile_contents:
#        if "SpectraID_List" in line:
#            frame_indices = np.asarray([int(value) for value in line.split("[")[1].split("]")[0].split(",")])
#            hdf5file["Science/SpectraID"] = frame_indices
#
#hdf5file.close()