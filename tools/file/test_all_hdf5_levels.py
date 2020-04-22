# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:05:28 2019

@author: iant

TEST SIGNAL LEVELS
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from hdf5_functions_v03 import makeFileList

DATA_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\test\iant\hdf5")
#DATA_DIRECTORY = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/")


"""compare Y values in different file levels"""
#filename_base = ["20180328_150554_", "_LNO_1_D_190"]
#file_levels_base = ["0p1d", "0p1e", "0p2a", "0p3a"]
#
#y_all = []
#for file_level_base in file_levels_base:
#
#    filename_full = filename_base[0] + file_level_base + filename_base[1]
#    fileLevel = "hdf5_level_"+file_level_base
#    
#    
#    year = filename_full[0:4] #get the date from the filename to find the file
#    month = filename_full[4:6]
#    day = filename_full[6:8]
#    
#    with h5py.File(os.path.join(DATA_DIRECTORY, fileLevel, year, month, day, filename_full + ".h5"), "r") as f:
#        y_all.append(f["Science/Y"][...])
#        
#for y in y_all:
#    print(np.sum(y))
#    
#    


"""compare Y in two files"""

filepaths = [
        r"w:\data\satellite\trace-gas-orbiter\nomad\test\iant\hdf5\hdf5_level_1p0a\2019\06\25\20190625_233600_1p0a_SO_A_E_136",
#        r"w:\data\satellite\trace-gas-orbiter\nomad\test\iant\hdf5\hdf5_level_0p3a\2019\06\25\20190625_233600_0p3a_SO_1_E_0",
        ]

y_all = []
for filepath in filepaths:
    
    with h5py.File(os.path.normcase(filepath + ".h5"), "r") as f:
        y = f["Science/Y"][...]
        bintops = f["Science/Bins"][:,0]
        y_all.append(y)
#        plt.plot(y[::4, 200])
        plt.plot(y[np.where(bintops==124)[0], 180])



"""compare X,Y in two files"""

#diffractionOrder = 134
#filepaths = [
#        r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5\hdf5_level_1p0a\2019\06\18\20190618_105903_1p0a_SO_A_E_%03i" %diffractionOrder,
#        r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5\hdf5_level_1p0b\2019\06\18\20190618_105903_1p0b_SO_A_E_%03i" %diffractionOrder,
#        r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5\hdf5_level_2p0a\2019\06\18\20190618_105903_2p0a_SO_A_E_%03i" %diffractionOrder,
#        ]
#
#plt.figure()
#x_all = []
#y_all = []
#frame_index = 61
#
#for filepath in filepaths:
#    
#    with h5py.File(os.path.normcase(filepath + ".h5"), "r") as f:
#        x = f["Science/X"][...]
#        x_all.append(x)
#        y = f["Science/Y"][...]
#        y_all.append(y)
#        
#        if "2p0a" not in filepath:        
#            firstPixel = f["Channel/FirstPixel"][0]
#            pixels = np.arange(320)
#            pixelSpectralCoefficients = f["Channel/PixelSpectralCoefficients"][0, :]
#            wavenumbers = np.polyval(pixelSpectralCoefficients, pixels + firstPixel) * np.float(diffractionOrder)
#            print(wavenumbers[100])
#        
#            plt.plot(x[frame_index, :], y[frame_index, :], label=os.path.basename(filepath))
##            plt.plot(x[frame_index, :] - 0.098, y[frame_index, :], label=os.path.basename(filepath))
#
#            print(f["Geometry/Point0/TangentAltAreoid"][61, 0])
#            
#            spectralResolution = f["Channel/SpectralResolution"][0]
#            print(spectralResolution)
#
#        else:
#            plt.plot(x[1, :], y[1, :], label=os.path.basename(filepath))
#            
#            yobs = f["Science/YObs"][...]
#            plt.plot(x[1, :], yobs[1, :], "--", label="%s YObs" %os.path.basename(filepath))
#
#        
##plt.plot(x_all[0][frame_index, :], y_all[0][frame_index, :], label=os.path.basename(filepaths[0]))
##plt.plot(x_all[1][frame_index, :], y_all[1][frame_index, :], label=os.path.basename(filepaths[1]))
#
#from plot_simulations_v01 import getSimulationDataNew
#
#vmr = 10.0
#alt = 12.0
#temperature = -5.0 #not important unless plotting vs. pixel
#methane_wavenumbers, methane_transmittance = getSimulationDataNew("so", "H2O", diffractionOrder, vmr, alt, temperature)
#
#plt.plot(methane_wavenumbers, methane_transmittance * 0.33, label="simulation")
#plt.legend()
#
#


