# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:34:34 2019

@author: iant

TEST FULLSCAN LINE POSITIONS
"""

#import h5py
#import os
import matplotlib.pyplot as plt
import numpy as np
import re

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList
#from get_hdf5_data_v01 import getLevel1Data


#hdf5_filename = "20180424_201833_0p3a_SO_1_S"
#hdf5_filename = "20180429_042658_0p3a_SO_1_S" #fast 
#hdf5_filename = "20180429_050758_0p3a_SO_1_S" #fast
regex = re.compile("20180813_223104_0p3a_SO_1_S")

fileLevel = "hdf5_level_0p3a"
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)



hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]



y = hdf5_file["Science/Y"][...]
bins = hdf5_file["Science/Bins"][:, 0]
orders = hdf5_file["Channel/DiffractionOrder"][...]

#if "20180424_201833" in hdf5_filename:
#    atmosIndices = list(range(5940, 6180))
#elif "20180429_042658" in hdf5_filename:
#    atmosIndices = list(range(6930, 7270))
#elif "20180813_223104" in hdf5_filename:
#    atmosIndices = list(range(1000, 1600))
#    
#    
#frameIndices = np.arange(len(y[:, 0]))
#
#frame_number = 120
#bin_index = 0
#
#
#uniqueBins = sorted(list(set(bins)))
#binIndices = np.where(bins == uniqueBins[bin_index])[0]
#
#
#plt.plot(frameIndices[binIndices], y[binIndices, 160])
#
#atmosBinIndices = [binIndex for binIndex in binIndices if binIndex in atmosIndices]
#
#if binIndices[1]-binIndices[0] == 24:
#
#    for index in atmosBinIndices:
#    
#        plt.figure()
#        plt.imshow(y[index:(index+23), :], aspect=5)
#        plt.title("order %i frame %i" %(orders[index], index))
#        plt.colorbar()
#        
#        

def fft_zerofilling(row, filling_amount):
    """apply fft, zero fill by a multiplier then reverse fft to give very high resolution spectrum"""
    rowrfft = np.fft.rfft(row, len(row))
    rowzeros = np.zeros(320 * filling_amount, dtype=np.complex)
    rowfft = np.concatenate((rowrfft, rowzeros))
    row_hr = np.fft.irfft(rowfft).real #get real component for reversed fft
    row_hr *= len(row_hr)/len(row) #need to scale by number of extra points

    pixels_hr = np.linspace(0, 320.0, num=len(row_hr))    
    return pixels_hr, row_hr
    

frame_numbers = [
#    [1080, 6648], #order 155
    [1104, 6672],
    [1128, 6696],
    [1152, 6720],

#    [1176, 6744],
#    [1200, 6768],
#    [1224, 6792],
#    [1248, 6816],
#    [1272, 6840],
#    [1296, 6864],
#    [1320, 6888],
#    [1344, 6912],
#    [1368, 4152],
#
    [1392, 4176],
    [1416, 4200],
    [1440, 4224], #order 170
    [1464, 4248],
#    [1488, 4272],

]

absorptionIndicesDict = {
1104:[range(9, 14), range(61, 66), range(69, 75), range(88,94), range(108, 116), range(127, 134), range(145, 154), range(165, 172), range(174, 179), range(184, 190), range(203, 210), range(214, 218), range(222, 227), range(238,246), range(251, 255), range(285, 290)], \
1128:[range(10, 14), range(29, 34), range(39, 43), range(49, 54), range(61, 66), range(78, 83), range(87, 91), range(95, 100), range(127, 132), range(144, 149), range(152, 157), range(160, 165), range(183, 187), range(192, 196), range(205, 210), range(215, 219), range(222, 226), range(237, 242), range(250, 255), range(285, 290)], \
1152:[range(9, 13), range(15, 20), range(31, 34), range(39, 44), range(51, 55), range(68, 72), range(77, 83), range(86, 91), range(101, 105), range(106, 110), range(113, 119), range(133, 137), range(143, 147), range(152, 157), range(169, 174), range(184, 187), range(187, 191), range(193, 197), range(203, 208), range(213, 217), range(222, 227), range(232, 236), range(240, 245), range(251, 255), range(269, 274), range(285, 290)], \
1176:[range(100,200)], \
1200:[range(100,200)], \
1224:[range(100,200)], \
1248:[range(100,200)], \
1272:[range(100,200)], \
1296:[range(100,200)], \
1320:[range(100,200)], \
1344:[range(100,200)], \
1368:[range(100,200)], \

1392:[range(5,9), range(10,14), range(21, 24), range(27, 32), range(33,37), range(44,48), range(60,65), range(87,92), range(103,107), range(112,116), range(143,148), range(184,188), range(223,227), range(258,263), range(273,277), range(283,287)], \
1416:[range(11,14), range(21,24), range(28,31), range(37, 41), range(43,47), range(60,65), range(87,92), range(102,107), range(112,115), range(200,204), range(249, 253), range(259, 263), range(273,278)], \
1440:[range(27,31), range(43,47), range(61,65), range(73,78), range(87,91), range(125,130), range(141,145), range(161,165), range(191,195), range(213,218), range(224,228)], \
1464:[range(5,8), range(47,50), range(77,81), range(102,107), range(205,209), range(221,225), range(240,244), range(250,254), range(281,286), range(291,295)], \
1488:[range(100,200)], \
}

ANCHOR_ROW = 12

pixelNumbersAll = []
absorptionSlopesAll = []

for atmos_index, sun_index in frame_numbers:
    
    fig1, ax1 = plt.subplots(figsize=(FIG_X + 6, FIG_Y+3))
    fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y+3))
    
    atmos_frame = y[atmos_index:(atmos_index+23), :]
    solar_frame = y[sun_index:(sun_index+23), :]
    frame = atmos_frame / (solar_frame + 0.01) #avoid div/0
    
    atmos_order = orders[atmos_index]
    sun_order = orders[sun_index]
    if atmos_order != sun_order:
        print("Error: diffraction order doesn't match")
    else:
        print("Diffraction order %s" %atmos_order)
    
    cmap = plt.get_cmap('Spectral')
    colours = [cmap(i) for i in np.arange(24.0)/24.0]
    
    absorptionIndicesAll = absorptionIndicesDict[atmos_index]
    
    coloursAbsorption = [cmap(i) for i in np.arange(len(absorptionIndicesAll))/len(absorptionIndicesAll)]
    
    
    for rowIndex, row in enumerate(frame):
        if rowIndex in range(4,21):
            ax1.plot(row, color=colours[rowIndex])
    ax1.set_title("%s fullscan order %i" %(hdf5_filename, atmos_order))
    
    for absorptionIndex, absorptionIndices in enumerate(absorptionIndicesAll):
    
        absorptionIndicesHr = np.arange(absorptionIndices[0], absorptionIndices[-1], 0.01)
        
        absorptionMinima = []
            
        for rowIndex, row in enumerate(frame):
            pixelHr, rowAbsorptionHr = fft_zerofilling(row, 100)
            absorptionStartHrIndex = np.argmin(pixelHr < absorptionIndices[0])
            absorptionEndHrIndex = np.argmin(pixelHr < absorptionIndices[-1])
            
            absorptionMinimumHrIndex = np.where(np.min(rowAbsorptionHr[absorptionStartHrIndex:absorptionEndHrIndex]) == rowAbsorptionHr[absorptionStartHrIndex:absorptionEndHrIndex])[0][0]
            absorptionMinimum = rowAbsorptionHr[absorptionStartHrIndex:absorptionEndHrIndex][absorptionMinimumHrIndex]
            absorptionMinimumPixel = pixelHr[absorptionStartHrIndex:absorptionEndHrIndex][absorptionMinimumHrIndex]
            
            absorptionMinima.append(absorptionMinimumPixel)
            if rowIndex in range(4,21): #only rows with signal
                ax1.scatter(absorptionMinimumPixel, absorptionMinimum, color=colours[rowIndex])
                ax1.plot(pixelHr[absorptionStartHrIndex:absorptionEndHrIndex], rowAbsorptionHr[absorptionStartHrIndex:absorptionEndHrIndex], color=colours[rowIndex])
        
        ax2.plot(absorptionMinima[4:21] - np.mean(absorptionMinima[(ANCHOR_ROW-2):(ANCHOR_ROW+3)]), list(range(24))[4:21], color=coloursAbsorption[absorptionIndex], label=absorptionMinima[ANCHOR_ROW])
        ax2.set_ylim([22, 4])
        ax2.set_ylabel("Detector Row")
        ax2.set_xlabel("Absorption minima pixel (delta from mean of illuminated rows)")
        ax2.set_title("Absorption positions %s" %hdf5_filename)
        plt.legend()
        
        #get slope of absorption across detector rows, where 0 = no slant.
        #x is normalised in 0-1 range so a gradient of 0.4 = pixel shift of 0.4 across entire range
        coeffs = np.polyfit(np.linspace(0, 1, num=len(np.arange(24)[4:21])), absorptionMinima[4:21], 1)
        pixelNumbersAll.append(absorptionMinima[ANCHOR_ROW])
        absorptionSlopesAll.append(coeffs[0])

plt.figure()
plt.title("Slope of absorption lines across detector rows")
plt.ylabel("Slope (pixel delta for 17 rows)")
plt.xlabel("Pixel Number")
plt.scatter(pixelNumbersAll, absorptionSlopesAll)

#y = frame[12, :]
#
#
#
#
#yfft = np.fft.fft(y, len(y))
#yrfft = np.fft.rfft(y, len(y))
#
#plt.figure()
#plt.plot(yfft)
#plt.plot(yrfft)
#
#yback = np.fft.irfft(yrfft)
#yrback = np.fft.ifft(yfft).real
#
#plt.figure()
#plt.plot(yback)
#plt.plot(yrback)
#
#yzeros = np.zeros(320*100, dtype=np.complex)
#yfft2 = np.concatenate((yrfft, yzeros))
#yrback2 = np.fft.irfft(yfft2).real
#
#plt.plot(np.linspace(0, 320, num=len(yrback2)), yrback2* (len(yrback2)/len(yrback)))


