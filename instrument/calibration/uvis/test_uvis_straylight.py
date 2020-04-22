# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:56:11 2019

@author: iant

UVIS Test
"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.ndimage import median_filter


FIG_X = 10
FIG_Y = 7


#frame_indices = [100]
#
#with h5py.File(r"C:\Users\iant\Documents\DATA\hdf5_copy_old\hdf5_level_0p2a\2019\01\18\20190118_054745_0p2a_UVIS_I.h5") as f:
#    y_all = f["Science/Y"][:, :, 8:1032]
#    alt = f["Geometry/Point0/TangentAltAreoid"][:, 0]
#    
#y_dark = y_all[1, :, :]
#y_all = y_all - y_dark
#    
#
#for frame_index in frame_indices:
#    y = y_all[frame_index]    
#    
#    y_corrected = np.zeros_like(y)
#    y_straylight = np.zeros_like(y)
#    
#    pixels = np.arange(1024)
#    
#    start_rows = [166, 165, 163, 158, 151, 149, 145]
#    end_rows = [180, 182, 184, 186, 188, 189, 190]
#    
#    
#    columns = [0, 116, 209, 444, 765, 843, 1023] #columns 0 to 1023
#    
#    corners = [[0, 140], [1023, 140], [1023, 205], [0, 205], [0, 140]]
#    
#    corners2 = corners[1::] + [corners[0]]
#    columns2 = columns[1::] + [columns[-1]]
#    start_rows2 = start_rows[1::] + [start_rows[-1]]
#    end_rows2 = end_rows[1::] + [end_rows[-1]]
#    
#    plt.figure(figsize=(FIG_X, FIG_Y))
#    
#    for column, column2, start, start2, end, end2 in zip(columns, columns2, start_rows, start_rows2, end_rows, end_rows2):
#        plt.scatter(column, start, c="b")
#        plt.scatter(column, end, c="b")
#        plt.plot([column, column2], [start, start2], "b")
#        plt.plot([column, column2], [end, end2], "b")
#    
#    start_rows_interp = np.interp(pixels, columns, start_rows)
#    end_rows_interp = np.interp(pixels, columns, end_rows)
#    for pixel_number, start_row, end_row  in zip(pixels, start_rows_interp, end_rows_interp):
#        plt.plot([pixel_number, pixel_number], [140, start_row], "r", linewidth=0.5)
#        plt.plot([pixel_number, pixel_number], [end_row, 205], "r", linewidth=0.5)
#        
#        straylight_line_top = y[140:int(start_row), pixel_number]
#        straylight_line_bottom = y[int(end_row):205, pixel_number]
#        
#        straylight_line_numbers = np.concatenate((np.arange(140, int(start_row)), np.arange(int(end_row), 205)))
#        straylight_line_values = np.concatenate((straylight_line_top, straylight_line_bottom))
#        straylight_fit_coeffs = np.polyfit(straylight_line_numbers, straylight_line_values, 1)
#        straylight_fit = np.polyval(straylight_fit_coeffs, np.arange(140, 205))
#        
#        y_straylight[140:205, pixel_number] = straylight_fit
#        
#        corrected_line = y[140:205, pixel_number] - straylight_fit
#        y_corrected[140:205, pixel_number] = corrected_line    
#    #for corner, corner2 in zip(corners, corners2):
#    #    plt.scatter(corner[0], corner[1], c="k")
#    #    plt.plot([corner[0], corner2[0]], [corner[1], corner2[1]], "k")
#        
#    plt.ylim([210, 135])
#    
#    plt.imshow(y, aspect = 5)
#    
#    plt.figure(figsize=(FIG_X, FIG_Y))
#    for pixel_number in [200, 500, 800, 1000]:
#        
#        plt.plot(range(140,205), y[140:205, pixel_number], "--", label="Pixel %i uncorrected" %pixel_number)
#        plt.plot(range(140,205), y_corrected[140:205, pixel_number], label="Pixel %i corrected" %pixel_number)
#        
##    corrected_spectra.append()
#    plt.legend()
#
#    fig, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
#
#    ax1.plot(y_corrected[171, :], "k", label="Y Corrected Row 171")
#    ax2 = ax1.twinx()
#    ax2.plot(y_straylight[171, :], "r", label="Y Straylight Row 171")
#    ax1.set_ylabel("Y Corrected", color="k")
#    ax2.set_ylabel("Y Straylight", color="r")
#    ax2.tick_params(axis='y', labelcolor="r")
#    
#    ax1.legend()
#    ax2.legend()
#    
    
    
    

    
    
    
    
    
    
from scipy.signal import savgol_filter

MEDIAN_FILTER_DEGREE = 3
#test UVIS nadir straylight

rs12_calibration_filename = r"C:\Users\iant\Documents\DATA\UVIS\20150402_111234_0p1a_UVIS.h5"

nadir_filename = r"C:\Users\iant\Documents\DATA\UVIS\20190110_001352_0p2a_UVIS_D.h5"

with h5py.File(rs12_calibration_filename, "r") as f:
    y_rs12 = f["Science/Y"][...]


with h5py.File(nadir_filename, "r") as f:
    y_nadir = f["Science/Y"][...]

v_start_rs12 = 107
v_end_rs12 = 230
 
frame_numbers = list(range(2,12))
frame = np.zeros((len(frame_numbers), 256, 1048)) * np.nan
y_dark = np.mean((median_filter(y_rs12[1, v_start_rs12:v_end_rs12, :], MEDIAN_FILTER_DEGREE), median_filter(y_rs12[12, v_start_rs12:v_end_rs12, :], MEDIAN_FILTER_DEGREE)), axis=0)

for frame_index, frame_number in enumerate(frame_numbers):
    y_frame = median_filter(y_rs12[frame_index, v_start_rs12:v_end_rs12, :], MEDIAN_FILTER_DEGREE) #107:230

    frame[frame_index, v_start_rs12:v_end_rs12, :] = y_frame - y_dark
    
mean_frame = np.mean(frame, axis=0) 

#plt.figure()
#plt.imshow(mean_frame)

plt.figure(figsize=(FIG_X, FIG_Y))
plt.xlabel("Detector Row")
plt.ylabel("Counts")
column = np.zeros_like(y_rs12[0, :, 0])
smoothed_rs12 = np.zeros_like(y_rs12[0, :, :400])
for pixel_number in range(400):
    
    column[107:230] = mean_frame[107:230, pixel_number]
    
    smoothed_rs12[107:230, pixel_number] = savgol_filter(column[107:230], 39, 2)
    if np.mod(pixel_number, 50) == 10:
        plt.plot(column, label="RS12 pixel %i" %pixel_number)
        plt.plot(smoothed_rs12[:, pixel_number], label="RS12 pixel %i smoothed" %pixel_number)
plt.legend()    

#get x from OU file
with h5py.File(r"C:\Users\iant\Documents\DATA\UVIS\from_OU\20180507_050656_1p0_UVIS_U.h5", "r") as f:
    wavelengths = f["Science/Wavelength"][...]

#now do nadir calibration



y_dark = y_nadir[1, :, :]
y_nadir = y_nadir - y_dark

v_start = 57
v_end = 239
    

for frame_index in [120]:
    
    y = np.zeros((256, 1048)) * np.nan

    
    y_filtered = median_filter(y_nadir[frame_index ,v_start:v_end, :], MEDIAN_FILTER_DEGREE)
    y[v_start:v_end, :] = y_filtered

#    plt.figure()
#    plt.imshow(y, aspect=3)
    
    y_corrected = np.zeros_like(y)
    y_straylight = np.zeros_like(y)
    
    pixels = np.arange(1024)
    
    start_rows = [120, 115, 113, 106, 104, 100]
    end_rows = [222, 227, 228, 232, 234, 235]
    
    columns = [0, 390, 444, 765, 843, 1023] #columns 0 to 1023
    
    corners = [[390, v_start], [1023, v_start], [1023, v_end], [390, v_end], [390, v_start]]
    
    corners2 = corners[1::] + [corners[0]]
    columns2 = columns[1::] + [columns[-1]]
    start_rows2 = start_rows[1::] + [start_rows[-1]]
    end_rows2 = end_rows[1::] + [end_rows[-1]]
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.imshow(y, aspect=3)
    
    for column, column2, start, start2, end, end2 in zip(columns, columns2, start_rows, start_rows2, end_rows, end_rows2):
        plt.scatter(column, start, c="b")
        plt.scatter(column, end, c="b")
        plt.plot([column, column2], [start, start2], "b")
        plt.plot([column, column2], [end, end2], "b")
    
    start_rows_interp = np.interp(pixels, columns, start_rows)
    end_rows_interp = np.interp(pixels, columns, end_rows)
    
    for pixel_number, start_row, end_row  in zip(pixels, start_rows_interp, end_rows_interp):

        start_row = int(start_row)
        end_row = int(end_row)
        if pixel_number > 389:
            plt.plot([pixel_number, pixel_number], [v_start, start_row], "r", linewidth=0.5)
            plt.plot([pixel_number, pixel_number], [end_row, v_end], "r", linewidth=0.5)
            
            straylight_line_top = y[v_start:start_row, pixel_number]
            straylight_line_bottom = y[end_row:v_end, pixel_number]
            
            straylight_line_numbers = np.concatenate((np.arange(v_start, start_row), np.arange(end_row, v_end)))
            straylight_line_values = np.concatenate((straylight_line_top, straylight_line_bottom))
            straylight_fit_coeffs = np.polyfit(straylight_line_numbers, straylight_line_values, 1)
            straylight_fit = np.polyval(straylight_fit_coeffs, np.arange(v_start, v_end))
            
            y_straylight[v_start:v_end, pixel_number] = straylight_fit
            
            corrected_line = y[v_start:v_end, pixel_number] - straylight_fit
            y_corrected[v_start:v_end, pixel_number] = corrected_line 
            
        else:
            #step 1 - rs12 fit
            line = y[:, pixel_number]
            rs12_line = smoothed_rs12[:, pixel_number]
            
            pixels_above = np.arange(v_start_rs12, start_row)
            line_above = line[v_start_rs12:start_row]
            rs12_line_above = rs12_line[v_start_rs12:start_row]
            pixels_below = np.arange(end_row, v_end_rs12)
            line_below = line[end_row:v_end_rs12]
            rs12_line_below = rs12_line[end_row:v_end_rs12]
            
            rs12_fit = rs12_line * np.mean((np.mean(line_above / rs12_line_above), (np.mean(line_below / rs12_line_below))))
            corrected_line = line - rs12_fit
            corrected_line[:v_start_rs12] = np.nan
            corrected_line[v_end_rs12:] = np.nan

            y_straylight[:, pixel_number] = rs12_fit
            y_corrected[:, pixel_number] = corrected_line
                
            plt.plot([pixel_number, pixel_number], [v_start_rs12, start_row], "g", linewidth=0.5)
            plt.plot([pixel_number, pixel_number], [end_row, v_end_rs12], "g", linewidth=0.5)
            
            
            #step 2 - linear interpolation
#            straylight_line_top = y_corrected[pixels_above, pixel_number]
#            straylight_line_bottom = y_corrected[pixels_below, pixel_number]
#            
#            straylight_line_numbers = np.concatenate((pixels_above, pixels_below))
#            straylight_line_values = np.concatenate((straylight_line_top, straylight_line_bottom))
#            straylight_fit_coeffs = np.polyfit(straylight_line_numbers, straylight_line_values, 1)
#            straylight_fit = np.polyval(straylight_fit_coeffs, np.arange(v_start_rs12, v_end_rs12))
#            
#            y_straylight[v_start_rs12:v_end_rs12, pixel_number] = straylight_fit
#            
#            corrected_line = y_corrected[v_start_rs12:v_end_rs12, pixel_number] - straylight_fit
#            y_corrected[v_start_rs12:v_end_rs12, pixel_number] = corrected_line 
            
            
    #for corner, corner2 in zip(corners, corners2):
    #    plt.scatter(corner[0], corner[1], c="k")
    #    plt.plot([corner[0], corner2[0]], [corner[1], corner2[1]], "k")
        
#    plt.ylim([210, 135])
    
    
    #plot continuum removal
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.xlabel("Detector Row")
    plt.ylabel("Counts")
    for pixel_number in [400, 500, 800, 1000]:
        
        plt.plot(y[:, pixel_number], "--", label="Pixel %i uncorrected" %pixel_number)
        plt.plot(y_corrected[:, pixel_number], label="Pixel %i corrected" %pixel_number)
        
#    corrected_spectra.append()
    plt.legend()
    
    #plot rs12 fit
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.xlabel("Detector Row")
    plt.ylabel("Counts")
    for pixel_number in [100, 200, 300]:
        
        plt.plot(y[:, pixel_number], "--", label="Pixel %i uncorrected" %pixel_number)
        plt.plot(y_corrected[:, pixel_number], label="Pixel %i corrected" %pixel_number)
    plt.legend()


    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.xlabel("Detector Row")
    plt.ylabel("Counts")

    plt.plot(line, label="UVIS detector column")
    plt.plot(rs12_line, label="RS12 detector column smoothed")
    plt.plot(rs12_fit, label="RS12 column scaled to UVIS")
#    plt.plot(rs12_corrected_line)
    plt.plot(pixels_above, line_above)
    plt.plot(pixels_above, rs12_line_above)
    plt.plot(pixels_below, line_below)
    plt.plot(pixels_below, rs12_line_below)
    plt.legend()

    fig, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
    ax1.plot(np.mean(y_corrected[150:200, :], axis=0), "k", label="Y Corrected Rows 150-200")
    ax2 = ax1.twinx()
    ax2.plot(y_straylight[171, :], "r", label="Y Straylight Row 171")
    ax1.set_ylabel("Y Corrected", color="k")
    ax2.set_ylabel("Y Straylight", color="r")
    ax2.tick_params(axis='y', labelcolor="r")
    ax1.set_xlabel("Pixel Number")
    ax1.set_ylim((0, 22000))
    ax2.set_ylim((0, 1750))
    ax1.legend()
    ax2.legend()
    
       