# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:07:05 2021

@author: iant

plot detector slant v2
"""

import numpy as np
# import os
import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import scipy.signal as ss

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist

from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.spectra.fit_polynomial import fit_polynomial
from tools.plotting.colours import get_colours

file_level = "hdf5_level_0p1a"

# regex = re.compile("20150404_072956_.*_SO_.") #order 134 CH4
# good_indices = range(240,250)
# max_chisq = 0.05
# absorption_indices = np.arange(133,141,1)
# absorption_indices = np.arange(182,191,1)
# absorption_indices = np.arange(305,314,1)


# regex = re.compile("20150404_083818_.*_SO_.") 
# good_indices = range(230,240)
# max_chisq = 0.005
# absorption_indices = np.arange(175,183,1)
# # absorption_indices = np.arange(308,315,1)


# regex = re.compile("20150404_114517_.*_SO_.")  #order 186 CO
# good_indices = range(120,130)
# max_chisq = 0.005
# absorption_indices = np.arange(104,115,1)
# # absorption_indices = np.arange(147,152,1)


#LNO post detector swap
regex = re.compile("20150425_074022_.*_LNO_.")  #order 129 CH4 T=-15
good_indices = range(120,130)
max_chisq = 0.005
absorption_indices = np.arange(242,249)

regex = re.compile("20150427_081547_.*_LNO_.")  #order 129 CH4
good_indices = range(120,130)
max_chisq = 0.005
absorption_indices = np.arange(265,271)


# regex = re.compile("20150425_130615_.*_LNO_.")  #order 159 CO2 T=-15
# good_indices = range(200,210,1)
# max_chisq = 0.005
# absorption_indices = np.arange(162,168)

# regex = re.compile("20150425_130615_.*_LNO_.")  #order 159 CO2T=-15
# good_indices = range(200,210,1)
# max_chisq = 0.005
# absorption_indices = np.arange(281,289)

# regex = re.compile("20150427_123133_.*_LNO_.")  #order 159 CO2 T=10
# good_indices = range(200,210,1)
# max_chisq = 0.005
# absorption_indices = np.arange(243,251)

regex = re.compile("20150427_123133_.*_LNO_.")  #order 159 CO2 T=10
good_indices = range(200,210,1)
max_chisq = 0.005
absorption_indices = np.arange(182,188)



hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    
    detector_rows = {"so":[128-8, 128+8], "lno":[152-72, 152+72]}[channel]
    


    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])
    

    dim = detector_data_all.shape
    n_rows_raw = dim[1] #data rows
    n_rows_binned = dim[1] * binning #pixel detector rows
    frame_indices = np.arange(dim[0])
    n_u = len(list(set(window_top_all))) #number of unique window tops
    n_ff = int(np.floor(dim[0]/n_u)) #number of full frames
    
    colours = get_colours(dim[1])
    colours2 = get_colours(len(good_indices), cmap="brg")
    
    #if not window stepping
    row_no = np.arange(window_top_all[0], window_top_all[0]+n_rows_binned, binning)


    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])

    minima = {}

    for index,i in enumerate(good_indices): #loop through good frames in file
        
        plt.figure(figsize = (FIG_X, FIG_Y))
        plt.title("Frame %i (%i)" %(i, orders[i]))
        y_mean = np.mean(detector_data_all[i, 12, 160:240])
        y_std = np.std(detector_data_all[i, 12, 160:240])
        
        minima[i] = {"row_no":[], "min_pixel":[], "colour":colours2[index]}
        
        
        for j in range(n_rows_raw): #loop through detector rows
            
            if np.mean(detector_data_all[i, j, 160:240]) > 0.3*y_mean:
                spectrum = detector_data_all[i, j, :]
            
                continuum = baseline_als(spectrum)
                absorption = spectrum/continuum
                
                plt.plot(absorption, color=colours[j], label="i=%i, row=%i" %(i,row_no[j]))
                
                # sav_gol = savgol_filter(spectrum, 9, 2)
                # plt.plot(sav_gol, color=colours[j], linestyle=":", label="i=%i, row=%i" %(i,row_no[j]))
                
                # oversampled = ss.resample(spectrum, 640)
                # plt.plot(np.arange(0, 320, 0.5), oversampled, color=colours[j], linestyle="--", label="i=%i, row=%i" %(i,row_no[j]))
                
                rel_indices = absorption_indices - np.mean(absorption_indices)
                gaussian = fit_gaussian_absorption(rel_indices, absorption[absorption_indices], error=True)
                # print("i=%i, row=%i" %(i,row_no[j]), gaussian[3])
                if len(gaussian[0])>0: #if not error
                    if gaussian[3] < max_chisq:
                        plt.plot(absorption_indices[0]-rel_indices[0]+gaussian[0], gaussian[1], color=colours[j], linestyle="--")
                        minima[i]["row_no"].append(row_no[j])
                        minima[i]["min_pixel"].append(absorption_indices[0]-rel_indices[0]+gaussian[2])
                    else:
                        print("chisq too high", i, row_no[j], gaussian[3])
                else:
                    print("fitting error", i, row_no[j])
            # else:
            #     print("Signal too small", i, row_no[j])
                        
        plt.ylim((0.6, 1.1))

    plt.legend()
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_title("Detector Slant, Ground Calibration Order %i, T=%0.1fC" %(orders[i], temperature))
    ax.set_xlabel("Pixel of absorption minimum")
    ax.set_ylabel("Detector row")
    for i in minima.keys():
        if len(minima[i]["min_pixel"])>0:
            ax.scatter(minima[i]["min_pixel"], minima[i]["row_no"], color=minima[i]["colour"], label="Frame %i" %(i))
            # ax.plot(minima[i]["min_pixel"], fit_polynomial(minima[i]["min_pixel"], minima[i]["row_no"]), color=minima[i]["colour"])

            # linear_coeffs = fit_polynomial(minima[i]["min_pixel"], minima[i]["row_no"], degree=2, coeffs=True)[1]
            # y_range = [detector_rows[0], detector_rows[-1]+1]
            # x_range = [(y_range[0]- linear_coeffs[1])/linear_coeffs[0], (y_range[-1]- linear_coeffs[1])/linear_coeffs[0]]
            # ax.plot(x_range, y_range, color=minima[i]["colour"])

    ax.axhline(y=detector_rows[0], color="k", linestyle="--")
    ax.axhline(y=detector_rows[1], color="k", linestyle="--")
    ax.invert_yaxis()
    ax.legend()

    print(temperature, "C")

