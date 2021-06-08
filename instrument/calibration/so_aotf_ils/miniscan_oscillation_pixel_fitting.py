# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:34:31 2021

@author: iant


MINSCAN OSCILLATION FITTING
"""
import numpy as np
import os
import re
# import json
from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
# import scipy.signal as ss
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths#, FIG_X, FIG_Y

from tools.plotting.colours import get_colours
# from tools.plotting.anim import make_line_anim

def sin(x, A, B, C, D):  
    return A * np.sin(B * x + C) + D

def sin_slope(x, A, B, C, D, E):  
    return A * np.sin(B * x + C) + D + E * x


def sin_in_sin(x, A, B, C, D, A2, B2, C2, D2):  
    return (A * np.sin(B * x + C) + D) + A2 * np.sin(B2 * x + C2) + D2

def sin_in_sin_slope(x, A, B, C, D, A2, B2, C2, D2, E):  
    return A * np.sin(B * x + C) + A2 * np.sin(B2 * x + C2) + D + E * x


file_level = "hdf5_level_0p2a"

CHOSEN_PIXELS = [10, 150, 250]


#1 khz 
# 20190223_054340_0p2a_SO_1_C # orders 125-127
# 20190223_054340_0p2a_SO_2_C
# 20190223_061847_0p2a_SO_1_C
# 20190223_061847_0p2a_SO_2_C
# regex = re.compile("20190223_(054340|061847)_0p2a_SO_._C")



#order 194:
#1 khz
# 20190416_020948_0p2a_SO_1_C

#2 khz
# 20181129_002850_0p2a_SO_2_C

#4 khz
# 20181010_084333_0p2a_SO_2_C
# 20190416_024455_0p2a_SO_1_C

#8 khz
# 20190107_015635_0p2a_SO_2_C
# 20190307_011600_0p2a_SO_1_C
#


# regex = re.compile("20190416_020948_0p2a_SO_1_C")
# regex = re.compile("20181129_002850_0p2a_SO_2_C")
regex = re.compile("20181010_084333_0p2a_SO_2_C")

# regex = re.compile("(20190416_020948_0p2a_SO_1_C|20181129_002850_0p2a_SO_2_C|20181010_084333_0p2a_SO_2_C|20190416_024455_0p2a_SO_1_C|20190107_015635_0p2a_SO_2_C|20190307_011600_0p2a_SO_1_C)")


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


pixels = np.arange(320)
colours = get_colours(len(hdf5Filenames), cmap="viridis")

# plt.figure()

fits = {"hdf5_filename":[], "y":[], "temperature":[], "aotf_stepping":[], "amp":[], "freq":[], "x_offset":[], "y_offset":[], "first_peak":[], "values_norm":[]}

for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    
    # detector_rows = {"so":[128-8, 128+8], "lno":[152-72, 152+72]}[channel]

    d = {"text":[], "text_position":[5,20000], "xlabel":"Pixel", "ylabel":"", "xlim":[0, 319]}
    d["legend"] = {"on":True, "loc":"lower right"}
    # d["keys"] = ["raw", "1_orders", "2_orders", "3_orders"]
    d["filename"] = hdf5_filename+"_new"
    d["format"] = "ffmpeg"
    d["save"] = False
    
    

    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    unique_aotf = sorted(list(set(aotf_freq)))

    unique_indices_all = [[i for i,v in enumerate(aotf_freq) if v==aotf] for aotf in unique_aotf]
    min_elements = min([len(i) for i in unique_indices_all])
    unique_indices = [i[0:min_elements] for i in unique_indices_all]
    
    unique_aotf_freqs = np.asfarray([aotf_freq[i[0]] for i in unique_indices])
    aotf_stepping = unique_aotf_freqs[1] - unique_aotf_freqs[0]
    
    d["x"] = {"spectrum_%i" %i:[] for i in range(min_elements) }
    d["y"] = {"spectrum_%i" %i:[] for i in range(min_elements) }
    
    
    
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])
    
    order_range_file = [min(orders), max(orders)]

    dim = detector_data_all.shape
    
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
    dim_rows = detector_centre_data.shape
    
    good_indices = range(dim[0])

 
    # d["title"] = "%s simulation orders %i-%i" %(hdf5_filename, order_range[0], order_range[1])

    d["text"] = ["A=%ikHz" %i for i in aotf_freq[good_indices]]


    
    for frame_index, frame_nos in enumerate(unique_indices):
 
        out = {}
        y_max_frame = []
        for n, frame_no in enumerate(frame_nos):
            std = np.std(detector_centre_data[frame_no, :, :], axis=0)
            mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)

            out["spectrum_%i" %n] = mean
        
            
            d["x"]["spectrum_%i" %n].append(pixels)
            d["y"]["spectrum_%i" %n].append(mean)
            
            y_max_frame.append(mean)
        
        # plt.plot(solar_norm, label="Solar spectrum full simulation")
        # plt.legend()

        
        filename = os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename, "%s_solar_simulation_%s_%04i_%ikHz.json" %(channel, hdf5_filename, frame_no, aotf_freq[frame_no]))
        
        # if not os.path.exists(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename)):
        #     os.makedirs(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename))
        # write_json(filename, out)
        
        
    y_max = np.max([np.max(d["y"][key]) for key in d["y"].keys()])
    d["ylim"] = [0, y_max]
    # d["y"]["simulation"] = [spectrum * y_max for spectrum in d["y"]["simulation"]]

    # make_line_anim(d)
    
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i, key in enumerate(d["y"].keys()):
        # if i == 0:
        for chosen_px in CHOSEN_PIXELS:
            
            sg_window = {1.0:19, 2.0:19, 4.0:19, 8.0:9}[aotf_stepping]
            
            px_values = np.asfarray(d["y"][key])[:, chosen_px]
            # px_values = np.mean(np.asfarray(d["y"][key]), axis=1)
            # sg = savgol_filter(px_values, sg_window, 1)
            
            
            plot_offset = float(file_index) * 0.05
            
            # plt.scatter(unique_aotf_freqs, values_norm + plot_offset, label="%ikHz stepping, T=%0.1fC" %(aotf_stepping, temperature), color=colours[file_index], alpha=0.7)
            # plt.plot(unique_aotf_freqs, values_norm + plot_offset, color=colours[file_index], alpha=0.7)
            

            amp = 4000.
            freq = 0.385
            x_offset = -180.0
            y_offset = 1.0

            amp2 = 35000.0
            freq2 = 0.04
            x_offset2 = 1.0
            y_offset2 = 150000.0
            slope2 = -0.2
            
            guess_sin_in_sin_slope = [amp, freq, x_offset, y_offset2, amp2, freq2, x_offset2, 0.0, slope2]
            guess_sin_slope = [amp2, freq2, x_offset2, y_offset2, slope2]
            
            x_hr = np.arange(26590.0, 28630.0, 0.1)
    
            #fit all in one
            # popt, pcov = curve_fit(sin_in_sin_slope, unique_aotf_freqs, px_values, p0=guess_sin_in_sin_slope)
            # fit_aotf_wave = sin_in_sin_slope(unique_aotf_freqs, *popt)
 
            #fit big curve + slope
            popt, pcov = curve_fit(sin_slope, unique_aotf_freqs, px_values, p0=guess_sin_slope)
            fit_aotf_wave = sin_slope(unique_aotf_freqs, *popt)

            values_aotf_wave = px_values / fit_aotf_wave
            
            ax1.plot(unique_aotf_freqs, px_values)
            ax1.plot(unique_aotf_freqs, fit_aotf_wave)
            
            # plt.figure()
            ax2.plot(unique_aotf_freqs, values_aotf_wave)
            
            # plt.plot(unique_aotf_freqs, sin_in_sin_slope(unique_aotf_freqs, *guess2))
            
            
            # # sg = savgol_filter(values_aotf_wave, sg_window, 1)
            
            # values_norm_aotf_wave = values_aotf_wave#/sg


            # guess = [amp, freq, x_offset, y_offset]


            # popt2, pcov2 = curve_fit(sin, unique_aotf_freqs, values_norm_aotf_wave, p0=guess)
            # fit_ringing = sin(unique_aotf_freqs, *popt2)
            # values_norm_ringing = values_norm_aotf_wave / fit_ringing
            
            # plt.plot(unique_aotf_freqs, values_norm_aotf_wave)
            # plt.plot(unique_aotf_freqs, fit_ringing)
            # plt.plot(unique_aotf_freqs, values_norm_ringing)
            
            # fit_hr = sin_slope(x_hr, *popt)
            # # plt.plot(x_hr, fit_hr + plot_offset, color=colours[file_index], linestyle=":", alpha=0.7)

            # fit_hr2 = sin(x_hr, *popt2)
            # # plt.plot(x_hr, fit_hr + fit_hr2 + plot_offset, color=colours[file_index], linestyle="--", alpha=0.7)

            # print(hdf5_filename, aotf_stepping, popt)
            
            
            # fits["y"].append(d["y"][key])
            # fits["values_norm"].append(values_norm)
            # fits["first_peak"].append(find_peaks(fit_hr)[0][0])
            # fits["amp"].append(popt[0])
            # fits["freq"].append(popt[1])
            # fits["hdf5_filename"].append(hdf5_filename)
            # fits["aotf_stepping"].append(aotf_stepping)
            # fits["temperature"].append(temperature)

# plt.legend()
# plt.xlabel("AOTF frequency")
# plt.ylabel("Normalised counts after continuum removal")
    
# plt.figure()
# plt.scatter(fits["temperature"], fits["first_peak"])

# polyfit = np.polyfit(fits["temperature"], fits["first_peak"], 1)
# plt.plot(fits["temperature"], np.polyval(polyfit, fits["temperature"]))
# plt.xlabel("Temperature")
# plt.ylabel("Position of first peak")

# #correct spectra
# index = 4

# amp = 0.018
# freq = 0.38492
# temp_coeffs = np.array([-7.84038467, 55.42655564])
# first_peak = np.polyval(temp_coeffs, fits["temperature"][index])
# x_offset = first_peak


# sin(x_hr, amp, freq, x_offset, 1.0)

# plt.figure()

# spectrum = np.asfarray(fits["y"][index])[:, 180]
# plt.plot(spectrum)

# spectra = np.asfarray(fits["y"][1])

# # plt.plot(spectra[:50, :].T)
# # plt.xlabel("Pixel")
# # plt.ylabel("Counts")

# plt.plot(spectra[:, 180])
# plt.xlabel("Frame number")
# plt.ylabel("Counts")












