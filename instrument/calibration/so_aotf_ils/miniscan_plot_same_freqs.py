# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:02:29 2021

@author: iant

PLOT ALL SPECTRA OF A SINGLE AOTF FREQ
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

from instrument.nomad_so_instrument import nu_grid, F_blaze, nu_mp, spec_res_order, F_aotf_goddard18b
from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.nu_hr_grid import nu_hr_grid




from tools.file.hdf5_functions import make_filelist
from tools.file.paths import paths#, FIG_X, FIG_Y

from tools.plotting.colours import get_colours
from tools.plotting.anim import make_line_anim

from tools.sql.get_sql_spectrum_temperature import get_sql_temperatures_all_spectra

chosen = {26666.:{"label":[], "spectrum":[], "temperature":[], "colour":[]}}
order_range = [192,197]

file_level = "hdf5_level_0p2a"


SIMULATE = True
# SIMULATE = False
SIMULATION_ADJACENT_ORDERS = 1

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


regex = re.compile("20190416_020948_0p2a_SO_1_C")
# regex = re.compile("20181129_002850_0p2a_SO_2_C")

# regex = re.compile("(20190416_020948_0p2a_SO_1_C|20181129_002850_0p2a_SO_2_C|20181010_084333_0p2a_SO_2_C|20190416_024455_0p2a_SO_1_C|20190107_015635_0p2a_SO_2_C|20190307_011600_0p2a_SO_1_C)")


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


pixels = np.arange(320)
colours = get_colours(len(hdf5Filenames), cmap="tab10")





for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    

    detector_data_all = hdf5_file["Science/Y"][...]
    # temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])
    
    temperatures = get_sql_temperatures_all_spectra(hdf5_file, channel)

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    unique_aotf = sorted(list(set(aotf_freq)))

    unique_indices_all = [[i for i,v in enumerate(aotf_freq) if v==aotf] for aotf in unique_aotf]
    min_elements = min([len(i) for i in unique_indices_all])
    unique_indices = [i[0:min_elements] for i in unique_indices_all]
    
    unique_aotf_freqs = np.asfarray([aotf_freq[i[0]] for i in unique_indices])
    aotf_stepping = unique_aotf_freqs[1] - unique_aotf_freqs[0]
    
    
   
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels


    # d = {i:[] for i in unique_aotf_freqs}
    for freq_index, frame_nos in enumerate(unique_indices):
        
        for chosen_aotf in chosen.keys():
            if chosen_aotf == unique_aotf_freqs[freq_index]:
                
                for n, frame_no in enumerate(frame_nos):
                    mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)
                    
                    chosen[chosen_aotf]["label"].append("%s: %i (T=%0.3f)" %(hdf5_filename[0:8], n, temperatures[frame_no]))
                    chosen[chosen_aotf]["spectrum"].append(mean)
                    chosen[chosen_aotf]["temperature"].append(temperatures[frame_no])
                    chosen[chosen_aotf]["colour"].append(colours[file_index])
     
 
    
 
if SIMULATE:
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
    
    c_order = int(np.mean(order_range))
    spec_res = spec_res_order(c_order)
      
    
            
        
for chosen_aotf in chosen.keys():
    plt.subplots()
    for label, spectrum, temperature, colour in zip(chosen[chosen_aotf]["label"], chosen[chosen_aotf]["spectrum"], chosen[chosen_aotf]["temperature"], chosen[chosen_aotf]["colour"]):
        plt.plot(spectrum, label=label, color=colour)
        
    
        """spectral grid and blaze functions of all orders"""
        if SIMULATE:
            dnu = 0.001
            nu_range = [
                nu_mp(order_range[0], [0.0], temperature)[0] - 5.0, \
                nu_mp(order_range[1], [319.0], temperature)[0] + 5.0            
                    ]
            nu_hr = np.arange(nu_range[0], nu_range[1], dnu)
            
            
            I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
            
            
            Nbnu_hr = len(nu_hr)
            NbP = len(pixels)
            
            sconv = spec_res/2.355
            W_conv_old = np.zeros((NbP,Nbnu_hr))
            W_conv_new = np.zeros((NbP,Nbnu_hr))
            print("###########")
            for iord in range(order_range[0], order_range[1]+1):
                print("Blaze order %i" %iord)
                nu_pm = nu_mp(iord, pixels, temperature)
                W_blaze_old = F_blaze(iord, pixels, temperature)
                W_blaze_new = F_blaze_goddard21(iord, pixels, temperature)
                for ip in pixels:
                    W_conv_old[ip,:] += (W_blaze_old[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
                    W_conv_new[ip,:] += (W_blaze_new[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
    
    
            W_aotf_old = F_aotf_goddard18b(0., nu_hr, temperature, A=aotf_freq[frame_no])
            I0_hr_old = W_aotf_old * I0_solar_hr
            I0_p_old = np.matmul(W_conv_old, I0_hr_old)
            solar_old = I0_p_old
    
            W_aotf_new = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[frame_no])
            I0_hr_new = W_aotf_new * I0_solar_hr
            I0_p_new = np.matmul(W_conv_new, I0_hr_new)
            solar_new = I0_p_new
            
            
            solar_norm_old = solar_old/max(solar_old)
            solar_norm_new = solar_new/max(solar_new)
            
            solar_scaled_old = solar_norm_old * np.max(chosen[chosen_aotf]["spectrum"])
            solar_scaled_new = solar_norm_new * np.max(chosen[chosen_aotf]["spectrum"])
    
            plt.plot(solar_scaled_old, label=label + " old sim", color=colour, linestyle="--")
            plt.plot(solar_scaled_new, label=label + " new sim", color=colour, linestyle=":")


plt.legend()
plt.xlabel("Pixel Number")
plt.ylabel("Counts")
plt.title("All miniscan spectra for AOTF frequency %0.0fkHz" %chosen_aotf)

#     # sg_window = {1.0:19, 2.0:19, 4.0:19, 8.0:9}[aotf_stepping]
#     sg_window = {1.0:19, 2.0:19, 4.0:19, 8.0:9}[aotf_stepping]
        
        
#     sum_spectra = []
#     for i, key in enumerate(d.keys()):
#         sum_spectra.append(d[key][0][180])
#         # sum_spectra.append(np.sum(d[key][0]))
#     sum_spectra = np.array(sum_spectra)
        
#     sg = savgol_filter(sum_spectra, sg_window, 1)
    
#     values_norm = sum_spectra/sg
#     plot_offset = float(file_index) * 0.05

#     amp = 0.018
#     freq = 0.38492
#     temp_coeffs = np.array([-7.84038467, 55.42655564])
#     first_peak = np.polyval(temp_coeffs, temperature)
#     x_offset = first_peak - 6.0
    
#     sine = sin(unique_aotf_freqs, amp, freq, x_offset, 1.0)

#     values_norm_corr = values_norm - sine

#     plt.scatter(unique_aotf_freqs, values_norm + plot_offset, label="%ikHz stepping, T=%0.1fC" %(aotf_stepping, temperature), color=colours[file_index], alpha=0.7)
#     plt.plot(unique_aotf_freqs, values_norm + plot_offset, color=colours[file_index], alpha=0.7)
#     plt.plot(unique_aotf_freqs, values_norm_corr + plot_offset + 1.05, color=colours[file_index], alpha=1.0)
#     plt.plot(unique_aotf_freqs, sine + plot_offset, color="b", alpha=1.0)
            
# # x_hr = np.arange(26590.0, 28630.0, 0.1)

