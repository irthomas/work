# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:47:33 2021

@author: iant

TEST AOTF FITTING. THIS VERSION CALCULATES THE TOTAL BLAZE + AOTF CONTRIBUTION FOR [0,1,2,3] ADJACENT ORDERS
AND EITHER:
    PLOTS THE ANIMATION OR
    SAVES THE RAW MINISCAN SPECTRA AND SIMULATIONS TO JSONS FOR PLOTING AN ANIMATION.

THIS VERSION AVOIDS THE STEP CHANGE, USING THE SAME WIDE RANGE SOLAR AND ADJACENT ORDERS FOR ALL AOTF FREQUENCIES 

"""




import numpy as np
import os
import re
import json
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import scipy.signal as ss

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist
from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict

from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from instrument.nomad_so_instrument import nu_grid, F_blaze, nu_mp, spec_res_order, F_aotf_goddard18b
from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.nu_hr_grid import nu_hr_grid


from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.spectra.fit_polynomial import fit_polynomial
from tools.plotting.colours import get_colours

from tools.plotting.anim import make_line_anim

from tools.sql.get_sql_spectrum_temperature import get_sql_temperatures_all_spectra


file_level = "hdf5_level_0p2a"


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
# regex = re.compile("20181010_084333_0p2a_SO_2_C|20190416_024455_0p2a_SO_1_C")

# regex = re.compile("(20181010_084333_0p2a_SO_2_C|20190416_024455_0p2a_SO_1_C|20190107_015635_0p2a_SO_2_C|20190307_011600_0p2a_SO_1_C)")


SIMULATION_ADJACENT_ORDERS = 2
SIMULATE = True
# SIMULATE = False
D_NU = 0.001

hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


linestyles = {0:"-", 1:":", 2:"--", 3:"-."}
pixels = np.arange(320)
ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    
    # detector_rows = {"so":[128-8, 128+8], "lno":[152-72, 152+72]}[channel]

    d = {"text":[], "text_position":[5,20000], "xlabel":"Pixel", "ylabel":"", "xlim":[0, 319]}
    d["legend"] = {"on":True, "loc":"lower right"}
    d["filename"] = hdf5_filename+"_new"
    d["format"] = "ffmpeg"

    d["save"] = True
    # d["save"] = False

    
    

    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    
    
    # temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])
    
    temperatures = get_sql_temperatures_all_spectra(hdf5_file, channel)
    temperature = np.mean(temperatures)

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    unique_aotf = sorted(list(set(aotf_freq)))

    unique_indices_all = [[i for i,v in enumerate(aotf_freq) if v==aotf] for aotf in unique_aotf]
    min_elements = min([len(i) for i in unique_indices_all])
    unique_indices = [i[0:min_elements] for i in unique_indices_all]
    
    d["x"] = {"spectrum_%i (%.1fC)" %(i, temperatures[unique_indices[0][i]]):[] for i in range(min_elements) }
    d["y"] = {"spectrum_%i (%.1fC)" %(i, temperatures[unique_indices[0][i]]):[] for i in range(min_elements) }
    
    if SIMULATE:
        d["x"]["simulation_old"] = []
        d["y"]["simulation_old"] = []
        d["x"]["simulation_new"] = []
        d["y"]["simulation_new"] = []
        d["x"]["simulation_new2"] = []
        d["y"]["simulation_new2"] = []
    
    
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])
    
    order_range_file = [min(orders), max(orders)]
    order_range = [order_range_file[0]- SIMULATION_ADJACENT_ORDERS, order_range_file[1] + SIMULATION_ADJACENT_ORDERS]
    c_order = int(np.mean(order_range))
    spec_res = spec_res_order(c_order)

    dim = detector_data_all.shape
    
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
    dim_rows = detector_centre_data.shape
    
    good_indices = range(dim[0])

    colours = get_colours(len(good_indices))

    d["title"] = "%s simulation orders %i-%i" %(hdf5_filename, order_range[0], order_range[1])

    d["text"] = ["i=%i, A=%ikHz" %(i, a) for i, a in enumerate(aotf_freq[good_indices])]


    """spectral grid and blaze functions of all orders"""
    if SIMULATE:
        dnu = D_NU

        nu_range = [
            nu_mp(order_range[0], [0.0], temperature)[0] - 5.0, \
            nu_mp(order_range[1], [319.0], temperature)[0] + 5.0            
                ]
        nu_hr = np.arange(nu_range[0], nu_range[1], dnu)
        
        
        I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
        
        
        #if convolution already saved to file
        h5_conv_filename = "conv_%s_order%i-%i_dnu%f_temp%f" %(hdf5_filename, order_range[0], order_range[1], D_NU, temperature)
        if os.path.exists(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename+".h5")):
            print("Reading W_conv from existing file")
            W_conv = read_hdf5_to_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename))[0]["W_conv"]

            
        else:
            print("Making file", h5_conv_filename)
            Nbnu_hr = len(nu_hr)
            NbP = len(pixels)
            
            #old and new blaze functions are functional identical - use 2021 function only
            sconv = spec_res/2.355
            W_conv = np.zeros((NbP,Nbnu_hr))
            for iord in range(order_range[0], order_range[1]+1):
                print("Blaze order %i" %iord)
                nu_pm = nu_mp(iord, pixels, temperature)
                W_blaze = F_blaze_goddard21(iord, pixels, temperature)
                for ip in pixels:
                    W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
                    
            W_conv[W_conv < 1.0e-5] = 0.0 #remove small numbers
                    
            write_hdf5_from_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename), {"W_conv":W_conv}, {}, {}, {})
          
    improvement = []
    for frame_index, frame_nos in enumerate(unique_indices):
 
        # if np.mod(frame_index, 10) == 0:
        #     print(frame_index)
            
        out = {}
        y_max_frame = []
        for n, frame_no in enumerate(frame_nos):
            std = np.std(detector_centre_data[frame_no, :, :], axis=0)
            mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)

            out["spectrum_%i" %n] = mean
        
            
            d["x"]["spectrum_%i (%.1fC)" %(n, temperatures[unique_indices[0][n]])].append(pixels)
            d["y"]["spectrum_%i (%.1fC)" %(n, temperatures[unique_indices[0][n]])].append(mean)
            
            y_max_frame.append(mean)
        
        if SIMULATE:
            W_aotf_old = F_aotf_goddard18b(0., nu_hr, temperature, A=aotf_freq[frame_no])
            I0_hr_old = W_aotf_old * I0_solar_hr
            I0_p_old = np.matmul(W_conv, I0_hr_old)
            solar_old = I0_p_old
    
            if frame_index == 0: #print values
                _ = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[frame_no], silent=False)
                # nu0: 4382.640722753343
                # sinc width: 20.739074862906232
                # sidelobe factor: 7.770908740766124
                # asymmetry: 0.28037507899728986

            W_aotf_new = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[frame_no])
            W_aotf_new2 = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[frame_no], wd=20.0)

            # plt.figure(); plt.plot(nu_hr, W_aotf_new); plt.xlabel("Wavenumber cm-1"); plt.title("Goddard AOTF 2021")
            I0_hr_new = W_aotf_new * I0_solar_hr
            I0_p_new = np.matmul(W_conv, I0_hr_new)
            solar_new = I0_p_new

            I0_hr_new2 = W_aotf_new2 * I0_solar_hr
            I0_p_new2 = np.matmul(W_conv, I0_hr_new2)
            solar_new2 = I0_p_new2
            
            
            solar_norm_old = solar_old/max(solar_old)
            solar_norm_new = solar_new/max(solar_new)
            solar_norm_new2 = solar_new2/max(solar_new2)
            
            solar_scaled_old = solar_norm_old * np.max(y_max_frame)
            solar_scaled_new = solar_norm_new * np.max(y_max_frame)
            solar_scaled_new2 = solar_norm_new2 * np.max(y_max_frame)
            
            d["x"]["simulation_old"].append(pixels)
            d["y"]["simulation_old"].append(solar_scaled_old)
            d["x"]["simulation_new"].append(pixels)
            d["y"]["simulation_new"].append(solar_scaled_new)
            d["x"]["simulation_new2"].append(pixels)
            d["y"]["simulation_new2"].append(solar_scaled_new2)
            
            abs_sum_new = np.sum(np.abs(mean - solar_scaled_new))
            abs_sum_new2 = np.sum(np.abs(mean - solar_scaled_new2))
            
            improvement.append(((abs_sum_new - abs_sum_new2)/abs_sum_new)*100.0)
            
            print("%i: %0.1f%% better" %(frame_index, improvement[-1]))
            
            # out["simulation"] = solar_norm_old
        
        # plt.plot(solar_norm, label="Solar spectrum full simulation")
        # plt.legend()

        
        # filename = os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename, "%s_solar_simulation_%s_%04i_%ikHz_new.json" %(channel, hdf5_filename, frame_no, aotf_freq[frame_no]))
        
        # if not os.path.exists(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename)):
        #     os.makedirs(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename))
        # write_json(filename, out)
        
        
    y_max = np.max([np.max(d["y"][key]) for key in d["y"].keys()])
    d["ylim"] = [0, y_max]
    # d["y"]["simulation"] = [spectrum * y_max for spectrum in d["y"]["simulation"]]
    
    print("Improvement = %0.1f%%" %np.mean(improvement))

    make_line_anim(d)

