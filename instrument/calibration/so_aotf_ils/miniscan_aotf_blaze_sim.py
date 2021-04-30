# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:47:33 2021

@author: iant

TEST AOTF FITTING. THIS VERSION CALCULATES THE TOTAL BLAZE + AOTF CONTRIBUTION FOR [0,1,2,3] ADJACENT ORDERS
AND SAVES THE RAW MINISCAN SPECTRA AND SIMULATIONS TO JSONS FOR PLOTING AN ANIMATION.

THERE IS A STEP WHERE THE ORDER CHANGES - NEXT VERSION WILL USE A CONSTANT SOLAR HR GRID AND ORDERS

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
from tools.file.json import write_json

from instrument.nomad_so_instrument import m_aotf as m_aotf_so
from instrument.nomad_lno_instrument import m_aotf as m_aotf_lno

from instrument.nomad_so_instrument import nu_grid, F_blaze, nu_mp, spec_res_order, F_aotf_goddard18b

from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.nu_hr_grid import nu_hr_grid


from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.spectra.fit_polynomial import fit_polynomial
from tools.plotting.colours import get_colours

from tools.plotting.anim import make_line_anim



file_level = "hdf5_level_0p2a"


#1 khz 
# 20190223_054340_0p2a_SO_1_C
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


# regex = re.compile("20190223_061847_0p2a_SO_1_C") #1khz simulations with various adjacent orders
regex = re.compile("20190307_011600_0p2a_SO_1_C")
# regex = re.compile("20210201_111011_0p2a_SO_2_C")  #order 188-202 8khz
# good_indices = range(0, 745, 1)


SIMULATION_ADJACENT_ORDERS = []#[0,1,2,3]

hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)


linestyles = {0:"-", 1:":", 2:"--", 3:"-."}
pixels = np.arange(320)

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    
    print(hdf5_filename)
    
    channel = hdf5_filename.split("_")[3].lower()
    
    # detector_rows = {"so":[128-8, 128+8], "lno":[152-72, 152+72]}[channel]
    


    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])

    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
    if channel == "so":
        orders = [m_aotf_so(a) for a in aotf_freq]
    elif channel == "lno":
        orders = [m_aotf_lno(a) for a in aotf_freq]
    print("Starting Order=%i" %orders[0])
    

    dim = detector_data_all.shape
    
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
    dim_rows = detector_centre_data.shape
    
    good_indices = range(dim[0])

    colours = get_colours(len(good_indices))


    # fig1 = plt.figure(figsize=(FIG_X, FIG_Y))
    # gs = fig1.add_gridspec(4,1)
    # ax1a = fig1.add_subplot(gs[0:3, 0])
    # ax2a = fig1.add_subplot(gs[3, 0], sharex=ax1a)
    
    d = {"x":[pixels], "y":[], "text":[], "text_position":[5,5000], "xlabel":"Pixel", "ylim":[0, 350000], \
         "filename":"%s_miniscan_orders_%i_to_%i" %(channel, min(orders), max(orders))}
        
    
    for frame_index, frame_no in enumerate(good_indices):
        if np.mod(frame_no, 10) == 0:
            print(frame_no)
            
            
        # ax1a.plot(pixels, detector_centre_data[frame_no, row_index, :].T, linestyle=linestyles[row_index], color=colours[frame_index])
        std = np.std(detector_centre_data[frame_no, :, :], axis=0)
        mean = np.mean(detector_centre_data[frame_no, :, :], axis=0)
        # ax2a.plot(pixels[50:], std[50:]/mean[50:], color=colours[frame_index])
        d["y"].append(mean)
        d["text"].append("%ikHz" %aotf_freq[frame_no])
        

        # plt.figure()
        # plt.plot(mean/max(mean), label="Raw")
        
        out = {}
        out["raw"] = mean
        
        c_order = orders[frame_no]
        for adj_orders in SIMULATION_ADJACENT_ORDERS:
        
            """simulation apprxoimation"""
            # px_sum = np.zeros(320)
            # for order in range(c_order-adj_orders, c_order+adj_orders+1, 1):
            #     nu_grid_px = nu_mp(order, pixels, temperature)
            #     blaze = F_blaze(order, pixels, temperature)
            
            #     f_aotf = F_aotf_goddard18b(0., nu=nu_grid_px, t=temperature, A=aotf_freq[frame_no])
                
            #     # solar = np.ones(len(nu_grid_px))
            
            #     solar = get_solar_hr(nu_grid_px)
            #     solar_norm = solar/max(solar)
                
                
            #     px_sum += solar_norm * blaze * f_aotf
                
                # if order == c_order:
                #     plt.plot(blaze, label="Blaze")
                #     plt.plot(f_aotf, label="AOTF")
            
            # plt.plot(px_sum/max(px_sum), label="Solar spectrum approximation")
            
            
            ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
            spec_res = spec_res_order(c_order)
            
            nu_hr, dnu = nu_hr_grid(c_order, adj_orders, temperature)
            I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
            
            
            Nbnu_hr = len(nu_hr)
            NbP = len(pixels)
            
            W_conv = np.zeros((NbP,Nbnu_hr))
            sconv = spec_res/2.355
            for iord in range(c_order-adj_orders, c_order+adj_orders+1):
                nu_pm = nu_mp(iord, pixels, temperature)
                W_blaze = F_blaze(iord, pixels, temperature)
                for ip in pixels:
                    W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
            
            W_aotf = F_aotf_goddard18b(0., nu_hr, temperature, A=aotf_freq[frame_no])
            I0_hr = W_aotf * I0_solar_hr
            I0_p = np.matmul(W_conv, I0_hr)
            solar = I0_p
            
            
            solar_norm = solar/max(solar)
            
            out["%i_orders" %adj_orders] = solar_norm
            
            # plt.plot(solar_norm, label="Solar spectrum full simulation")
            # plt.legend()

        
        filename = os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename, "%s_solar_simulation_%s_%04i_%ikHz.json" %(channel, hdf5_filename, frame_no, aotf_freq[frame_no]))
        
        if not os.path.exists(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename)):
            os.makedirs(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename))
        write_json(filename, out)
        
        

        # make_line_anim(d)

