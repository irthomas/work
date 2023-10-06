# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:13:30 2023

@author: iant

STEP 3: CALCULATE AOTFS WHEN ALL SOLAR LINE COEFFS HAVE BEEN DEFINED
ANALYSE THE AOTF AND BLAZE FUNCTIONS AND SAVE THEM TO A FILE
"""


import os
import h5py
import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit import Model
from scipy import integrate

from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu0_aotf


from tools.general.progress_bar import progress

from instrument.calibration.so_lno_2023.fit_absorption_miniscan_array import trap_absorption
from instrument.calibration.so_lno_2023.solar_line_dict import solar_line_dict




plot = ["absorptions"]
naxes = [1,1]

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

channel = "so"
# channel = "lno"

AOTF_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_aotfs.txt" %channel)
# BLAZE_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_blazes.txt" %channel)



#clear files
with open(AOTF_OUTPUT_PATH, "w") as f:
    f.write("")
# with open(BLAZE_OUTPUT_PATH, "w") as f:
#     f.write("")





loop = 0
for h5_prefix, solar_line_data_all in solar_line_dict.items(): #loop through files
    channel = h5_prefix.split("-")[0].lower()


    #get data from miniscan file
    with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" %h5_prefix), "r") as f:
        
        keys = list(f.keys())
        n_reps = len([i for i, key in enumerate(keys) if "array" in key])
        
        aotfs = []
        for i in range(n_reps):
            aotfs.append(f["aotf%02i" %i][...])
        
        arrs = []
        for i in range(n_reps):
            arrs.append(f["array%02i" %i][...])

        t_ranges = []
        for i in range(n_reps):
            t_ranges.append(f["t%02i" %i][...])
    


            
    aotf_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all]
    # blaze_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all if "blaze_rows" in solar_line_data.keys()][0]
    
    
    for line_ix, solar_line_data in enumerate(aotf_solar_line_data): #loop through list of dictionaries, one per absorption line
    
        if "good" in solar_line_data.keys():
            if not solar_line_data["good"]:
                print("Skipping file %s (not good)" %h5_prefix)
                continue
        else:
            print("Warning: no quality flag for file %s" %h5_prefix)
            
    
        abs_region_rows = solar_line_data["abs_region_rows"][:]
        abs_region_cols = solar_line_data["abs_region_cols"][:]
                
        #change to arr size if -1s
        if abs_region_rows[1] == -1:
            abs_region_rows[1] = np.min([a.shape[0] for a in arrs])
        if abs_region_cols[1] == -1:
            abs_region_cols[1] = np.min([a.shape[1] for a in arrs])

        
    
        
        if len(plot) > 0:
            fig1, ax1 = plt.subplots(figsize=(14, 10), ncols=naxes[0], nrows=naxes[1], squeeze=0)#, constrained_layout=True)
            if len(ax1) != 1:
                ax1 = ax1.flatten()
        
        for rep, (arr, aotf, t_range) in enumerate(zip(arrs, aotfs, t_ranges)):
        
            print(loop, h5_prefix, rep)
            loop += 1
                
            # HR_SCALER = int(arr.shape[1]/320)
            
            #cut arrays to leave region around the absorption feature
            arr_cut = arr[abs_region_rows[0]:abs_region_rows[1], abs_region_cols[0]:abs_region_cols[1]]

            #cut the unwanted edges from the array to speed up sine fitting
            aotf_cut = aotf[abs_region_rows[0]:abs_region_rows[1], abs_region_cols[0]:abs_region_cols[1]]
            
            #get middle temperature from the range
            t = np.mean(t_range)
            
            # calculate band depth    
            aotf_d = trap_absorption(arr_cut, aotf_cut, t, channel, h5_prefix, abs_region_cols, abs_region_rows)
            
            
            
            if "absorptions" in plot:
                #only if not defining edges
                ax1[0][0].plot(aotf_d["mean"]["aotf_nu"], aotf_d["mean"]["raw"], color="C%i" %rep, label="t=%0.1f" %(np.mean(t_range)))
                # ax1[0][0].plot(aotf_d["mean"]["aotf_nu"], aotf_d["mean"]["smooth"], color="C%i" %rep, label="t=%0.1f" %(np.mean(t_range)))
                # ax1[0][0].plot(aotf_d["mean"]["aotf_nu"], aotf_d["centre"], color="C%i" %rep, linestyle="--")
                ax1[0][0].fill_between(aotf_d["min"]["aotf_nu"], aotf_d["min"]["smooth"], aotf_d["max"]["smooth"], color="C%i" %rep, alpha=0.3)
                
        
            aotf_max = np.max(aotf_d["mean"]["smooth"])
            line = "\t".join(["%0.4f" %i for i in aotf_d["mean"]["aotf_nu"]]) + "\t" + "\t".join(["%0.6f" %(i/aotf_max) for i in aotf_d["mean"]["smooth"]])
            with open(AOTF_OUTPUT_PATH, "a") as f:
                f.write(line+"\n")

        ax1[0][0].set_title("AOTF functions %s line %i" %(h5_prefix, line_ix))
        ax1[0][0].axhline(y=0, color="k", linestyle="--")
        ax1[0][0].grid()
        ax1[0][0].legend()

    # #find blaze functions
    # blaze_row_indices = blaze_solar_line_data["blaze_rows"]
    # for blaze_row_index in blaze_row_indices:
    #     for rep, arr in enumerate(arrs):
    #         blaze_hr = arr[blaze_row_index, :]
    
    #         savgol = savgol_filter(blaze_hr, 125, 2)
    #         blaze_max = np.max(savgol)
            
            
    #         aotf_central_col = aotf[blaze_row_index, int(aotf.shape[1]/2)]
            
    #         line = "%0.4f\t" %aotf_central_col + "\t".join(["%0.6f" %(i/blaze_max) for i in savgol])
    #         with open(BLAZE_OUTPUT_PATH, "a") as f:
    #             f.write(line+"\n")

