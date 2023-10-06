# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:53:42 2023

@author: iant

GET 2D ARRAY CONTAINING SOLAR LINE IN CENTRE AND NORMAL MINISCAN PATTERN ON EDGES

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import savgol_filter

from tools.plotting.colours import get_colours

from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu0_aotf

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")



def trap_absorption(arr_abs, aotf_cut, t, channel, h5_prefix, abs_region_cols, abs_region_rows, plot=[], nstd=1.0):
    
    """trapezium rule area under the curve
    arr is the subarray around the absorption, cut using abs_region_cols and abs_region_rows"""

    
    
    """make the interpolated continuum arrays"""
    #load smoothed blaze function from file
    fit_coeff0 = np.loadtxt(os.path.join(MINISCAN_PATH, channel, "%s_fit_coeff0.txt" %h5_prefix))

    
    #blaze function for scaling each column
    scalars = fit_coeff0[abs_region_cols[0]:abs_region_cols[1]] / fit_coeff0[(abs_region_cols[1]-1)]
    #convert to 2D array
    arr_scalars = np.repeat(scalars, arr_abs.shape[0]).reshape((-1, arr_abs.shape[0])).T
    #divide by the fit0 blaze mean to correct for columns having higher signal than others
    arr_div = arr_abs / arr_scalars
    
    
    #there is a small difference still - need to linearly interpolate
    mean_cont_left = np.mean(arr_div[:, 0:5], axis=1)
    mean_cont_right = np.mean(arr_div[:, -6:-1], axis=1)
    
    polyfits = np.polyfit([2.5, arr_div.shape[1]-2.5], [mean_cont_left, mean_cont_right], 1)
    polyvals = np.asarray([np.polyval(polyfit, np.arange(arr_div.shape[1])) for polyfit in polyfits.T])
    
    arr_div /= polyvals
    
    
    # plot absorptions scaled by fit coeffs (row values flattened)
    # plt.figure()
    # plt.imshow(arr_div, aspect="auto")


    #get maximum value in each row, divide by this to normalise each row
    div_max = np.max(arr_div, axis=1)
    arr_div_max = np.repeat(div_max, arr_div.shape[1]).reshape((-1, arr_div.shape[1]))
    arr_norm = arr_div / arr_div_max
    
    
    #get the centre column
    abs_centre_col = 1.0 - arr_norm[:, int(arr_div.shape[1]/2)]
    abs_centre_col /= np.max(abs_centre_col)
    
       
    
    mean_cont_both_sides = np.concatenate([arr_div[:, 0:5], arr_div[:, -6:-1]], axis=1)
    
    #average to make mean col
    mean_cont = np.mean(mean_cont_both_sides, axis=1)
    #stdev to get uncertainty
    std_cont = np.std(mean_cont_both_sides, axis=1) * nstd
    #convert to 2D array, one for min, mean and, max within stdev
    arr_cont = [
        np.tile(mean_cont, (arr_abs.shape[1], 1)).T, #mean must be first
        np.tile(mean_cont - std_cont, (arr_abs.shape[1], 1)).T,
        np.tile(mean_cont + std_cont, (arr_abs.shape[1], 1)).T,
    ]
    
    out_d = {"centre":abs_centre_col}
    
    for i, name in enumerate(["mean", "min", "max"]):
    #subtract to leave the continuum-corrected absorption
        arr_sub = arr_cont[i] - arr_div
        
        
        #integrate under the curve
        traps = integrate.trapezoid(arr_sub, axis=1)
        
        #get aotf frequency and wavenumbers
        max_abs_col_ix = np.argmax(np.mean(arr_sub, axis=0)) #column ix with deepest absorption
        aotf_peak_col = aotf_cut[:, max_abs_col_ix]
        
        if channel == "lno":
            aotf_nu = [nu0_aotf(A) for A in aotf_peak_col]
        if channel == "so":
            aotf_nu = [aotf_peak_nu(A, t) for A in aotf_peak_col] #needs temperature
            
        
    
    
    
        savgol = savgol_filter(traps, 125, 2)


        #normalise to 1 using the mean value
        if name == "mean":
            max_raw = np.max(traps)
            max_smooth = np.max(savgol)
            
            traps /= np.max(traps)
            savgol /= np.max(savgol)
            
        else:
            traps /= np.max(max_raw)
            savgol /= np.max(max_smooth)
            
        
        out_d[name] = {"raw":traps, "smooth":savgol, "aotf_nu":aotf_nu}
     
    return out_d# aotf_nu, traps, savgol
