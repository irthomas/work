# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:53:42 2023

@author: iant

GET 2D ARRAY CONTAINING SOLAR LINE IN CENTRE AND NORMAL MINISCAN PATTERN ON EDGES

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import savgol_filter

from tools.plotting.colours import get_colours

from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu0_aotf





def trap_absorption(arr, aotf_cut, channel, h5_prefix, abs_region_cols, abs_region_rows, arr_region_cols, arr_region_rows, plot=[]):

    # abs_region_cols = [680, 750]
    # abs_region_rows = [0, 1997]
    
    # arr_region_cols = [500, 900]
    # arr_region_rows = [0, 1997]
    
    
    #get indices of edges within the subarray
    col_edges = [abs_region_cols[0]-arr_region_cols[0], abs_region_cols[1]-arr_region_cols[0]]
    row_edges = [abs_region_rows[0]-arr_region_rows[0], abs_region_rows[1]-arr_region_rows[0]]
    # col_abs = range(col_edges[0], col_edges[1])
    row_abs = range(row_edges[0], row_edges[1])
    
    #arr of continuum rows
    # arr_cont = np.array([np.interp(col_abs, col_edges, col_side_values[i, :]) for i in range(col_side_values.shape[0])])
    
    #arr of rows containing the absorption
    arr_abs = arr[row_edges[0]:row_edges[1], col_edges[0]:col_edges[1]]
    
    
    
    """make the interpolated continuum arrays"""
    #make the smoothed blaze function from the fits
    # fit_coeff0 = savgol_filter(fits2[0, :], 99, 1)
    
    #or load from file
    # np.savetxt("fit_coeff0_178.txt", fit_coeff0)
    fit_coeff0 = np.loadtxt("%s_fit_coeff0.txt" %h5_prefix)
    
    #blaze function for scaling each column
    scalars = fit_coeff0[abs_region_cols[0]:abs_region_cols[1]] / fit_coeff0[(abs_region_cols[1]-1)]
    #convert to 2D array
    arr_scalars = np.repeat(scalars, arr_abs.shape[0]).reshape((-1, arr_abs.shape[0])).T
    #divide by blaze mean to correct for columns having higher signal than others
    arr_div = arr_abs / arr_scalars
    
    
    # plt.figure()
    # plt.imshow(arr_div, aspect="auto")
    
    
    
    # colours = get_colours(10, cmap="brg")
    # plt.figure()
    # for i in range(10):
    #     plt.plot(arr_div[:, i], alpha=0.3, color=colours[i], label=i)
    # for i in range(10):
    #     plt.plot(arr_div[:, 60+i], alpha=0.3, color=colours[i], linestyle="--", label=60+i)
        
    # plt.legend()
    
    
    # plt.figure()
    # plt.plot(range(10), arr_div[420, 0:10])
    # plt.plot(range(60, 70), arr_div[420, 60:70])
    
    #make mean columns from left and right side
    mean_cont_left = np.mean(arr_div[:, 0:5], axis=1)
    mean_cont_right = np.mean(arr_div[:, -6:-1], axis=1)
    #average to make mean col
    mean_cont = np.mean([mean_cont_left, mean_cont_right], axis=0)
    #convert to 2D array
    arr_cont = np.tile(mean_cont, (arr_abs.shape[1], 1)).T
    
    #subtract to leave the continuum-corrected absorption
    arr_sub = arr_cont - arr_div
    
    
    #integrate under the curve
    traps = integrate.trapezoid(arr_sub, axis=1)
    
    #get aotf frequency and wavenumbers
    max_abs_col_ix = np.argmax(np.mean(arr_sub, axis=0)) #column ix with deepest absorption
    aotf_peak_col = aotf_cut[:, max_abs_col_ix + col_edges[0]]
    
    if channel == "lno":
        aotf_nu = [nu0_aotf(A) for A in aotf_peak_col[row_abs]]
    if channel == "so":
        aotf_nu = [aotf_peak_nu(A, 0.0) for A in aotf_peak_col[row_abs]] #needs temperature
        
        
    
    #test: fit sine to edges, try to fit first onto second
    #two steps: multiply by blaze from coefficient 0, then correct for small slope difference (coeff 4)
    
    #test2: assume slope difference between columns is linear
    # find slope difference from linear polyfit to plt.plot(col_side_values[:, 1]-col_side_values[:, 0])
    # plt.figure()
    # xs = []
    # ys = []
    # for i in range(2):
    #     y = col_side_values[:, i]
    #     x = np.arange(len(y))
        
        
    #     if i == 0:
    #         coeff4 = [-0.725064, 413.378]
    #         scaler = 89954.07674925645 / 77807.4163744896
    #     if i == 1:
    #         coeff4 = [0.0]
    #         scaler = 1.0
    #     slope = np.polyval(coeff4, x)
    #     y *= scaler
    #     y -= slope
        
    #     # smodel = Model(sinefunction)
    #     # result = smodel.fit(y, x=x, a=sinc_apriori["a"], b=sinc_apriori["b"], c=sinc_apriori["c"], d=sinc_apriori["d"], e=sinc_apriori["e"])
    #     # yfit = result.best_fit
    
    #     plt.plot(x, y)
    #     # plt.plot(x, yfit)
    #     # plt.title("Fit vs raw columns %i and %i" %(solar_line_data["abs_region_cols"][0], solar_line_data["abs_region_cols"][1]))
        
    #     xs.append(x)
    #     ys.append(y)
        
    # xs = np.asarray(xs)
    # ys = np.asarray(ys)
    # plt.figure()
    # plt.plot(xs[0, :], ys[0, :]-ys[1, :])
    
    # coeffs = np.polyfit(xs[0, :], ys[0, :]-ys[1, :], 1)
    # polyval = np.polyval(coeffs, xs[0, :])
    # plt.plot(xs[0, :], polyval)
    
    
    
    
    savgol = savgol_filter(traps, 125, 2)
    
    # plt.figure()
    # plt.plot(aotf_nu, traps)
    # plt.plot(aotf_nu, savgol)
    
    #normalise to 1
    traps /= np.max(traps)
    savgol /= np.max(savgol)
    
    
    return aotf_nu, traps, savgol
