# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:14:38 2023

@author: iant

CO ORDERS OCCULTATION
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# from analysis.so_lno_2023.functions.h5 import read_h5
from tools.file.h5_obj import h5_obj
from tools.file.hdf5_functions import open_hdf5_file

from tools.general.get_minima_maxima import get_local_minima_or_equals


good_px_ixs = np.arange(50, 320)


# to be replaced when fast fullscans are reprocessed
#fullscans with orders 185 to 195
# 20230826_015002	I
# 20230826_212855	I
# 20230828_172637	E
# 20230831_231912	I
# 20230920_225731	I
# 20230921_223324	I


h5s = ["20230122_055302_1p0a_SO_I_S"]
h5_no_order = "20230122_055302_1p0a_SO_A_I_"
orders = [186]#np.arange(186, 196)
h5s = ["%s%i" %(h5_no_order, order) for order in orders]
bin_ixs = [0]


low_alt_ix = 10


# orders = [186]#np.arange(183, 187)
# orders = np.arange(183, 187)
# bin_ixs = [0]
# low_alt_ix = 10
# h5_no_order = "20220301_114833_1p0a_SO_A_I_"
# h5s = ["%s%i" %(h5_no_order, order) for order in orders]


#solar calibration
# h5s = ["20211105_155547_1p0a_SO_1_CM"]
# bin_ixs = [6,7,8,9,10,11,12,13,14,15,16,17,18]


#assuming first 50 pixels removed
abs_line_points = {
    183:[95.443, 137.048, 177.873, 217.829],
    184:[76.91540426844962, 114.88315004791231, 189.86206428016794],
    185:[3.3050990035902257, 77.30923093995291, 112.16091686858721, 145.67586364327533, 173.18314437648354, 178.61095030677595],
    186:[17.835, 144.926, 190.194]
}

#assuming first 50 pixels removed, corresponding to absorption indices above
baseline_points = {
    183:[4, 8, 26, 47, 65, 81, 88, 109, 123, 129, 149, 160, 170, 194, 203, 224, 229, 239, 246, 262],
    184:[0, 5, 8, 13, 27, 46, 83, 92, 108, 118, 123, 130, 142, 159, 165, 181, 194, 205, 215, 230, 237, 246, 262],
    185:[8, 13, 14, 22, 27, 33, 46, 47, 48, 83, 91, 92, 108, 117, 128, 129, 139, 157, 182, 204, 205, 215, 230, 237, 246, 261],
    186:[6, 10, 22, 27, 48, 66, 72, 82, 92, 108, 120, 130, 158, 165, 182, 194, 206, 217, 247, 261],
}


fig1, ax1 = plt.subplots()





h5s_d = {}
for h5 in h5s:


    h5f = open_hdf5_file(h5, path=r"C:\Users\iant\Documents\DATA\hdf5")
    
    h5o = h5_obj(h5)
    h5o.set_h5_path(r"C:\Users\iant\Documents\DATA\hdf5")
    
    h5o.h5_to_dict(bin_ixs)
    h5o.cut_pixels(bin_ixs, good_px_ixs)
    
    if h5o.calibration:
        h5o.solar_cal(bins_to_average=bin_ixs)
    else:
        h5o.trans_recal(bin_ixs=bin_ixs, top_of_atmosphere=110.0)
    
    
    
    for bin_ix in bin_ixs:
        h5_d = h5o.h5_d[bin_ix]
    
        y = h5_d["y_mean"]
        alts_all = h5_d["alt"]
        order = h5o.h5_d["orders"][0]
        x = h5o.h5_d["x"]
        
        """spectral calibration"""
        #fit to abs lines
        #find local minima

        #find indices where 5% < Trans <95%
        atmos_ixs = np.where((np.max(y[:, 160:240], axis=1) > 0.2) & (np.max(y[:, 160:240], axis=1) < 0.95))[0]
        
        #quick baseline fit
        y_simp_norms = np.zeros((len(atmos_ixs[2:]), y.shape[1]))
        for i, atmos_ix in enumerate(atmos_ixs[2:]):
            y_simp_baseline = np.polyval(np.polyfit(np.arange(y.shape[1]), y[atmos_ix, :], 9), np.arange(y.shape[1]))
            y_simp_norm = y[atmos_ix, :] / y_simp_baseline
            y_simp_norms[i, :] = y_simp_norm
            
        y_av_norm = np.mean(y_simp_norms, axis=0)
        
        y_av_std = np.std(y_av_norm)
        y_av_mean = np.mean(y_av_norm)


        low_alt_ix = atmos_ixs[0]-1

        
        
        #find absorption line indices - iterate until at least 3 lines found
        abs_ixs = []
        n_stds = 5
        
        while len(abs_ixs) < 3:
        
            n_stds -= 0.5
            
            below_std_ixs = np.where(y_av_norm < (y_av_mean - y_av_std*n_stds))[0]
            minima_ixs = get_local_minima_or_equals(y_av_norm)
            
            abs_ixs = sorted(list(set(below_std_ixs).intersection(minima_ixs)))
        
        #now find exact pixel number of centre of lines
        # plt.figure()
        min_ixs = []
        min_nus = []
        for abs_ix in abs_ixs:
            abs_ix_around = [abs_ix-1, abs_ix, abs_ix+1]
            
            #quadratic to find minima
            abs_polyfit = np.polyfit([-1., 0., 1.], y_av_norm[abs_ix_around], 2)
            abs_polyval = np.polyval(abs_polyfit, [-1., 0., 1.])
            abs_min = -abs_polyfit[1]/(2.0 * abs_polyfit[0])

            # plt.scatter(abs_ix_around, abs_polyval)
            # plt.scatter(np.arange(abs_ix-1, abs_ix+1, 0.01), np.polyval(abs_polyfit, np.arange(-1, 1, 0.01)))
            # plt.axvline(abs_min + abs_ix)
            
            min_ixs.append(abs_min + abs_ix)
            min_nus.append(x[abs_ix] + abs_min * (x[abs_ix]-x[abs_ix-1]))
        
        print(h5, "min_ixs:", min_ixs)
        
        # plt.plot(x, y_av_norm)
        # plt.axhline((y_av_mean - y_av_std*n_stds))
        # for min_nu in min_nus:
        #     plt.axvline(min_nu)

        # plt.plot(y_av_norm)
        # plt.axhline((y_av_mean - y_av_std*n_stds))

    


    
        if order in baseline_points.keys():
            #get continuum indices
            baseline_ixs = np.asarray(baseline_points[order])
            
            #correct for temperature shift
            #get corresponding absorption indices from a reference observation
            abs_line_ixs = np.asarray(abs_line_points[order])
            
            #find difference between current observation and the reference obs
            ix_shifts = np.asarray(min_ixs) - abs_line_ixs
            
            #check not too far away (indicates error)
            if np.std(ix_shifts) < 0.2:
                ix_shift = np.mean(ix_shifts)
                print("Temperature shift from reference spectrum:", ix_shift, "pixels")
            else:
                print("Error in temperature shift correction:", min_ixs, abs_line_ixs)
            #now shift the continuum indices by this amount
            
            baseline_ixs += int(np.round(ix_shift))
            
            #remove negatives
            baseline_ixs = baseline_ixs[baseline_ixs >= 0.0]
                
            plt.figure()
            for atmos_ix in atmos_ixs:
                plt.plot(y[atmos_ix, :])
            plt.grid()
            plt.xlabel("Pixel number")
            plt.ylabel("Transmittance")
            for min_ix in min_ixs:
                plt.axvline(min_ix)

        else:
            print("Points not found")
            plt.plot(y.T)
            sys.exit()
    
        # ratio_2d = ratio[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :]
        # y *= ratio_2d.T
        
        
        """normalise baseline"""
        polyfits = np.polyfit(baseline_ixs, y[:, baseline_ixs].T, 5)
    
        #interpolate to spectral continuum
        y_baseline = np.zeros_like(y)
        for row_ix in np.arange(y.shape[0]):
            y_baseline[row_ix, :] = np.polyval(polyfits[:, row_ix], np.arange(y.shape[1]))
            
        # plt.figure()
        # plt.title("Y and Y baseline fits")
        # plt.plot(y.T)
        # plt.plot(y_baseline.T)
    
        #normalise baseline to 1, remove lowest altitudes (noise only)
        y_norm = y[low_alt_ix:, :]/y_baseline[low_alt_ix:, :]
        alts = alts_all[low_alt_ix:]
            
        y_smooth = np.zeros_like(y_norm)
        row_ixs = np.arange(y_norm.shape[0])
        for ix in np.arange(y_norm.shape[1]):
            y_smooth[:, ix] = np.polyval(np.polyfit(row_ixs, y_norm[:, ix], int(np.floor(y_norm.shape[0]/10.))), row_ixs)
            
        plt.figure()
        # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y_norm.shape[0])[np.newaxis, :], y_smooth.T, alpha=0.1)
        plt.plot(y_smooth.T, alpha=0.1)
        
        plt.xlabel("Pixel number")
        plt.ylabel("Transmittance")
        plt.title("%s: baseline corrected and smoothed" %h5)
        plt.grid()
        
        # plt.figure()
        # plt.imshow(y_smooth)
        
        
        # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :], y[:, px_ixs].T)
        # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :], y_baseline.T)
        
        
        
        # plt.figure()
        # plt.plot(y_norm)
        
        # if order == 185:
        #     for i, px_ix in enumerate([68, 74, 127, 196, 223, 229, 284]):
        #         ix = np.where(px_good_ixs == px_ix)[0][0]
        #         # plt.plot(y_norm[:, px_ix])
        #         ax1.plot(alts, y_smooth[:, ix], label="Selected pixel %i" %px_ix, c="C%i" %i)
        for i, px_ix in enumerate(np.round(min_ixs)):
            ax1.plot(alts, y_smooth[:, int(px_ix)], label="Selected pixel %i" %px_ix, c="C%i" %i)
            ax1.scatter()
        
        ax1.set_title(h5)
        ax1.set_xlabel("Tangent altitude")
        ax1.set_ylabel("Transmittance of line")
        ax1.grid()
        ax1.legend()
        
    
        # for 
        # plt.ylim((0.5, 1.1))