# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:50:33 2022

@author: iant

GET COUNTS PER PIXEL PER SECOND FROM GROUND CAL
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths
from tools.plotting.colours import get_colours


from instrument.nomad_lno_instrument_v01 import m_aotf, nu_mp



fullscan_h5s = {
"20180702_112352_1p0a_LNO_1_CF":-4.49,
"20181101_213226_1p0a_LNO_1_CF":4.40,
"20190314_021825_1p0a_LNO_1_CF":-2.14,
"20190609_011514_1p0a_LNO_1_CF":-10.35,
"20190921_222413_1p0a_LNO_1_CF":-8.88,
"20191207_051654_1p0a_LNO_1_CF":-9.64,
"20200105_132318_1p0a_LNO_1_CF":-4.24,
"20200324_145739_1p0a_LNO_1_CF":2.65,
"20201115_013915_1p0a_LNO_1_CF":-16.24,
"20201222_114725_1p0a_LNO_1_CF":-7.64,
"20210325_135449_1p0a_LNO_1_CF":-16.07,
"20220101_030345_1p0a_LNO_1_CF":-9.09,
"20220616_112436_1p0a_LNO_1_CF":-13.94,
}




def planck(xscale, temp): #planck function W/cm2/sr/cm-1
    c1=1.191042e-5
    c2=1.4387752
    return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2





def read_cal_file(cal_h5):
    
    h5_f = open_hdf5_file(cal_h5)

    y_f = h5_f["Science/Y"][...]
    bins = h5_f["Science/Bins"][:, 0]
    
    orders = h5_f["Channel/DiffractionOrder"][...]
    it_f = h5_f["Channel/IntegrationTime"][0]
    binning_f = h5_f["Channel/Binning"][0]
    naccs_f = h5_f["Channel/NumberOfAccumulations"][0]
    t_f = h5_f["Temperature/NominalLNO"][...]
    
    
    
    it = np.float32(it_f) / 1.0e3 #microseconds to seconds
    naccs = np.float32(naccs_f)/2.0 #assume LNO nadir background subtraction is on
    binning = np.float32(binning_f) + 1.0 #binning starts at zero
    t = np.mean(t_f)
    
   
    # print("integration time file = %i" %it_f)
    # print("integration time = %0.4f" %it)
    # print("n accumulation = %i" %naccs)
    # print("binning = %i" %binning)
    
    #normalise to 1s integration time per pixel
    y_norm = y_f / (it * naccs * binning)
    
    h5_f.close()

    return {"orders":orders, "y_norm":y_norm, "t":t, "bins":bins}





def rad_cal_order(cal_h5, order, centre_indices=None):

    d = read_cal_file(cal_h5)
    
    unique_bins = sorted(list(set(d["bins"])))
    centre_bins = unique_bins[6:-6]
    
    ix = [i for i,(bin_, order_) in enumerate(zip(d["bins"],d["orders"])) if bin_ in centre_bins and order_ == order]
    # print(order, ix[0:10])
    y_frame = d["y_norm"][ix, :]
    
    #mean of repeated order spectra    
    y_spectrum = np.mean(y_frame, axis=0)

    #get mean of spectrum (either all or specific pixels)
    if not centre_indices:
        y_centre_mean = np.mean(y_spectrum)
    else:
        y_centre_mean = np.mean(y_spectrum[centre_indices])
    
    #convert to wavenumbers
    x = nu_mp(order, np.arange(320.), d["t"])
    
    # return {"x":x, "y_frame":y_frame, "y_spectrum":y_spectrum, "solar_b":solar_b, "counts_per_rad":counts_per_rad}
    return {"x":x, "y_spectrum":y_spectrum, "x_mean":np.mean(x), "y_centre_mean":y_centre_mean}



# cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
# order = 169

# d = rad_cal_order(cal_h5, order)

