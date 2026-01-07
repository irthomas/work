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

cal_h5s = {
    "20150426_054602_0p1a_LNO_1":423.0,
    "20150426_030851_0p1a_LNO_1":423.0,
} #150C BB cold instrument






def planck(xscale, temp): #planck function W/cm2/sr/cm-1
    c1=1.191042e-5
    c2=1.4387752
    return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2




def csl_window_trans():
    #0:Wavelength, 1:Lens ZnSe, 2:Lens Si, 3: Lens Ge, 4:AOTF, 5:Par mirror, 6:Planar miror, 7:Detector, 8:Cold filter, 9:Window transmission function
    #10:CSL sapphire window
    optics_all = np.loadtxt(paths["BASE_DIRECTORY"]+os.sep+"reference_files"+os.sep+"nomad_optics_transmission.csv", skiprows=1, delimiter=",")
    optics_transmission_total = optics_all[:,10]
    optics_wavenumbers =  10000. / optics_all[:,0]
    return optics_wavenumbers, optics_transmission_total




def read_cal_file(cal_h5):
    
    h5_f = open_hdf5_file(cal_h5)

    # x = h5_f["Science/X"][...]
    y_f = h5_f["Science/Y"][...]
    aotf_f = h5_f["Channel/AOTFFrequency"][...]
    it_f = h5_f["Channel/IntegrationTime"][0]
    binning_f = h5_f["Channel/Binning"][0]
    naccs_f = h5_f["Channel/NumberOfAccumulations"][0]
    t_f = h5_f["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
    
    
    # nSpectra = y_f.shape[0]
    
    
    it = np.float32(it_f) / 1.0e3 #microseconds to seconds
    naccs = np.float32(naccs_f)/2.0 #assume LNO nadir background subtraction is on
    binning = np.float32(binning_f) + 1.0 #binning starts at zero
    t = np.mean(t_f[2:10])
    
    orders = np.array([m_aotf(i) for i in aotf_f])
    
    # print("integration time file = %i" %it_f)
    # print("integration time = %0.2f" %it)
    # print("n accumulation = %i" %naccs)
    # print("binning = %i" %binning)
    
    #normalise to 1s integration time per pixel
    y_norm = y_f / (it * naccs * binning)
    
    h5_f.close()

    return {"orders":orders, "y_norm":y_norm, "t":t}





def rad_cal_order(cal_h5, order):
    
    d = read_cal_file(cal_h5)
    orders = d["orders"]
    y_norm = d["y_norm"]
    t = d["t"]
    
    bb_temp = cal_h5s[cal_h5]
    
    #get first frame index of fullscan
    ix = np.where(order == orders)[0][0]

    #convert to wavenumbers
    x = nu_mp(order, np.arange(320.), t)
    
    bb_b = planck(x, bb_temp)
    
    """include CSL window transmission in expected radiance signal"""
    opticsWavenumbers, cslWindowTransmission = csl_window_trans()
    cslWindowInterp = np.interp(x, opticsWavenumbers, cslWindowTransmission)
    bb_window_b = bb_b * cslWindowInterp         
    
    
    y_frame = y_norm[ix, :, :]
    y_spectrum = np.mean(y_frame, axis=0)
    
    counts_per_rad = y_spectrum / bb_window_b
    
    return {"x":x, "y_frame":y_frame, "y_spectrum":y_spectrum, "bb_window_b":bb_window_b, "counts_per_rad":counts_per_rad}





def plot_ground_rad_cal(cal_h5):
        
    file_d = read_cal_file(cal_h5)
    orders = file_d["orders"]

    fig1, (ax1a, ax1b, ax1c) = plt.subplots(figsize=(15, 10), nrows=3, sharex=True)
    colours = get_colours(len(orders))
    
    
    measured_orders = [] #some orders are repeated, skip if already done

    # convert to a function given order input
    for order_ix, order in enumerate(orders):
        
        if order in measured_orders:
            continue
        else:
            measured_orders.append(order)

        d = rad_cal_order(cal_h5, order)
        
        
        for row_ix in range(d["y_frame"].shape[0]):
            ax1a.plot(d["x"], d["y_frame"][row_ix, :], color=colours[order_ix])
            
        ax1a.plot(d["x"], d["y_spectrum"], "k--")
        ax1b.plot(d["x"], d["bb_window_b"], color=colours[order_ix])
        ax1c.plot(d["x"], d["counts_per_rad"], color=colours[order_ix])
        
    fig1.suptitle("LNO Radiance Calibration")
    ax1a.set_title("Counts when viewing blackbody")
    ax1b.set_title("Radiance of blackbody")
    ax1c.set_title("Counts per radiance")
    
    ax1a.grid()
    ax1b.grid()
    ax1c.grid()
    
    ax1c.set_xlabel("Wavenumbers cm-1")
    ax1a.set_ylabel("Counts per pixel per second")
    ax1b.set_ylabel("Radiance W/cm2/sr/cm-1")
    ax1c.set_ylabel("Counts per unit radiance")
    
    plt.savefig("lno_radiance_cal_%s.png" %cal_h5)
    
    
if __name__ == "__main__":
    for cal_h5 in cal_h5s:
        plot_ground_rad_cal(cal_h5)