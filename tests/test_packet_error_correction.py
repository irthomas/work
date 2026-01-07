# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:32:23 2023

@author: iant

DETECT AND CORRECT PACKET TIMESTAMP ERRORS
"""



import numpy as np
import matplotlib.pyplot as plt


ts = np.loadtxt(r"\\wsl.localhost\Debian\bira-iasb\projects\NOMAD\Instrument\SOFTWARE-FIRMWARE\nomad_ops\20230715_102013_SO.txt")



def correct_ts_error(h5_basename, ts, plot=False):

    diff = np.diff(ts)
    
    #seconds spacing between consecutive subdomains
    median = np.median(diff)
    
    
    ts_err = 0.05
    
    #find indices within acceptable range
    ixs1 = np.where((diff > (median - ts_err)) & (diff < (median + ts_err)))[0]
    #indices outside the first range
    not_ixs1 = [i for i in range(len(diff)) if i not in ixs1]
    
    #ix spacing between indices
    remaining_ix_spacing = np.diff(not_ixs1)
    remaining_ix_spacing_median = int(np.median(remaining_ix_spacing))
    
    #seconds spacing between last subdomains
    remaining_median = np.median(diff[not_ixs1])
    
    
    #now create simulated diff
    sim_diff = []
    for i in range(len(diff)):
        if np.mod(i+1, remaining_ix_spacing_median) == 0:
            sim_diff.append(remaining_median)
        else:
            sim_diff.append(median)
    
    #calculate the expected timestamp for each packet
    expected_ts = []
    for i in range(len(ts)):
        if i == 0:
            #assume first value is good
            expected_ts.append(ts[i])
    
        elif np.abs(diff[i-1] - sim_diff[i-1]) < ts_err:
            expected_ts.append(expected_ts[-1] + diff[i-1])
            
        else:
            expected_ts.append(expected_ts[-1] + sim_diff[i-1])
    
    
    #difference from expected timestamp
    ts_expected_diff = np.abs(ts - expected_ts)
    
    
    #check if error is not in first packet!
    #find timestamps outside error bounds
    error_ixs = [i for i,v in enumerate(ts_expected_diff) if v > ts_err]
    
    ratio = len(error_ixs) / len(ts)
    
    if ratio > 0.1:
        print("Warning: %s ratio is %0.2f" %ratio)
        np.savetxt()
    # print(ratio)
    
    
    corrected_ts = []
    for i in range(len(ts)):
        
        if i == 0:
            #assume first value is good
            corrected_ts.append(ts[i])
        
        elif ts_expected_diff[i] < ts_err:
            #if good match
            corrected_ts.append(ts[i])
            # print("<error", i, ts[i], expected_ts[i], ts_expected_diff[i], ts_err)
        else:
            corrected_ts.append(expected_ts[i])
            # print(">error", i, ts[i], expected_ts[i], ts_expected_diff[i], ts_err)
    
    
    # ts_diff = np.abs(ts - corrected_ts)

    if plot:
        plt.figure()
        plt.plot(ts)
        plt.plot(corrected_ts)


    return corrected_ts