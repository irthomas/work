# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:25:30 2020

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



for npoints in [100, 1000, 100000]:
    #npoints = 200
    mean = 16.5
    stdev = 3.3
    gauss = np.random.normal(loc=mean, scale=stdev, size=npoints)
    
    bin_starts = np.arange(0.0, 40.0)
    bin_delta = bin_starts[1]-bin_starts[0]
    bin_ends = np.arange(1.0, 41.0)
    bin_starts_hr = np.linspace(bin_starts[0], bin_starts[-1], num=1000)
    
    bins = []
    for start, end in zip(bin_starts, bin_ends):
        bins.append([start, end])
        
    binnedGauss = np.histogram(gauss, bins=bin_starts)[0]
    
    plt.figure(figsize=[10,3])
    plt.bar(bin_starts[0:-1]+ bin_delta/2.0, binnedGauss)
    
    
    plt.plot(bin_starts_hr, (norm.pdf(bin_starts_hr, mean, stdev)/ np.max(norm.pdf(bin_starts_hr, mean, stdev)) * np.max(binnedGauss)), "r")
    plt.ylabel("N points in each bin")
    plt.xlabel("Values, binned")
    plt.title("Normal distribution: mean=%0.1f, stdev=%0.1f, number of points=%i" %(mean, stdev, npoints))
    
    plt.tight_layout()
    plt.savefig("normal_dist_npoints=%i.png" %npoints)