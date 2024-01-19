# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:10:01 2023

@author: iant

CHECK POLISH CH4 DETECTION

#july 21, 2021, z=4.4km, T=7-8%, h5=20210721_132621_1p0a_SO_A_E_134

"""


import re
import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist



# regex = re.compile("20210721_......_.*_SO_.*_134")
regex = re.compile("20210721_132621_.*_SO_.*_134")


file_level = "hdf5_level_1p0a"

h5_fs, h5s, _ = make_filelist(regex, file_level)


bin_colours = {120:"C0", 124:"C1", 128:"C2", 132:"C3"}

for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):
    
    lats = h5_f["Geometry/Point0/Lat"][:, 0]
    lons = h5_f["Geometry/Point0/Lon"][:, 0]
    alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
    y = h5_f["Science/Y"][...]
    x = h5_f["Science/X"][...]
    bins = h5_f["Science/Bins"][:, 0]


    

    ixs = np.where((alts > 4) & (alts < 5))[0]
    
    print(h5, ixs)
    
    plt.figure()
    plt.title(h5)
    plt.xlabel("Wavenumber cm-1")
    plt.ylabel("Transmittance")
    plt.grid()
    
    for ix in ixs:
        
        bin_ = bins[ix]
        alt = alts[ix]
        
        plt.plot(x[0, :], y[ix, :], color=bin_colours[bin_], alpha=0.5, label="ix=%i, bin=%i, z=%0.5fkm" %(ix, bin_, alt))
        
    plt.legend()
    
    