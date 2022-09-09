# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:11:07 2022

@author: iant
"""


import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist



regex = re.compile("20200525_184425_1p0a_UVIS_I")
file_level = "hdf5_level_1p0a"



h5_fs, h5s, _ = make_filelist(regex, file_level)
spectra_dict = {}

for i, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):
    
    alts_all = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
    pointing_deviation = h5_f["Geometry/FOVSunCentreAngle"][:, 0]
    y = h5_f["Science/Y"][...]
    x = h5_f["Science/X"][0, :]
    
    ixs = np.where((x > 500) & (x < 600))[0]
    
    y_binned = np.mean(y[:, ixs], axis=1)

    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig1.suptitle(h5)
    ax1.plot(alts_all, y_binned, label="Mean transmittance 500-600nm")
    ax2.plot(alts_all, pointing_deviation, label="Pointing deviation")
    
    ax1.set_ylabel("Transmittance")
    ax2.set_ylabel("Pointing deviation (arcminutes)")
    ax2.set_xlabel("Tangent Altitude (km)")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    
    fig1.savefig("%s_pointing_deviation.png" %h5)
    