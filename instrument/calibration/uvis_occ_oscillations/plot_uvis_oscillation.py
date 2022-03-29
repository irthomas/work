# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:52:46 2022

@author: iant
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist


regex = re.compile("20180628_014311_.*_UVIS_E")

file_level = "hdf5_level_0p3k"



hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

for file_index, (h5, h5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
    if np.mod(file_index, 100) == 0:
        print(h5_filename)

    # binning = h5["Channel/HorizontalAndCombinedBinningSize"][0]
    mode = h5["Channel/AcquisitionMode"][0]
    
    #if not full frame, skip file
    if mode != 0:
        continue

    #load spectra in pandas, set rows to wavelength and columns to tangent alts
    df_raw = pd.DataFrame(np.array(h5['Science/Y']).T, index=np.array(h5['Science/X'][0,:]), columns=np.array(h5['Geometry/Point0/TangentAltAreoid'][:, 0]))

    if -999.0 in df_raw.columns:
       df_raw = df_raw.drop(columns=[-999.0]) #get rid of bad tangent alts
    
    #approx solar calibration - mean of 10 highest alt spectra
    max_alts = sorted(list(df_raw.columns), reverse=True)[:10]
    sun_mean = df_raw[max_alts].mean(axis=1)


    df_sun = df_raw.copy()
    for i in df_raw.columns:
        df_sun[i] = sun_mean
        
    df = df_raw / df_sun
    
    
    df_500 = df[(df.index > 500) & (df.index < 600)]
    
    #if no visible region spectra acquired
    if len(df_500.index) == 0:
        continue
    
    df_rolling = df_500.rolling(5, axis=0, center=True).mean()
    
    
    
    
    df_poly = df_500.copy()
    for i in df_500.columns:
        df_poly[i] = np.polyval(np.polyfit(np.arange(df_500.shape[0]), df_500[i], 4), np.arange(df_500.shape[0]))
    
    fig, ax = plt.subplots()
    df.plot(colormap=plt.cm.get_cmap("Spectral"), ax=ax)
    # df_rolling.plot(ax=ax)
    # df_poly.plot(ax=ax)
