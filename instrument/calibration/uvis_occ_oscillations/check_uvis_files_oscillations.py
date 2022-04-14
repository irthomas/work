# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:08:52 2022

@author: iant

CHECK UVIS 0.3K FILES FOR OSCILLATIONS
"""


import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist


regex = re.compile("20181029_203917_.*_UVIS_E")
# regex = re.compile("20180622_......_.*_UVIS_E")
# regex = re.compile("2018...._......_.*_UVIS_E")

file_level = "hdf5_level_0p3k"



hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)


transmittance_dict = {}

with PdfPages("uvis_oscillations.pdf") as pdf: #open pdf
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
        
        #select 500 to 600nm region
        df_500 = df[(df.index > 500) & (df.index < 600)]
        
        #if no visible region spectra acquired
        if len(df_500.index) == 0:
            continue
        
        #apply rolling mean (follows oscillation pattern)
        df_rolling = df_500.rolling(5, axis=0, center=True).mean()
        
        #new dataframe with 4th order polynomial fit to spectra continuum (no fitting to oscillations)
        df_poly = df_500.copy()
        for i in df_500.columns:
            df_poly[i] = np.polyval(np.polyfit(np.arange(df_500.shape[0]), df_500[i], 4), np.arange(df_500.shape[0]))
        
        #subtract rolling mean from raw spectra
        df_rolling_sub = df_500 - df_rolling
        #subtract polynomial fit from raw spectra
        df_poly_sub = df_500 - df_poly
        
        
        
        #standard deviations
        rolling_std = df_rolling_sub.std()
        poly_std = df_poly_sub.std()
        
        #mean transmittance of polynomial (continuum) fit
        mean_trans = df_poly.mean()
        
        #
        sigma1 = (poly_std / rolling_std)
        # sigma2 = (poly_std / rolling_std) / mean_trans
        
        
        max_sigma = max(sigma1)

        alt_max_sigma = sigma1.index[sigma1 == max_sigma]
        index_max_sigma = sigma1.index.get_loc(alt_max_sigma[0])
        
        next_alt_max_sigma = sigma1.index[index_max_sigma+1]
        prev_alt_max_sigma = sigma1.index[index_max_sigma-1]
        
        trans_max_sigma = mean_trans[alt_max_sigma]
        next_trans_max_sigma = mean_trans[next_alt_max_sigma]
        prev_trans_max_sigma = mean_trans[prev_alt_max_sigma]


        transmittance_dict[h5_filename] = {"cont_trans":trans_max_sigma.item(), "alt":alt_max_sigma.item(), \
                                           "max_sigma":max_sigma, "next_cont_trans":next_trans_max_sigma, "prev_cont_trans":prev_trans_max_sigma}
        
        plt.figure(figsize=(10, 8))
        plt.title(h5_filename)
        plt.axhline(y=alt_max_sigma, c="k", linestyle="--")
        plt.plot(sigma1, sigma1.index, label="Oscillation strength")
        plt.plot(mean_trans, mean_trans.index, label="Mean transmission 500-600nm")
        plt.text(trans_max_sigma+0.01, trans_max_sigma.index[0]+3, "T=%0.2f" %trans_max_sigma, )
        plt.legend(loc="upper right")
        plt.ylim([-5, 270])
        plt.xlim([-0.05, 2.5])
        
        plt.grid()
        plt.xlabel("Strength / transmittance")
        plt.ylabel("Tangent Alt Areoid (km)")
        pdf.savefig()
        # plt.close()
