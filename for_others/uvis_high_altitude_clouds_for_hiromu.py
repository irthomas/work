# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:46:08 2022

@author: iant

CHECK UVIS SPECTRA FOR HIGH ALTITUDE CLOUDS
"""


# import os
import re
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime

# from tools.file.paths import paths
from tools.file.hdf5_functions import make_filelist, open_hdf5_file
# from tools.general.get_nearest_index import get_nearest_index
# from tools.spectra.baseline_als import baseline_als
# from tools.plotting.colours import get_colours
# from tools.general.get_minima_maxima import get_local_minima, get_local_maxima

# from tools.spectra.fft_zerofilling import fft_hr_nu_spectrum
# from tools.spectra.savitzky_golay import savitzky_golay
# from tools.spectra.fit_gaussian_absorption import fit_gauss, make_gauss

from matplotlib.backends.backend_pdf import PdfPages

from tools.general.progress_bar import progress_bar

regex = re.compile("20......_......_.*_UVIS_[IE]")
file_level = "hdf5_level_0p3k"

#set top of atmosphere to be 100. Check all spectra above this level
toa = 100.0

def make_solar_spectra_obs_dict(regex, file_level, toa):

    _, h5s, _ = make_filelist(regex, file_level, open_files=False)
    spectra_dict = {}
    
    print("Collecting spectra from files")
    # for i, h5 in enumerate(h5s):
    for h5 in progress_bar(h5s):
        
        # if np.mod(i, 100) == 0:
        #     print("%i/%i" %(i, len(h5s)))
        
        h5_f = open_hdf5_file(h5)
    
        mode = h5_f["Channel/AcquisitionMode"][0]
        #if not full frame, skip file
        if mode != 0:
            continue
        
        # x = h5_f["Science/X"][0, :]
        
        # #most observations have binning=7
        # if len(x) != 128:
        #     continue

        alts_all = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        pointing_deviation = h5_f["Geometry/FOVSunCentreAngle"][:, 0]

        toa_ix = np.argmin(alts_all > toa)
        
        #if too close to start/end of file
        if toa_ix < 10 or toa_ix > len(alts_all) - 10:
            continue
        
        if alts_all[0] > alts_all[1]: #if ingress
            spectra = h5_f["Science/Y"][:toa_ix, :]
            pointing_error = pointing_deviation[:toa_ix]
            alts = alts_all[:toa_ix]
        else:
            spectra = h5_f["Science/Y"][toa_ix:, :]
            pointing_error = pointing_deviation[toa_ix:]
            alts = alts_all[toa_ix:]
    
        #if all nans
        if np.all(np.isnan(spectra)):
            continue
    
        spectra_dict[h5] = {"spectra":spectra, "pointing_error":pointing_error, "alts":alts}
        
        h5_f.close()
        
    return spectra_dict


spectra_dict = make_solar_spectra_obs_dict(regex, file_level, toa)

with PdfPages("high_altitude_clouds.pdf") as pdf: #open pdf
    for h5 in spectra_dict.keys():
        spectra = spectra_dict[h5]["spectra"]
        
        #take the difference between consecutive spectra
        diff = np.diff(spectra.T)
        # plt.plot(diff.T)
        
        shape = spectra.shape
        start_row = int(400 * shape[1] / 1024)
        end_row = int(1000 * shape[1] / 1024)
        row_delta = int((end_row - start_row - 2)/10)
        #find the mean difference for the most illuminated part of the detector
        diff_mean = np.mean(diff[start_row:end_row, :], axis=0)
        # plt.plot(diff)
        # plt.plot(diff_mean)
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        
        #apply smoothing to the mean difference to remove small pixel-to-pixel variations        
        diff_mean_convolved = np.convolve(diff_mean, kernel, mode='same')
        
        # plt.plot(diff_mean_convolved, label=h5)
        
        #if the smoothed difference is high enough, check if associated to a high pointing error
        if np.max(diff_mean_convolved) > 10:
            if np.max(spectra_dict[h5]["pointing_error"]) < 1.0:
                
                #if not, plot the data and save to pdf
                alts = spectra_dict[h5]["alts"]
                fig, (ax1a, ax1b, ax1c) = plt.subplots(figsize=(12, 10), nrows=3, sharex=True)
                fig.suptitle(h5)
                ax1a.plot(alts[:-1], diff_mean)
                ax1b.plot(alts, spectra[:, np.arange(start_row, end_row, row_delta)]/np.mean(spectra[:, np.arange(start_row, end_row, row_delta)], axis=0), label=np.arange(start_row, end_row, row_delta))
                ax1c.plot(alts, spectra_dict[h5]["pointing_error"])
                
                ax1c.set_xlabel("Tangent Altitude (km)")
                ax1a.set_ylabel("Mean difference between consecutive spectra (smoothed)")
                ax1b.set_ylabel("Counts of selected pixels")
                ax1c.set_ylabel("TGO pointing error")
                
                ax1a.grid()
                ax1b.grid()
                ax1c.grid()
                
                ax1b.legend(loc="upper right")
    
                pdf.savefig()
                plt.close()
