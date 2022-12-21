# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:50:38 2022

@author: iant

PHOBOS FOV LINESCAN ANALYSIS
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp

from datetime import datetime


from instrument.nomad_lno_instrument_v01 import m_aotf
from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
from tools.general.normalise_values_to_range import normalise_values_to_range
from tools.general.get_minima_maxima import get_local_maxima
from tools.spectra.running_mean import running_mean_1d
from tools.spice.load_spice_kernels import load_spice_kernels


# plot_level = -1
# plot_level = 0
plot_level = 2



bad_pixel_d = {
    (144, 2):[12, 102, 312],
    (146, 2):[218, ],
    (148, 2):[],
    (150, 2):[],
    (152, 2):[37, 40, 245, ],
    (154, 2):[],
    (156, 2):[209, ],
    (158, 2):[],
    
}



subtract_last_bin = True #bad
subtract_last_bin = False

zero_lhs = True
# zero_lhs = False #bad


obs_types = {
    # "LNO prime Phobos linescan 1":{"h5":"20221113_202115_0p1a_LNO_1", "bins":[2, 3, 4]},
    "UVIS prime Phobos linescan 1":{"h5":"20221116_170814_0p1a_LNO_1", "bins":[3,4,5,6]},
}




colour_bin_dict = {
    144:{"colour":"C0"},
    146:{"colour":"C1"},
    148:{"colour":"C2"},
    150:{"colour":"C3"},
    152:{"colour":"C4"},
    154:{"colour":"C5"},
    156:{"colour":"C6"},
    158:{"colour":"C7"},
}


# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_PHOBOS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 401
# et2lst: form of longitude supplied by the variable lon
SPICE_LONGITUDE_FORM = "PLANETOCENTRIC"
# spkpos: reference frame relative to which the output position vector
# should be expressed
SPICE_REFERENCE_FRAME = "J2000"
#et2utc: string format flag describing the output time string. 'C' Calendar format, UTC
SPICE_STRING_FORMAT = "C"
# et2utc: number of decimal places of precision to which fractional seconds
# (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed
SPICE_TIME_PRECISION = 3

SPICE_TARGET = "PHOBOS"
SPICE_OBSERVER = "-143"

SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"

# SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
# SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"

DREF = "TGO_NOMAD_LNO_OPS_NAD"

sp.kclear()
load_spice_kernels()

# sp.pdpool("INS-143310_FOV_REF_ANGLE", [4.000000] )
# sp.pdpool("INS-143311_FOV_REF_ANGLE", [4.000000] )
# sp.pdpool("INS-143312_FOV_REF_ANGLE", [4.000000] )
# "INS-143311_FOV_REF_ANGLE = (  4.000000 )",
# "INS-143312_FOV_REF_ANGLE = (  4.000000 )",
# ])


# PHOBOS_OFFSET_ARCMINS = 0
PHOBOS_OFFSET_ARCMINS = -3.25

new_vector = 0.37000276139181304 + PHOBOS_OFFSET_ARCMINS / 60.0

sp.pdpool("TKFRAME_-143311_ANGLES", [new_vector, 0.6000003670468607, 0.00000000000000000] )

 

for obs_type in obs_types.keys():
    
    h5 = obs_types[obs_type]["h5"]

    h5_f = open_hdf5_file(h5, path=r"E:\DATA\hdf5_phobos") #no detector offset!)


    phobos_bins = np.array(obs_types[obs_type]["bins"])


    y = h5_f["Science/Y"][...]
    unique_bins = sorted(h5_f["Science/Bins"][0, :, 0])
    binning = unique_bins[1] - unique_bins[0]
    
    aotf_f = h5_f["Channel/AOTFFrequency"][...]
    unique_freqs = sorted(list(set(aotf_f)))
    
    orders = np.array([m_aotf(i) for i in aotf_f])
    unique_orders = sorted(list(set(orders)))
    
    obs_dt_str = h5_f["Geometry/ObservationDateTime"][...]
    # dt_str = obs_dt_str[0, 0].decode()
    
    dt_s = [datetime.strptime(s[0].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]
    dt_e = [datetime.strptime(s[1].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]
    
    dt_means = [((e - s) / 2) + s for s, e in zip(dt_s, dt_e)]
    
    
    print(obs_type, "orders:", ", ".join([str(i) for i in unique_orders]))
    
    
    #add offset?
    y += 0#6.72
    
    
    #correct bad pixels
    for i, unique_bin in enumerate(unique_bins):
        bad_pixels = bad_pixel_d[(unique_bin, binning)]
        y[:, i, bad_pixels] = np.nan


    #remove offsets by forcing the left hand side to zero?
    if zero_lhs:
        y_lhs = np.nanmean(y[:, :, 0:50], axis=2)
        y -= np.repeat(y_lhs, (y.shape[2])).reshape(y.shape[0], y.shape[1], -1)
        
        # plt.figure()
    
    
    y_spectral_mean = np.nanmean(y[:, :, 100:300], axis=2) #spectral mean
    
    #subtract the first bin? Only for the spectrally binned case, pixel by pixel sub too noisy
    if subtract_last_bin:
        y_spectral_mean -= np.tile(y_spectral_mean[:, -1], (len(unique_bins), 1)).T #subtract the first bin
    
    
    y_column_mean = np.nanmean(y_spectral_mean, axis=1) #mean of rows
    
    
    # im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
    # stop()
    
    
    if plot_level < -1:
        #plot raw data for every bin to check for noisy regions
        plt.plot(y_spectral_mean)
    
    
    
    if plot_level < 0:
        #plot all good spectra for a specific bin
        for bin_ in range(y.shape[1]):
            plt.figure(figsize=(8, 4))
            plt.title("Raw spectra for bin %i" %bin_)
            plt.xlabel("Pixel number")
            plt.ylabel("Signal (counts)")
            for i, y_spectrum in enumerate(y[:, bin_, :]):
                colour = "C0"
                plt.plot(y_spectrum, alpha=0.3, color=colour)
                if np.any(y_spectrum > 100):
                    plt.plot(y_spectrum, color="k")
                    print(unique_bins[bin_])
    
    
        # sys.exit()
   
    
    if plot_level < 1:
        #plot raw y data spectrally binned
        plt.figure(figsize=(12, 3))
        plt.title("%s: raw Y data, spectrally binned" %h5)
        plt.xlabel("Frame index (all orders)")
        plt.ylabel("Bin number")
        im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
        cbar = plt.colorbar(im)
        cbar.set_label("Signal (counts)", rotation=270, labelpad=10)
        plt.subplots_adjust(bottom=0.15)
    
    
    phobos_bin_means = y_spectral_mean[:, phobos_bins]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    phobos_mean = np.nanmean(phobos_bin_means, axis=1)
    for bin_ix, phobos_bin in enumerate(phobos_bins):
        plt.plot(dt_means, phobos_bin_means[:, bin_ix], alpha=0.33)
    # plt.plot(phobos_mean, "k--")
    phobos_running_mean = running_mean_1d(phobos_mean, 19)
    ax.plot(dt_means, phobos_running_mean, "k-", label="LNO signal (smoothed)")
    
    phobos_local_max_ix = get_local_maxima(phobos_running_mean)
    
    phobos_obs_ix = np.where(phobos_running_mean > 2.5)[0]
    
    phobos_max_ix = [i for i in phobos_local_max_ix if i in phobos_obs_ix]
    for ix, frame_ix in enumerate(phobos_max_ix):
        
        if ix == 0:
            label = "Max LNO signal"
        else:
            label = ""
            
        #remove local minima
        if phobos_running_mean[frame_ix] > phobos_running_mean[frame_ix -2] and phobos_running_mean[frame_ix] > phobos_running_mean[frame_ix+2]:
        
            ax.axvline(x=dt_means[frame_ix], color="k", linestyle="--", label=label)
            ax.text(dt_means[frame_ix], 4.5-ix, dt_means[frame_ix])
            print(dt_means[frame_ix])
        
    
    
    ets = [sp.utc2et(datetime.strftime(s, "%Y %b %d %H:%M:%S.%f")) for s in dt_means]
    phobos_pos_lno = np.array([sp.spkpos(SPICE_TARGET, et, DREF, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0] for et in ets])
    
    phobos_vec_lno = np.array([s / sp.vnorm(s) for s in phobos_pos_lno])
    
    # plt.figure()
    # plt.plot(phobos_vec_lno[:, 0])
    ax2 = ax.twinx()
    ax2.plot([dt_means[0], dt_means[-1]], [0, 0], "b--", label="Zero offset from Phobos centre")
    ax2.plot(dt_means, phobos_vec_lno[:, 1], "b", label="LNO offset from Phobos centre")
    # plt.plot(phobos_vec_lno[:, 2])

    ax.set_xlabel("Observation datetime")
    ax.set_ylabel("LNO signal in illuminated bins")
    ax.set_title("LNO Phobos linescan with %0.2f arcminute boresight offset" %PHOBOS_OFFSET_ARCMINS)
    ax2.set_ylabel("LNO vector offset from Phobos centre with %0.2f arcminute offset" %PHOBOS_OFFSET_ARCMINS, color="b")
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # ax.legend()
    # ax2.legend()
    ax2.grid()
    
    # plt.savefig("LNO_phobos_linescan_%0.2f_arcminute_offset.png" %PHOBOS_OFFSET_ARCMINS)