# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:43:09 2021

@author: iant

SIMULATION CONFIG
"""

import numpy as np

AOTF_OFFSET_SHAPE = "Gaussian"
# AOTF_OFFSET_SHAPE = "Constant"

# BLAZE_WIDTH_FIT = False
BLAZE_WIDTH_FIT = True

AOTF_FROM_FILE = True
# AOTF_FROM_FILE = False


sim_parameters = {
    4383.5:{
        "centre_order":194, 
        "order_range":[185, 205], 
        "nu_range":[4150., 4650.], 
        "pixels":np.arange(320), 
        "pixels_solar_line_area":np.arange(160, 320, 1), #only find area for solar line on right of detector
        "d_nu":0.005, 
        "solar_line_aotf_range":[26560, 26640],
        "solar_line_nu_range":[4381.74, 4384.385],
        "histogram_bins":np.linspace(4250., 4550., 501),
        "error_n_medians":50,
        "filter_smoothing":99,
        "filenames":[
            "20180716_000706_0p2a_SO_1_C", #(approx. orders 188-195) in steps of 4kHz
            "20181010_084333_0p2a_SO_2_C", #(approx. orders 194-201) in steps of 4kHz
            "20181129_002850_0p2a_SO_2_C", #(approx. orders 194-197) in steps of 2kHz
            "20181206_171850_0p2a_SO_2_C", #(approx. orders 191-198) in steps of 4kHz
            "20190416_020948_0p2a_SO_1_C", #(approx. orders 194-195) in steps of 1kHz
            "20190416_024455_0p2a_SO_1_C", #(approx. orders 194-201) in steps of 4kHz
            "20210201_111011_0p2a_SO_2_C", #(approx. orders 188-202) in steps of 8kHz
            "20210226_085144_0p2a_SO_2_C", #(approx. orders 188-195) in steps of 4kHz
        ],
        "solar_spectra":{
            "ACE":"Solar_irradiance_ACESOLSPEC_2015.dat", 
            "PFS":"pfsolspec_hr.dat",
        },

    },
    4276.1:{
        "centre_order":189, 
        "order_range":[180, 200], 
        "nu_range":[4040., 4535.], 
        "pixels":np.arange(320),
        "pixels_solar_line_area":np.arange(160, 320, 1), #only find area for solar line on right of detector
        "d_nu":0.005, 
        "solar_line_aotf_range":[25923-40, 25923+40],
        "solar_line_nu_range":[4275.6, 4276.8],
        "histogram_bins":np.linspace(4150., 4400., 501),
        "error_n_medians":20,
        "filter_smoothing":99,
        "filenames":[
            "20180716_000706_0p2a_SO_1_C", #25719-26738kHz (approx. orders 188-195) in steps of 4kHz
            "20190819_001536_0p2a_SO_2_C", #24851-26890kHz (approx. orders 182-196) in steps of 8kHz
            "20190819_001536_0p2a_SO_1_C", #23981-26020kHz (approx. orders 176-190) in steps of 8kHz
            "20200606_205016_0p2a_SO_1_C", #23981-26020kHz (approx. orders 176-190) in steps of 8kHz
            "20200606_205016_0p2a_SO_2_C", #24851-26890kHz (approx. orders 182-196) in steps of 8kHz
            "20210201_111011_0p2a_SO_2_C", #25719-27758kHz (approx. orders 188-202) in steps of 8kHz
            "20210226_085144_0p2a_SO_1_C", #25285-26304kHz (approx. orders 185-192) in steps of 4kHz
            "20210226_085144_0p2a_SO_2_C", #25719-26738kHz (approx. orders 188-195) in steps of 4kHz
        ],
        "solar_spectra":{
            "ACE":"Solar_irradiance_ACESOLSPEC_2015.dat", 
            "PFS":"pfsolspec_hr.dat",
        },
    },
    3787.9:{
        "centre_order":168, 
        "order_range":[160, 180], 
        # "nu_range":[4150., 4650.], 
        "pixels":np.arange(320), 
        "d_nu":0.005, 
        "absorption_line_aotf_range":[26560, 26640],
        "error_n_medians":50,
    },
}


ORDER_RANGE = [185, 205]
nu_range = [4150., 4650.]


pixels = np.arange(320)

# nu_range = [4309.7670539950705, 4444.765043408191]
D_NU = 0.005

abs_aotf_range = [26560, 26640]