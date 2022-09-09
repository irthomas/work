# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:27:04 2022

@author: iant
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import spiceypy as sp
from datetime import datetime

from tools.file.hdf5_functions import make_filelist

from tools.spice.load_spice_kernels import load_spice_kernels




ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

NA_VALUE = -999 #value to be used for NaN

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

# SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
# SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"

SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"




regex = re.compile(".*_UVIS_P")
# regex = re.compile("20210927_224950_.*_UVIS_P")
# regex = re.compile("20210921_132947_0p2a_UVIS_P")

# h5 = "20220301_063212_0p3k_SO_A_E_185"


# h5_f = open_hdf5_file(h5)


h5_fs, h5s, _= make_filelist(regex, "hdf5_level_0p2a")

load_spice_kernels()


with PdfPages("uvis_boresight_phobos_observations.pdf") as pdf: #open pdf

    for h5, h5_f in zip(h5s, h5_fs):
    
        it = h5_f["Channel/IntegrationTime"][0]    
        
        
            
        ymdhm = [int(i) for i in [h5[0:4], h5[4:6], h5[6:8], h5[9:11], h5[11:13]]]
        dt = datetime(*ymdhm)
        
        observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
        
        flags = h5_f["Channel"]["ReverseFlagAndDataTypeFlagRegister"][...]
    
        #make non-point dictionary
        d = {}
        d_tmp = {}
    
        d["et_s"] = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
        d["et_e"] = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
        d["et"] = np.vstack((d["et_s"], d["et_e"])).T
        
    
        dref = "TGO_NOMAD_UVIS_NAD"
        channelId = sp.bods2c(dref) #find channel id number
        [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)
    
    
        fov = boresightVector
    
    
        d_tmp["obs_subpnt_s"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_s"]]
        d_tmp["obs_subpnt_e"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_e"]]
    
    
        dp = {}
        dp_tmp = {}
    
        point = 0
        dp[point] = {}
        dp_tmp[point] = {}
    
        dp_tmp[point]["sincpt_s"] = []
        dp_tmp[point]["surf_s"] = []
        dp_tmp[point]["ilumin_s"] = []
        dp[point]["ph_angle_s"] = []
        dp[point]["inc_angle_s"] = []
        dp[point]["em_angle_s"] = []
    
        #check each fov corner individually to see which hit phobos
        for flag, et in zip(flags, d["et_s"]):
            
            if flag != 4:
                continue
            
            try:
                sincpt = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, dref, fov)
            except sp.stypes.NotFoundError:
                sincpt = (np.zeros(3) + NA_VALUE, NA_VALUE, np.zeros(3) + NA_VALUE)
    
            if sincpt[1] != NA_VALUE:
                ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, sincpt[0])
                dp[point]["ph_angle_s"].append(ilumin[2] * SP_DPR)
                dp[point]["inc_angle_s"].append(ilumin[3] * SP_DPR)
                dp[point]["em_angle_s"].append(ilumin[4] * SP_DPR)
            else:
                dp[point]["ph_angle_s"].append(NA_VALUE)
                dp[point]["inc_angle_s"].append(NA_VALUE)
                dp[point]["em_angle_s"].append(NA_VALUE)
    
            
            # y = h5_f["Science/Y"][...]
            # y_frame = y[20, :, :]
            # plt.imshow(y_frame)
            # y_bin = np.mean(y[:, 100:150, 500:1000], axis=(1,2))
            # plt.plot(y_bin)        
            # for y_ in y_bin:
                # plt.plot(y_)
    
        #if more than 1/3 pointing to moon
        if len([i for i in dp[point]["ph_angle_s"] if i > -998.]) > 5:
            plt.figure(figsize=(8,6), constrained_layout=True)
            plt.title("%s: Integration time %is" %(h5, it/1000))
            plt.ylabel("Angle at centre of FOV")
            plt.xlabel("Frame number")
            plt.grid()
            plt.plot(dp[point]["ph_angle_s"], label="Phase angle")
            plt.plot(dp[point]["inc_angle_s"], label="Solar incidence angle")
            plt.plot(dp[point]["em_angle_s"], label="Emission angle")
            plt.legend()
            plt.ylim((0, 90))


            pdf.savefig()
            plt.close()

            print("%s: Integration time %is" %(h5, it/1000))
