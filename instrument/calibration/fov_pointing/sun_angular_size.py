# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301
"""
Created on Mon May  7 09:31:36 2018

@author: iant
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp

BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
#BASE_DIRECTORY = os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")
#KERNEL_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\kernels\mk")
KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
#KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
#KERNEL_DIRECTORY = os.path.normcase(r"D:\kernels\kernels\mk")
METAKERNEL_NAME = "em16_plan.tm"


FIG_X = 15
FIG_Y = 9

def et_2_utc(ephemeris_time):
    """convert ephemeris time to utc"""
    formatstr = "C"
    return sp.et2utc(ephemeris_time, formatstr, 0)

def get_sun_sizes(utc_start, utc_end, step_size):
    """get sun angular size and time steps given start and end times"""
    #spice constants
    abcorr = "None"
    #tolerance = "1"
    #method = "Intercept: ellipsoid"
    #prec = 3
    #shape = "Ellipsoid"


    #load spiceypy kernels
    os.chdir(KERNEL_DIRECTORY)
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    os.chdir(BASE_DIRECTORY)
    utctimestart = sp.str2et(utc_start)
    utctimeend = sp.str2et(utc_end)


    durationseconds = utctimeend - utctimestart
    nsteps = int(np.floor(durationseconds/step_size))
    timesteps = np.arange(nsteps) * step_size + utctimestart

    ref = "J2000"
    observer = "-143"
    target = "SUN"
    #get TGO-SUN pos
    tgo2sunpos = [sp.spkpos(target, time, ref, abcorr, observer)[0] for time in timesteps]
    sunaxes = sp.bodvrd("SUN", "RADII", 3)[1][0] #get mars axis values

    return ([np.arctan((sunaxes*2.0)/np.linalg.norm(tgo2sunVector))*sp.dpr()*60.0 \
                for tgo2sunVector in tgo2sunpos], timesteps)


TIME_STEP_SIZE = 60*60*24*30
SUN_SIZES, TIMES = get_sun_sizes("2018APR01-00:00:00 UTC", "2019DEC01-00:00:00 UTC", TIME_STEP_SIZE)
TIME_STRINGS = [et_2_utc(time)[0:9] for time in TIMES]


plt.figure(figsize=(FIG_X, FIG_Y))
plt.scatter(TIMES, SUN_SIZES)
plt.xticks(TIMES, TIME_STRINGS, rotation=90)
#plt.xlabel("Time")
plt.ylabel("Solar disk angular size (arcmins)")
plt.savefig(BASE_DIRECTORY+os.sep+"Sun_angular_size.png")