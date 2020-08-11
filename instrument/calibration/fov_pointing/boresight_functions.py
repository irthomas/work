# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:03:33 2020

@author: iant

Field of view simulations


"""
#import os
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#from datetime import datetime
import spiceypy as sp

#from tools.file.paths import paths, FIG_X, FIG_Y
#from tools.file.hdf5_functions import open_hdf5_file, get_files_from_datastore
from tools.spice.load_spice_kernels import load_spice_kernels
#from tools.spice.datetime_functions import et2utc, utc2et


DETECTOR_CENTRE_LINES = {"so":128, "lno":152}
#boresights = {"so":[-0.92156, -0.38819, 0.00618]}
BORESIGHTS = {"so":[-0.9218973, -0.38738526, 0.00616719]} #June 2018 onwards

SPICE_TOLERANCE = "1"

channel = "so"

load_spice_kernels()



#test stuff
#obs=-143
#ref="J2000"
#abcorr = "NONE"
#targ = "SUN"
#inst = -143000
#tolerance = "1"
#
#et = 579302388.5957096 #centre of MTP001 scan 2
#sctol = sp.sctiks(obs, tolerance)
#scticks = sp.sce2c(obs, et)
#cmatrix, sc_time = sp.ckgp(inst, scticks, sctol, ref)
#
#sun_pos = sp.spkpos(targ, et, ref, abcorr, str(obs))[0]
#sun_pos_norm = sun_pos / np.linalg.norm(sun_pos)
#
#boresight = np.asfarray([-0.9218973, -0.38738526, 0.00616719])
#boresight_j2000 = np.dot(np.transpose(cmatrix), boresight)
#
#sp.vsep(boresight_j2000, sun_pos_norm)
#lon_lat = sp.reclat(boresight)[1:3]
#lon_lat_j2000 = sp.reclat(boresight_j2000)[1:3]
#stop()


def find_cmatrix(ets_in):
    """find list of c-matrices given ephemeris times and time errors"""

    obs="-143"
    ref="J2000"
    inst = int(obs) * 1000
    matrices = []
    sctol = sp.sctiks(int(obs), SPICE_TOLERANCE)

    for et in ets_in:
        scticks = sp.sce2c(int(obs), et)
        [matrix, sc_time] = sp.ckgp(inst, scticks, sctol, ref)
        matrices.append(matrix)
    return matrices


def rotation_matrix(theta):
    rot_matrix = np.asfarray([[1., 0., 0.], [0., np.cos(theta), np.sin(theta)], [0., -1.*np.sin(theta), np.cos(theta)]])
    return rot_matrix    
    


def find_boresight(ets_in, boresight_in, theta=0.):
    """return array of boresight pointing directions given ephemeris times, 
    time errors and TGO-to-channel boresight vector"""
    cmatrices = find_cmatrix(ets_in)
    boresights = np.zeros((len(cmatrices), 3))
    
    rot_matrix = rotation_matrix(theta)


    for index, cmatrix in enumerate(cmatrices):
        boresights[index,:] = np.dot(rot_matrix, np.dot(np.transpose(cmatrix), boresight_in))
    return boresights


def find_rotation_angle(et_start, et_end, scan_index, channel):
    """unreliable"""
    
    boresight = BORESIGHTS[channel]
    
    x1 = sp.reclat(find_boresight([et_start], boresight)[0])[1]
    x2 = sp.reclat(find_boresight([et_end], boresight)[0])[1]
    y1 = sp.reclat(find_boresight([et_start], boresight)[0])[2]
    y2 = sp.reclat(find_boresight([et_end], boresight)[0])[2]
    
    #find angle to vertical
    if scan_index == 0:
        rotation_angle = np.arctan((y2-y1)/(x2-x1))
    if scan_index == 1:
    #find angle to horizontal
        rotation_angle = np.arctan((y2-y1)/(x2-x1)) - np.pi/2.
    print(rotation_angle * 180 / np.pi)
    return rotation_angle







        
