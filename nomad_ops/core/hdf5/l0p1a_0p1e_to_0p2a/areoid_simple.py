# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:05:15 2020

@author: iant

AREOID SIMPLIFIED MODEL - LOOP THROUGH LONS AND LATS AND INTERPOLATE WITH 4PPD AREOID MODEL

"""
#import os
import numpy as np
#import h5py

import logging

#from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import PFM_AUXILIARY_FILES    
    
__project__   = "NOMAD"
__author__    = "Ian Thomas, Justin Erwin"
__contact__   = "ian.thomas@aeronomie.be"

logger = logging.getLogger( __name__ )

def geoid_4ppd(lons, lats, ra, nan_indices):
    """strip out all other options and pass h5 data to function"""

    ires = 4.

    ras = np.ones_like(lons) * -999.0
    for i, (lon, lat, flag) in enumerate(zip(lons, lats, nan_indices)):
        if flag:
            lon = np.mod(lon, 360.)
            assert lat >= -90. and lat <= 90.
        
            ilon = int(np.floor(lon*ires))
            ilat = int(np.floor((lat+90.)*ires))
        
            lon1 = ilon/ires
            lon2 = (ilon+1)/ires
            lat1 = ilat/ires - 90.
            lat2 = (ilat+1)/ires - 90.
            w_lat_lon = bilinear_spherical_interp_weights([lat1,lat2],[lon1,lon2],lat,lon)
        
            ras[i] = (np.sum(w_lat_lon*ra[ilat:ilat+2,ilon:ilon+2])) / 1.0e3

    return ras


def bilinear_spherical_interp_weights(Lats, Lons, lat, lon):
    """ Compute the weights for a Langrange interpolation """

    Thetas = np.pi/2. - np.deg2rad(Lats)
    theta = np.pi/2. - np.deg2rad(lat)
    Phis = np.deg2rad(Lons)
    phi = np.deg2rad(lon)

    if Phis[0] > Phis[1]:
        if phi >= Phis[1]:
            phi = phi - 2*np.pi
        Phis[0] = Phis[0] - 2*np.pi
    #print(Thetas, theta, Phis, phi)

    w = np.outer([abs(np.cos(theta)-np.cos(Thetas[1])), abs(np.cos(theta)-np.cos(Thetas[0]))],
                    [abs(phi-Phis[1]), abs(phi-Phis[0])])
    w /= w.sum()

    return w
