# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:29:22 2023

@author: iant

READ IN HDF5 FILE

get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst)
"""

import os
import h5py

from analysis.so_lno_2023.config import ROOT_DATA_DIR


def make_filepath(h5):
    
    year = h5[0:4]
    month = h5[4:6]
    day = h5[6:8]
    
    h5_filepath = os.path.join(ROOT_DATA_DIR, year, month, day, "%s.h5" %h5)
    return h5_filepath
    




def read_h5(h5):
    
    h5_filepath = make_filepath(h5)
    
    with h5py.File(h5_filepath, "r") as h5_f:
        
        y = h5_f["Science/Y"][...]
        y_raw = h5_f["Science/YUnmodified"][...]
        x = h5_f["Science/X"][0, :]
        order = h5_f["Channel/DiffractionOrder"][0]
        aotf_freq = h5_f["Channel/AOTFFrequency"][0]
        
        nomad_t = h5_f["Channel/MeasurementTemperature"][0][0]
        
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        lats = h5_f["Geometry/Point0/Lat"][:, 0]
        lons = h5_f["Geometry/Point0/Lon"][:, 0]
        
        myear = 36
        ls = h5_f["Geometry/LSubS"][0, 0] #first value
        lst = h5_f["Geometry/Point0/LST"][:, 0]
        
    return {"y":y, "y_raw":y_raw, "x":x, "order":order, "aotf_freq":aotf_freq, "nomad_t":nomad_t, "alts":alts, "lats":lats, "lons":lons, "myear":myear, "ls":ls, "lst":lst}
    
    
    

