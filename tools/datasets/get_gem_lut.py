# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:11:45 2023

@author: iant

GEM LUT
IF OBSERVATION GEOMETRY ALREADY IN FILE, GET DATA FROM LUT
IF OBSERVATION IS NOT FOUND, GET DATA AND ADD TO LUT
"""

import os
import h5py
import numpy as np

from tools.file.paths import paths

from tools.datasets.get_gem_data import get_gem_data



# GEM_KEYS = ['z', 'p', 't', 'nd', 'co2', 'co', 'h2o', 'o2', 'o3', 'ext_dust', 'ext_h2o_ice']

def group_name(lat, lon, lst):
    
    return "lat%0.0f_lon%0.0f_lst%0.0f" %(np.round(lat), np.round(lon), np.round(lst))




def get_gem_lut(myear, ls, lat, lon, lst, plot=False):
    """get data from LUT if it exists, if not add it to the GEM LUT"""
    
    if ls > 359.5:
        ls = 0.1
        myear += 1
    
    
    gem_filename = "gem_my%i.h5" %(myear)
    gem_filepath = os.path.join(paths["LOCAL_DIRECTORY"], "lut", gem_filename)
    
    
    # make empty file if it doesn't exist
    if not os.path.exists(gem_filepath):    
        with h5py.File(gem_filepath, "w") as h5f:
            for ls_all in np.arange(360.0):
                h5f.create_group("%0.0f" %ls_all)
    
    
    
    
    with h5py.File(gem_filepath, "a") as h5f:
        #get list of group names for given Ls
        existing_groups = list(h5f["%0.0f" %np.round(ls)].keys())
        
        new_obs = group_name(lat, lon, lst)
        
        if new_obs not in existing_groups:
            #if not in LUT, get data
            atmos_dict = get_gem_data(myear, ls, lat, lon, lst, plot=plot)
            
            #add to lut
            #make new group for lat/lon/lst
            h5f["%0.0f" %np.round(ls)].create_group(new_obs)
            #add each item to group
            for key, value in atmos_dict.items():
                h5f["%0.0f" %np.round(ls)][new_obs].create_dataset(key, dtype=np.float32, data=value, compression="gzip", shuffle=True)
                
        else:
            #if already in LUT, just get the data
            atmos_dict = {}
            for key in h5f["%0.0f" %np.round(ls)][new_obs].keys():
                atmos_dict[key] = h5f["%0.0f" %np.round(ls)][new_obs][key][...]
                
    return atmos_dict



def get_gem_lut_tpvmr(molecule, alt_grid, myear, ls, lat, lon, lst, plot=False):
    """get temperature, pressure, mol ppmv and co2 ppmv for given altitude(s), time and location from lut"""
    
    gem_d = get_gem_lut(myear, ls, lat, lon, lst, plot=plot)
    
    t = np.interp(alt_grid, gem_d["z"][::-1], gem_d["t"][::-1])
    pressure = np.interp(alt_grid, gem_d["z"][::-1], gem_d["p"][::-1]) / 101300. #pa to atmosphere
    mol_ppmv = np.interp(alt_grid, gem_d["z"][::-1], gem_d[molecule.lower()][::-1]) #ppmv
    co2_ppmv = np.interp(alt_grid, gem_d["z"][::-1], gem_d["co2"][::-1]) #ppmv

    return t, pressure, mol_ppmv, co2_ppmv


#testing

# list of GEM temperatures and pressures for a time/location
# myear = 35
# ls = 90.0
# lat = 0.0
# lon = 0.0
# lst = 12.0


# atmos_dict = get_gem_lut(myear, ls, lat, lon, lst)