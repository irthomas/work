# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:33:01 2022

@author: iant

READ IN MCD TERMINATOR OCCULTATION FILES

18 Times (Ls X+2.5 deg to X+90-2.5 deg in steps of 5 deg)
42 altitude layers (-5km to 200km in steps of 5km)
49 latitudes (90 to -90 deg in steps of -3.75 deg)
65 longitudes (-180 to +180 in steps of 5.625 deg)

num_co, num_co2, num_h2o_vap, num_o3, pressure, temp, h2o_ice, dustq
VMRx = num_x / (7.2429e16. × pressure/temp)

gcm_so_evening_MY34_LS_180-270_Ls.nc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime




from scipy.io import netcdf
from scipy.interpolate import interpn

from tools.file.paths import paths
from tools.spice.get_mars_year_ls import get_mars_year_ls


DUST_STORM_YEARS = [34]
DEFAULT_NON_STORM_YEAR = 35


conversion_dict = {
    "co":"num_co",
    "co2":"num_co2",
    "nd":"num_co2",
    "h2o":"num_h2o_vap",
    "o2":"num_o2",
    "o3":"num_o3",
    "p":"pressure",
    "t":"temp",
    "ext_h2o_ice":"h2o_ice",
    "ext_dust":"dustq"
    }


MCD_DIRPATH = os.path.join(paths["GCM_DIRECTORY"], "mcd", "ls_5deg_bins")




def get_surrounding_indices(array, value):
    
    ix = np.searchsorted(array, value)
    n_points = len(array)
    ixs = [max([0, ix - 1]), min([n_points - 1, ix])]
    
    return ixs




def interpolate_nearest_points(dataset_name, myear, ls, lat, lon, nc_dict):
    
    #get indices before and after point
    
    ls_ixs = get_surrounding_indices(nc_dict["Time"][...], ls)
    lat_ixs = get_surrounding_indices(nc_dict["latitude"][...], lat)
    lon_ixs = get_surrounding_indices(nc_dict["longitude"][...], lon)

    mcd_name = conversion_dict[dataset_name]
    
    #get data
    data_values = nc_dict[mcd_name][...]
    
    points = (nc_dict["Time"][...][ls_ixs], nc_dict["altitude"][...], nc_dict["latitude"][...][lat_ixs], nc_dict["longitude"][...][lon_ixs])
    values = data_values[ls_ixs[0]:(ls_ixs[1]+1), :, lat_ixs[0]:(lat_ixs[1]+1), lon_ixs[0]:(lon_ixs[1]+1)]
    point = (ls, nc_dict["altitude"][...], lat, lon)
    
    #do interpolation between 2 points in time/lat/lon space
    value = interpn(points, values, point)
    
    return value



def open_mcd_file(myear, ls, lst):
    """get mcd data from netcdf files. 
    Make corrections: remove negative altitude values, flip latitude axes to be ascending
    Take first/last values from next/previous nc file if ls range close to start/end of file
    """

    #choose correct Martian year
    if int(myear) in DUST_STORM_YEARS:
        myear = 34
    else:
        myear = DEFAULT_NON_STORM_YEAR
    
    if lst > 12.0:
        time = "evening"
    else:
        time = "morning"
        
    ls_start = int(np.floor(ls / 90.) * 90)
    ls_end = int(np.floor(ls / 90.) * 90) + 90
    
    mcd_filename = f"gcm_so_{time}_MY{myear}_LS_{ls_start}-{ls_end}_Ls.nc"
    filepath = os.path.join(MCD_DIRPATH, mcd_filename)
    nc_file = netcdf.NetCDFFile(filepath,'r')
    
    nc_dict = nc_file.variables
    
    append_to = "none"
    #special cases - interpolate between first/last points in adjacent files
    if ls < ls_start + 2.5: #interpolate to previous file
        append_to = "start"
        ls_start2 = ls_start - 90
        ls_end2 = ls_start
        myear2 = myear
        
        #year before
        if ls_start2 < 0:
            ls_start2 = 270
            ls_end2 = 360
            myear2 = myear - 1
    
    elif ls > ls_end - 2.5: #interpolate to next file
        append_to = "end"
        ls_start2 = ls_start + 90
        ls_end2 = ls_start
        myear2 = myear
        
        #year after
        if ls_start2 > 359:
            ls_start2 = 0
            ls_end2 = 90
            myear2 = myear + 1
    
    #open next/previous file and get data for first/last Ls
    if append_to in ["start", "end"]:
        mcd_filename2 = f"gcm_so_{time}_MY{myear2}_LS_{ls_start2}-{ls_end2}_Ls.nc"
        filepath2 = os.path.join(MCD_DIRPATH, mcd_filename2)
        nc_file2 = netcdf.NetCDFFile(filepath2,'r')
        nc_dict2 = nc_file2.variables
        
    
        #join dictionaries. Only need variables of dimensions 1 and 4
        nc_dict_joined = {}
        for key, value in nc_dict.items():
            shape = list(value.shape)
            
            if len(shape) == 1:
                if key == "Time":
                    shape[0] += 1
                    nc_dict_joined[key] = np.zeros(shape)
                    
                    if append_to == "start":
                        nc_dict_joined[key][0] = nc_dict2[key][-1]
                        nc_dict_joined[key][1:] = nc_dict[key][:]
                    elif append_to == "end":
                        nc_dict_joined[key][-1] = nc_dict2[key][0]
                        nc_dict_joined[key][:-1] = nc_dict[key][:]
                        
                    
                elif key == "altitude": #chop off negative altitude
                    nc_dict_joined[key] = nc_dict[key][1:]
    
                elif key == "latitude": #flip latitudes
                    nc_dict_joined[key] = nc_dict[key][::-1]
                    
                else:
                    nc_dict_joined[key] = value[:]
    
            if len(shape) == 4:
                shape[0] += 1
                shape[1] -= 1
                nc_dict_joined[key] = np.zeros(shape)
                if append_to == "start":
                    nc_dict_joined[key][0, :, :, :] = nc_dict2[key][-1, 1:, ::-1, :]
                    nc_dict_joined[key][1:, :, :, :] = nc_dict[key][:, 1:, ::-1, :]
                elif append_to == "end":
                    nc_dict_joined[key][-1, :, :, :] = nc_dict2[key][0, 1:, ::-1, :]
                    nc_dict_joined[key][:-1, :, :, :] = nc_dict[key][:, 1:, ::-1, :]
    
                nc_dict_joined[key][np.where(nc_dict_joined[key] > 9.9e19)] = np.nan
                
                
        nc_file.close()
        nc_file2.close()
        return nc_dict_joined
    
    else:
        nc_dict_single = {}
        for key, value in nc_dict.items():
            shape = list(value.shape)
            if len(shape) == 1:
                if key == "altitude": #chop off negative altitude
                    nc_dict_single[key] = nc_dict[key][1:]
            
                elif key == "latitude": #flip latitudes
                    nc_dict_single[key] = nc_dict[key][::-1]

                else:
                    nc_dict_single[key] = value[:]

            if len(shape) == 4:
                nc_dict_single[key] = nc_dict[key][:, 1:, ::-1, :]

        
        return nc_dict_single
    
    

def get_mcd_data(myear, ls, lat, lon, lst, plot=False):
    
    nc_dict = open_mcd_file(myear, ls, lst)

    p = interpolate_nearest_points("p", myear, ls, lat, lon, nc_dict)
    t = interpolate_nearest_points("t", myear, ls, lat, lon, nc_dict)
    
    atmos_dict = {"z":nc_dict["altitude"][...]/1000., "p":p, "t":t}
    atmos_dict["nd"] = interpolate_nearest_points("nd", myear, ls, lat, lon, nc_dict) * 1.0e6
    for molecule in ["co2", "co", "h2o", "o2", "o3"]:
        vmr = interpolate_nearest_points(molecule, myear, ls, lat, lon, nc_dict) / (7.2429e16 * (p / t)) * 1.0e6 #ppm
        atmos_dict[molecule] = vmr
        
    for ext in ["ext_dust", "ext_h2o_ice"]:
        atmos_dict[ext] = interpolate_nearest_points(ext, myear, ls, lat, lon, nc_dict)
    


    if plot:
        plt.figure(constrained_layout=True)
        plt.title(f"MY{myear}, Ls={ls}, ({lat}, {lon})")
        plt.grid()
        
        
        for header in atmos_dict.keys():
            if header != "z":
                plt.plot(atmos_dict[header]/np.max(atmos_dict[header]), atmos_dict["z"], label=header)
                # plt.plot(atmos_dict[header], atmos_dict["z"], label=header)
                
        plt.legend()
        # plt.xscale("log")
    
    return atmos_dict

# myear=35
# ls=185.0
# lst=6.0
# lat=0.0
# lon=0.0
# lst=12.0
# atmos_dict = get_mcd_data(myear, ls, lat, lon, lst, plot=True)




def get_mcd_data_from_h5(h5_f, reference_altitude=50.0, index=None, plot=False):
    """Get mcd data from the geometry of a hdf5 file.
    If index not specified:
    If occultation file, get geometry at the tangent point corrsponding to the reference altitude,
    else get geometry from minimum incidence angle
    """

    if not index:
        if "TangentAltAreoid" in h5_f["Geometry/Point0"].keys():
            #get index closest to reference altitude
            alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            index = np.abs(alts - reference_altitude).argmin()
        
        else:
            incid = h5_f["Geometry/Point0/IncidenceAngle"][:, 0]
            index = np.argmin(incid)
            
            print(incid[index])
            
        
    
    lon = h5_f["Geometry/Point0/Lon"][index, 0]
    lat = h5_f["Geometry/Point0/Lat"][index, 0]
    lst = h5_f["Geometry/Point0/LST"][index, 0]
    
    dt_str = h5_f["Geometry/ObservationDateTime"][index, 0].decode()
    
    dt = datetime.strptime(dt_str, "%Y %b %d %H:%M:%S.%f")
    
    myear, ls = get_mars_year_ls(dt)
    
    return get_mcd_data(myear, ls, lat, lon, lst, plot=plot)
    
    
    