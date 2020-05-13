# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:56:52 2020

@author: iant

FIND LNO OBSERVATIONS MATCHING SEARCH CRITERIA AND CONVERT TO RADIANCE FACTOR

"""
from other.orbit.plot_lno_groundtracks import plot_lno_groundtracks
#from instrument.calibration.lno_radiance_factor.convert_lno_rad_fac import convert_lno_rad_fac
from tools.file.hdf5_functions import get_filepath
from instrument.calibration.lno_radiance_factor.l0p3a_to_1p0a import convert

search_dict ={
        134:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},

        168:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,-2], "latitude":[-15,5], "longitude":[127,147]},
            
        188:{"n_orders":[0,4]},
        189:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[1,2], "latitude":[-10,10], "longitude":[-10,10]},
        
        193:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,5]},#, "latitude":[-90,90], "longitude":[-180,180]},
}



file_level = "hdf5_level_0p3a"
diffraction_order = 134

hdf5_filenames = plot_lno_groundtracks(search_dict, file_level, diffraction_order)

for hdf5_filename in hdf5_filenames[0:1]:
    hdf5file_path = get_filepath(hdf5_filename)

    convert(hdf5file_path)
#    convert_lno_rad_fac(hdf5_filename)
