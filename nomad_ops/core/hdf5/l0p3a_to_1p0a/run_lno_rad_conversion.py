# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:56:52 2020

@author: iant

FIND LNO OBSERVATIONS MATCHING SEARCH CRITERIA AND CONVERT TO RADIANCE FACTOR

"""
import os

from other.orbit.plot_lno_groundtracks import plot_lno_groundtracks
from tools.file.hdf5_functions import get_filepath
from nomad_ops.core.hdf5.l0p3a_to_1p0a.l0p3a_to_1p0a_v23 import convert

search_dict ={
        119:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
        134:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},

        168:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,-2], "latitude":[-15,5], "longitude":[127,147]},
            
        188:{"n_orders":[0,4]},
        189:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[1,2], "latitude":[-10,10], "longitude":[-10,10]},
        
        193:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,5]},#, "latitude":[-90,90], "longitude":[-180,180]},
}

for i in range(115, 200):
    if i not in search_dict.keys():
        search_dict[i] = {"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]}
#        search_dict[i] = {"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]}


file_level = "hdf5_level_0p3a"
diffraction_order = 119

hdf5_filenames = plot_lno_groundtracks(search_dict, file_level, diffraction_order)

for hdf5_filename in hdf5_filenames[0:1]:
    hdf5file_path = get_filepath(hdf5_filename)

    hdf5file_paths_out = convert(hdf5file_path)
    for hdf5file_path_out in hdf5file_paths_out:
        print("%s converted" %os.path.basename(hdf5file_path_out))
    





#    #min_lat, max_lat, min_lon, max_lon, max_incidence_angle, min_temperature, max_temperature, max_orders
#    obsSearchDict = {
#            118:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            120:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            126:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            130:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            133:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            142:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            151:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            156:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#
#
#            162:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
#            163:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
#            164:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            166:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            167:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            168:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            169:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#
#            173:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            174:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            178:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            179:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            180:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            182:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            184:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            189:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            194:[-90, 90, 127, 147, 90, -30, 30, 4], #good
#            195:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            196:[-90, 90, 127, 147, 90, -30, 30, 4], #could be improved
#            }
#
#