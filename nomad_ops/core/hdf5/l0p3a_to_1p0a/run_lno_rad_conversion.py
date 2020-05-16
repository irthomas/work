# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:56:52 2020

@author: iant

FIND LNO OBSERVATIONS MATCHING SEARCH CRITERIA AND CONVERT TO RADIANCE FACTOR

"""
import os
from datetime import datetime

from other.orbit.plot_lno_groundtracks import plot_lno_groundtracks
from tools.file.hdf5_functions import get_filepath
from nomad_ops.core.hdf5.l0p3a_to_1p0a.l0p3a_to_1p0a_v23 import convert

#orders with data: 119, 121, 124, 127, 131, 134, 136, 149, 153, 158, 161-169, 

#search_dict ={
#        119:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#        121:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#        124:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#        127:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#
#        131:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#        134:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#        136:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#
#        149:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#
#        153:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#        158:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#
#
#        161:{"n_orders":[0,7], "incidence_angle":[0,30], "temperature":[-30,15]},
#        162:{"n_orders":[0,7], "incidence_angle":[0,30], "temperature":[-30,15]},
#        163:{"n_orders":[0,7], "incidence_angle":[0,30], "temperature":[-30,15]},
#        164:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        165:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        166:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        167:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#        168:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#        169:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#
#        171:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},
#            
#        186:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        187:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        188:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        189:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-10,10], "longitude":[-10,10]},
#
#        190:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-10,10], "longitude":[-10,10]},
#        191:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-10,10], "longitude":[-10,10]},
#        
#        193:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        194:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        196:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#        197:{"n_orders":[0,7], "incidence_angle":[0,10], "temperature":[-30,15]},
#        198:{"n_orders":[0,7], "incidence_angle":[0,50], "temperature":[-30,15]},
#        199:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15]},
#}


#get all data
search_dict = {}
for order in range(115, 200, 1):
    search_dict[order] = {"n_orders":[0,7], "incidence_angle":[0,90], "temperature":[-30,15]}


file_level = "hdf5_level_0p3a"

#for diffraction_order in [198]:
for diffraction_order in search_dict.keys():

    hdf5_filenames = plot_lno_groundtracks(search_dict, file_level, diffraction_order, plot_fig=False)
    
    for hdf5_filename in hdf5_filenames:#[0:5]:
        hdf5file_path = get_filepath(hdf5_filename)
    
        hdf5file_paths_out = convert(hdf5file_path)
        for hdf5file_path_out in hdf5file_paths_out:
            print("%s: %s done" %(("%s" %datetime.now())[:-7], os.path.basename(hdf5file_path_out)))
    


