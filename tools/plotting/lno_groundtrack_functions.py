# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:45:56 2020

@author: iant
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from tools.sql.obs_database import obs_database 
from tools.sql.make_obs_dict import make_obs_dict
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths

from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import FIG_X, FIG_Y




def make_query_from_search_dict(search_dict, table_name, diffraction_order):
    """build obs database sql query from items in search dictionary"""

    search_query = "SELECT * from %s WHERE diffraction_order == %i " %(table_name, diffraction_order)
    
    for key, value in search_dict[diffraction_order].items():
        search_query += "AND %s > %i AND %s < %i " %(key, value[0], key, value[1])
    
    return search_query



def plot_lno_groundtracks(search_dict, file_level, diffraction_order, plot_fig=True, save_fig=True):
    """search obs database for lno nadirs matching the search parameters
    plot groundtracks and output list of hdf5 filenames satisfying criteria"""

    #search database for parameters
    search_query = make_query_from_search_dict(search_dict, file_level, diffraction_order)
    
    database_name = "lno_nadir_%s" %file_level
    db_obj = obs_database(database_name, silent=True)
    query_output = db_obj.query(search_query)
    db_obj.close()
    
    #test new calibrations on a nadir observation
    obs_data_dict = make_obs_dict("lno", query_output, filenames_only=True)

    #get filenames matching search parameters
    hdf5_filenames = obs_data_dict["filename"]
    
    if plot_fig:
        fig0, ax0 = plt.subplots(figsize=(FIG_X-4, FIG_Y-3))
    
        #draw rectangle on search area
        if "longitude" in search_dict[diffraction_order].keys() and "latitude" in search_dict[diffraction_order].keys():
            rectangle = np.asarray([
                [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][0]], \
                [search_dict[diffraction_order]["longitude"][1], search_dict[diffraction_order]["latitude"][0]], \
                [search_dict[diffraction_order]["longitude"][1], search_dict[diffraction_order]["latitude"][1]], \
                [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][1]], \
                [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][0]], \
            ])
            ax0.plot(rectangle[:, 0], rectangle[:, 1], "k")
        
        
        
        for hdf5_filename in hdf5_filenames:
            
            hdf5_file = open_hdf5_file(hdf5_filename)
            
            lat = hdf5_file["Geometry/Point0/Lat"][:, 0]
            lon = hdf5_file["Geometry/Point0/Lon"][:, 1]
    
            if len(search_query)>200:
                midpoint = int(len(search_query)/2)
                ax0.set_title(search_query[:midpoint]+"\n"+search_query[midpoint:])
            else:
                ax0.set_title(search_query)
            ax0.scatter(lon, lat, label=hdf5_filename)
            ax0.set_xlabel("Longitude")
            ax0.set_ylabel("Latitude")
        ax0.set_xlim((-180, 180))
        ax0.set_ylim((-90, 90))
        ax0.legend()
        ax0.grid()
        
        if save_fig:
            fig0.savefig(os.path.join(paths["BASE_DIRECTORY"], "output", "groundtrack_order_%i.png" %diffraction_order))
    
    return hdf5_filenames




def get_hdf5_filename_list(search_dict, file_level, diffraction_order):
    """search obs database for lno nadirs matching the search parameters
    output list of hdf5 filenames satisfying criteria"""

    #search database for parameters
    search_query = make_query_from_search_dict(search_dict, file_level, diffraction_order)
    
    database_name = "lno_nadir_%s" %file_level
    db_obj = obs_database(database_name)
    query_output = db_obj.query(search_query)
    db_obj.close()
    
    #test new calibrations on a nadir observation
    obs_data_dict = make_obs_dict("lno", query_output, filenames_only=True)

    #get filenames matching search parameters
    hdf5_filenames = obs_data_dict["filename"]
    
    return hdf5_filenames


