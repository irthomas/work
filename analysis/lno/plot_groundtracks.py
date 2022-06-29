# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:47:41 2022

@author: iant

PLOT LNO GROUNDTRACKS

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from tools.file.paths import paths

from tools.sql.sql import sql_db
from tools.sql.obs_db_functions import get_match_data, get_files_match_data

from tools.file.hdf5_functions import open_hdf5_file



search_tuple = ("lno_d", {"diffraction_order":[167,170], "longitude":[0.0, 5.0], "latitude":[0.0, 5.0]})




def plot_nadir_tracks(search_tuple):
    """find and plot all nadir observation points within a bounding box"""
    
    db_path = os.path.join(paths["DB_DIRECTORY"], "obs_db.db")
    with sql_db(db_path) as db:
        #search database for parameters
        match_dict = get_match_data(db, search_tuple)
    
    
    fig0, ax0 = plt.subplots(figsize=(10, 8))
    
    search_dict = search_tuple[1]
    #draw rectangle on search area
    if "longitude" in search_dict.keys() and "latitude" in search_dict.keys():
        rectangle = np.asarray([
            [search_dict["longitude"][0], search_dict["latitude"][0]], \
            [search_dict["longitude"][1], search_dict["latitude"][0]], \
            [search_dict["longitude"][1], search_dict["latitude"][1]], \
            [search_dict["longitude"][0], search_dict["latitude"][1]], \
            [search_dict["longitude"][0], search_dict["latitude"][0]], \
        ])
        ax0.plot(rectangle[:, 0], rectangle[:, 1], "k")
            
            
        points = plt.scatter(match_dict["longitude"], match_dict["latitude"], c=match_dict["incidence_angle"])
        fig0.colorbar(points)
    
        ax0.set_xlabel("Longitude")
        ax0.set_ylabel("Latitude")
    
    




def plot_nadir_tracks2(search_tuple, plot_fig=True, save_fig=True):
    """find and plot all nadir observation points for any observations that cross a bounding box"""


    db_path = os.path.join(paths["DB_DIRECTORY"], "obs_db.db")
    with sql_db(db_path) as db:
        #search database for parameters
        match_dict = get_match_data(db, search_tuple)
    
    
    
    
    #get filenames matching search parameters
    hdf5_filenames = sorted([os.path.splitext(s)[0] for s in list(set(match_dict["filename"]))])

    search_dict = search_tuple[1]

    if plot_fig:
        fig0, ax0 = plt.subplots(figsize=(10, 8))
    
        #draw rectangle on search area
        if "longitude" in search_dict.keys() and "latitude" in search_dict.keys():
            rectangle = np.asarray([
                [search_dict["longitude"][0], search_dict["latitude"][0]], \
                [search_dict["longitude"][1], search_dict["latitude"][0]], \
                [search_dict["longitude"][1], search_dict["latitude"][1]], \
                [search_dict["longitude"][0], search_dict["latitude"][1]], \
                [search_dict["longitude"][0], search_dict["latitude"][0]], \
            ])
            ax0.plot(rectangle[:, 0], rectangle[:, 1], "k")
        
        
        
        for hdf5_filename in hdf5_filenames:
            
            try:
                hdf5_file = open_hdf5_file(hdf5_filename)
            except FileNotFoundError:
                print("Error: file %s not found" %hdf5_filename)
                continue
            
            lat = hdf5_file["Geometry/Point0/Lat"][:, 0]
            lon = hdf5_file["Geometry/Point0/Lon"][:, 1]
    
            # if len(search_query)>200:
            #     midpoint = int(len(search_query)/2)
            #     ax0.set_title(search_query[:midpoint]+"\n"+search_query[midpoint:])
            # else:
            #     ax0.set_title(search_query)
            ax0.scatter(lon, lat, label=hdf5_filename)
        ax0.set_xlabel("Longitude")
        ax0.set_ylabel("Latitude")
        ax0.set_xlim((-180, 180))
        ax0.set_ylim((-90, 90))
        ax0.legend()
        ax0.grid()
        
        if save_fig:
            fig0.savefig(os.path.join(paths["BASE_DIRECTORY"], "output", "groundtrack.png"))
    
    return hdf5_filenames




def plot_nadir_spectra(search_tuple):
    
    db_path = os.path.join(paths["DB_DIRECTORY"], "obs_db.db")
    with sql_db(db_path) as db:
        #search database for parameters
        match_dict = get_match_data(db, search_tuple)
    
    #get filenames matching search parameters
    # hdf5_filenames = sorted([os.path.splitext(s)[0] for s in list(set(match_dict["filename"]))])
    hdf5_filenames = sorted(list(set(match_dict["filename"])))

    fig0, ax0 = plt.subplots(figsize=(10, 8))


    #now for each filename, find indices of spectra within the bounding box
    for hdf5_filename in hdf5_filenames:
        match_indices = [i for i, s in enumerate(match_dict["filename"]) if s == hdf5_filename]
        frame_indices = [match_dict["frame_id"][i] for i in match_indices]
        print(hdf5_filename, frame_indices)


        try:
            hdf5_file = open_hdf5_file(os.path.splitext(hdf5_filename)[0])
        except FileNotFoundError:
            print("Error: file %s not found" %hdf5_filename)
            continue
        
        x = hdf5_file["Science/X"][:] #1d 
        y = hdf5_file["Science/YReflectanceFactor"][frame_indices, :]
        
        for spectrum in y:
            ax0.plot(x, spectrum, label=hdf5_filename)
        ax0.set_xlabel("Wavenumber cm-1")
        ax0.set_ylabel("Reflectance Factor")
    ax0.legend()
    ax0.grid()

    
