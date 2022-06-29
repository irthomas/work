# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:50:51 2022

@author: iant

MONITOR NADIRS AND OCCULTATIONS:
PRODUCE COVERAGE MATCH FOR EACH MONTH OF OPERATIONS

"""


import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

from tools.file.paths import paths

from tools.sql.sql import sql_db
from tools.sql.obs_db_functions import get_files_match_data

from tools.file.hdf5_functions import open_hdf5_file







def plot_files_tracks(search_tuple, save_fig=True):
    """find and plot all nadir observation points within a bounding box"""
    
    db_path = os.path.join(paths["DB_DIRECTORY"], "obs_db.db")
    with sql_db(db_path) as db:
        #search database for parameters
        match_dict = get_files_match_data(db, search_tuple)
    
    
    
    fig0, ax0 = plt.subplots(figsize=(10, 8))
    dt_str_start = datetime.strftime(search_tuple[1]["utc_start_time"][0], "%Y-%m-%d")
    dt_str_end = datetime.strftime(search_tuple[1]["utc_start_time"][1], "%Y-%m-%d")
    title = "Nadir and occultation tracks: %s to %s" %(dt_str_start, dt_str_end)
    ax0.set_title(title)
    
    # search_dict = search_tuple[1]

    #get filenames matching search parameters
    h5s = sorted([os.path.splitext(s)[0] for s in list(set(match_dict["filename"]))])
            
    # points = plt.scatter(match_dict["longitude"], match_dict["latitude"], c=match_dict["incidence_angle"])
    # fig0.colorbar(points)

    h5s_plotted = []
    for h5 in h5s:
        if h5[15] not in h5s_plotted:
            channel = h5.split("_")[3]
            
            try:
                h5_f = open_hdf5_file(h5)
            except FileNotFoundError:
                print("Error: file %s not found" %h5)
                continue
            
            lat = h5_f["Geometry/Point0/Lat"][:, 0]
            lon = h5_f["Geometry/Point0/Lon"][:, 1]
            
            #plot only when contiguous i.e. avoid longitude wrapping
            lon_diff = np.abs(np.diff(lon))
            wrap_ixs = np.where(lon_diff > 150)[0]
            
            colour = {"SO":"b", "LNO":"r"}[channel]
            if len(wrap_ixs) > 0:
                previous_wrap_ix = 0
                #loop through contiguous data
                for wrap_ix in wrap_ixs:
                    ax0.plot(lon[previous_wrap_ix:wrap_ix], lat[previous_wrap_ix:wrap_ix], color=colour, alpha=0.1)
                    previous_wrap_ix = wrap_ix + 1
                #plot end of file after final wrap
                ax0.plot(lon[previous_wrap_ix:], lat[previous_wrap_ix:], color=colour, alpha=0.1)
            else:
                ax0.plot(lon, lat, color=colour, alpha=0.1)
            
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_xlim((-180, 180))
    ax0.set_ylim((-90, 90))
    ax0.grid()
    
    if save_fig:
        fig0.savefig(title.replace(" ","_").replace(":","") + ".png")
    
    return h5s



def plot_3d_files_tracks(search_tuple):
    """find and plot all nadir observation points within a bounding box"""
    
    db_path = os.path.join(paths["DB_DIRECTORY"], "obs_db.db")
    with sql_db(db_path) as db:
        #search database for parameters
        match_dict = get_files_match_data(db, search_tuple)
    
    
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    
    # search_dict = search_tuple[1]

    #get filenames matching search parameters
    h5s = sorted([os.path.splitext(s)[0] for s in list(set(match_dict["filename"]))])
            
    # points = plt.scatter(match_dict["longitude"], match_dict["latitude"], c=match_dict["incidence_angle"])
    # fig0.colorbar(points)

    h5s_plotted = []
    for h5 in h5s:
        if h5[15] not in h5s_plotted:
            channel = h5.split("_")[3]
            
            try:
                h5_f = open_hdf5_file(h5)
            except FileNotFoundError:
                print("Error: file %s not found" %h5)
                continue
            
            lat = h5_f["Geometry/Point0/Lat"][:, 0]
            lon = h5_f["Geometry/Point0/Lon"][:, 0]
            if channel in "SO":
                alt = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            else:
                alt = 0.0

            R = 3389.5
            x = R * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
            y = R * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
            z = (R + alt) * np.sin(np.radians(lat))
            
            colour = {"SO":"b", "LNO":"r"}[channel]
            
            ax.plot(x, y, z, color=colour, alpha=0.5)
    ax.grid()

# search_tuple = ("files", {"utc_start_time":[datetime(2019, 9, 1), datetime(2020, 1, 1)]})
# plot_3d_files_tracks(search_tuple)
# stop()

# dt = datetime(2018, 3, 1)
# data = True
# while data:
#     dt_end = dt + relativedelta(months=3)
#     search_tuple = ("files", {"utc_start_time":[dt, dt_end]})
#     h5s = plot_files_tracks(search_tuple)
#     plt.close()

#     if len(h5s) == 0:
#         data = False
#     dt = dt_end

# for 