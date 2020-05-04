# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:50:48 2020

@author: iant
"""
import numpy as np
import h5py

from tools.sql.sql_table_fields import sql_table_fields
from tools.file.hdf5_functions import get_filepath



def make_obs_dict(channel, query_output, filenames_only=True):
    """make observation dictionary from sql query, add x and y data from files"""
    obs_dict = {}
    
    table_fields = sql_table_fields(channel)
    #make empty dicts
    for field_dict in table_fields:
        obs_dict[field_dict["name"]] = []
        
    #add sql db data to dictionary
    for output_row in query_output:
        for i in range(len(output_row)):
            obs_dict[table_fields[i]["name"]].append(output_row[i])
        
    #convert to arrays
    for field_dict in table_fields:
        obs_dict[field_dict["name"]] = np.asarray(obs_dict[field_dict["name"]])

    #add data from hdf5 files to dictionary
    obs_dict["x"] = []
    obs_dict["y"] = []
    obs_dict["filepath"] = []
    obs_dict["file_index"] = []
    
    hdf5_filenames = sorted(list(set(obs_dict["filename"]))) #unique matching filenames
    if filenames_only:
        return {"filename":hdf5_filenames}
        
    for file_index, hdf5_filename in enumerate(hdf5_filenames):
#        with h5py.File(getFilePath(hdf5Filename)) as f:
        hdf5_filepath = get_filepath(hdf5_filename)
        with h5py.File(hdf5_filepath, "r") as f: #open file
            for filename, frame_index in zip(obs_dict["filename"], obs_dict["frame_id"]):
                if filename == hdf5_filename:
                    x = f["Science/X"][frame_index, :]
                    y = f["Science/Y"][frame_index, :]
                    
                    integrationTimeRaw = f["Channel/IntegrationTime"][0]
                    numberOfAccumulationsRaw = f["Channel/NumberOfAccumulations"][0]
                    integrationTime = np.float(integrationTimeRaw) / 1.0e3 #microseconds to seconds
                    numberOfAccumulations = np.float(numberOfAccumulationsRaw)/2.0 #assume LNO nadir background subtraction is on
                    measurementPixels = 144.0
                    measurementSeconds = integrationTime * numberOfAccumulations
                    
                    y = y / measurementPixels / measurementSeconds
                    obs_dict["x"].append(x)
                    obs_dict["y"].append(y)
                    obs_dict["filepath"].append(hdf5_filepath)
                    obs_dict["file_index"].append(file_index)
            print("measurementPixels=", measurementPixels, "; measurementSeconds=", measurementSeconds)

    obs_dict["file_index"] = np.asarray(obs_dict["file_index"])

    return obs_dict
