# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:50:48 2020

@author: iant
"""
import numpy as np
import h5py

from tools.sql.sql_table_fields import sql_table_fields
from tools.file.hdf5_functions import get_filepath



def make_obs_dict(channel, query_output, add_data=True):
    """make observation dictionary from sql query, add x and y data from files"""
    
    table_fields = sql_table_fields(channel)
    obsDict = {}
    #make empty dicts
    for fieldDict in table_fields:
        obsDict[fieldDict["name"]] = []
        
    for output_row in query_output:
        for i in range(len(output_row)):
            obsDict[table_fields[i]["name"]].append(output_row[i])

    obsDict["x"] = []
    obsDict["y"] = []
    obsDict["filepath"] = []
    
    hdf5_filenames = set(obsDict["filename"]) #unique matching filenames
    
    for hdf5_filename in hdf5_filenames:
#        with h5py.File(getFilePath(hdf5Filename)) as f:
        hdf5_filepath = get_filepath(hdf5_filename)
        with h5py.File(hdf5_filepath, "r") as f: #open file
            for filename, frameIndex in zip(obsDict["filename"], obsDict["frame_id"]):
                if filename == hdf5_filename:
                    x = f["Science/X"][frameIndex, :]
                    y = f["Science/Y"][frameIndex, :]
                    
                    integrationTimeRaw = f["Channel/IntegrationTime"][0]
                    numberOfAccumulationsRaw = f["Channel/NumberOfAccumulations"][0]
                    integrationTime = np.float(integrationTimeRaw) / 1.0e3 #microseconds to seconds
                    numberOfAccumulations = np.float(numberOfAccumulationsRaw)/2.0 #assume LNO nadir background subtraction is on
                    measurementPixels = 144.0
                    measurementSeconds = integrationTime * numberOfAccumulations
                    
                    y = y / measurementPixels / measurementSeconds
                    obsDict["x"].append(x)
                    obsDict["y"].append(y)
                    obsDict["filepath"].append(hdf5_filepath)
            print("measurementPixels=", measurementPixels, "; measurementSeconds=", measurementSeconds)

    return obsDict
