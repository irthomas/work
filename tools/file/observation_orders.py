# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:13:56 2020

@author: iant
"""
import os

from tools.file.paths import paths



def list_other_measured_orders(hdf5_filenames):
    """given a list of observations, find orders that were measured at the same time. 
    Note: also finds ingress / egress where datetime in filename is the same e.g. merged occultations"""
    
    d = {}
    for hdf5_filename in hdf5_filenames:
        
        #get info from filename
        hdf5_datetime = hdf5_filename[0:15]
        hdf5_file_level = "hdf5_level_" + hdf5_filename[16:20]
        year = hdf5_filename[0:4]
        month = hdf5_filename[4:6]
        day = hdf5_filename[6:8]
        channel = hdf5_filename.split("_")[3]
        
        #get folder path
        folder_path = os.path.join(paths["DATA_DIRECTORY"], hdf5_file_level, year, month, day)
        
        #list files in directory
        directory_list = [s for s in os.listdir(folder_path) if channel in s]
        
        #find matches with same datetime
        matches = [s.replace(".h5","") for s in directory_list if hdf5_datetime in s]
        
        #extract order number as a string
        orders_text = [s.split("_")[-1] for s in matches]
        
        #convert orders to integers
        orders = ([int(i) for i in orders_text])
        
        #save to dictionary
        d[hdf5_filename] = orders
    return d


# #test
# hdf5_filenames = ["20191014_175211_1p0a_SO_A_I_134"]
# orders_measured = list_other_measured_orders(hdf5_filenames)
# for k,v in orders_measured.items():
#     print(k, "orders measured at same time:", v)