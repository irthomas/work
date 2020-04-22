# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:13:56 2020

@author: iant
"""
import os

from tools.file.paths import paths



def list_other_measured_orders(hdf5_filenames):
    """given a list of observations, find orders that were measured at the same time"""
    
    orders = []
    for hdf5_filename in hdf5_filenames:
        hdf5_datetime = hdf5_filename[0:15]
        hdf5_file_level = "hdf5_level_" + hdf5_filename[16:20]
        year = hdf5_filename[0:4]
        month = hdf5_filename[4:6]
        day = hdf5_filename[6:8]
        folder_path = os.path.join(paths["DATA_DIRECTORY"], hdf5_file_level, year, month, day)
        directory_list = os.listdir(folder_path)
        matches = [i.replace(".h5","") for i in directory_list if hdf5_datetime in i]
        orders_text = [i.split("_")[-1] for i in matches]
        orders.append([int(i) for i in orders_text])
    return orders


def simultaneous_orders(diffraction_orders):
    """find observations where the given orders are measured at the same time"""
    
    return 0

#hdf5_filenames = ["20191014_175211_1p0a_SO_A_I_134"]
#orders_measured = list_other_measured_orders(hdf5_filenames)