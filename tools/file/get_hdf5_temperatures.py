# -*- coding: utf-8 -*-
"""
Created on Wed Aug 4 14:54:31 2021

@author: iant

# example usage

import h5py
hdf5_file = h5py.File(r"<path_to_file>.h5", "r")
so = get_interpolated_temperatures(hdf5_file, "so")
lno = get_interpolated_temperatures(hdf5_file, "lno")
uvis = get_interpolated_temperatures(hdf5_file, "uvis")

"""


from datetime import datetime
from scipy import interpolate
from scipy.signal import medfilt
# import matplotlib.pyplot as plt

def get_interpolated_temperatures(hdf5_file, channel):
    """give an open hdf5 file and return a temperature for each spectrum in the file, interpolated from TGO readouts"""
    
    #get observation times from file
    utc_times = [i.decode() for i in hdf5_file["Geometry/ObservationDateTime"][:, 0]]
    utc_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in utc_times]
    
    
    #get temperature times from file
    utc_t_times = [i.decode() for i in hdf5_file["Temperature/TemperatureDateTime"][...]]
    utc_t_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in utc_t_times]
    
    
    
    #convert to seconds
    utc_start_datetime = utc_datetimes[0]
    frame_delta_seconds = [(i-utc_start_datetime).total_seconds() for i in utc_datetimes]
    temperature_delta_seconds = [(i-utc_start_datetime).total_seconds() for i in utc_t_datetimes]
    
    #get channel dataset
    channel_temperatures = hdf5_file["Temperature/Nominal%s" %channel.upper()][...]
    temperatures_median = medfilt(channel_temperatures, 5)
    # plt.figure()
    # plt.plot(channel_temperatures)
    # plt.plot(temperatures_median)

    
    #interpolate onto observation times
    f = interpolate.interp1d(temperature_delta_seconds, temperatures_median, kind="quadratic")
    
    frame_temperatures = f(frame_delta_seconds)
    return frame_temperatures


