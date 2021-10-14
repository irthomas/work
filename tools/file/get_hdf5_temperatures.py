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


from datetime import datetime, timedelta
from scipy import interpolate
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np

def get_interpolated_temperatures(hdf5_file, channel, plot=False, sensor="", t_filter="median", precooling=False):
    """give an open hdf5 file and return a temperature for each spectrum in the file, interpolated from TGO readouts"""
    
    #get observation times from file
    if precooling:
        utc_start_datetime = datetime.strptime(hdf5_file["Housekeeping/DateTime"][0].decode(), "%Y %b %d %H:%M:%S.%f")
        utc_datetimes = [utc_start_datetime + timedelta(seconds=i) for i in range(600)]
        
    else:        
        utc_datetime_strings = [i.decode() for i in hdf5_file["Geometry/ObservationDateTime"][:, 0]]
        utc_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in utc_datetime_strings]

    if sensor=="":
        sensor = "Temperature/Nominal%s" %(channel.upper())
        sensor_dt = "Temperature/TemperatureDateTime"
    elif "Temperature/" in sensor: 
        sensor_dt = "Temperature/TemperatureDateTime"
    elif "Housekeeping/" in sensor: 
        sensor_dt = "Housekeeping/DateTime"
    
    
    #get temperature times from file
    utc_t_times = [i.decode() for i in hdf5_file[sensor_dt][...]]
    utc_t_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in utc_t_times]
    
    
    
    #convert to seconds
    utc_start_datetime = utc_datetimes[0]
    frame_delta_seconds = np.array([(i - utc_start_datetime).total_seconds() for i in utc_datetimes])
    temperature_delta_seconds = np.array([(i - utc_start_datetime).total_seconds() for i in utc_t_datetimes])

    # print(frame_delta_seconds)
    # print(temperature_delta_seconds)
    
    #get channel dataset
    channel_temperatures = hdf5_file[sensor][...]
    
    if t_filter=="median":
        temperatures = medfilt(channel_temperatures, 5)
    if t_filter=="quadratic":
        x = np.arange(len(channel_temperatures))
        temperatures = np.polyval(np.polyfit(x, channel_temperatures, 2), x)
    if t_filter=="m+q":
        ts = medfilt(channel_temperatures, 5)
        x = np.arange(len(ts))
        temperatures = np.polyval(np.polyfit(x, ts, 2), x)
    
    #datetime elements can be duplicated - remove duplicates
    vals, idx = np.unique(temperature_delta_seconds, return_index=True)

    #interpolate onto observation times
    f = interpolate.interp1d(temperature_delta_seconds[idx], temperatures[idx], kind="quadratic", fill_value=(np.nan, np.nan), bounds_error=False)
    
    frame_temperatures = f(frame_delta_seconds)

    if plot:
        plt.figure()
        plt.plot(temperature_delta_seconds[idx], channel_temperatures[idx], label="Raw values")
        plt.plot(temperature_delta_seconds[idx], temperatures[idx], label="%s filtered" %t_filter)
        if t_filter=="m+q":
            plt.plot(temperature_delta_seconds[idx], ts[idx], label="Median filtered")
        plt.plot(frame_delta_seconds, frame_temperatures, label="%s filtered & interpolated" %t_filter)
        plt.legend()
        plt.title("TGO Temperature Readouts")
        plt.xlabel("Measurement Number")
        plt.ylabel("Temperature")


    return frame_temperatures


