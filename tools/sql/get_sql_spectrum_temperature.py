# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:27:12 2020

@author: iant


GET TEMPERATURES FROM SQL DATABASE. NOTE THIS HAS LARGELY BEEN SUPERSEDED BY THE DATA IN THE HDF5 FILES
"""
from datetime import datetime, timedelta
from scipy import interpolate

from tools.general.get_nearest_datetime import get_nearest_datetime
from tools.sql.heaters_temp import get_temperature_range #taken from data pipeline


def get_sql_spectrum_temperature(hdf5_file, frame_index, channel="lno"):
    """get nominal temperature readout closest to chosen frame in hdf5_file"""
    
    
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0].decode()
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, 0].decode()
    utc_start_datetime = datetime.strptime(utc_start_time, "%Y %b %d %H:%M:%S.%f")
    utc_end_datetime = datetime.strptime(utc_end_time, "%Y %b %d %H:%M:%S.%f")
    temperatures = get_temperature_range(utc_start_datetime, utc_end_datetime)

    utc_obs_time = hdf5_file["Geometry/ObservationDateTime"][frame_index, 0].decode()
    utc_obs_datetime = datetime.strptime(utc_obs_time, "%Y %b %d %H:%M:%S.%f")

    #get index
    obs_temperature_index = get_nearest_datetime([i[0] for i in temperatures], utc_obs_datetime)
    #get column number for channel
    index = {"so":1, "lno":2, "uvis":3}[channel]
    
    measurement_temperature = float(temperatures[obs_temperature_index][index])

    return measurement_temperature



def get_sql_temperatures_all_spectra(hdf5_file, channel):
    """read in a hdf5 file and return a temperature for each spectrum in the file, interpolated from TGO readouts"""
    
    #get observation times from file
    utc_times = [i.decode() for i in hdf5_file["Geometry/ObservationDateTime"][:, 0]]
    utc_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in utc_times]
    utc_start_datetime = utc_datetimes[0]
    utc_end_datetime = utc_datetimes[-1]
    
    #convert to seconds
    frame_delta_seconds = [(i-utc_start_datetime).total_seconds() for i in utc_datetimes]
    
    #get data from SQL for given range + 3 minutes
    temperature_data = get_temperature_range(utc_start_datetime - timedelta(minutes=3), utc_end_datetime + timedelta(minutes=3))
    temperature_delta_seconds = [(i[0]-utc_start_datetime).total_seconds() for i in temperature_data]
    
    #choose channel
    index = {"so":1, "lno":2, "uvis":3}[channel]
    channel_temperatures = [float(i[index]) for i in temperature_data]
    
    #interpolate onto observation times
    f = interpolate.interp1d(temperature_delta_seconds, channel_temperatures, kind="quadratic")
    
    frame_temperatures = f(frame_delta_seconds)
    return frame_temperatures