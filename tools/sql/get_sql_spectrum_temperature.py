# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:27:12 2020

@author: iant
"""
import datetime

from tools.general.get_nearest_datetime import get_nearest_datetime
from tools.sql.heaters_temp import get_temperature_range


def get_sql_spectrum_temperature(hdf5_file, frame_index, channel="lno"):
    """get LNO nominal temperature readout closest to chosen frame in hdf5_file"""
    
    
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0].decode()
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, 0].decode()
    utc_start_datetime = datetime.datetime.strptime(utc_start_time, "%Y %b %d %H:%M:%S.%f")
    utc_end_datetime = datetime.datetime.strptime(utc_end_time, "%Y %b %d %H:%M:%S.%f")
    temperatures = get_temperature_range(utc_start_datetime, utc_end_datetime)

    utc_obs_time = hdf5_file["Geometry/ObservationDateTime"][frame_index, 0].decode()
    utc_obs_datetime = datetime.datetime.strptime(utc_obs_time, "%Y %b %d %H:%M:%S.%f")

    obs_temperature_index = get_nearest_datetime([i[0] for i in temperatures], utc_obs_datetime)
    #get LNO nominal temperature
    index = {"so":1, "lno":2, "uvis":3}[channel]
    
    measurement_temperature = float(temperatures[obs_temperature_index][index])

    return measurement_temperature

