# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:46:01 2021

@author: iant

GET TGO TEMPERATURES FROM SQL
"""

import numpy as np
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.signal import medfilt
import matplotlib.pyplot as plt



# from nomad_ops.core.pipeline.extras.heaters_temp import get_temperature_range
from tools.sql.heaters_temp import get_temperature_range


FORMAT_STR_SECONDS = "%Y %b %d %H:%M:%S.%f"
NA_VALUE = -999.


#get TGO temperature readouts for this observation
def get_tgo_readouts(beg_datetimestring, end_datetimestring, delta_minutes=11.0):
    beg_dt = datetime.strptime(beg_datetimestring.decode(), FORMAT_STR_SECONDS) - timedelta(minutes=delta_minutes)
    end_dt = datetime.strptime(end_datetimestring.decode(), FORMAT_STR_SECONDS) + timedelta(minutes=delta_minutes)
    temperature_db_data = get_temperature_range(beg_dt, end_dt)
    
    datetimestring = []
    so_nom = []
    lno_nom = []
    so_red = []
    lno_red = []
    uvis_nom = []
    for temperature_db_row in temperature_db_data:
        datetimestring.append(datetime.strftime(temperature_db_row[0], FORMAT_STR_SECONDS).encode())
        so_nom.append(temperature_db_row[1])
        lno_nom.append(temperature_db_row[2])
        so_red.append(temperature_db_row[3])
        lno_red.append(temperature_db_row[4])
        uvis_nom.append(temperature_db_row[5])
    return {"TemperatureDateTime":datetimestring, 
            "NominalSO":np.asfarray(so_nom), 
            "NominalLNO":np.asfarray(lno_nom), 
            "RedundantSO":np.asfarray(so_red), 
            "RedundantLNO":np.asfarray(lno_red),
            "NominalUVIS":np.asfarray(uvis_nom)}







# def get_interpolated_temperatures(frame_dt_strings, tgo_dt_strings, tgo_sensor_temps, plot=False, t_filter="m+q"):
def get_interpolated_temperatures(frame_dt_strings, tgo_dt_strings, tgo_sensor_temps, plot=True, t_filter="m+q"):
    """give an open hdf5 file and return a temperature for each spectrum in the file, interpolated from TGO readouts"""
    
    #convert observation times
    utc_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in frame_dt_strings]
    
    
    #convert temperature times
    utc_t_datetimes = [datetime.strptime(i, "%Y %b %d %H:%M:%S.%f") for i in tgo_dt_strings]
    
    
    
    #convert to seconds
    utc_start_datetime = utc_datetimes[0]
    frame_delta_seconds = np.array([(i - utc_start_datetime).total_seconds() for i in utc_datetimes])
    temperature_delta_seconds = np.array([(i - utc_start_datetime).total_seconds() for i in utc_t_datetimes])
    
    print(frame_delta_seconds)
    print(temperature_delta_seconds)
    
    #apply filter(s)
    if t_filter=="median":
        temperatures = medfilt(tgo_sensor_temps, 5)
    if t_filter=="quadratic":
        x = np.arange(len(tgo_sensor_temps))
        temperatures = np.polyval(np.polyfit(x, tgo_sensor_temps, 2), x)
    if t_filter=="m+q":
        ts = medfilt(tgo_sensor_temps, 5)
        x = np.arange(len(ts))
        temperatures = np.polyval(np.polyfit(x, ts, 2), x)
        
    
    #doesn't work well - need to check delta between values and remove those <1 second
    #datetime elements can be duplicated - remove duplicates
    vals, idx = np.unique(np.round(temperature_delta_seconds), return_index=True)
    
    
    #interpolate onto observation times
    f = interpolate.interp1d(temperature_delta_seconds[idx], temperatures[idx], kind="quadratic")
    
    frame_temperatures = f(frame_delta_seconds)

    print("tgo_sensor_temps")
    print(tgo_sensor_temps)
    print("temperatures")
    print(temperatures)
    print("frame_temperatures")
    print(frame_temperatures)
    print("temperature_delta_seconds")
    print(temperature_delta_seconds)
    print("frame_delta_seconds")
    print(frame_delta_seconds)

    if plot:
        plt.figure()
        plt.plot(temperature_delta_seconds[idx], tgo_sensor_temps[idx], label="Raw values")
        plt.plot(temperature_delta_seconds[idx], temperatures[idx], label="%s filtered" %t_filter)
        if t_filter=="m+q":
            plt.plot(temperature_delta_seconds[idx], ts[idx], label="Median filtered")
        plt.plot(frame_delta_seconds, frame_temperatures, label="%s filtered & interpolated" %t_filter)
        plt.legend()
        plt.title("TGO Temperature Readouts")
        plt.xlabel("Measurement Number")
        plt.ylabel("Temperature")
        # plt.savefig("/bira-iasb/projects/work/NOMAD/test/iant/.tmp/temperature.png")


    return frame_temperatures



def collect_temperatures(hdf5FileIn, channel, hdf5_basename):
    
    """get temperatures from SQL, make frame-interpolated nominal temperature array"""
    
    ydimensions = hdf5FileIn["Science/Y"].shape
    nSpectra = ydimensions[0]
    
    
    #read in observation start and end times
    if "Geometry/ObservationDateTime" in hdf5FileIn.keys():
        observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]
        observationEndTime = hdf5FileIn["Geometry/ObservationDateTime"][-1,-1]
    else:
        observationStartTime = hdf5FileIn["DateTime"][0]
        observationEndTime = hdf5FileIn["DateTime"][-1]
    
    error = False
    temperatureDictionary = get_tgo_readouts(observationStartTime, observationEndTime)
    temperatureData = temperatureDictionary["Nominal%s" %channel.upper()]
    if len(temperatureData)>0:
        output_msg = ""
        
        #get frame interpolated temperatures
        tgo_sensor_temps = np.asfarray(temperatureData)
        tgo_dt_strings = [i.decode() for i in temperatureDictionary["TemperatureDateTime"]]
        frame_dt_strings = [i.decode() for i in hdf5FileIn["Geometry/ObservationDateTime"][:, 0]]
        frame_temperatures = get_interpolated_temperatures(frame_dt_strings, tgo_dt_strings, tgo_sensor_temps)
    
        #mean temperature (single value for all spectra)
        measurementTemperature = np.mean(frame_temperatures)
    
        
    else:
        output_msg = "No TGO temperatures available for %s. Skipping file generation" %hdf5_basename
        measurementTemperature = NA_VALUE
        frame_temperatures = np.tile(measurementTemperature,(nSpectra,1))
        error = True
        
    MeasurementTemperature = np.tile(measurementTemperature,(nSpectra,1))
    
    temperature_data = [temperatureDictionary, frame_temperatures, MeasurementTemperature, error, output_msg]
        
    return temperature_data



    
def add_temperatures_to_hdf5(hdf5FileOut, temperature_data):

    temperatureDictionary, frame_temperatures, MeasurementTemperature, error, output_msg = temperature_data

    if not error:
        #add tgo temperatures to file
        for key, values in temperatureDictionary.items():
            if "DateTime" not in key:
                if len(values) > 0:
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype=np.float32,
                                               data=values, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                else: #if no data for given time period
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype=np.float32, data=-999.0)

            else:
                if len(values) > 0:
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype="S27",
                                               data=values, compression="gzip", shuffle=True)
                else: #if no data for given time period
                    hdf5FileOut.create_dataset("Temperature/%s" %key, dtype="S27", data=["-999".encode()])

        #add interpolated temperatures if no error
        hdf5FileOut.create_dataset("Channel/InterpolatedTemperature", dtype=np.float,
                        data=frame_temperatures, fillvalue=NA_VALUE, compression="gzip", shuffle=True)

    
    
    hdf5FileOut.create_dataset("Channel/MeasurementTemperature", dtype=np.float,
                            data=MeasurementTemperature, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
 
    
 
    return output_msg


import h5py
hdf5_basename = "20190820_093004_0p3a_SO_1_E_140"
channel = "so"
hdf5FileIn = h5py.File("%s.h5" %hdf5_basename, "r")

temperatureDictionary, frame_temperatures, MeasurementTemperature, error, output_msg = collect_temperatures(hdf5FileIn, channel, hdf5_basename)