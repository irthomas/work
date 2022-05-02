# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:49:12 2022

@author: iant

NEW OBSERVATION DATABASE
"""

import numpy as np
import os
import datetime
import decimal
import re
import h5py
import sqlite3 as sql

import platform


# from tools.file.paths import paths
# from tools.file.passwords import passwords
# from tools.sql.sql_table_fields import obs_database_fields
# from tools.sql.read_cache_db import get_filenames_from_cache

# from tools.file.hdf5_functions import make_filelist, get_filepath

# from tools.spectra.baseline_als import baseline_als



from tools.sql.sql import sql_db



SPICE_DATETIME_FORMAT = "%Y %b %d %H:%M:%S.%f"
SQL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"
ARG_FORMAT = "%Y-%m-%d"

db_path = "test.db"

if platform.system() == "Windows":

    cache_db_path = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared\db\hdf5_level_1p0a.db"
else:
    cache_db_path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_1p0a/cache.db"



def get_obs_duration(hdf5_file):
    
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0]
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, -1]
    utc_start_datetime = datetime.datetime.strptime(utc_start_time.decode(), SPICE_DATETIME_FORMAT)
    utc_end_datetime = datetime.datetime.strptime(utc_end_time.decode(), SPICE_DATETIME_FORMAT)
    total_seconds = (utc_end_datetime - utc_start_datetime).total_seconds()
    
    return total_seconds


file_dict = {
    "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
    "orbit":["INTEGER NOT NULL", []],
    "filename":["TEXT NOT NULL", []],
    "utc_start_time":["TIMESTAMP NOT NULL", []],
    "duration":["FLOAT NOT NULL", []],
    "n_spectra":["INTEGER NOT NULL", []],
    "n_orders":["INTEGER NOT NULL", []],
    "bg_subtraction":["INTEGER NOT NULL", []],
}   


lno_obs_dict = {
    "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
    "frame_id":["INTEGER NOT NULL", []],
    "temperature":["FLOAT NOT NULL", []],
    "diffration_order":["INTEGER NOT NULL", []], #order is a protected keyword!
    "longitude":["FLOAT NOT NULL", []],
    "latitude":["FLOAT NOT NULL", []],
    "incidence_angle":["FLOAT NOT NULL", []],
    "local_time":["FLOAT NOT NULL", []],
    "file_id":["INTEGER NOT NULL", []],
}

so_obs_dict = {
    "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
    "frame_id":["INTEGER NOT NULL", []],
    "temperature":["FLOAT NOT NULL", []],
    "diffration_order":["INTEGER NOT NULL", []],
    "bin_index":["INTEGER NOT NULL", []],
    "altitude":["FLOAT NOT NULL", []],
    "longitude":["FLOAT NOT NULL", []],
    "latitude":["FLOAT NOT NULL", []],
    "local_time":["FLOAT NOT NULL", []],
    "file_id":["INTEGER NOT NULL", []],
}



def clear_db():
    with sql_db(db_path) as db:
        db.populate_db("files", file_dict, clear=True)
        db.populate_db("so_ie", so_obs_dict, clear=True)
        db.populate_db("lno_d", lno_obs_dict, clear=True)
        # a = db.get_all_rows(table_name)



def files_for_period(cur, beg_dtime=None, end_dtime=None):

    sql_tests = []
    sql_params = {}
    if beg_dtime:
        sql_tests.append("(end_dtime >= :beg_dtime)")
        sql_params["beg_dtime"] = beg_dtime
    if end_dtime:
        sql_tests.append("(beg_dtime <= :end_dtime)")
        sql_params["end_dtime"] = end_dtime

    sql_comm = "select * from files"
    if sql_tests:
        sql_comm += " where " + " and ".join(sql_tests)

    cur.execute(sql_comm, sql_params)
    file_info_list = []
    for rec in cur.fetchall():
        file_info_list.append(rec)
    return file_info_list



def get_data_from_cache(cache_db_path, beg_dtime=None, end_dtime=None):
    """get filenames from cache.db"""
        
    con = sql.connect(cache_db_path)
    print("Getting data from %s" %os.path.basename(cache_db_path))
    
    cur = con.cursor()
    
    
    rows = files_for_period(cur, beg_dtime=beg_dtime, end_dtime=end_dtime)
    con.close()
    
    
    return rows





# filepaths = [filepath[0] for filepath in rows]
# filenames = [os.path.split(filepath)[1] for filepath in filepaths]






# def fill_db():
    
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('observation_type', type=str, help='Enter command: so_occultation, lno_nadir')
    parser.add_argument('--beg', type=str, default=None, help='Enter start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='Enter end date YYYY-MM-DD')
    parser.add_argument('-regenerate', action='store_true', help='Delete table and regenerate. Always use on first run')
    parser.add_argument('-silent', action='store_true', help='Output messages')
    parser.add_argument('--regex', type=str, default="", help='Match regex')
    args = parser.parse_args()
    observation_type = args.observation_type
else:
    observation_type = ""





"""make database containing info about all spectra in a channel for a particular observation type"""






# filenames = get_filenames_from_cache(cache_db_path)
# #make datetime from hdf5 filenames, find those that match the beg/end times
# hdf5_datetimes = [datetime.datetime.strptime(i[:15], HDF5_FILENAME_FORMAT) for i in hdf5_filenames]
# matching_hdf5_filenames = [hdf5_filename for hdf5_datetime, hdf5_filename in zip(hdf5_datetimes, hdf5_filenames) if beg_datetime < hdf5_datetime < end_datetime]






if args.silent:
    silent = True
else:
    silent = False
    
if args.regenerate:
    print("Deleting table")
    clear_db()
    
    print("Regenerating table")
    
    
print("Getting file list")
observation_type = args.observation_type

if args.beg:
    beg_dt = datetime.datetime.strptime(args.beg, ARG_FORMAT)
if args.end:
    end_dt = datetime.datetime.strptime(args.end, ARG_FORMAT)


rows = get_data_from_cache(cache_db_path, beg_dtime=beg_dt, end_dtime=end_dt)
hdf5_filepaths = [row[0] for row in rows]
utc_start_times = [row[2] for row in rows]

print("%i files found in directory" %len(hdf5_filepaths))

for file_index, (utc_start_time, hdf5_filepath) in enumerate(zip(utc_start_times, hdf5_filepaths)):
    
    hdf5_filename = os.path.basename(hdf5_filepath)
    
    if not silent:
        print("Collecting data: file %i/%i: %s" %(file_index, len(hdf5_filepaths), hdf5_filename))

    
    file_dict["filename"][1].append(hdf5_filename)
    file_dict["utc_start_time"][1].append(utc_start_time)
    
    if not os.path.exists(hdf5_filepath):
        print("Error: %s does not exist" %hdf5_filename)
        
    elif "UVIS" in hdf5_filename:
        continue
    
    else:
        with h5py.File(hdf5_filepath, "r") as hdf5_file:
            
            n_spectra = len(hdf5_file["Geometry/ObservationDateTime"][:, 0])
            
            file_dict["orbit"][1].append(hdf5_file.attrs["Orbit"])
            file_dict["duration"][1].append(get_obs_duration(hdf5_file))
            file_dict["n_spectra"][1].append(n_spectra)
            file_dict["n_orders"][1].append(hdf5_file.attrs["NSubdomains"])
            file_dict["bg_subtraction"][1].append(hdf5_file["Channel/BackgroundSubtraction"][0])
            
            file_id = file_index #update this
            
            if "InterpolatedTemperature" in hdf5_file["Channel"].keys():
                temperatures = hdf5_file["Channel/InterpolatedTemperature"][...]
            else:
                temperatures = hdf5_file["Channel/MeasurementTemperature"][...]
                
            diffraction_orders = hdf5_file["Channel/DiffractionOrder"][...]
            bg_subtractions = hdf5_file["Channel/BackgroundSubtraction"][0]
            longitudes = hdf5_file["Geometry/Point0/Lon"][:, 0]
            latitudes = hdf5_file["Geometry/Point0/Lat"][:, 0]
            local_times = hdf5_file["Geometry/Point0/LST"][:, 0]
    
            if "SO" in hdf5_filename:
                
                bin_index = hdf5_file["Channel/IndBin"][...]
                altitudes = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
    
                for i in range(n_spectra):
                
                    so_obs_dict["file_id"][1].append(file_id)
                    so_obs_dict["frame_id"][1].append(i)
                    so_obs_dict["temperature"][1].append(1)
                    so_obs_dict["diffration_order"][1].append(diffraction_orders[i])
                    so_obs_dict["bin_index"][1].append(bin_index[i])
                    so_obs_dict["altitude"][1].append(altitudes[i])
                    so_obs_dict["longitude"][1].append(longitudes[i])
                    so_obs_dict["latitude"][1].append(latitudes[i])
                    so_obs_dict["local_time"][1].append(local_times[i])
                    
            elif "LNO" in hdf5_filename:
    
                incidence_angles = hdf5_file["Geometry/Point0/IncidenceAngle"][:, 0]
                
                for i in range(n_spectra):
                
                    lno_obs_dict["file_id"][1].append(file_id)
                    lno_obs_dict["frame_id"][1].append(i)
                    lno_obs_dict["temperature"][1].append(1)
                    lno_obs_dict["diffration_order"][1].append(diffraction_orders[i])
                    lno_obs_dict["longitude"][1].append(longitudes[i])
                    lno_obs_dict["latitude"][1].append(latitudes[i])
                    lno_obs_dict["incidence_angle"][1].append(incidence_angles[i])
                    lno_obs_dict["local_time"][1].append(local_times[i])
    
                
        
with sql_db(db_path) as db:
    db.populate_db("files", file_dict, clear=False)
    db.populate_db("so_ie", so_obs_dict, clear=False)
    db.populate_db("lno_d", lno_obs_dict, clear=False)
