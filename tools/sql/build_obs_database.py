# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:49:12 2022

@author: iant

NEW OBSERVATION DATABASE

GET FILE LIST FROM CACHE.DB, POPULATE MASTER TABLE WITH FILENAMES AND INDIVIDUAL SPECTRA IN SUB TABLES
"""

import numpy as np
import os
import datetime
# import decimal
import re
import h5py
import sqlite3 as sql

import platform


# from tools.file.paths import paths
# from tools.file.passwords import passwords
# from tools.sql.read_cache_db import get_filenames_from_cache

# from tools.file.hdf5_functions import make_filelist, get_filepath

# from tools.spectra.baseline_als import baseline_als



from tools.sql.table_dicts import obs_file_dicts
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
    
    """calculate duration of observation. For SO it is an approximation, as the file is ordered by altitude"""
    
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0]
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, -1]
    utc_start_datetime = datetime.datetime.strptime(utc_start_time.decode(), SPICE_DATETIME_FORMAT)
    utc_end_datetime = datetime.datetime.strptime(utc_end_time.decode(), SPICE_DATETIME_FORMAT)
    total_seconds = np.abs((utc_end_datetime - utc_start_datetime).total_seconds())
    
    return total_seconds




def clear_db():
    with sql_db(db_path) as db:
        db.populate_db("files", obs_file_dicts["files"], clear=True)
        db.populate_db("so_ie", obs_file_dicts["so_ie"], clear=True)
        db.populate_db("lno_d", obs_file_dicts["lno_d"], clear=True)
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






    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('observation_type', type=str, help='Enter command: so_occultation, lno_nadir')
    parser.add_argument('--beg', type=str, default=None, help='Enter start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='Enter end date YYYY-MM-DD')
    parser.add_argument('-regenerate', action='store_true', help='Delete table and regenerate. Always use on first run')
    parser.add_argument('-silent', action='store_true', help='Output messages')
#     parser.add_argument('--regex', type=str, default="", help='Match regex')
    args = parser.parse_args()
    observation_type = args.observation_type
else:
    observation_type = ""








def populate_db(args):
    """make database containing info about all spectra in a channel for a particular observation type"""
    
    if args.silent:
        silent = True
    else:
        silent = False
        
    if args.regenerate:
        print("Deleting table")
        clear_db()
        
        print("Regenerating table")
        
        
    print("Getting file list")
    # observation_type = args.observation_type
    
    if args.beg:
        beg_dt = datetime.datetime.strptime(args.beg, ARG_FORMAT)
    if args.end:
        end_dt = datetime.datetime.strptime(args.end, ARG_FORMAT)
    
    
    rows = get_data_from_cache(cache_db_path, beg_dtime=beg_dt, end_dtime=end_dt)
    
    regex = re.compile(".*(_SO_._[IEG]|_LNO_._D[PF]).*")
    rows = sorted([row for row in rows if regex.match(os.path.basename(row[0]))])
    
    
    hdf5_filepaths = [row[0] for row in rows]
    utc_start_times = [row[2] for row in rows]
    
    print("%i files found in directory" %len(hdf5_filepaths))
    
    for file_index, (utc_start_time, hdf5_filepath) in enumerate(zip(utc_start_times, hdf5_filepaths)):
        
        hdf5_filename = os.path.basename(hdf5_filepath)
        
        """TODO: check if already present; if not, get next file_id"""
        
        obs_file_dicts["files"]["filename"][1].append(os.path.splitext(hdf5_filename)[0]) #strip .h5 from filename
        obs_file_dicts["files"]["utc_start_time"][1].append(utc_start_time)
        
        if not os.path.exists(hdf5_filepath):
            print("Error: %s does not exist" %hdf5_filename)
            
        elif "UVIS" in hdf5_filename:
            continue
        
        else:
    
            if not silent:
                if np.mod(file_index, 100) == 0:
                    print("Collecting data: file %i/%i: %s" %(file_index, len(hdf5_filepaths), hdf5_filename))
    
    
            with h5py.File(hdf5_filepath, "r") as hdf5_file:
                
                n_spectra = len(hdf5_file["Geometry/ObservationDateTime"][:, 0])
                
                obs_file_dicts["files"]["orbit"][1].append(hdf5_file.attrs["Orbit"])
                obs_file_dicts["files"]["duration"][1].append(get_obs_duration(hdf5_file))
                obs_file_dicts["files"]["n_spectra"][1].append(n_spectra)
                obs_file_dicts["files"]["n_orders"][1].append(int(hdf5_file.attrs["NSubdomains"]))
                obs_file_dicts["files"]["bg_subtraction"][1].append(int(hdf5_file["Channel/BackgroundSubtraction"][0]))
                
                file_id = file_index #update this
                
                if "InterpolatedTemperature" in hdf5_file["Channel"].keys():
                    temperatures = hdf5_file["Channel/InterpolatedTemperature"][...]
                else:
                    temperatures = hdf5_file["Channel/MeasurementTemperature"][...]
                    
                diffraction_orders = hdf5_file["Channel/DiffractionOrder"][...]
                longitudes = hdf5_file["Geometry/Point0/Lon"][:, 0]
                latitudes = hdf5_file["Geometry/Point0/Lat"][:, 0]
                local_times = hdf5_file["Geometry/Point0/LST"][:, 0]
        
                if "SO" in hdf5_filename:
                    
                    bin_index = hdf5_file["Channel/IndBin"][...]
                    altitudes = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
        
                    for i in range(n_spectra):
                    
                        obs_file_dicts["so_ie"]["file_id"][1].append(file_id+1)
                        obs_file_dicts["so_ie"]["frame_id"][1].append(i)
                        obs_file_dicts["so_ie"]["temperature"][1].append(float(temperatures[i]))
                        obs_file_dicts["so_ie"]["diffraction_order"][1].append(int(diffraction_orders[i]))
                        obs_file_dicts["so_ie"]["bin_index"][1].append(int(bin_index[i]))
                        obs_file_dicts["so_ie"]["altitude"][1].append(altitudes[i])
                        obs_file_dicts["so_ie"]["longitude"][1].append(longitudes[i])
                        obs_file_dicts["so_ie"]["latitude"][1].append(latitudes[i])
                        obs_file_dicts["so_ie"]["local_time"][1].append(local_times[i])
                        
                elif "LNO" in hdf5_filename:
        
                    incidence_angles = hdf5_file["Geometry/Point0/IncidenceAngle"][:, 0]
                    
                    for i in range(n_spectra):
                    
                        obs_file_dicts["lno_d"]["file_id"][1].append(file_id+1)
                        obs_file_dicts["lno_d"]["frame_id"][1].append(i)
                        obs_file_dicts["lno_d"]["temperature"][1].append(float(temperatures[i]))
                        obs_file_dicts["lno_d"]["diffraction_order"][1].append(int(diffraction_orders[i]))
                        obs_file_dicts["lno_d"]["longitude"][1].append(longitudes[i])
                        obs_file_dicts["lno_d"]["latitude"][1].append(latitudes[i])
                        obs_file_dicts["lno_d"]["incidence_angle"][1].append(incidence_angles[i])
                        obs_file_dicts["lno_d"]["local_time"][1].append(local_times[i])
        
                    
            
    with sql_db(db_path) as db:
        db.populate_db("files", obs_file_dicts["files"], clear=False)
        db.populate_db("so_ie", obs_file_dicts["so_ie"], clear=False)
        db.populate_db("lno_d", obs_file_dicts["lno_d"], clear=False)


populate_db(args)