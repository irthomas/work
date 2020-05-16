# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:35:00 2019

@author: iant

MAKE SO/LNO AND TGO TEMPERATURE DATABASES
"""


import numpy as np
import os
import datetime
#import matplotlib.pyplot as plt
import decimal
import re
import h5py

from tools.file.paths import paths
from tools.file.passwords import passwords
from tools.sql.sql_table_fields import sql_table_fields

from tools.file.hdf5_functions import make_filelist, get_filepath



#SERVER_DB = True
SERVER_DB = False


if SERVER_DB:
    import MySQLdb
    from MySQLdb import OperationalError
    SERVER = "sqldatadev2-ae"
else:
    import sqlite3
    SERVER = paths["DB_DIRECTORY"]


SPICE_DATETIME_FORMAT = "%Y %b %d %H:%M:%S.%f"
SQL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"
ARG_FORMAT = "%Y-%m-%d"


def get_obs_duration(hdf5_file):
    
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0]
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, -1]
    utc_start_datetime = datetime.datetime.strptime(utc_start_time.decode(), SPICE_DATETIME_FORMAT)
    utc_end_datetime = datetime.datetime.strptime(utc_end_time.decode(), SPICE_DATETIME_FORMAT)
    total_seconds = (utc_end_datetime - utc_start_datetime).total_seconds()
    
    return total_seconds




class obs_database(object):
    def connect(self, server_name):
        
        if SERVER_DB:
            """replace with ini script reader"""
            if not self.silent:
                print("Connecting to central database %s" %SERVER)
            host = SERVER
            user = "nomad_user"
            passwd = passwords["nomad_user"]
            db = "data_nomad"
            self.db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)
            self.cursor = self.db.cursor()
        else:
            if not self.silent:
                print("Connecting to database file %s" %server_name)
            server_path = os.path.join(paths["DB_DIRECTORY"], server_name+".db")
            if not os.path.exists(server_path):
                open(server_path, 'w').close()
            self.db = sqlite3.connect(server_path, detect_types=sqlite3.PARSE_DECLTYPES)
            
        
    def __init__(self, server_name, silent=False):
        self.silent = silent
        self.connect(server_name)

    def close(self):
        if not self.silent:
            print("Disconnecting from mysql database")
        if SERVER_DB:
            self.cursor.close()
        self.db.close()
        

    def query(self, input_query):
#        print(input_query)
        if SERVER_DB:
            try:
                self.cursor.execute((input_query))
            except OperationalError:
                print(input_query)
            output = self.cursor.fetchall()
        else:
            c = self.db.execute(input_query)
            self.db.commit()
            output = c.fetchall()
        return output
    
    def temperature_query(self, datetime_string_from_file):
        datetime_from_file = datetime.datetime.strptime(datetime_string_from_file.decode(), SPICE_DATETIME_FORMAT)
        datetime_string = datetime.datetime.strftime(datetime_from_file, SQL_DATETIME_FORMAT)
        query_string = ("SELECT * FROM temperatures ORDER BY ABS(JULIANDAY(utc_start_time) - JULIANDAY('%s')) LIMIT 1" %datetime_string)
#        print("query=", query_string)
        output = self.query(query_string)
#        print("output=", output)
        return output


    def read_table(self, table_name):
        query_string = "SELECT * FROM %s" %table_name
        table = self.query(query_string)

        new_table = []
        for row in table:
            new_table.append([float(element) if type(element) == decimal.Decimal else element for element in row])
        
        return new_table
    
    def convert_table_datetimes(self, table_fields, table_rows):
        """convert all spice format strings to datetimes in preparation for writing sql"""
        table_fields_not_key_datetimes = [True if ("datetime" in field["type"]) or ("timestamp" in field["type"]) else False for field in table_fields if "primary" not in field.keys()]
        table_rows_datetime = []
        for table_row in table_rows:
            table_row_datetime = []
            for table_element, table_is_datetime in zip(table_row, table_fields_not_key_datetimes):
                if table_is_datetime and table_element != "-": #normal datetimes
                    table_row_datetime.append(datetime.datetime.strptime(table_element, SPICE_DATETIME_FORMAT))
                elif table_element == "-": #any blank values in datetime or other
                    table_row_datetime.append("NULL")
                else:
                    table_row_datetime.append(table_element)
            table_rows_datetime.append(table_row_datetime)
        
        return table_rows_datetime

    
    def insert_rows(self, table_name, table_fields, table_rows, check_duplicates=False, duplicate_columns=[]):
        if SERVER_DB:
            table_fields_not_key = [field["name"] for field in table_fields if "primary" not in field["type"]]
            if len(table_fields_not_key) != len(table_rows[0]):
                print("Error: Field names and data are not the same length")
                
            if check_duplicates: #check if any column value already exists. Use on datetimes primarily
                existing_table = self.read_table(table_name) 
                
    
            print("Inserting %i rows into table %s" %(len(table_rows), table_name))
            for row_index, table_row in enumerate(table_rows):
                duplicates = 0
                if check_duplicates:
                    for existing_row in existing_table:
                        for column_number in duplicate_columns:
                            if table_row[column_number] == existing_row[column_number+1]:
                                duplicates += 1
                                
                if duplicates > 0:
                    print("Error: Row %i contains elements matching existing rows" %row_index)
    
                else:
                    query_string = "INSERT INTO %s (" %table_name
                    for table_field in table_fields_not_key:
                        query_string += "%s, " %table_field
                    query_string = query_string[:-2]
                    query_string += ") VALUES ("
                    for table_element in table_row:
                        if type(table_element) == str:
                            if table_element == "NULL":
                                query_string += "%s, " %table_element #nulls must not have inverted commas
                            else:
                                query_string += "\"%s\", " %table_element #other strings must have inverted commas
                        elif type(table_element) == datetime.datetime: #datetimes must be written as strings
                            query_string += "\"%s\", " %table_element
                        else: #values must not have inverted commas
                            query_string += "%s, " %table_element
                    query_string = query_string[:-2]
                    query_string += ")"
                    if np.mod(row_index, 100000) == 0:
                        print("Adding line %i/%i to db" %(row_index, len(table_rows)))
                    self.query(query_string)
        else:
            n_fields = len(table_fields)
            query_string = "INSERT INTO %s (" %table_name
            for table_field in table_fields:
                query_string += "%s, " %table_field["name"]
            query_string = query_string[:-2]
            query_string += ") VALUES (" + "?," * n_fields
            query_string = query_string[:-1]
            query_string += ")"
#            print(query_string)
#            for table_row in table_rows:
#                print(table_row)
#                self.db.execute(query_string, table_row)
#                self.db.commit()
            
            self.db.executemany(query_string, table_rows)
            self.db.commit()
            
            
    def new_table(self, table_name, table_fields):
        if SERVER_DB:
            table_not_key = []
            for field in table_fields:
                if "primary" in field["type"]:
                    table_key = field["name"]
                else:
                    table_not_key.append(field["name"])
            
            query_string = "CREATE TABLE %s" %table_name + "(%s INT NOT NULL AUTO_INCREMENT" %table_key + ", PRIMARY KEY (%s), " %table_key
            for field in table_fields:
                if field["name"] != table_key:
                    query_string += "%s %s, " %(field["name"], field["type"])
            query_string = query_string[:-2]
            query_string += ")"
        else:
            query_string = "CREATE TABLE %s (" %table_name
            for field in table_fields:
                query_string += "%s %s, " %(field["name"], field["type"])
            query_string = query_string[:-2]
            query_string += ")"
        print("Creating table %s" %table_name)
        self.query(query_string)

    
    def drop_table(self, table_name):
        query_string = "DROP TABLE IF EXISTS %s" %table_name
        print("Dropping table %s" %table_name)
        self.query(query_string)
        
        
    def create_database(self, db_name):
        query_string = "CREATE DATABASE %s" %db_name
        print("Creating database %s" %db_name)
        self.query(query_string)
        
        
    def check_if_table_exists(self, table_name):
        if SERVER_DB:
            output = self.query("SHOW TABLES") #returns nested tuple
            output_flat = [each_output[0] for each_output in output] #flatten
            if len(output_flat) > 0:
                if table_name in output_flat:
                    print("%s already exists in database" %table_name)
                else:
                    print("%s does not exist in database" %table_name)
            else:
                print("no tables in database")
        
        else:
            output = self.query("SELECT name FROM sqlite_master WHERE type='table'")
            if len(output) > 0:
                if table_name in output[0]:
                    print("%s already exists in database" %table_name)
                else:
                    print("%s does not exist in database" %table_name)
            else:
                print("no tables in database")





    def process_channel_data(self, args, silent=True):
        

        table_fields = sql_table_fields(server_db=SERVER_DB)
        table_name = args.level
        if args.regenerate:
            print("Deleting and regenerating table")
            self.check_if_table_exists(table_name)
            self.drop_table(table_name)
            self.new_table(table_name, table_fields)
            
        print("Getting file list")
        if args.command == "lno_nadir":
            regex = re.compile("20.*_LNO.*_D.*")
        elif args.command == "so_occultation":
            regex = re.compile("20.*_SO.*_[IE].*")
        elif args.command == "uvis_nadir":
            regex = re.compile("20.*_UVIS.*_D")
        elif args.command == "uvis_occultation":
            regex = re.compile("20.*_UVIS.*_[IE]")
            
        beg_datetime = datetime.datetime.strptime(args.beg, ARG_FORMAT)
        end_datetime = datetime.datetime.strptime(args.end, ARG_FORMAT)

        _, hdf5Filenames, _ = make_filelist(regex, args.level, silent=silent, open_files=False)
        
        print("%i files found in directory" %len(hdf5Filenames))
        #make datetime from hdf5 filenames, find those that match the beg/end times
        hdf5_datetimes = [datetime.datetime.strptime(i[:15], HDF5_FILENAME_FORMAT) for i in hdf5Filenames]
        
#        hdf5_file_indices = [i for i, hdf5_datetime in enumerate(hdf5_datetimes) if beg_datetime < hdf5_datetime < end_datetime]
        matching_hdf5_filenames = [hdf5_filename for hdf5_datetime, hdf5_filename in zip(hdf5_datetimes, hdf5Filenames) if beg_datetime < hdf5_datetime < end_datetime]
        
        print("Adding %i files between %s and %s to database" %(len(matching_hdf5_filenames), args.beg, args.end))
        for fileIndex, hdf5Filename in enumerate(matching_hdf5_filenames):
            
            hdf5Filepath = get_filepath(hdf5Filename)
            if not silent:
                print("Collecting data: file %i/%i: %s" %(fileIndex, len(hdf5Filenames), hdf5Filename))


            with h5py.File(hdf5Filepath, "r") as hdf5File:
                orbit = hdf5File.attrs["Orbit"]
                filename = hdf5Filename
                
                diffraction_order = hdf5File["Channel/DiffractionOrder"][0]
                sbsf = hdf5File["Channel/BackgroundSubtraction"][0]
                utc_start_times = hdf5File["Geometry/ObservationDateTime"][:, 0]
                duration = get_obs_duration(hdf5File)
                n_spectra = len(utc_start_times)
                n_orders = hdf5File.attrs["NSubdomains"]
                longitudes = hdf5File["Geometry/Point0/Lon"][:, 0]
                latitudes = hdf5File["Geometry/Point0/Lat"][:, 0]
                if args.command == "lno_nadir":        
                    mean_temperature_tgo = 1.0 + np.mean(hdf5File["Temperature/NominalLNO"][...])
                    bin_index = np.ones(n_spectra)
                    incidence_angles = hdf5File["Geometry/Point0/IncidenceAngle"][:, 0]
                    altitudes = np.zeros(n_spectra) - 999.0
                elif args.command == "so_occultation":
                    mean_temperature_tgo = 1.0 + np.mean(hdf5File["Temperature/NominalSO"][...])
                    bin_index = hdf5File["Channel/IndBin"][...]
                    incidence_angles = np.zeros(n_spectra) - 999.0
                    altitudes = hdf5File["Geometry/Point0/TangentAltAreoid"][:, 0]
                local_times = hdf5File["Geometry/Point0/LST"][:, 0]

                
                sql_table_rows = []
                for i in range(n_spectra):
                    sql_table_rows.append([None, orbit, filename, i, mean_temperature_tgo, \
                       int(diffraction_order), int(sbsf), int(bin_index[i]), utc_start_times[i].decode(), \
                       duration, int(n_spectra), int(n_orders), longitudes[i], latitudes[i], \
                       altitudes[i], incidence_angles[i], local_times[i]])
                sql_table_rows_datetime = self.convert_table_datetimes(table_fields, sql_table_rows)
#                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=False)
                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=True)



#    def processTemperatureData(self, overwrite=False):
#        
#        
#        def prepExternalTemperatureReadingsWebsite():
#            """read in LNO channel temperatures from file. Only use in every 20 values to avoid saturating website plot"""
#            utc_datetimes = []
#            temperatures = []
#            
#            filenames = HEATER_EXPORTS
#            for filename in filenames:
#                with open(os.path.join(LOCAL_DIRECTORY, "reference_files", filename)) as f:
#                    lines = f.readlines()
#                    
#                if SERVER_DB:
#                    line_step = 20
#                else:
#                    line_step = 1
#                        
#                for line in lines[1:len(lines):line_step]:
#                    split_line = line.split(",")
#                    utc_datetimes.append(datetime.datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
#                    temperatures.append(split_line[column_number])
#            
#            return np.asarray(utc_datetimes), np.asfarray(temperatures)
#
#
#        if SERVER_DB:
#            table_fields = [
#                {"name":"row_id", "type":"integer primary key"}, \
#                {"name":"utc_start_time", "type":"datetime"}, \
#                {"name":"temperature_so", "type":"real"}, \
#                {"name":"temperature_lno", "type":"real"}, \
#            ]
#        else:
#            table_fields = [
#                {"name":"row_id", "type":"integer primary key"}, \
#                {"name":"utc_start_time", "type":"timestamp"}, \
#                {"name":"temperature_so", "type":"real"}, \
#                {"name":"temperature_lno", "type":"real"}, \
#            ]
#        table_name = "temperatures"
#        if overwrite:
#            self.checkIfTableExists(table_name)
#            self.drop_table(table_name)
#            self.new_table(table_name, table_fields)
#            
#        print("Getting temperature data")
#        utc_datetimes, so_temperatures = prepExternalTemperatureReadingsWebsite(1) #SO baseplate nominal
#        utc_datetimes, lno_temperatures = prepExternalTemperatureReadingsWebsite(2) #LNO baseplate nominal
#                
#        sql_table_rows = []
#        for i in range(len(utc_datetimes)):
#            if np.mod(i, 100000) == 0:
#                print("Collecting data: line %i/%i" %(i, len(utc_datetimes)))
#            
#            if SERVER_DB:
#                sql_table_rows.append([utc_datetimes[i], so_temperatures[i].astype(np.float64), lno_temperatures[i].astype(np.float64)])
#            else:
#                sql_table_rows.append([None, utc_datetimes[i], so_temperatures[i].astype(np.float64), lno_temperatures[i].astype(np.float64)])
#        self.insert_rows(table_name, table_fields, sql_table_rows, check_duplicates=False)





