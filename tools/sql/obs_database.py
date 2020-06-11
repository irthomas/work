# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:35:00 2019

@author: iant

DATABASE READER AND WRITER. CAN BE USED TO MAKE THE OBSERVATION DATABASES
"""


import numpy as np
import os
import datetime
import decimal
import re
import h5py

from tools.file.paths import paths
from tools.file.passwords import passwords
from tools.sql.sql_table_fields import obs_database_fields

from tools.file.hdf5_functions import make_filelist, get_filepath

from tools.spectra.baseline_als import baseline_als

import MySQLdb
from MySQLdb import OperationalError
import sqlite3


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
        
        if self.bira_server:

            host = "sqldatadev2-ae"
            user = "nomad_user"
            passwd = passwords["nomad_user"]
            db = "data_nomad"

            if not self.silent:
                print("Connecting to central database %s" %host)
 
            self.db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)
            self.cursor = self.db.cursor()
        else:
            

            if not self.silent:
                print("Connecting to database file %s" %server_name)
            server_path = os.path.join(paths["DB_DIRECTORY"], server_name+".db")
            if not os.path.exists(server_path):
                open(server_path, 'w').close()
            self.db = sqlite3.connect(server_path, detect_types=sqlite3.PARSE_DECLTYPES)
            
        
    def __init__(self, server_name, bira_server=False, silent=False):
        self.silent = silent
        self.bira_server = bira_server
        self.connect(server_name)

    def close(self):
        if not self.silent:
            print("Disconnecting from mysql database")
        if self.bira_server:
            self.cursor.close()
        self.db.close()
        

    def query(self, input_query):
#        print(input_query)
        if self.bira_server:
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
        if self.bira_server:
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
        if self.bira_server:
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
        if self.bira_server:
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





    def process_channel_data(self, args):
        """make database containing info about all spectra in a channel for a particular observation type"""

        self.level = args.level
        self.command = args.command
        if args.silent:
            silent = True
        else:
            silent = False
            
        table_fields = obs_database_fields(self.level, bira_server=self.bira_server)
        table_name = self.level
        if args.regenerate:
            print("Deleting and regenerating table")
            self.check_if_table_exists(table_name)
            self.drop_table(table_name)
            self.new_table(table_name, table_fields)
            
        print("Getting file list")
        if args.regex:
            regex = re.compile(args.regex)
        else:
            if self.command == "lno_nadir":
                if self.level == "hdf5_level_1p0a":
                    regex = re.compile("20.*_LNO.*_D(P|F).*")
                else:
                    regex = re.compile("20.*_LNO.*_D.*")
            elif self.command == "so_occultation":
                regex = re.compile("20.*_SO.*_[IE].*")
            elif self.command == "uvis_nadir":
                regex = re.compile("20.*_UVIS.*_D")
            elif self.command == "uvis_occultation":
                regex = re.compile("20.*_UVIS.*_[IE]")
            
        beg_datetime = datetime.datetime.strptime(args.beg, ARG_FORMAT)
        end_datetime = datetime.datetime.strptime(args.end, ARG_FORMAT)

        _, hdf5Filenames, _ = make_filelist(regex, self.level, silent=silent, open_files=False)
        
        print("%i files found in directory" %len(hdf5Filenames))
        #make datetime from hdf5 filenames, find those that match the beg/end times
        hdf5_datetimes = [datetime.datetime.strptime(i[:15], HDF5_FILENAME_FORMAT) for i in hdf5Filenames]
        
#        hdf5_file_indices = [i for i, hdf5_datetime in enumerate(hdf5_datetimes) if beg_datetime < hdf5_datetime < end_datetime]
        matching_hdf5_filenames = [hdf5_filename for hdf5_datetime, hdf5_filename in zip(hdf5_datetimes, hdf5Filenames) if beg_datetime < hdf5_datetime < end_datetime]
        
        print("Adding %i files between %s and %s to database" %(len(matching_hdf5_filenames), args.beg, args.end))
        for fileIndex, hdf5Filename in enumerate(matching_hdf5_filenames):
            
            hdf5Filepath = get_filepath(hdf5Filename)
            if not silent:
                print("Collecting data: file %i/%i: %s" %(fileIndex, len(matching_hdf5_filenames), hdf5Filename))


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
                if self.command == "lno_nadir":        
                    mean_temperature_tgo = np.mean(hdf5File["Temperature/NominalLNO"][...])
                    bin_index = np.ones(n_spectra)
                    incidence_angles = hdf5File["Geometry/Point0/IncidenceAngle"][:, 0]
                    altitudes = np.zeros(n_spectra) - 999.0
                elif self.command == "so_occultation":
                    mean_temperature_tgo = np.mean(hdf5File["Temperature/NominalSO"][...])
                    bin_index = hdf5File["Channel/IndBin"][...]
                    incidence_angles = np.zeros(n_spectra) - 999.0
                    altitudes = hdf5File["Geometry/Point0/TangentAltAreoid"][:, 0]
                local_times = hdf5File["Geometry/Point0/LST"][:, 0]
                
                sql_table_rows = []

                #get mean of y radiance factor continuum
                if self.level == "hdf5_level_1p0a":
                    y = hdf5File["Science/YRadianceFactor"][:, :]
                    for i in range(n_spectra):
                        if incidence_angles[i] < 80.0:
                            continuum = baseline_als(y[i, :])
                            y_mean = np.mean(continuum[160:240])
    
                            sql_table_rows.append([None, orbit, filename, i, mean_temperature_tgo, \
                               int(diffraction_order), int(sbsf), int(bin_index[i]), utc_start_times[i].decode(), \
                               duration, int(n_spectra), int(n_orders), longitudes[i], latitudes[i], \
                               altitudes[i], incidence_angles[i], local_times[i], float(y_mean)])
                else:
                
                    for i in range(n_spectra):
                        sql_table_rows.append([None, orbit, filename, i, mean_temperature_tgo, \
                           int(diffraction_order), int(sbsf), int(bin_index[i]), utc_start_times[i].decode(), \
                           duration, int(n_spectra), int(n_orders), longitudes[i], latitudes[i], \
                           altitudes[i], incidence_angles[i], local_times[i]])
                sql_table_rows_datetime = self.convert_table_datetimes(table_fields, sql_table_rows)
#                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=False)
                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=True)





