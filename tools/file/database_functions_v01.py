# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:35:00 2019

@author: iant

MAKE SO/LNO AND TGO TEMPERATURE DATABASES
"""


import numpy as np
import os
from datetime import datetime, timedelta
#import matplotlib.pyplot as plt
import decimal
import re
import h5py

from tools.file.hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, LOCAL_DIRECTORY, DB_DIRECTORY, makeFileList #FIG_X, FIG_Y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help='Enter command: temperature_db, so_db, or lno_db')
    args = parser.parse_args()
    command = args.command
else:
    command = ""



#SERVER_DB = True
SERVER_DB = False

CHANNEL = "so"
CHANNEL = "lno"

column_number = {"so":1, "lno":2}[CHANNEL]

HEATER_EXPORTS = ["heaters_temp_2016-04-01T000153_to_2017-03-31T235852.csv", "heaters_temp_2018-03-01T000137_to_2020-01-29T235955.csv"]

if SERVER_DB:
    import MySQLdb
    from MySQLdb import OperationalError
    SERVER = "sqldatadev2-ae"
else:
    import sqlite3
    SERVER = DB_DIRECTORY


SPICE_DATETIME_FORMAT = "%Y %b %d %H:%M:%S.%f"
SQL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"



#def prepExternalTemperatureReadings(column_number):
#    """read in TGO channel temperatures from file (only do once)"""
#    utc_datetimes = []
#    temperatures = []
#    
#    filenames = ["heaters_temp_2016-04-01T000153_to_2017-03-31T235852.csv", "heaters_temp_2018-03-24T000131_to_2019-12-25T235954.csv"]
#    for filename in filenames:
#        with open(os.path.join(LOCAL_DIRECTORY, "reference_files", filename)) as f:
#            lines = f.readlines()
#                
#        for line in lines[1:]:
#            split_line = line.split(",")
#            utc_datetimes.append(datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
#            temperatures.append(split_line[column_number])
#    
#    return np.asarray(utc_datetimes), np.asfarray(temperatures)
# 
#
#if "EXTERNAL_TEMPERATURE_DATETIMES" not in globals():
#    print("Reading in TGO temperatures")
#    EXTERNAL_TEMPERATURE_DATETIMES, EXTERNAL_TEMPERATURES = prepExternalTemperatureReadings(column_number) #1=SO nominal, 2=LNO nominal
#
#
#
#def getExternalTemperatureReadings(utc_string, column_number): #input format 2015 Mar 18 22:41:03.916651
#    """get TGO readout temperatures. Input SPICE style datetime, output in Celsius
#    column 1 = SO baseplate nominal, 2 = LNO baseplate nominal"""
#    utc_datetime = datetime.strptime(utc_string[:20].decode(), "%Y %b %d %H:%M:%S")
#    
#    external_temperature_datetimes = EXTERNAL_TEMPERATURE_DATETIMES
#    external_temperatures = EXTERNAL_TEMPERATURES
#    
#    closestIndex = np.abs(external_temperature_datetimes - utc_datetime).argmin()
#    closest_time_delta = np.min(np.abs(external_temperature_datetimes[closestIndex] - utc_datetime).total_seconds())
#    if closest_time_delta > 60 * 5:
#        print("Error: time delta %0.1f too high" %closest_time_delta)
#        print(external_temperature_datetimes[closestIndex])
#        print(utc_datetime)
#    else:
#        closestTemperature = np.float(external_temperatures[closestIndex])
#    
#    return closestTemperature
    


def getTableFields(channel):

    if channel == "so":
        if SERVER_DB:
            table_fields = [
                    {"name":"frame_id", "type":"int NOT NULL AUTO_INCREMENT", "primary":True}, \
                    {"name":"obs_id", "type":"int NOT NULL"}, \
                    {"name":"filename", "type":"varchar(100) NULL DEFAULT NULL"}, \
                    {"name":"temperature", "type":"decimal NOT NULL"}, \
                    {"name":"temperature_tgo", "type":"decimal NOT NULL"}, \
                    {"name":"diffraction_order", "type":"int NOT NULL"}, \
                    {"name":"sbsf", "type":"int NOT NULL"}, \
                    {"name":"bin_number", "type":"int NOT NULL"}, \
                    {"name":"utc_start_time", "type":"datetime NOT NULL"}, \
                    {"name":"longitude", "type":"decimal NOT NULL"}, \
                    {"name":"latitude", "type":"decimal NOT NULL"}, \
                    {"name":"altitude", "type":"decimal NOT NULL"}, \
                    {"name":"local_time", "type":"decimal NOT NULL"}, \
                    ]
        else:
            table_fields = [
                    {"name":"row_id", "type":"integer primary key"}, \
                    {"name":"file_id", "type":"integer"}, \
                    {"name":"frame_id", "type":"integer"}, \
                    {"name":"filename", "type":"text"}, \
                    {"name":"temperature", "type":"real"}, \
                    {"name":"temperature_tgo", "type":"real"}, \
                    {"name":"diffraction_order", "type":"integer"}, \
                    {"name":"sbsf", "type":"integer"}, \
                    {"name":"bin_number", "type":"integer"}, \
                    {"name":"utc_start_time", "type":"timestamp"}, \
                    {"name":"longitude", "type":"real"}, \
                    {"name":"latitude", "type":"real"}, \
                    {"name":"altitude", "type":"real"}, \
                    {"name":"local_time", "type":"real"}, \
                    ]

    if channel == "lno":
        if SERVER_DB:
            table_fields = [
                    {"name":"frame_id", "type":"int NOT NULL AUTO_INCREMENT", "primary":True}, \
                    {"name":"obs_id", "type":"int NOT NULL"}, \
                    {"name":"filename", "type":"varchar(100) NULL DEFAULT NULL"}, \
                    {"name":"temperature", "type":"decimal NOT NULL"}, \
                    {"name":"temperature_tgo", "type":"decimal NOT NULL"}, \
                    {"name":"diffraction_order", "type":"int NOT NULL"}, \
                    {"name":"n_orders", "type":"int NOT NULL"}, \
                    {"name":"utc_start_time", "type":"datetime NOT NULL"}, \
                    {"name":"longitude", "type":"decimal NOT NULL"}, \
                    {"name":"latitude", "type":"decimal NOT NULL"}, \
                    {"name":"incidence_angle", "type":"decimal NOT NULL"}, \
                    {"name":"local_time", "type":"decimal NOT NULL"}, \
                    ]
        else:
            table_fields = [
                    {"name":"row_id", "type":"integer primary key"}, \
                    {"name":"file_id", "type":"integer"}, \
                    {"name":"frame_id", "type":"integer"}, \
                    {"name":"filename", "type":"text"}, \
                    {"name":"temperature", "type":"real"}, \
                    {"name":"temperature_tgo", "type":"real"}, \
                    {"name":"diffraction_order", "type":"integer"}, \
                    {"name":"n_orders", "type":"integer"}, \
                    {"name":"utc_start_time", "type":"timestamp"}, \
                    {"name":"longitude", "type":"real"}, \
                    {"name":"latitude", "type":"real"}, \
                    {"name":"incidence_angle", "type":"real"}, \
                    {"name":"local_time", "type":"real"}, \
                    ]
    return table_fields
    

def getFilePath(hdf5_filename):
    """get full file path from name"""
    
    file_level = "hdf5_level_%s" %hdf5_filename[16:20]
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]

    filename = os.path.join(DATA_DIRECTORY, file_level, year, month, day, hdf5_filename+".h5") #choose a file
    
    return filename

PASSWORD = ""
def getFileFromDatastore(hdf5_filepath):

    import pysftp
    import getpass
    global PASSWORD
    from hdf5_functions_v04 import DATASTORE_SERVER, DATASTORE_DIRECTORY

    if PASSWORD == "":
        PASSWORD = getpass.getpass('Password:')
    

    future_path = os.path.dirname(hdf5_filepath)
    print("Making directory", future_path)
    os.makedirs(future_path, exist_ok=True)
    os.chdir(future_path)
    
    hdf5_filepath_split = hdf5_filepath.split(os.sep)
    
    levelFolderIndex = [index for index, value in enumerate(hdf5_filepath_split) if "hdf5_level_" in value][0]
    
    
    file_level = hdf5_filepath_split[levelFolderIndex]
    year_in = hdf5_filepath_split[levelFolderIndex+1]
    month_in = hdf5_filepath_split[levelFolderIndex+2]
    day_in = hdf5_filepath_split[levelFolderIndex+3]
    filename_in = hdf5_filepath_split[levelFolderIndex+4]
    
    with pysftp.Connection(DATASTORE_SERVER[0], username=DATASTORE_SERVER[1], password=PASSWORD) as sftp:
        with sftp.cd(DATASTORE_DIRECTORY+"/"+file_level): # temporarily chdir to public
            pathToDatastoreFile = "%s/%s/%s/%s/%s/%s" %(DATASTORE_DIRECTORY, file_level, year_in, month_in, day_in, filename_in)
            print(pathToDatastoreFile)
            sftp.get(pathToDatastoreFile) # get a remote file
            print("File %s transferred" %filename_in)
    os.chdir(BASE_DIRECTORY)


def findHdf5File(hdf5_filename):
    """check if file exists in data directory; if not download from server"""

    #assume directory structure    
    hdf5_filepath = getFilePath(hdf5_filename)
    if os.path.exists(hdf5_filepath):
        print("File %s found" %(hdf5_filename))

    else:
        print("File %s not found. Getting from datastore" %(hdf5_filename))
        getFileFromDatastore(hdf5_filepath)

    return hdf5_filepath



class obsDB(object):
    #db=MySQLdb.connect(host="sqlprod1-ae",user='nomad_user',passwd='Kr7NkoaN',db='nomad')
    def connect(self, server_name):
        
        if SERVER_DB:
            """replace with ini script reader"""
            print("Connecting to central database %s" %SERVER)
            host = SERVER
            user = "nomad_user"
            passwd = "Bt11Mw5X"
            db = "data_nomad"
            self.db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)
            self.cursor = self.db.cursor()
        else:
            print("Connecting to database file %s" %server_name)
            server_path = os.path.join(DB_DIRECTORY, server_name+".db")
            if not os.path.exists(server_path):
                open(server_path, 'w').close()
            self.db = sqlite3.connect(server_path, detect_types=sqlite3.PARSE_DECLTYPES)
            
        
    def __init__(self, server_name):
        self.connect(server_name)

    def close(self):
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
        datetime_from_file = datetime.strptime(datetime_string_from_file.decode(), SPICE_DATETIME_FORMAT)
        datetime_string = datetime.strftime(datetime_from_file, SQL_DATETIME_FORMAT)
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
                    table_row_datetime.append(datetime.strptime(table_element, SPICE_DATETIME_FORMAT))
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
                        elif type(table_element) == datetime: #datetimes must be written as strings
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
        
        
    def checkIfTableExists(self, table_name):
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



    def processTemperatureData(self, overwrite=False):
        
        
        def prepExternalTemperatureReadingsWebsite(column_number):
            """read in TGO channel temperatures from file. Only head in every 20 values to avoid saturating website plot"""
            utc_datetimes = []
            temperatures = []
            
            filenames = HEATER_EXPORTS
            for filename in filenames:
                with open(os.path.join(LOCAL_DIRECTORY, "reference_files", filename)) as f:
                    lines = f.readlines()
                    
                if SERVER_DB:
                    line_step = 20
                else:
                    line_step = 1
                        
                for line in lines[1:len(lines):line_step]:
                    split_line = line.split(",")
                    utc_datetimes.append(datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
                    temperatures.append(split_line[column_number])
            
            return np.asarray(utc_datetimes), np.asfarray(temperatures)


        if SERVER_DB:
            table_fields = [
                {"name":"row_id", "type":"integer primary key"}, \
                {"name":"utc_start_time", "type":"datetime"}, \
                {"name":"temperature_so", "type":"real"}, \
                {"name":"temperature_lno", "type":"real"}, \
#                {"name":"temperature_uvis", "type":"real"}, \
            ]
        else:
            table_fields = [
                {"name":"row_id", "type":"integer primary key"}, \
                {"name":"utc_start_time", "type":"timestamp"}, \
                {"name":"temperature_so", "type":"real"}, \
                {"name":"temperature_lno", "type":"real"}, \
            ]
        table_name = "temperatures"
        if overwrite:
            self.checkIfTableExists(table_name)
            self.drop_table(table_name)
            self.new_table(table_name, table_fields)
            
        print("Getting temperature data")
        utc_datetimes, so_temperatures = prepExternalTemperatureReadingsWebsite(1) #SO baseplate nominal
        utc_datetimes, lno_temperatures = prepExternalTemperatureReadingsWebsite(2) #LNO baseplate nominal
                
        sql_table_rows = []
        for i in range(len(utc_datetimes)):
            if np.mod(i, 100000) == 0:
                print("Collecting data: line %i/%i" %(i, len(utc_datetimes)))
            
            if SERVER_DB:
                sql_table_rows.append([utc_datetimes[i], so_temperatures[i].astype(np.float64), lno_temperatures[i].astype(np.float64)])
            else:
                sql_table_rows.append([None, utc_datetimes[i], so_temperatures[i].astype(np.float64), lno_temperatures[i].astype(np.float64)])
        self.insert_rows(table_name, table_fields, sql_table_rows, check_duplicates=False)



    def processChannelData(self, channel, fileLevel, regex, temp_db_obj, overwrite=False):

        table_fields = getTableFields(channel)
        table_name = {"so":"so_occultation", "lno":"lno_nadir"}[channel]
        if overwrite:
            self.checkIfTableExists(table_name)
            self.drop_table(table_name)
            self.new_table(table_name, table_fields)
            
        print("Getting file list")
        hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel, open_files=False)
    
        for fileIndex, hdf5Filename in enumerate(hdf5Filenames):
            
            hdf5Filepath = getFilePath(hdf5Filename)
            print("Collecting data: file %i/%i: %s" %(fileIndex, len(hdf5Filenames), hdf5Filename))

            if channel == "so":        
                with h5py.File(hdf5Filepath, "r") as hdf5File:
                    diffraction_orders = hdf5File["Channel/DiffractionOrder"][...]
                    temperature = np.mean(hdf5File["Housekeeping/SENSOR_1_TEMPERATURE_SO"][2:10])
                    sbsf = hdf5File["Channel/BackgroundSubtraction"][0]
                    utc_start_time = hdf5File["Geometry/ObservationDateTime"][:, 0]
#                    temperature_tgo = np.asfarray([getExternalTemperatureReadings(utc_time, 1) for utc_time in utc_start_time])
                    
                    midpoint_index = int(np.round(len(utc_start_time) / 2))
                    temperature_tgo = temp_db_obj.temperature_query(utc_start_time[midpoint_index])[0][2]
#                    print(temperature_tgo)                    
                    longitude = hdf5File["Geometry/Point0/Lon"][:, 0]
                    latitude = hdf5File["Geometry/Point0/Lat"][:, 0]
                    altitude = np.mean(hdf5File["Geometry/Point0/TangentAltAreoid"][:, :], axis=1)
                    local_time = hdf5File["Geometry/Point0/LST"][:, 0]
                    bin_number = hdf5File["Channel/IndBin"][...]
                
                nSpectra = len(diffraction_orders)
                
                sql_table_rows = []
                for i in range(nSpectra):
                    sql_table_rows.append([None, fileIndex, i, hdf5Filename, temperature.astype(np.float64), temperature_tgo, int(diffraction_orders[i]), int(sbsf), int(bin_number[i]), utc_start_time[i].decode(), longitude[i], latitude[i], altitude[i], local_time[i]])
                sql_table_rows_datetime = self.convert_table_datetimes(table_fields, sql_table_rows)
                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=False)

            elif channel == "lno":        
                with h5py.File(hdf5Filepath, "r") as hdf5File:
                    diffraction_orders = hdf5File["Channel/DiffractionOrder"][...]
                    temperature = np.mean(hdf5File["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][2:10])
                    n_orders = hdf5File.attrs["NSubdomains"]
                    utc_start_time = hdf5File["Geometry/ObservationDateTime"][:, 0]
#                    temperature_tgo = np.asfarray([getExternalTemperatureReadings(utc_time, 2) for utc_time in utc_start_time])

#                    midpoint_index = int(np.round(len(utc_start_time) / 2))
#                    temperature_tgo = temp_db_obj.temperature_query(utc_start_time[midpoint_index])[0][3]
                    temperature_tgo = np.asfarray([temp_db_obj.temperature_query(utc_time)[3] for utc_time in utc_start_time])

                    longitude = hdf5File["Geometry/Point0/Lon"][:, 0]
                    latitude = hdf5File["Geometry/Point0/Lat"][:, 0]
                    incidence_angle = hdf5File["Geometry/Point0/IncidenceAngle"][:, 0]
                    local_time = hdf5File["Geometry/Point0/LST"][:, 0]
                
                nSpectra = len(diffraction_orders)
                
                sql_table_rows = []
                for i in range(nSpectra):
                    sql_table_rows.append([None, fileIndex, i, hdf5Filename, temperature.astype(np.float64), temperature_tgo[i], int(diffraction_orders[i]), int(n_orders), utc_start_time[i].decode(), longitude[i], latitude[i], incidence_angle[i], local_time[i]])
                sql_table_rows_datetime = self.convert_table_datetimes(table_fields, sql_table_rows)
                self.insert_rows(table_name, table_fields, sql_table_rows_datetime, check_duplicates=False)







def makeObsDict(channel, query_output, add_data=True):
    """make observation dictionary from sql query, add x and y data from files"""
    
    table_fields = getTableFields(channel)
    obsDict = {}
    #make empty dicts
    for fieldDict in table_fields:
        obsDict[fieldDict["name"]] = []
        
    for output_row in query_output:
        for i in range(len(output_row)):
            obsDict[table_fields[i]["name"]].append(output_row[i])

    obsDict["x"] = []
    obsDict["y"] = []
    obsDict["filepath"] = []
    
    hdf5_filenames = set(obsDict["filename"]) #unique matching filenames
    
    for hdf5_filename in hdf5_filenames:
#        with h5py.File(getFilePath(hdf5Filename)) as f:
        hdf5_filepath = findHdf5File(hdf5_filename)
        with h5py.File(hdf5_filepath, "r") as f: #open file
            for filename, frameIndex in zip(obsDict["filename"], obsDict["frame_id"]):
                if filename == hdf5_filename:
                    x = f["Science/X"][frameIndex, :]
                    y = f["Science/Y"][frameIndex, :]
                    
                    integrationTimeRaw = f["Channel/IntegrationTime"][0]
                    numberOfAccumulationsRaw = f["Channel/NumberOfAccumulations"][0]
                    integrationTime = np.float(integrationTimeRaw) / 1.0e3 #microseconds to seconds
                    numberOfAccumulations = np.float(numberOfAccumulationsRaw)/2.0 #assume LNO nadir background subtraction is on
                    measurementPixels = 144.0
                    measurementSeconds = integrationTime * numberOfAccumulations
                    
                    y = y / measurementPixels / measurementSeconds
                    obsDict["x"].append(x)
                    obsDict["y"].append(y)
                    obsDict["filepath"].append(hdf5_filepath)
            print("measurementPixels=", measurementPixels, "; measurementSeconds=", measurementSeconds)

    return obsDict


print("Running command", command)

if command == "temperature_db":
    """make TGO temperature database"""
    dbName = "tgo_temperatures"
    db_obj = obsDB(dbName)
    db_obj.processTemperatureData(overwrite=True)
    db_obj.close()
elif command == "lno_db":
    """Add LNO data to sql"""
    channel = "lno"
    dbName = "lno_0p3a"
    fileLevel = "hdf5_level_0p3a"
    regex = re.compile("201[89][0-9][0-9][0-9][0-9]_.*LNO.*_D_.*")
    #regex = re.compile("201808[0-9][0-9]_.*LNO.*_D_.*")
    db_obj = obsDB(dbName)
    
    tempDbName = "tgo_temperatures"
    temp_db_obj = obsDB(tempDbName)
    db_obj.processChannelData(channel, fileLevel, regex, temp_db_obj, overwrite=True)
    temp_db_obj.close()
    db_obj.close()
elif command == "so_db":
    """Add SO data to sql"""
    channel = "so"
    dbName = "so_1p0a"
    fileLevel = "hdf5_level_1p0a"
    #regex = re.compile("201804[0-9][0-9]_.*_SO_A_[IE]_13[2-7]")
    regex = re.compile("20[1-2][0-9][0-9][0-9][0-9][0-9]_.*_SO_A_[IE]_13[2-7]")
    db_obj = obsDB(dbName)
    
    tempDbName = "tgo_temperatures"
    temp_db_obj = obsDB(tempDbName)
    db_obj.processChannelData(channel, fileLevel, regex, temp_db_obj, overwrite=True)
    temp_db_obj.close()
    db_obj.close()

"""query temperature database"""

#dbName = "tgo_temperatures"
#db_obj = obsDB(dbName)
##queryOutput = db_obj.query("SELECT * FROM temperatures ORDER BY ABS(JULIANDAY(utc_start_time) - JULIANDAY('2018-10-07 04:23:19')) LIMIT 1")
##queryOutput = db_obj.query("SELECT * FROM temperatures WHERE utc_start_time BETWEEN '2018-04-22 00:00:00' AND '2018-04-22 01:00:00'")
#out = db_obj.temperature_query('2018-10-07 04:23:19')[0][2] #SO
#db_obj.close()




"""get full table"""
#db_obj = obsDB()
#table = db_obj.read_table("lno_nadir")
#db_obj.close()


"""get dictionary from query and plot"""
#channel = "lno"
#db_obj = obsDB(dbName)
#CURIOSITY = -4.5895, 137.4417
#searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir where latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND incidence_angle < 40 and diffraction_order == 134")
#obsDict = makeObsDict(channel, searchQueryOutput)
#db_obj.close()
#
#plt.figure()
#plt.scatter(obsDict["longitude"], obsDict["latitude"])
#
#    
#fig1, ax1 = plt.subplots()
#
#
#for frameIndex, (x, y) in enumerate(zip(obsDict["x"], obsDict["y"])):
#    ax1.plot(x, y, alpha=0.3)
#
#yMean = np.mean(np.asfarray(obsDict["y"])[:, :], axis=0)
#xMean = np.mean(np.asfarray(obsDict["x"])[:, :], axis=0)
#ax1.plot(xMean, yMean, "k")
#
#yMean = np.mean(np.asfarray(obsDict["y"])[145:, :], axis=0)
#xMean = np.mean(np.asfarray(obsDict["x"])[145:, :], axis=0)
#ax1.plot(xMean, yMean, "r")



"""write so occultations to text file for Sebastien"""
#channel = "so"
#dbName = "so_1p0a"
#db_obj = obsDB(dbName)
#searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE diffraction_order == 134")
##obsDict = makeObsDict(channel, searchQueryOutput)
#db_obj.close()
#
#table_headers = getTableFields(channel)
#headers = [value["name"] for value in table_headers]
#header_types = [value["type"] for value in table_headers]
#
#with open(os.path.join(BASE_DIRECTORY, "order_134_occultations.txt"), "w") as f:
#    lines = ["%s, %s, %s, %s, %s\n" %(headers[9], headers[10], headers[11], headers[12], headers[13])]
#    for queryLine in searchQueryOutput:
#        lines.append("%s, %0.3f, %0.3f, %0.3f, %0.3f\n" 
#             %(queryLine[9], queryLine[10], queryLine[11], queryLine[12], queryLine[13]))
#    for line in lines:
#        f.write(line)



"""check for errors in timestamps of TGO readouts in heaters_temp"""
dbName =  "heaters_temp"
db_obj = obsDB(dbName)
db_obj.query("SELECT ts from heaters_temp")
data = db_obj.query("SELECT ts from heaters_temp")
db_obj.close()
ts = [i[0] for i in data[540000:]]
delta = [ts[i+1]-ts[i] for i in range(len(ts[:-1]))]
ind = np.where(np.array(delta)>timedelta(minutes=20))[0] #find all gaps>20 minutes
for i in ind:
    print(ts[i], "-", ts[i+1])
