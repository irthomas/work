# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:35:00 2019

@author: iant

DATABASE READER AND WRITER. CAN BE USED TO MAKE THE OBSERVATION DATABASES
"""


#import numpy as np
import os
#import datetime
import decimal
#import re
#import h5py
import platform

if platform.system() == "Windows":
    from tools.file.paths import paths
    from tools.file.passwords import passwords

elif os.path.isdir("tools"):
    from tools.file.paths import paths
    from tools.file.passwords import passwords
    
else: #if running in pipeline
    paths = {}
    with open("passwords.txt", "r") as f:  passwords = eval("".join(f.readlines()))

import MySQLdb
from MySQLdb import OperationalError
import sqlite3


SQL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"




class database(object):
    def __init__(self, server_name, bira_server=False, silent=False):
        self.silent = silent
        self.bira_server = bira_server
        self.connect(server_name)


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
    


    def read_table(self, table_name):
        query_string = "SELECT * FROM %s" %table_name
        table = self.query(query_string)

        new_table = []
        for row in table:
            new_table.append([float(element) if type(element) == decimal.Decimal else element for element in row])
        
        return new_table


    def new_table(self, table_name, table_fields): #name must not contain spaces!
        if self.bira_server:
            table_not_key = []
            for field in table_fields:
                if "primary" in field["type"] or "primary" in field.keys():
                    if not self.silent:
                        print("%s is the primary key" %field["name"])
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
        if not self.silent:
            print(query_string)
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
                    if not self.silent:
                        print("%s already exists in database" %table_name)
                    return True
                else:
                    if not self.silent:
                        print("%s does not exist in database" %table_name)
                    return False
            else:
                if not self.silent:
                    print("no tables in database")
                return False
        
        else:
            output = self.query("SELECT name FROM sqlite_master WHERE type='table'")
            if len(output) > 0:
                if table_name in output[0]:
                    if not self.silent:
                        print("%s already exists in database" %table_name)
                    return True
                else:
                    if not self.silent:
                        print("%s does not exist in database" %table_name)
                    return False
            else:
                if not self.silent:
                    print("no tables in database")
                return False



