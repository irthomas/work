# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:37:43 2022

@author: iant

SQLITE DATABASE FUNCTIONS

THE CLASS USES ENTER AND EXIT FUNCTIONS SO CAN BE USED AS FOLLOWS:
    
with sql_db(<path>) as db:
    db.get_all_rows(<table_name>)

IN THIS WAY THE DB IS AUTOMATICALLY OPENED AND CLOSED CORRECTLY
    
"""



import sqlite3 as sql


class sql_db(object):
    
    def __init__(self, db_path):
        self.db_path = db_path
        
    def __enter__(self):
        print("Connecting to database %s" %self.db_path)
        self.connect_db()
        return self

    def connect_db(self):
        self.con = sql.connect(self.db_path, detect_types=sql.PARSE_DECLTYPES)
        self.cur = self.con.cursor()

    def __exit__(self, type, value, traceback):
        print("Closing database %s" %self.db_path)
        self.con.close()



    def get_all_rows(self, table_name):
        """get data from all rows of the given table"""
        self.cur.execute('SELECT * FROM {}'.format(table_name))
        self.con.commit()
        rows = self.cur.fetchall()
        return rows

    def query(self, query):
        #if query is only a string
        if isinstance(query, str):
            self.cur.execute(query)
            
        #if query is a string and a list of variables
        elif len(query) == 2:
            self.cur.execute(query[0], query[1])

        self.con.commit()
        rows = self.cur.fetchall()
        return rows
        


    def empty_table(self, table_name, table_dict):
        """delete table and rebuild empty"""


        #convert a dictionary of field names and formats into an SQL query to make the table
        #table_dict = {"id":["INTEGER PRIMARY KEY AUTOINCREMENT"], <field1>:["TEXT NOT NULL", ...], <field2>:["REAL NOT NULL", ...], ...}

        query = "CREATE TABLE %s (" %table_name

        for key, value in table_dict.items():
            query += "%s %s, " %(key, value[0])
        query = query[:-2] #remove last comma and space
        query += ");"
    
        print("Deleting and rebuilding table", table_name)
        self.cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
        self.cur.execute(query)
        self.con.commit()


    def populate_db(self, table_name, table_dict, clear=False):
        
        
        if clear:
            self.empty_table(table_name, table_dict)
       
        print("Populating db with dictionary values")
        
        fields = list(table_dict.keys())
        fields_not_primary = fields[1:]

        field_names_text = ", ".join(fields_not_primary)
        
        questions_str = ",".join(["?"] * len(fields_not_primary))

        
        field_not_primary = fields_not_primary[0]
        for i in range(len(table_dict[field_not_primary][1])):
            
            values = [table_dict[key][1][i] for key in fields_not_primary]
       
            self.cur.execute("INSERT INTO {} (%s) VALUES (%s)".format(table_name) %(field_names_text, questions_str), values)
        self.con.commit()


#Example code: clear table h5 (if it exists) and repoplulate with 3 fields. Add 3 rows to the table, then exit

# table_name = "h5"
# table_dict = {
#     "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
#     "h5_filename":["TEXT NOT NULL", ["filename1", "filename2", "filename3"]],
#     "h5_filepath":["TEXT NOT NULL", ["filepath1", "filepath2", "filepath3"]],
#     }


# db_path = "test.db"
# with sql_db(db_path) as db:
#     db.populate_db(table_name, table_dict, clear=True)
#     a = db.get_all_rows(table_name)
    
