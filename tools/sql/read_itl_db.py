# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:54:53 2022

@author: iant

READ OBSERVATIONS FROM ITL SQL DB 
"""


import sqlite3 as sql



def connect_db(db_path):
    con = sql.connect(db_path, detect_types=sql.PARSE_DECLTYPES)
    return con

def close_db(con):
    con.close()
    


   
    
def get_all_rows_from_db(db_path, table_name):
    con = sql.connect(db_path, detect_types=sql.PARSE_DECLTYPES)
    con.row_factory = sql.Row
       
    cur = con.cursor()
    #first get column names
    cur.execute("PRAGMA table_info({})".format(table_name))
    rows = cur.fetchall()
    field_names = []
    field_types = []
    for row in rows:
        field_names.append(row[1])
        field_types.append(row[2])
    
    #now get all data
    cur.execute('SELECT * FROM {}'.format(table_name))
       
    rows = cur.fetchall()
    con.close()
    
    #make dictionary
    dictionary = {k:[] for k in field_names}
    for row in rows:
        for field_name, field_type in zip(field_names, field_types):
            dictionary[field_name].append(row[field_name])
        

    return dictionary
    

def get_itl_dict(db_path):
    print("Getting data from itl table")
    table_name = "obs_from_itl"
    d_itl = get_all_rows_from_db(db_path, table_name)
    
    #sort itl dictionary by tc20_exec_start
    sort_indices = [i[1] for i in sorted((e,i) for i,e in enumerate(d_itl["tc20_exec_start"]))]
    for key in d_itl.keys():
        d_itl[key] = [d_itl[key][i] for i in sort_indices]
    
    return d_itl

