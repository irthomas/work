# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:58:01 2022

@author: iant

GET DATA FROM CACHE.DB FILES
"""

import sqlite3
import re
import os



def connect_db(db_path):
    print("Connecting to database %s" %db_path)
    con = sqlite3.connect(db_path)
    return con

def close_db(con):
    con.close()

def get_filenames_from_cache(db_path):
    """get filenames from cache.db"""
    
    print("Getting data from %s" %os.path.basename(db_path))
    localpath = os.path.join(db_path)
    
    con = connect_db(localpath)
    cur = con.cursor()
    cur.execute('SELECT path FROM files')
    rows = cur.fetchall()
    filepaths = [filepath[0] for filepath in rows]
    close_db(con)
    filenames = [os.path.split(filepath)[1] for filepath in filepaths]
    
    if "spacewire" in os.path.basename(db_path):
        return [] * len(filenames), filenames

    channels = [re.search("(SO|LNO|UVIS)", filename).group() for filename in filenames]
    return channels, filenames


