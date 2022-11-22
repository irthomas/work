# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:26:34 2021

@author: iant

PSA CAL LOG DB FUNCTIONS
"""



import sqlite3 as sql
import os
import hashlib

from nomad_ops.core.psa_transfer.config import PATH_PSA_LOG_DB

from nomad_ops.core.psa_transfer.get_psa_logs import \
    get_log_list, extract_log_info, get_log_datetime, get_log_version



    
    
def connect_db(db_path):
    print("Connecting to db %s" %db_path)
    con = sql.connect(db_path, detect_types=sql.PARSE_DECLTYPES)
    return con

def close_db(con):
    con.close()



def md5sum(filepath):
    
    md5 = hashlib.md5(open(filepath,'rb').read()).hexdigest()
    return md5






def get_db_rows(con, table_name):
    cur = con.cursor()
    cur.execute("SELECT * FROM {}".format(table_name))


    rows = cur.fetchall()
    # close_db(con)
    return rows




def empty_table(con, table_name):
    """delete table and rebuild empty"""
    print("Deleting and rebuilding table", table_name)

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS {}".format(table_name))
    
    if table_name == "logs":
        query = """CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            md5 TEXT NOT NULL, \
            version TEXT NOT NULL) """
        
        cur.execute(query)

    elif table_name == "pass":
        query = """CREATE TABLE pass (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            version TEXT NOT NULL, \
            lid TEXT NOT NULL) """

        cur.execute(query)

    elif table_name == "fail":
        query = """CREATE TABLE fail (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            version TEXT NOT NULL, \
            lid TEXT NOT NULL) """

        cur.execute(query)



def delete_rows(con, table_name, log_filename):
    """delete log entries from table"""

    cur = con.cursor()
    cur.execute("DELETE FROM {} WHERE log=?".format(table_name), (log_filename,))
    con.commit()



def make_db(clear=False):
    
    print("Making PSA calibration log database")
    
    log_filepath_list = get_log_list()
    populate_log_db(log_filepath_list, clear=clear)
    





def populate_log_db(log_filepath_list, clear=False):
    """read in psa log files and add entries to db"""
   
    
    db_path = os.path.join(PATH_PSA_LOG_DB)
    if not os.path.exists(db_path):
        print("%s doesn't exist: creating" %db_path)
        clear = True
    con = connect_db(db_path)
    
    
    if clear:
        print("clearing tables from %s" %db_path)
        for table_name in ["logs", "pass", "fail"]:
            empty_table(con, table_name)
    
    #get data from table of existing log filenames
    rows = get_db_rows(con, "logs")
    existing_logs = {row[1]:row[2] for row in rows} #make dict: log name:md5 checksum
    cur = con.cursor()
    
    for new_log_path in log_filepath_list:
        new_log = os.path.basename(new_log_path)
    
        md5 = md5sum(new_log_path)
        

        log_datetime = get_log_datetime(new_log_path)
        log_version = get_log_version(log_datetime)
        
        
        if new_log not in existing_logs.keys():
            add_log = True

            print("Adding new log %s to db" %new_log)
            #add to list and parse
            cur.execute("INSERT INTO logs (log, version, md5) VALUES (?,?,?)", (new_log, log_version, md5))



            #if log already in db, check md5 matches
        elif md5 == existing_logs[new_log]:
            add_log = False
            #already in db
            print("Log %s already in db" %new_log)
        
        else:
            add_log = True

            #file different -> remove and reprocess
            print("Log %s in db but md5 mismatch" %new_log)
            for table_name in ["logs", "pass", "fail"]:
                delete_rows(con, table_name, new_log)

            

        if add_log:
            pass_dict, fail_dict = extract_log_info(new_log_path)
    
            print("Pass: Adding %i lids to %s" %(len(pass_dict.keys()), new_log_path))
            for lid in pass_dict.keys():
                cur.execute("INSERT INTO pass (log, version, lid) VALUES (?,?,?)", (new_log, pass_dict[lid]["version"], lid))

            print("Fail: Adding %i lids to %s" %(len(fail_dict.keys()), new_log_path))
            for lid in fail_dict.keys():
                cur.execute("INSERT INTO fail (log, version, lid) VALUES (?,?,?)", (new_log, fail_dict[lid]["version"], lid))
    
            con.commit()
            
            
                
            
    con.commit()
    close_db(con)
    
    
    
    

def get_lids_of_version(table_name, version):
    
    db_path = os.path.join(PATH_PSA_LOG_DB)
    con = connect_db(db_path)

    cur = con.cursor()
    cur.execute("SELECT lid FROM {} WHERE version IS ?".format(table_name), (version,))

    rows = cur.fetchall()
    close_db(con)
    
    lids = [i[0] for i in rows]
    return lids
