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
    get_log_list, extract_log_info, get_versions_from_zip, get_log_datetime, get_log_version



    
    
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
            versions TEXT NOT NULL) """
        
        cur.execute(query)

    elif table_name == "pass":
        query = """CREATE TABLE pass (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            versions TEXT NOT NULL, \
            lid TEXT NOT NULL) """

        cur.execute(query)

    elif table_name == "fail":
        query = """CREATE TABLE fail (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            versions TEXT NOT NULL, \
            lid TEXT NOT NULL) """

        cur.execute(query)

    elif table_name == "error":
        query = """CREATE TABLE error (id INTEGER PRIMARY KEY AUTOINCREMENT, \
            log TEXT NOT NULL, \
            versions TEXT NOT NULL, \
            lid TEXT NOT NULL, \
            error TEXT NOT NULL) """
        
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
        clear = True
    con = connect_db(db_path)
    
    
    if clear:
        for table_name in ["logs", "pass", "fail", "error"]:
            empty_table(con, table_name)
    
    rows = get_db_rows(con, "logs")
    existing_logs = {row[1]:row[2] for row in rows}
    # existing_md5s = [row[2] for row in rows]
    cur = con.cursor()
    
    previous_versions = []
    
    for new_log_path in log_filepath_list:
        new_log = os.path.basename(new_log_path)
    
        md5 = md5sum(new_log_path)
        
        if new_log not in existing_logs.keys():
            print("Adding new log %s to db" %new_log)
            #add to list and parse
            
            log_dict = extract_log_info(new_log_path)
            
            unique_versions1 = get_versions_from_zip(log_dict["zip_filenames_received"])
            unique_versions2 = get_versions_from_zip(log_dict["zip_filenames_transferred"])
            unique_versions3 = get_versions_from_zip(log_dict["zip_filenames_expanded"])
            unique_versions = sorted(list(set(unique_versions1 + unique_versions2 + unique_versions3)))
            
            
            #if no version found in file, use version from previous file
            if len(unique_versions) == 0:
                print("Warning: no version numbering found in file. Using previous file version")
                unique_versions = previous_versions
            else:
                previous_versions = unique_versions.copy()

            log_datetime = get_log_datetime(new_log_path)
            log_version = get_log_version(log_datetime)
            versions_str = ",".join(unique_versions)
            if log_version != versions_str:
                print("Error: log version %s does not match expected %s" %(log_version, versions_str))


            
            
                    
            cur.execute("INSERT INTO logs (log, md5, versions) VALUES (?,?,?)", (new_log, md5, versions_str))
    
            for lid in log_dict["validator_lids_pass"]:
                cur.execute("INSERT INTO pass (log, versions, lid) VALUES (?,?,?)", (new_log, versions_str, lid))
            for lid in log_dict["validator_lids_fail"]:
                cur.execute("INSERT INTO fail (log, versions, lid) VALUES (?,?,?)", (new_log, versions_str, lid))
            for lid, error in zip(log_dict["validator_lids_error"], log_dict["validator_errors"]):
                cur.execute("INSERT INTO error (log, versions, lid, error) VALUES (?,?,?,?)", (new_log, versions_str, lid, error))
    
            con.commit()
            
            # stop()
            
        else:
            #if log already in db, check md5 matches
            if md5 == existing_logs[new_log]:
                #already in db
                print("Log %s already in db" %new_log)
            
            else:
                #file different -> remove and reprocess
                print("Log %s in db but md5 mismatch" %new_log)
                
                for table_name in ["logs", "pass", "fail", "error"]:
                    delete_rows(con, table_name, new_log)
    
                print("Adding new log %s to db" %new_log)
                #add to list and parse
                
                log_dict = extract_log_info(new_log_path)
                
                unique_versions = get_versions_from_zip(log_dict["zip_filenames_received"])
                #if no version found in file, use version from previous file
                if len(unique_versions) == 0:
                    print("Warning: version not found in new file, getting version from UPLOAD_DATES")
                    log_datetime = get_log_datetime(new_log_path)
                    log_version = get_log_version(log_datetime)
                    versions_str = log_version
                else:
                    versions_str = ",".join(unique_versions)
                
                        
                cur.execute("INSERT INTO logs (log, md5, versions) VALUES (?,?,?)", (new_log, md5, versions_str))
        
                for lid in log_dict["validator_lids_pass"]:
                    cur.execute("INSERT INTO pass (log, versions, lid) VALUES (?,?,?)", (new_log, versions_str, lid))
                for lid in log_dict["validator_lids_fail"]:
                    cur.execute("INSERT INTO fail (log, versions, lid) VALUES (?,?,?)", (new_log, versions_str, lid))
                for lid, error in zip(log_dict["validator_lids_error"], log_dict["validator_errors"]):
                    cur.execute("INSERT INTO error (log, versions, lid, error) VALUES (?,?,?,?)", (new_log, versions_str, lid, error))
            
                con.commit()
                
            
    con.commit()
    close_db(con)
    
    
    
    

def get_lids_of_version(table_name, version):
    
    db_path = os.path.join(PATH_PSA_LOG_DB)
    con = connect_db(db_path)

    cur = con.cursor()
    cur.execute("SELECT lid FROM {} WHERE versions IS ?".format(table_name), (version,))

    rows = cur.fetchall()
    close_db(con)
    
    lids = [i[0] for i in rows]
    return lids


