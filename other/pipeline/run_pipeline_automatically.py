#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:10:20 2020

@author: iant

RUN CONTINUOUSLY - UPDATE DATASTORE AT MIDNIGHT

#STEPS: 
1. DOWNLOAD SPICE KERNELS
2. GET DATA FROM SERVER
3. SORT INTO DB DIRECTORIES
3. CHECK WHERE SCI DATA ENDS (WITH MARGIN) OR 10 DAYS ELAPSED
4. CHECK IF CORRESPONDING MTL IN DB
5. IF NOT, GET NEW FILE FROM GIT MTPXXX_NOMAD BRANCH (OR STOP)
6. ADD MTL TO DB (OR STOP)
7. INSERT PSA PAR
7a. CHECK IF HEATERS_TEMP_DB IS UP TO DATE
7b. IF NOT, UPDATE DB
8. MAKE ALL FILES
9. MAKE REPORTS
10.SEND EMAIL IF ERROR
11.TRANSFER TO FTP
12.MAKE PSA FILES
13.DOWNLOAD AND CHECK PSA LOGS FOR ERRORS
13.TRANSFER TO ESA NEW FILES AND PREVIOUSLY FAILED FILES


"""
#TESTING=True
TESTING=False

import os
import sys
import smtplib
import textwrap
import datetime
import time
import subprocess
from email.mime.text import MIMEText
import posixpath
import sqlite3
import ftplib
import platform

#SIMULATE = True
WAIT_FOR_MIDNIGHT = False

SIMULATE = False
#WAIT_FOR_MIDNIGHT = True


MAKE_REPORTS = True
MAKE_EXPORTS = True
MAKE_HDF5 = True
FTP_UPLOAD = True

LOGGER_LEVEL = "info"

SQL_UPDATE = True

if platform.system() == "Windows":
    TESTING = True #always


if TESTING:
    SIMULATE = True
    OBS_TYPE_DB = r"C:\Users\iant\Documents\DATA\db\obs_type.db"
    PATH_EDDS_SPACEWIRE = r"C:\Users\iant\Documents\DATA\db"
    HDF5_L10A_FILE_DESTINATION = r"C:\Users\iant\Documents\DATA\db\l1"
    HDF5_RAW_FILE_DESTINATION = r"C:\Users\iant\Documents\DATA\db\l0"
    PATH_EXPORT_HEATERS_TEMP = r"C:\Users\iant\Documents\DATA\db"
    from tools.file.passwords import passwords
    
else:
#    SIMULATE = False
    os.environ["FS_MODEL"] = "False"
    os.environ["NMD_OPS_PROFILE"] = "default"
    sys.path.append('.')

    from nomad_ops.config import RUN_PIPELINE_LOG_DIR, OBS_TYPE_DB, \
        PATH_EDDS_SPACEWIRE, HDF5_L10A_FILE_DESTINATION, \
        HDF5_RAW_FILE_DESTINATION, PATH_EXPORT_HEATERS_TEMP

    with open("passwords.txt", "r") as f:  passwords = eval("".join(f.readlines()))
    

#print("Current directory = ", os.getcwd())

SENDER = "nomadr@aeronomie.be"
RECIPIENTS = ["ian.thomas@aeronomie.be"]
MESSAGE = """
    *** NOMAD PIPELINE STATUS ***

    There were errors during the last run.
    Please check the error log.

    Errors occured during:
    {}
"""

FORMAT_STR_DAYS = "%Y-%m-%d"
FORMAT_STR_SECONDS = "%Y-%m-%d %H:%M:%S"


def check_if_after_ref_time(interval, hour=0, minute=0, second=0):
    
    while True:
        now = datetime.datetime.now()
        ref_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        seconds = (now - ref_time).seconds

        if seconds < interval *2:
            return True
        else:
            print("Time is %s; waiting for %s" \
                  %(datetime.datetime.strftime(now, FORMAT_STR_SECONDS), \
                    datetime.datetime.strftime(ref_time, "%H:%M:%S")))
            time.sleep(interval)
    

def get_end_datetime(delta_days):
    end_dt = datetime.datetime.today() - datetime.timedelta(days=delta_days)
    return end_dt, datetime.date.strftime(end_dt, FORMAT_STR_DAYS)


def check_which_mtp(datetime_in):
    """find mtp start/end times and ls for an mtp"""
    
    mtp0Start = datetime.datetime(2018, 3, 24)
    mtpTimeDelta = datetime.timedelta(days=28)
    
    datetime_calc = mtp0Start
    mtpNumber = -1
    while datetime_in >= datetime_calc:
        mtpNumber += 1
        datetime_calc = datetime_calc + mtpTimeDelta
#        print(mtpNumber, datetime_calc)
        
    mtpStartDatetime = datetime_calc - mtpTimeDelta
    mtpEndDatetime = datetime_calc
    
    return mtpNumber, mtpStartDatetime, mtpEndDatetime




def check_for_itl(end_mtp):
    """check if itl for mtp can be found in database"""
    
    itl_present = False
    with sqlite3.connect(OBS_TYPE_DB, detect_types=sqlite3.PARSE_DECLTYPES) as db:
        db_out = db.execute( """SELECT filename FROM itl_files""")
        itl_filepaths = sorted([i[0] for i in db_out.fetchall()])
    #search for MTP in ITL db
    for itl_filepath in itl_filepaths:
        itl_filename = posixpath.split(itl_filepath)[1]
        if "MITL_M%03i_NOMAD" %end_mtp in itl_filename:
            itl_present = True
            print("####### %s for MTP%03i found in database" %(itl_filename, end_mtp))
    return itl_present



def check_ftp():
    
    last_day = datetime.datetime(2020, 1, 3)
    return last_day




def notify_mailer(errors):
    try:
        smtp_obj = smtplib.SMTP("smtp.oma.be", port=25)
        mesg = MIMEText(textwrap.dedent(MESSAGE).format('\n'.join(errors)))
        mesg['Subject'] = "*** NOMAD PIPELINE STATUS ***"
        mesg['From'] = SENDER
        mesg['To'] = ', '.join(RECIPIENTS)
        smtp_obj.send_message(mesg)
        print("Notified recipients of pipeline errors.")
    except smtplib.SMTPException:
        print("Error while sending mail.")

def check_log():
    return os.stat(os.path.join(RUN_PIPELINE_LOG_DIR, "run_pipeline.log")).st_mtime


repeat = True

while repeat:
    #wait for start time
    #check_if_after_ref_time(10, hour=15, minute=59, second=50)
    
    if WAIT_FOR_MIDNIGHT:
        check_if_after_ref_time(60 * 60) #wait for midnight, checking every 60 minutes
    print("####### Starting script at %s" %datetime.datetime.strftime(datetime.datetime.now(), FORMAT_STR_SECONDS))
    script_start_time = time.time()
    errs = []
    rp = ["scripts/run_pipeline.py", "--log", LOGGER_LEVEL]
    rp_info = ["scripts/run_pipeline.py", "--log", "INFO"]
    
    command = ["scripts/sync_bira_esa.py","--transfer","esa_to_bira"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
    
    
    
    command = rp_info + ["config", "sync"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    command = rp_info + ["config", "itl"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    #check ITL db for corresponding MTP
    end_datetime, end_datetime_string = get_end_datetime(10)
    end_mtp, mtp_start_datetime, mtp_end_datetime = check_which_mtp(end_datetime)
    print("####### End datetime %s is in MTP%03i" %(end_datetime_string, end_mtp))
    
    print("####### Checking ITL db for ITL file up to MTP%03i" %end_mtp)
    itl_present = check_for_itl(end_mtp)


    if not itl_present:
        errs.append("No ITL")
        print("####### Error: Update ITL")
        sys.exit()


    
    command = rp + ["make", "--to", "inserted"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    
    #check if data is available in db up to end_datetime
    #TODO: check if not just a random file but all data available
    edds_db = os.path.join(PATH_EDDS_SPACEWIRE, "cache.db")
    with sqlite3.connect(edds_db, detect_types=sqlite3.PARSE_DECLTYPES) as db:
        db_out = db.execute( """SELECT end_dtime FROM files""")
        edds_end_datetimes = [i[0] for i in db_out.fetchall()]
    edds_end_datetime = sorted(edds_end_datetimes)[-1]
    edds_end_datetime_minus_1_day = edds_end_datetime - datetime.timedelta(days=1)
    
    correct_end_datetime = min(edds_end_datetime_minus_1_day, end_datetime)
    correct_end_datetime_string = datetime.datetime.strftime(correct_end_datetime, FORMAT_STR_DAYS)
    print("####### EDDS last entry (-1 day): %s" %str(edds_end_datetime_minus_1_day))
    print("####### Date 10 days ago: %s" %end_datetime_string)
    if correct_end_datetime == edds_end_datetime_minus_1_day:
        print("####### EDDS backlog. Using %s as true end date" %correct_end_datetime_string)
    if correct_end_datetime == end_datetime:
        print("####### EDDS up to date. Using T-10 days (%s) as true end date" %correct_end_datetime_string)
    
            
    command = rp + ["insert_psa"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    
    #check where to start from, searching L1 for latest file
    raw_db = os.path.join(HDF5_RAW_FILE_DESTINATION, "cache.db")
    with sqlite3.connect(raw_db, detect_types=sqlite3.PARSE_DECLTYPES) as db:
        db_out = db.execute( """SELECT end_dtime FROM files""")
        l01a_end_datetimes = [i[0] for i in db_out.fetchall()]
    l01a_end_datetime = sorted(l01a_end_datetimes)[-1]
    l01a_end_datetime_minus_1_day = l01a_end_datetime - datetime.timedelta(days=1)
    l01a_correct_datetime_string = datetime.datetime.strftime(l01a_end_datetime_minus_1_day, FORMAT_STR_DAYS)
    
    
    command = rp + ["make", "--from", "inserted", "--to", "hdf5_l01a", "--beg", "%s" %l01a_correct_datetime_string, "--end", "%s" %correct_end_datetime_string, "--all", "--n_proc=8"]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    if MAKE_REPORTS:
        command = rp + ["make", "--from", "raw", "--to", "raw_reports"]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)
            
        command = rp + ["make", "--from", "raw", "--to", "raw_anomalies"]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)

    if MAKE_EXPORTS:    
        #scripts/manage_esa_data.py export nomad_pwr --from 2020-03-21 --to 2020-04-19
        #scripts/manage_esa_data.py export sc_deck_temp --from 2020-03-21 --to 2020-04-19
        #scripts/manage_esa_data.py export heaters_temp --from 2020-03-21 --to 2020-04-19

        exp = ["scripts/manage_esa_data.py", "export"]

        #subtract one day when making reports 
        #so that when new MTP starts the full report is made for the previous MTP
        end_datetime_minus_1day, _ = get_end_datetime(1)
        end_mtp_minus_1day, mtp_start_datetime_minus_1day, mtp_end_datetime_minus_1day = check_which_mtp(end_datetime)


        mtp_start_date = datetime.datetime.strftime(mtp_start_datetime_minus_1day, FORMAT_STR_DAYS)
        mtp_end_date = datetime.datetime.strftime(mtp_end_datetime_minus_1day, FORMAT_STR_DAYS)
        
        command = exp + ["nomad_pwr", "--from", mtp_start_date, "--to", mtp_end_date]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)
            
        command = exp + ["sc_deck_temp", "--from", mtp_start_date, "--to", mtp_end_date]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)

        command = exp + ["heaters_temp", "--from", mtp_start_date, "--to", mtp_end_date]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)
    
    ### update heaters_temp_db
    heaters_temp_db = os.path.join(PATH_EXPORT_HEATERS_TEMP, "heaters_temp.db")        
    with sqlite3.connect(heaters_temp_db, detect_types=sqlite3.PARSE_DECLTYPES) as db:
        db_out = db.execute( """SELECT ts FROM heaters_temp ORDER BY ts DESC LIMIT 1""")
        heaters_temp_end_datetime_tuple = db_out.fetchone()
    heaters_temp_end_datetime = heaters_temp_end_datetime_tuple[0]
    heaters_temp_datetime_minus_1_day = heaters_temp_end_datetime - datetime.timedelta(days=1)
    heaters_temp_correct_datetime_string = datetime.datetime.strftime(heaters_temp_datetime_minus_1_day, FORMAT_STR_DAYS)
    
    heaters_temp_today_string = datetime.datetime.strftime(datetime.datetime.now(), FORMAT_STR_DAYS)
    
    #run_pipeline.py extras --update_heaters_db --beg 2020-02-16 --end 2020-04-03
    command = rp + ["extras", "--update_heaters_db", "--beg", "%s" %heaters_temp_correct_datetime_string, "--end", "%s" %heaters_temp_today_string]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.run(command)
        err_log_ts = check_log()
        if check_log() > err_log_ts:
            errs.append(command_string)
    
    
    if MAKE_HDF5:
    
        l10a_db = os.path.join(HDF5_L10A_FILE_DESTINATION, "cache.db")
        with sqlite3.connect(l10a_db, detect_types=sqlite3.PARSE_DECLTYPES) as db:
            db_out = db.execute( """SELECT end_dtime FROM files""")
            l10a_end_datetimes = [i[0] for i in db_out.fetchall()]
        l10a_end_datetime = sorted(l10a_end_datetimes)[-1]
        l10a_end_datetime_minus_1_day = l10a_end_datetime - datetime.timedelta(days=1)
        l10a_correct_datetime_string = datetime.datetime.strftime(l10a_end_datetime_minus_1_day, FORMAT_STR_DAYS)
        
        
        command = rp + ["make", "--from", "hdf5_l01a", "--to", "hdf5_l10a", "--beg", "%s" %l10a_correct_datetime_string, "--end", "%s" %correct_end_datetime_string, "--all", "--n_proc=8"]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)

    if FTP_UPLOAD:
    
        SC_FTP_ADR = "ftp-ae.oma.be"
        SC_FTP_USR = "nomadadm"
        SC_FTP_PWD = passwords["nomadadm"]
        SC_FTP_ROOT = "/Data"
        SC_FTP_LEVEL = "hdf5_level_1p0a"
        
        def open_ftp(server_address, username, password):
            # Open an FTP connection with specified credentials
            ftp_conn = ftplib.FTP(server_address)
            try:
                ftp_conn.login(user=username, passwd=password)
            except ftplib.all_errors as e:
                print("FTP error ({0})".format(e.message))
            return ftp_conn
        
        
        def dir_list(ftp_conn, path):
            # Return a list of all files and subdirs in specified path (non-recursive)
            dirs = []
            files = []
            try:
                dir_cont = ftp_conn.mlsd(path)
            except ftplib.all_errors as e:
                print("FTP error ({0})".format(e.message))
            for i in dir_cont:
                if i[1]['type'] == "dir":
                    dirs.append(i[0])
                elif i[1]['type'] == "file":
                    files.append(i[0])
            return (dirs, files)
        
        
        def ftp_walk(ftp_conn, path):
            # Recursively scan FTP starting from path
            (dirs, files) = dir_list(ftp_conn, path)
            yield (path, dirs, files)
            for i in dirs:
                path = posixpath.join(path, i)
                yield from ftp_walk(ftp_conn, path)
                path = posixpath.dirname(path)
        
        def get_remote_list(ftp_conn, path, relative=False):
            # Return a flat list containing all paths present on the FTP
            file_list = []
            for i in ftp_walk(ftp_conn, path):
                if not i[1]:
                    if relative:
                        file_list += [posixpath.relpath(posixpath.join(i[0], j), path) for j in i[2]]
                    else:
                        file_list += [posixpath.join(i[0], j) for j in i[2]]
            return file_list
        
        
        ftp_conn = open_ftp(SC_FTP_ADR, SC_FTP_USR, SC_FTP_PWD)
        
        ftp_day_found = False
        ftp_end_datetime = correct_end_datetime
        while not ftp_day_found:
        
            year = "%04i" %ftp_end_datetime.year
            month = "%02i" %ftp_end_datetime.month
            day = "%02i" %ftp_end_datetime.day
            
            #a = get_remote_list(ftp_conn, "/Data/hdf5_level_1p0a/")
            ftp_path = posixpath.join(SC_FTP_ROOT, SC_FTP_LEVEL)
            ftp_list = list(ftp_conn.mlsd(path=ftp_path))
            ftp_items = [i[0] for i in ftp_list if i[0] not in [".",".."]]
            if year in ftp_items:
                ftp_path = posixpath.join(SC_FTP_ROOT, SC_FTP_LEVEL, year)
                ftp_list = list(ftp_conn.mlsd(path=ftp_path))
                ftp_items = [i[0] for i in ftp_list if i[0] not in [".",".."]]
                if month in ftp_items:
                    ftp_path = posixpath.join(SC_FTP_ROOT, SC_FTP_LEVEL, year, month)
                    ftp_list = list(ftp_conn.mlsd(path=ftp_path))
                    ftp_items = [i[0] for i in ftp_list if i[0] not in [".",".."]]
                    if day in ftp_items:
                        ftp_day_found = True
                        
            print("%s not found on ftp server. Checking day before" %datetime.datetime.strftime(ftp_end_datetime, FORMAT_STR_DAYS))
            ftp_end_datetime = ftp_end_datetime - datetime.timedelta(days=1)
        
        print("####### Last day on ftp is %s. Copying from day before" %datetime.datetime.strftime(ftp_end_datetime, FORMAT_STR_DAYS))
        
        ftp_end_datetime = ftp_end_datetime - datetime.timedelta(days=1)
        ftp_end_datetime_string = datetime.datetime.strftime(ftp_end_datetime, FORMAT_STR_DAYS)
        
        
        #transfer all files between two dates (overwrite last day to be sure)
        command = rp + ["transfer", "ftp", "--from", "hdf5_l10a", "--to", "hdf5_l10a", "--beg", "%s" %ftp_end_datetime_string]
        command_string = " ".join(command)
        print("#######", command_string)
        if not SIMULATE:
            subprocess.run(command)
            err_log_ts = check_log()
            if check_log() > err_log_ts:
                errs.append(command_string)
        
        
            
    #
    #command = rp + ["transfer", "bira_to_esa"]
    #command_string = " ".join(command)
    #print(command_string)
    #if not SIMULATE:
    #    subprocess.run(command)
    #    err_log_ts = check_log()
    #    if check_log() > err_log_ts:
    #        errs.append(command_string)
    
    if errs: 
        print("####### Errors found")
        if not SIMULATE:
            notify_mailer(errs)
    
    script_end_time = time.time()
    script_elapsed_time = script_end_time - script_start_time
    print("Processing finished at %s (duration = %s)" %(datetime.datetime.now(), str(datetime.timedelta(seconds=script_elapsed_time)).split(".")[0]))

    if SIMULATE:
        repeat = False
    else:
        time.sleep(600) #wait 10 minutes until next check
