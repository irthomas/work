# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:30:03 2021

@author: iant
"""


TESTING = False
#TESTING = True

import os
import re
import sys
import h5py
import sqlite3
import logging
from datetime import datetime
from collections import OrderedDict
import platform

if platform.system() == "Windows":
    OBS_TYPE_DB = "obs_type.db"
    MITL_PATH = "."
    TESTING = True

if not TESTING:
    from nomad_ops.config import OBS_TYPE_DB, MITL_PATH



__project__   = "NOMAD"
__author__    = "Bram Beeckman"
__contact__   = "bram.beeckman@aeronomie.be"

#OBS_TYPE_DB = "/home/brambe/obs_type.db"

ITL_DT_FORMAT = "%Y %b %d %H:%M:%S"
PDHU_FILENAME_RE = re.compile("(?<=(?:XF500A01|XF001A01) = )([\dA-F]{6})")
TC20_START_RE = re.compile("(?<= TC20 execution start time \(UTC\): )\d{4}\s[A-Z]{3}\s\d{2}\s\d{2}\:\d{2}\:\d{2}")
OBS_CH_ORB_RE = re.compile("(?:#\s)(.*)(?:\swith\s)(LNO|SO|UVIS)(?:\sand\s)*(UVIS)*(?:\schannel\(s\) in orbit nr\. )(\d{4,5})")

MITL_RE = re.compile("MITL_M\d{3}_NOMAD\.ITL")

LNO_SO_OBS = {
    "FULLSCAN" :    {("Dayside Nadir","Nightside Nadir","Limb","Dayside Limb","Dayside True Limb") : "F",
                    ("Ingress Solar Occultation","Egress Solar Occultation","Merged Solar Occultation","Grazing Solar Occultation") : "S",
                    ("Calibration") : "C", 
                    ("Nightside Nadir") : "N", 
                    ("Nightside Limb","Nightside True Limb") : "O",},
                     
    "SCIENCE" :     {("Dayside Nadir")  : "D",
                    ("Nightside Nadir") : "N",
                    ("Dayside Limb","Dayside True Limb") : "L",
                    ("Nightside Limb","Nightside True Limb") : "O",
                    ("Ingress Solar Occultation","Merged Solar Occultation","Grazing Solar Occultation") : "I",
                    ("Egress Solar Occultation") : "E",
                    ("Phobos") : "P",
                    ("Deimos") : "M",
                    ("Calibration") : "C"},
    "MINISCAN" :    {("Calibration") : "C"},
    "CALIBRATION" : {("Solar line scan","Calibration") : "C"} # Use get() to default to "C" if key is not found
}

UVIS_OBS = {
    ("Dayside Nadir") : "D",
    ("Nightside Nadir") : "N",
    ("Ingress Solar Occultation","Merged Solar Occultation","Grazing Solar Occultation") : "I",
    ("Egress Solar Occultation") : "E",
    ("Dayside Limb") : "TBD", # See
    ("Dayside True Limb") : "L",
    ("Nightside True Limb") : "O",
    ("Solar line scan","Calibration") : "C"
}


SPECIAL_OBS = ["Phobos", "Deimos"]

DELTA_LIMIT = 120

logger = logging.getLogger( __name__ )
handler = logging.StreamHandler(sys.stdout)

class NomadItlDB(object):
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            open(db_path, 'w').close()
            self.create_itl_db(db_path)
        self.__db_path = db_path
        self._conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    @classmethod
    def create_itl_db(cls, db_path):
        cache_db = cls(db_path)
        cache_db._conn.execute( """CREATE TABLE itl_files (
                                filename text primary key,
                                date_modified timestamp
                                )""" )
        cache_db._conn.execute( """CREATE TABLE obs_from_itl (
                                tc20_exec_start timestamp primary key,
                                orbit text,
                                tc20_obs_type text,
                                channels text,
                                pdhu_filename text
                                )""" )
        cache_db._conn.execute( """CREATE TABLE IF NOT EXISTS obs_type (
                                tc20_exec_start timestamp,
                                channel text,
                                orbit text,
                                pdhu_filename text,
                                obs_type text,
                                PRIMARY KEY (tc20_exec_start, channel)
                                )""" )
        cache_db._conn.commit()
        return cache_db

    def close(self):
        self._conn.close()

    def insert_obs(self, obs):
        self._conn.execute("INSERT INTO obs_from_itl VALUES (:tc20_start,:orbit,:obs_desc,:channels,:pdhu_filename)", obs)
        self._conn.commit()

    def insert_obs_letter(self, itl_obs, dt, ch, letter):
        # (datetime, channel, orbit, pdhu filename, obs letter)
        obs = (dt, ch, itl_obs[1], itl_obs[4], letter)
        self._conn.execute("INSERT OR IGNORE INTO obs_type VALUES (?,?,?,?,?)", obs)
        self._conn.commit()

    def insert_obs_file(self, itl_file):
        # (filename, st_mtime)
        self._conn.execute("INSERT INTO itl_files VALUES (?,?)", itl_file)
        self._conn.commit()

    def get_itl_files(self):
        c = self._conn.execute("SELECT * FROM itl_files")
        return set(c.fetchall())

    def get_obs_type(self, dt, ch):
        c = self._conn.execute("SELECT * FROM obs_type WHERE tc20_exec_start == :dt AND channel == :ch", {"dt" : dt, "ch" : ch})
        return c.fetchone()

    def clear_itl_tables(self):
        self._conn.execute("DROP TABLE itl_files")
        self._conn.execute("DROP TABLE obs_from_itl")
        self._conn.commit()
        self.create_itl_db(self.__db_path)

    def clear_obs_type_table(self):
        self._conn.execute("DROP TABLE obs_type")
        self._conn.execute( """CREATE TABLE obs_type (
                            tc20_exec_start timestamp,
                            channel text,
                            orbit text,
                            pdhu_filename text,
                            obs_type text,
                            PRIMARY KEY (tc20_exec_start, channel)
                            )""" )
        self._conn.commit()

    def get_closest_obs(self, dt, ch):
        query = """SELECT *, ABS(strftime('%s',v) - strftime('%s',:dt)) AS delta FROM (
                    SELECT *, MIN(tc20_exec_start) AS v FROM obs_from_itl WHERE tc20_exec_start >= :dt
                    UNION SELECT *, MAX(tc20_exec_start) AS v FROM obs_from_itl WHERE tc20_exec_start <= :dt)
                WHERE channels LIKE :ch
                ORDER BY delta LIMIT 1"""
        c = self._conn.execute(query, {"dt" : dt, "ch" : "%"+ch+"%"})
        return c.fetchone()

def clear_obs(db_path):
    logger.info("Clearing the observation type table from the database.")
    db_obj = NomadItlDB(db_path)
    db_obj.clear_obs_type_table()
    db_obj.close()

def parse_itl(itl_file):
    # { "PDHU_filename" :
    #   [ { "tc_20_start" : datetime(),
    #       "obs_desc" : "Egress Solar Occultation",
    #       "channels": ["LNO", "UVIS"],
    #       "orbit" : "XXXX"}
    #   ]
    # }
    itl_dict = OrderedDict()
    curr_dict_key = "dummy"
    itl_dict[curr_dict_key] = []
    try:
        for line in open(itl_file):
            # if "Phobos" in line:
            #     print(line)
            fname = PDHU_FILENAME_RE.search(line.rstrip('\n'))
            if fname:
                curr_dict_key = fname.group()
                if curr_dict_key not in itl_dict: # MTP000 duplicate filenames
                    itl_dict[curr_dict_key] = []
            else:
                obs = OBS_CH_ORB_RE.findall(line.rstrip('\n'))
                if obs:
                    obs = obs[0]
                    # print(obs)
                    for special_obs_name in SPECIAL_OBS:
                        if special_obs_name in obs[0]:
                            obs = list(obs)
                            obs[0] = special_obs_name
                            print(obs)
                    itl_dict[curr_dict_key].append({})
                    itl_dict[curr_dict_key][-1]["obs_desc"] = obs[0]
                    itl_dict[curr_dict_key][-1]["orbit"] = obs[3]
                    if obs[2]:
                        itl_dict[curr_dict_key][-1]["channels"] = ', '.join(obs[1:3])
                    else:
                        itl_dict[curr_dict_key][-1]["channels"] = obs[1]

                else:
                    dt = TC20_START_RE.search(line.rstrip('\n'))
                    if dt:
                        ts = datetime.strptime(dt.group(), ITL_DT_FORMAT)
                        itl_dict[curr_dict_key][-1]["tc20_start"] = ts
    except Exception as e:
        logger.error(e)

    del itl_dict["dummy"]
    return OrderedDict({k: v for k, v in itl_dict.items() if v})

def get_tup(tups, el):
    res = [i for i in tups if i[0] == el] # Alternatively use dict(tups) and dict.get(el)
    if len(res) == 1:
        return res[0]
    else:
        logger.error("Multiple tuple candidates.")
        return None

def construct_db(db_path):
    db_obj = NomadItlDB(db_path)
    # List of local files and their mtime
    itl_files = set()
    for f in os.scandir(MITL_PATH):
        if MITL_RE.match(f.name):
            itl_files.add((f.path, datetime.fromtimestamp(f.stat().st_mtime)))
    # Compare local with db
    itl_files_db = db_obj.get_itl_files()
    if itl_files - itl_files_db:
        # Check if a new itl file exists (append) or if there was a modification (regenerate)
        new_files = {t[0] for t in itl_files} - {t[0] for t in itl_files_db}
        if new_files:
            # New file, append to database
            for f in new_files:
                logger.info("New ITL file %s found, inserting into database." % f)
                itl_dict = parse_itl(f)
                obs_list = [{**obs, **{"pdhu_filename" : k}} for k,v in itl_dict.items() for obs in v]
                for obs in obs_list:
                    db_obj.insert_obs(obs)
                db_obj.insert_obs_file(get_tup(itl_files, f))
        else:
            logger.info("Found discrepancies between database and ITL files, regenerating...")
            print(*(itl_files-itl_files_db), sep='\n')
            # Clean databases and rescan ITL files
            db_obj.clear_itl_tables()
            for tup in itl_files:
                logger.info("Inserting ITL file %s into database." % tup[0])
                itl_dict = parse_itl(tup[0])
                obs_list = [{**obs, **{"pdhu_filename" : k}} for k,v in itl_dict.items() for obs in v]
                for obs in obs_list:
                    db_obj.insert_obs(obs)
                db_obj.insert_obs_file(tup)
    else:
        # Keep the database as-is if an ITL file was removed
        pass
    db_obj.close()

def deduce_obs(db_obj, dt, ch):
    # db_res = (tc20 datetime, orbit, obs type, channels, pdhu, v, delta)
    db_res = db_obj.get_closest_obs(dt, ch)
    if db_res[6] > DELTA_LIMIT:
        logger.error("Closest observation delta exceeded.")
        return None
    # Get rid of the text after "Calibration"
    if "Calibration" in db_res[2]:
        return (db_res[0], db_res[1], "Calibration", db_res[3], db_res[4], db_res[5], db_res[6])
    return db_res

def get_obs_type(hdf5_file):
    h5f = h5py.File(hdf5_file, 'r')
    hdf5_name = os.path.basename(hdf5_file)
    db_obj = NomadItlDB(OBS_TYPE_DB)
    m_dt = re.match("\d{8}_\d{6}", hdf5_name)
    m_ch = re.match("(?:.*)(SO|LNO|UVIS)(?:_\d{1})?\.h5", hdf5_name)
    dt = datetime.strptime(m_dt.group(), "%Y%m%d_%H%M%S")
    ch = m_ch.group(1)
    if dt < datetime(2016,1,1):
        return (dt, ch, -1, "", "C")
    itl_obs = deduce_obs(db_obj, dt, ch)

    try:
        # Check database if observation type letter was already determined
        db_obs = db_obj.get_obs_type(dt, ch)
        if db_obs:
            return db_obs
        # Determine observation type letter and insert in database
        elif ch in ["SO", "LNO"]:
            n_sub = h5f.attrs["NSubdomains"]
            desc = h5f.attrs["Desc"]
            if n_sub == 1:
                # Check for stepping
                if "AOTF_IX" in desc:
                    t_str = "FULLSCAN"
                elif "AOTF_FREQ" in desc:
                    t_str = "MINISCAN"
                else:
                    t_str = "CALIBRATION"
            elif n_sub == 0:
                logger.error("No subdomains for %s" % hdf5_file)
                return None
            else:
                t_str = "SCIENCE"

            # Determine the observation type letter
            for k in LNO_SO_OBS[t_str].keys():
                if itl_obs[2] in k:
                    obs_letter = LNO_SO_OBS[t_str].get(k, "C") # Default to C
                    logger.info("Observation type %s found for %s" % (obs_letter, itl_obs))
                    db_obj.insert_obs_letter(itl_obs, dt, ch, obs_letter)
                    return db_obj.get_obs_type(dt, ch)
            logger.error("No observation type letter found for %s" % str(itl_obs))

        elif ch == "UVIS":
            mode = h5f["Channel/Mode"][0]
            acq_mode = h5f["Channel/AcquisitionMode"][0]
            cop_row = [int(i) for i in re.match(".*(?=\s#)", h5f.attrs["Desc"]).group().split(',')]
            dark_so_steps = cop_row[16]
            dark_nadir_steps = cop_row[17]
            flag_register = h5f["Channel/FlagRegister"][0]
            ccd_calibration = bin(flag_register)[-3]

            if dark_so_steps == 0 and dark_nadir_steps == 0:
                t_str = "CALIBRATION"
            elif mode > 2:
                t_str = "CALIBRATION"
            elif ccd_calibration == 1:
                t_str = "CALIBRATION"
            elif acq_mode in [1,2]:
                t_str = "BINNED"
            elif acq_mode == 0:
                t_str = "UNBINNED"
            else:
                logger.error("UVIS acquistion mode unknown.")
                return None

            if t_str == "CALIBRATION":
                obs_letter = "C"
            else:
                # Determine the observation type letter
                for k in UVIS_OBS.keys():
                    if itl_obs[2] in k:
                        if itl_obs[2] == "Dayside Limb" and mode == 1:
                            obs_letter = "L"
                        elif itl_obs[2] == "Dayside Limb" and mode == 2:
                            obs_letter = "D"
                        else:
                            obs_letter = UVIS_OBS.get(k, "C") # Default to C

            if obs_letter:
                logger.info("Observation type %s found for %s" % (obs_letter, itl_obs))
                db_obj.insert_obs_letter(itl_obs, dt, ch, obs_letter)
                return db_obj.get_obs_type(dt, ch)
            logger.error("No observation type letter found for %s" % itl_obs)
        else:
            logger.error("Channel could not be determined from filename.")
            return None

    except Exception as e:
        logger.error(e.args[0])

    finally:
        db_obj.close()



if TESTING:
    print("Making db")
    # clear_obs(OBS_TYPE_DB)
    construct_db(OBS_TYPE_DB)
