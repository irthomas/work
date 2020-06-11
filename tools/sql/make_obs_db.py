# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:20:45 2020

@author: iant

HEATERS_TEMP_DB IS NOW CREATED BY PIPELINE
TGO TEMPERATURES ARE NOW IN HDF5 FILES. GET FROM FILE INSTEAD OF MAKING DB

python3 tools/sql/obs_db_functions.py lno_nadir hdf5_level_0p3a 2018-03-01 2030-01-01 --regenerate=True
python3 tools/sql/obs_db_functions.py lno_nadir hdf5_level_1p0a 2018-03-01 2018-04-01 --regenerate=True

runfile('C:/Users/iant/Dropbox/NOMAD/Python/tools/sql/make_obs_db.py', args='lno_nadir hdf5_level_1p0a 2018-03-01 2018-07-01 -regenerate --regex=".*LNO.*_134"')
python3 tools/sql/make_obs_db.py lno_nadir hdf5_level_1p0a 2018-03-01 2021-01-01 -regenerate

python3 tools/sql/obs_db_functions.py so_occultation hdf5_level_1p0a 2018-03-01 2030-01-01 --regenerate=True

"""
#import datetime
#import re

from tools.sql.obs_database import obs_database

#import sys
#sys.path.append(r"C:\Users\iant\Dropbox\NOMAD\Python")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help='Enter command: so_occultation, lno_nadir')
    parser.add_argument('level', type=str, help='Enter full level name: e.g. hdf5_level_0p3a')
    parser.add_argument('beg', type=str, help='Enter start date YYYY-MM-DD')
    parser.add_argument('end', type=str, help='Enter end date YYYY-MM-DD')
    parser.add_argument('-regenerate', action='store_true', help='Delete table and regenerate. Always use --regenerate=True on first run')
    parser.add_argument('-silent', action='store_true', help='Output messages')
    parser.add_argument('--regex', type=str, default="", help='Output messages')
    args = parser.parse_args()
    command = args.command
else:
    command = ""




print("Running command", command)

if command != "":
    dbName = "%s_%s" %(command, args.level)
    db_obj = obs_database(dbName)
    db_obj.process_channel_data(args)
    db_obj.close()

