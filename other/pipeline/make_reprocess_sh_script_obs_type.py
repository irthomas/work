# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:21:53 2022

@author: iant

MAKE REPROCESSING SHELL SCRIPT FOR A SPECIFIC OBSERVATION TYPE
"""

import posixpath
import paramiko
import os
from datetime import datetime, timedelta
import sqlite3
import re
import platform

from tools.sql.read_itl_db import get_itl_dict
from tools.file.passwords import passwords


# dev
# nomadr = False
# profile = "ian_0p1d"

# prod
nomadr = True
profile = ""

log_level = "INFO"
# log_level = "WARNING"


if nomadr:
    command = f"./scripts/run_as_nomadr ./scripts/run_pipeline.py --log {log_level} make --from %s --to %s --beg %s --end %s --filter='%s' --n_proc=8 --all\n"
else:
    command = "scripts/run_pipeline.py --profile {profile} --log {log_level} make --from %s --to %s --beg %s --end %s --filter='%s' --n_proc=8 --all\n"

SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"
REMOTE_DATA_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/"
HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"


# get list of file datetimes to reprocess from ITL db or from a hdf5 level cache.db?
# this is the level to search for matching filenames, not the level to start the pipeline
# e.g. use this to select specific obs types to reprocess for 0.1a where the letter is not yet present in the filename

# source = "ITL"
source = "hdf5_level_0p2a"
# source = "hdf5_level_0p2a"
# source = "hdf5_level_1p0a"

# hdf5_filter = ".*(_SO|_LNO)_._S.*"
# hdf5_filter = ".*_SO_._S.*"
# hdf5_filter = ".*_C.*"
# hdf5_filter = ".*_LNO_._CM.*"
# hdf5_filter = ".*_UVIS_GI.*"
hdf5_filter = ".*_UVIS_[PQ]"


transfer_db = True
# transfer_db = False

start_dt = datetime(2018, 4, 21, 12, 0, 0)


levels = [
    # {"in":"inserted",  "out":"raw",       "filter":""},
    # {"in":"raw",       "out":"hdf5_l01a", "filter":""},
    # {"in":"hdf5_l01a", "out":"hdf5_l01d", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01d", "out":"hdf5_l01e", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01e", "out":"hdf5_l02a", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01a", "out":"hdf5_l02a", "filter":".*UVIS.*"},
    # {"in":"hdf5_l02a", "out":"hdf5_l02b", "filter":".*UVIS.*"},
    # {"in":"hdf5_l02b", "out":"hdf5_l03b", "filter":".*UVIS.*"},
    # {"in":"hdf5_l03b", "out":"hdf5_l03c", "filter":".*UVIS_D.*"},
    # {"in":"hdf5_l03c", "out":"hdf5_l10a", "filter":".*UVIS_D.*"},
    # {"in":"hdf5_l02a", "out":"hdf5_l03a", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l03a", "out":"hdf5_l10a", "filter":".*(SO|LNO).*"},


    # {"in": "hdf5_l01a", "out": "hdf5_l10a", "filter": ".*(SO|LNO).*"},

    # {"in":"hdf5_l01a", "out":"hdf5_l10b", "filter":".*(_SO|_LNO).*"},
    # {"in":"hdf5_l01a", "out":"hdf5_l10b", "filter":".*_UVIS.*"},

    {"in": "hdf5_l01a", "out": "hdf5_l03c", "filter": ".*_UVIS.*"},


]


s = "#!/bin/bash\n\n"
s += '#generated automatically by make_reprocess_sh_script_obs_type.sh\n\n'
s += '#cd /bira-iasb/projects/NOMAD/Instrument/SOFTWARE-FIRMWARE/nomad_ops\n\n'


def transfer_cache_db(level):
    """copy cache db for chosen level from remote path to local computer"""
    localpath = os.path.join(SHARED_DIR_PATH, "db", level + ".db")

    remotepath = posixpath.join(REMOTE_DATA_PATH, level, "cache.db")

    print("Connecting to hera")
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect("hera.oma.be", port=22, username="iant", password=passwords["hera"])

    sftp = p.open_sftp()
    print("Downloading cache.db for level %s" % level)
    sftp.get(remotepath, localpath)
    sftp.close()
    p.close()


def get_filenames_from_cache(level):
    """get filenames from cache.db"""

    def connect_db(db_path):
        print("Connecting to database %s" % db_path)
        con = sqlite3.connect(db_path)
        return con

    def close_db(con):
        con.close()

    print("Getting data for level %s from cache.db" % level)
    localpath = os.path.join(SHARED_DIR_PATH, "db", level + ".db")

    con = connect_db(localpath)
    cur = con.cursor()
    cur.execute('SELECT path FROM files')
    rows = cur.fetchall()
    filepaths = [filepath[0] for filepath in rows]
    close_db(con)
    filenames = [os.path.split(filepath)[1] for filepath in filepaths]

    channels = [re.search("(SO|LNO|UVIS)", filename).group() for filename in filenames]
    return channels, filenames


# def make_reprocessing_script(datetime_strings):
#     """print a list of reprocessing lines to add to shell script
#     Copy to script x.sh in master branch nomad_ops directory, then run using command ./x.sh"""

#     pipeline_filter = ".*UVIS.*"
#     pipeline_from = "hdf5_l03c"
#     pipeline_to = "hdf5_l10a"
#     pipeline_nproc = 8
#     #make reprocessing script
#     for file_datetime_string in datetime_strings:
#         file_datetime = datetime.strptime(file_datetime_string, HDF5_FILENAME_FORMAT)

#         file_minus_5_mins = file_datetime - timedelta(minutes=5)
#         file_plus_5_mins = file_datetime + timedelta(minutes=5)

#         pipeline_beg = datetime.strftime(file_minus_5_mins, "%Y-%m-%dT%H:%M:%S")
#         pipeline_end = datetime.strftime(file_plus_5_mins, "%Y-%m-%dT%H:%M:%S")

#         print("python3 scripts/run_pipeline.py --log INFO make --from %s --to %s --beg %s --end %s --all --n_proc=%i --filter=\"%s\"" \
#               %(pipeline_from, pipeline_to, pipeline_beg, pipeline_end, pipeline_nproc, pipeline_filter))


def make_obs_start_end_dts(hdf5_dt_strs, start_dt):

    pipeline_begs = []
    pipeline_ends = []

    for file_datetime_string in hdf5_dt_strs:
        file_datetime = datetime.strptime(file_datetime_string, HDF5_FILENAME_FORMAT)

        # ignore all entries before this date
        if file_datetime > start_dt:

            file_minus_5_mins = file_datetime - timedelta(minutes=5)
            file_plus_5_mins = file_datetime + timedelta(minutes=5)

            pipeline_beg = datetime.strftime(file_minus_5_mins, "%Y-%m-%dT%H:%M:%S")
            pipeline_end = datetime.strftime(file_plus_5_mins, "%Y-%m-%dT%H:%M:%S")

            pipeline_begs.append(pipeline_beg)
            pipeline_ends.append(pipeline_end)

    return pipeline_begs, pipeline_ends

    # print("python3 scripts/run_pipeline.py --log INFO make --from %s --to %s --beg %s --end %s --all --n_proc=%i --filter=\"%s\"" \
    #       %(pipeline_from, pipeline_to, pipeline_beg, pipeline_end, pipeline_nproc, pipeline_filter))


if source == "ITL":

    obs_type = "Calibration"

    d_itl = get_itl_dict(os.path.join(SHARED_DIR_PATH, "db", "obs_type.db"))

    indices = [i for i, v in enumerate(d_itl["tc20_obs_type"]) if obs_type in v]

    exec_times = [v for i, v in enumerate(d_itl["tc20_exec_start"]) if i in indices]

    with open("missing_calibrations.txt", "w") as f:
        for exec_time in exec_times:

            start = str(exec_time - timedelta(seconds=60)).replace(" ", "T")
            end = str(exec_time + timedelta(seconds=60)).replace(" ", "T")

            f.write(command % (start, end))


elif "hdf5" in source:

    if transfer_db:
        transfer_cache_db(source)

    channels, hdf5_filenames = get_filenames_from_cache(source)

    # apply filter
    re_filter = re.compile(hdf5_filter)
    hdf5_filenames_filtered = [s for s in hdf5_filenames if re_filter.match(s)]

    # get dt strings from filenames
    hdf5_dt_strs = [s[0:15] for s in hdf5_filenames_filtered]

    # remove duplicates
    hdf5_dt_strs_unique = sorted(list(set(hdf5_dt_strs)))

    # find start/end times +- N minutes
    begs, ends = make_obs_start_end_dts(hdf5_dt_strs_unique, start_dt)

    s += 'python3 scripts/pipeline_log.py "Starting reprocessing, filter=%s"\n' % (hdf5_filter)
    s += 'python3 scripts/check_pipeline_log.py\n'

    print("%i files found" % len(begs))

    for beg, end in zip(begs, ends):
        for level in levels:

            # reprocess all
            s += command % (level["in"], level["out"], beg, end, level["filter"])

    s += 'python3 scripts/pipeline_log.py "Reprocessing done, filter=%s"\n' % (hdf5_filter)
    s += 'python3 scripts/check_pipeline_log.py\n'


if platform.system() == "Windows":
    shell_output_path = "reprocessing_script_obstype.sh"
else:
    shell_output_path = "/home/iant/reprocessing_script_obstype.sh"

print("Writing shell script to %s" % shell_output_path)
with open(shell_output_path, "w") as f:
    f.write(s)
