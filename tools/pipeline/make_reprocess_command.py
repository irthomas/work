# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:50:00 2023

@author: iant
"""

from datetime import datetime, timedelta




def h5_filename_to_datetime(h5):
    
    dt = datetime(int(h5[0:4]), int(h5[4:6]), int(h5[6:8]), int(h5[9:11]), int(h5[11:13]), int(h5[13:15]))
    return dt




def make_reprocess_command(level_from, level_to, h5_prefix=None, h5_exectime=None, filter_=None, filepath=None, seconds=30):        

    """print and/or save to file the command for reprocessing an observation in the data pipeline
    Inputs are:
        Either the h5_prefix e.g. 20230222_151005 or the execution time if no H5 file is available
        level_from and level_to in the form "hdf5_l01a" or "hdf5_l10a"
        If a filepath is given, the command is appended to the file
        seconds in the +- seconds to search for the file in the datastore
    """
    
    if h5_prefix:
        dt = h5_filename_to_datetime(h5_prefix)
    
    elif h5_exectime:
        dt = datetime.strptime(h5_exectime, "%Y-%m-%d %H:%M:%S")
    
    else:
        print("Error: must give prefix or execution time")
        
    dt_start = datetime.strftime(dt - timedelta(seconds=seconds), "%Y-%m-%dT%H:%M:%S")
    dt_end = datetime.strftime(dt + timedelta(seconds=seconds), "%Y-%m-%dT%H:%M:%S")
    
    if filter_:
        command = f"./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from {level_from} --to {level_to} --beg {dt_start} --end {dt_end} --n_proc=1 --filter='{filter_}' --all"
    else:
        command = f"./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from {level_from} --to {level_to} --beg {dt_start} --end {dt_end} --n_proc=1 --all"
    
    if filepath:
        with open(filepath, "a") as f:
            f.write("%s\n" %command)
    else:
        print(command)
