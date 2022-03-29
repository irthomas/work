# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:52:03 2022

@author: iant

LIST CALIBRATION FILES AND MAKE REPROCESSING TIMES
"""

import os
from datetime import timedelta

from tools.sql.read_itl_db import get_itl_dict


SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"



d_itl = get_itl_dict(os.path.join(SHARED_DIR_PATH, "db", "obs_type.db"))

indices = [i for i,v in enumerate(d_itl["tc20_obs_type"]) if "Calibration" in v]

exec_times = [v for i,v in enumerate(d_itl["tc20_exec_start"]) if i in indices]

command = "scripts/run_pipeline.py --profile ian --log INFO make --from hdf5_l01a --to hdf5_l10a --beg %s --end %s --n_proc=8 --all\n"
# command = "./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from hdf5_l01a --to hdf5_l10a --beg %s --end %s --n_proc=8 --all\n"
with open("missing_calibrations.txt", "w") as f:
    for exec_time in exec_times:
        
        start = str(exec_time - timedelta(seconds=60)).replace(" ", "T")
        end = str(exec_time + timedelta(seconds=60)).replace(" ", "T")
    
        f.write(command %(start, end))
