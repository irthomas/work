# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:51:09 2022

@author: iant
"""

import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

import platform

if platform.system() == "Windows":
    shell_output_path = "reprocessing_script.sh"
else:
    shell_output_path = "/home/iant/reprocessing_script.sh"

#dev
# nomadr = False
# profile = "ian"

#prod
nomadr = True
profile = ""

# log_level = "INFO"
log_level = "WARNING"



start_month = "10-2019"
end_month = "11-2023" #not included


# level_start = "raw"
# level_end = "hdf5_l01a"

# level_start = "raw"
# level_end = "hdf5_l02a"

# level_start = "hdf5_l01a"
# level_end = "hdf5_l02a"

level_start = "hdf5_l02a"
level_end = "hdf5_l10a"

# level_start = "hdf5_l01a"
# level_end = "hdf5_l10a"

# level_start = "hdf5_l10a"
# level_end = "hdf5_l10b"

# level_start = "hdf5_l03k"
# level_end = "hdf5_l10a"


# filter_ = "........_......_.*_(SO|LNO)_._[IEGS].*"
filter_ = ".*"

n_proc = 8

#delete_cache # on first run should the cache be deleted?
# delete_cache = True
delete_cache = False


delete = True
# delete = False


if nomadr:
    command_prefix = "./scripts/run_as_nomadr ./"
else:
    command_prefix = "python3 "
    

if profile == "":
    profile_text = ""
else:
    profile_text = "--profile ian "




key = (level_start, level_end)


### enter all intermediate levels here
levels_to_delete_d = {
    ("raw", "hdf5_l01a"):["hdf5_l01a"],
    ("raw", "hdf5_l02a"):["hdf5_l01a", "hdf5_l01d", "hdf5_l01e", "hdf5_l02a"],
    ("hdf5_l01a", "hdf5_l02a"):["hdf5_l01d", "hdf5_l01e", "hdf5_l02a"],
    ("hdf5_l02a", "hdf5_l10a"):["hdf5_l02b", "hdf5_l03a", "hdf5_l03b", "hdf5_l03c", "hdf5_l03i", "hdf5_l03j", "hdf5_l03k", "hdf5_l10a"],
    ("hdf5_l01a", "hdf5_l10a"):["hdf5_l01d", "hdf5_l01e", "hdf5_l02a", "hdf5_l02b", "hdf5_l03a", "hdf5_l03b", "hdf5_l03c", "hdf5_l03i", "hdf5_l03j", "hdf5_l03k", "hdf5_l10a"],
    ("hdf5_l03k", "hdf5_l10a"):["hdf5_l10a"],
    ("hdf5_l10a", "hdf5_l10b"):["hdf5_l10b"],
}

if key in levels_to_delete_d.keys():
    levels_to_delete = levels_to_delete_d[key]
else:
    print("Error: intermediate levels not yet defined")
    sys.exit()


start_dt = datetime.strptime(start_month, "%m-%Y")
end_dt = datetime.strptime(end_month, "%m-%Y")

dt = start_dt

loop = 0

with open(shell_output_path, "w") as f:

    f.write("\n")
    f.write("#!/bin/bash\n")
    f.write("#generated by other.pipeline.make_reprocessing_sh_script.py\n")
    f.write("\n")

    f.write('python3 scripts/check_pipeline_log.py\n')
    
    while dt < end_dt:
        month_start = datetime.strftime(dt, "%m")
        year_start = datetime.strftime(dt, "%Y")
        
        month_end = datetime.strftime(dt + relativedelta(months=1), "%m")
        year_end = datetime.strftime(dt + relativedelta(months=1), "%Y")
        
        start = "%s-%s-01" %(year_start, month_start)
        end = "%s-%s-01" %(year_end, month_end)
        
        if delete:
            f.write('python3 scripts/pipeline_log.py "Deleting levels %s to %s for time period %s to %s"\n' %(level_start, level_end, start, end))
            for level_to_delete in levels_to_delete:
                if loop == 0 and delete_cache:
                    f.write(command_prefix + 'scripts/delete_month_data.py %s %s %s -delete_cache\n' %(month_start, year_start, level_to_delete))
                else:
                    f.write(command_prefix + 'scripts/delete_month_data.py %s %s %s\n' %(month_start, year_start, level_to_delete))
            f.write("\n")


        f.write('python3 scripts/pipeline_log.py "Starting reprocessing of levels %s to %s for time period %s to %s"\n' %(level_start, level_end, start, end))
        f.write(command_prefix + 'scripts/run_pipeline.py %s--log %s make --from %s --to %s --beg %s --end %s --filter="%s" --n_proc=%i --all\n' \
              %(profile_text, log_level, level_start, level_end, start, end, filter_, n_proc))
        
        f.write('python3 scripts/check_pipeline_log.py\n')
        f.write('python3 scripts/pipeline_log.py "Reprocessing levels %s to %s for time period %s to %s done"\n' %(level_start, level_end, start, end))
        f.write('sleep 2\n')
        f.write("\n")
        f.write("\n")
        f.write("\n")
    
        dt += relativedelta(months=1)
        loop += 1
     

