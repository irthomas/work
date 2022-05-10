# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:51:09 2022

@author: iant
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

start_month = "04-2018"
end_month = "03-2019"
level_start = "hdf5_l01a"
level_end = "hdf5_l02a"

levels_to_delete = ["hdf5_l01d", "hdf5_l01e", "hdf5_l02a"]

start_dt = datetime.strptime(start_month, "%m-%Y")
end_dt = datetime.strptime(end_month, "%m-%Y")

dt = start_dt

loop = 0

while dt < end_dt:
    month_start = datetime.strftime(dt, "%m")
    year_start = datetime.strftime(dt, "%Y")
    
    month_end = datetime.strftime(dt + relativedelta(months=1), "%m")
    year_end = datetime.strftime(dt + relativedelta(months=1), "%Y")
    
    start = "%s-%s-01" %(year_start, month_start)
    end = "%s-%s-01" %(year_end, month_end)
    print('python3 scripts/pipeline_log.py "Deleting levels %s to %s for time period %s to %s"' %(level_start, level_end, start, end))
    for level_to_delete in levels_to_delete:
        if loop == 0:
            print('./scripts/run_as_nomadr ./scripts/delete_month_data.py %s %s %s -delete_cache' %(month_start, year_start, level_to_delete))
        else:
            print('./scripts/run_as_nomadr ./scripts/delete_month_data.py %s %s %s' %(month_start, year_start, level_to_delete))

    print('python3 scripts/pipeline_log.py "Reprocessing levels %s to %s for time period %s to %s"' %(level_start, level_end, start, end))
    print('./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from %s --to %s --beg %s --end %s --n_proc=8 --all' \
          %(level_start, level_end, start, end))
    
    print('python3 scripts/check_pipeline_log.py')
    print('python3 scripts/pipeline_log.py "Reprocessing levels %s to %s for time period %s to %s done"' %(level_start, level_end, start, end))
    print('sleep 2')
    print("")

    dt += relativedelta(months=1)
    loop += 1
 
       
