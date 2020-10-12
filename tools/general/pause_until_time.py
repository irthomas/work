# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:17:35 2020

@author: iant
"""

import datetime
import time
import numpy as np

FORMAT_STR_SECONDS = "%Y-%m-%d %H:%M:%S"

def check_if_after_ref_time(interval, hour=0, minute=0, second=0):
    """wait until reference time is reached, by waiting for n intervals of length interval seconds"""
    
    #get current time
    now = datetime.datetime.now()
    #replace by midnight of the same day
    ref_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    #add 1 day
    ref_time += datetime.timedelta(days=1)
    #calculate seconds until next midnight
    seconds = (ref_time - now).seconds
    
    #calculate number of intervals until reference time
    n_intervals = int(np.ceil(seconds / interval))

    #loop through intervals, when limit is reached then exit function
    for interval_number in range(n_intervals):
        
        print("Time is %s; waiting for %s" \
              %(datetime.datetime.strftime(now, FORMAT_STR_SECONDS), \
                datetime.datetime.strftime(ref_time, FORMAT_STR_SECONDS)))
        time.sleep(interval)
    return True
    
