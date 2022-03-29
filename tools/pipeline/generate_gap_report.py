# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:05:59 2022

@author: iant


WRITE GAP REPORT FOR .EXM FILES

BEFORE RUNNING ENSURE THAT CACHE.DB AND REAL FILE TREE ARE IDENTICAL (RUN COMPARE_TREE_TO_CACHE.PY WITH CORRECT LEVEL)
"""

import os
import numpy as np

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib


from tools.sql.transfer_dbs import transfer_cache_db, transfer_obs_type_db
from tools.sql.read_cache_db import get_filenames_from_cache
from tools.sql.read_itl_db import get_itl_dict
from tools.file.esac_remote_tree import get_esac_tree_filenames


SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"


level = "hdf5_level_0p1a"



#update local cache db
transfer_cache_db(level)
#update local obs type db
transfer_obs_type_db()
#update local spacewire cache db
transfer_cache_db("spacewire")


#read cache.db
cache = get_filenames_from_cache(os.path.join(SHARED_DIR_PATH, "db", level + ".db"))
cache_filenames = sorted([s.replace(".h5","") for s in cache[1]])


#read spacewire cache.db
cache = get_filenames_from_cache(os.path.join(SHARED_DIR_PATH, "db", "spacewire.db"))
cache_spacewire_filenames = sorted([s.replace(".EXM","") for s in cache[1]])



#read obs_type db
d_itl = get_itl_dict(os.path.join(SHARED_DIR_PATH, "db", "obs_type.db"))



#read ESAC tree filenames
esac = get_esac_tree_filenames("iant", "spacewire")
esac_filenames = sorted([s.replace(".EXM","") for s in esac])



#check for mismatch between ESAC and spacewire cache.db
def return_not_matching(a, b):
    """compare list 1 to list 2 and return non matching entries"""
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

a,b = return_not_matching(cache_spacewire_filenames, esac_filenames)




#convert filenames to datetime
filename_dts = []
filenames = []
for filename in cache_filenames:
    dt = datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
    # if dt > datetime(2018, 3, 14, 12, 30, 0):
    if dt > datetime(2018, 4, 21):
        filenames.append(filename)
        filename_dts.append(dt)


#define possible timedelta range
timedelta_ranges = [
    [datetime(2018, 3, 1), datetime(2018, 4, 1), -120, 0],
    [datetime(2018, 4, 1), datetime(2018, 4, 5), -80, 20],
    [datetime(2018, 4, 5), datetime(2018, 8, 25), -10, 20],
    [datetime(2018, 8, 25), datetime(2018, 11, 5), -5, 50],
    [datetime(2018, 11, 5), datetime(2021, 9, 1), -25, 25],
    [datetime(2021, 9, 1), datetime(2025, 1, 1), -35, 35],
]


#define known off-periods
off_periods = [
    [datetime(2018, 6, 1, 13, 0, 0), datetime(2018, 6, 2, 1, 0, 0), "Malargue ground station failure", True],
    [datetime(2019, 5, 18, 10, 14, 0), datetime(2019, 5, 25, 19, 0, 0), "TGO safe mode", True],
    [datetime(2020, 3, 28, 17, 0, 0), datetime(2020, 4, 11, 12, 0, 0), "Covid-19 at MOC", True],
    [datetime(2020, 6, 21, 10, 0, 0), datetime(2020, 6, 21, 11, 0, 0), "MITL change, data received", False],
    [datetime(2020, 6, 23, 9, 0, 0), datetime(2020, 6, 23, 10, 0, 0), "MITL change, data received", False],
] 


#go through itl observations

#first, find first match between itl and tree
i = -1
j = -1

matches = {}


#loop through 
print("Comparing ITL db and %s cache.db filenames" %level)
while i < len(filename_dts) - 1:
    
    i += 1
    found = False

    while not found:
        
        # i += 1
        j += 1
        
        
        itl_dt = d_itl["tc20_exec_start"][j]
        channels = d_itl["channels"][j].split(", ")
        
        filename_dt = filename_dts[i]
        filename = filenames[i]
    
        tdelta = (itl_dt - filename_dt).total_seconds()
        
        #get timedelta range:
        for timedelta_range in timedelta_ranges:
            if timedelta_range[0] < filename_dt < timedelta_range[1]:
                start = timedelta_range[2]
                end = timedelta_range[3]
                
        
        if start < tdelta < end:
            for channel in channels:
                if channel in filename:
                    if np.mod(i, 5000) == 0:
                        print(tdelta, itl_dt, channel, filename)
                    key = "%s_%s" %(itl_dt, channel)
                    matches[key] = [filename, tdelta, channel]
                    
                    found = True
                    j -= 3


print("Finding missing observations")
missing_dts = {}
for i, (itl_dt, channels) in enumerate(zip(d_itl["tc20_exec_start"], d_itl["channels"])):
    if np.mod(i, 1000) == 0:
        print(itl_dt, channels)
    if itl_dt > datetime(2018, 4, 21): #only consider from start of mission
        if itl_dt < max(filename_dts): #ignore future observations
            for channel in channels.split(", "):
                key = "%s_%s" %(itl_dt, channel)
                if key not in matches.keys():
                    if itl_dt in missing_dts.keys():
                        missing_dts[itl_dt]["channels"].append(channel)
                    else:
                        missing_dts[itl_dt] = {"channels":[], "reason":[]}
                        missing_dts[itl_dt]["channels"] = [channel]


colour_dict = {
    "tab:blue":["LNO"],
    "tab:green":["SO"],
    "tab:red":["UVIS"],
    "tab:purple":["LNO", "UVIS"],
    "tab:brown":["SO", "UVIS"],
    }


fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax.set_title("Timeline of missing %s files" %level)
for colour, channels in colour_dict.items():
    idx = [i for i,(k,v) in enumerate(missing_dts.items()) if v["channels"] == channels]
    print(", ".join(channels), len(idx))
    ax.scatter([dt for i,dt in enumerate(missing_dts.keys()) if i in idx], np.zeros(len(idx)), color=colour, label=", ".join(channels))
    
    texts = [k for i,(k,v) in enumerate(missing_dts.items()) if v["channels"] == channels]
    for text in texts:
        ax.text(text, 0.1, text, rotation=90)
ax.legend()
ax.set_ylim([-0.1, 1])
ax.set_xlabel("Observation Date")
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=1))
ax.tick_params(axis='x', labelrotation=90)
ax.grid()
fig.savefig("missing_file_timeline.png")


def match_pdhu_filename(dt, d_itl):
    """get start of PDHU filenames from ITL db"""
    #e.g. SCI__DNMD__03479A01_2022-001T16-36-43__00001.EXM
    
    if dt in d_itl["tc20_exec_start"]:
        pdhu_match = [d_itl["pdhu_filename"][i] for i,v in enumerate(d_itl["tc20_exec_start"]) if v == dt]
        if len(pdhu_match) == 1:
            
            dt_minus1 = dt - timedelta(days=1)
            #make filename for day and day before (in case file starts before midnight)
            pdhu_strs = [
                "SCI__DNMD__03%s_%04i-%sT" %(pdhu_match[0], dt.year, dt.strftime('%j')),
                "SCI__DNMD__03%s_%04i-%sT" %(pdhu_match[0], dt_minus1.year, dt_minus1.strftime('%j')),
            ]
        else:
            print("Error: multiple PDHU filenames found")
            pdhu_strs = ["",""]
    else:
        print("Error: Execution time not found")
        pdhu_strs = ["",""]

    return pdhu_strs


print("Writing missing files to file")
# h = "<!DOCTYPE html><html><head><title>%s GAP REPORT</title>\n</head><body>\n" %level.upper()
h = ""
h += "<table border=1>\n"
h += "<tr><th>Expected TC20 Execution Time</th><th>Channel(s) Affected</th><th>Reason for Failure</th><th>Expected EXM filename prefix (if missing)</th></tr>\n"
for missing_dt in missing_dts.keys():

    reason = ""
    pdhu_filenames = ["",""]

    #check for known off-periods
    include = True
    for off_period in off_periods:
        if off_period[0] < missing_dt < off_period[1]:
            include = off_period[3]
            reason = off_period[2]
            
    #if not a known off-period
    if reason == "":
        pdhu_filenames = match_pdhu_filename(missing_dt, d_itl)
        
        #now check if PDHU substrings appear in spacewire cache db
        strings1 = [
            [s for s in cache_spacewire_filenames if pdhu_filenames[0] in s], 
            [s for s in cache_spacewire_filenames if pdhu_filenames[1] in s],
        ]

        #now check if PDHU substring in ESAC tree
        strings2 = [
            [s for s in esac_filenames if pdhu_filenames[0] in s],
            [s for s in esac_filenames if pdhu_filenames[0] in s],
        ]
        
        #if neither string is found in cache or at ESAC
        if (len(strings1[0]) == 0 and len(strings1[1]) == 0) and (len(strings2[0]) == 0 and len(strings2[0]) == 0):
            reason = "EXM not found in datastore or on ESAC server"

        #if neither string is found in cache but one is found at ESAC
        elif (len(strings1[0]) == 0 and len(strings1[1]) == 0) and (len(strings2[0]) == 1 or len(strings2[0]) == 1):
            reason = "EXM not found in datastore but is present on ESAC server"

        #if neither string is found in cache but two or more are found at ESAC
        elif (len(strings1[0]) == 0 and len(strings1[1]) == 0) and (len(strings2[0]) > 1 or len(strings2[0]) > 1):
            reason = "EXM not found in datastore but multiple versions are present on ESAC server"

        #if either string is found in cache
        elif (len(strings1[0]) == 1 or len(strings1[1]) == 1):
            reason = "EXM file received, HDF5s not generated due to another error"

        #if either string is found in cache multiple times
        elif (len(strings1[0]) > 1 or len(strings1[1]) > 1):
            reason = "Multiple EXM files found in datastore, HDF5s not generated due to another error"
            print(missing_dt, strings1)

        else:
            reason = "Another error occurred"
        

    if include:
        channels = missing_dts[missing_dt]["channels"]
        h += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" %(missing_dt, ", ".join(channels), reason, pdhu_filenames[0])
        
    missing_dts[missing_dt]["reason"] = reason

h += "</table><br>\n"
h += "Made by tools.pipeline.generate_gap_report.py<br>\n"
h += "Last updated %s\n" %str(datetime.now())[:19]
# h += "</body></html>"

with open("missing_%s.html" %level, "w") as f:
    f.writelines(h)


command = "./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from inserted --to hdf5_l10a --beg %s --end %s --n_proc=8 --all\n"
with open("missing_%s_commands.txt" %level, "w") as f:
    for missing_dt, channels_reasons in missing_dts.items():
        
        if "EXM file received, HDF5s not generated" in channels_reasons["reason"]:
            
            start = str(missing_dt - timedelta(seconds=60)).replace(" ", "T")
            end = str(missing_dt + timedelta(seconds=60)).replace(" ", "T")
    
            f.write(command %(start, end))




#now check if available on ESAC server
