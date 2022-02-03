# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:53:13 2022

@author: iant
"""

import os
import sqlite3
import re
import matplotlib.pyplot as plt

# from datetime import datetime, timedelta

# from tools.file.paths import paths
# from tools.file.list_files import list_files



SHARED_DIR_PATH = r"C:\Users\iant\Dropbox\NOMAD\Python\web_dev\shared"


level = "hdf5_level_1p0a"
regex = re.compile("(....)(..).._......_1p0a_SO_._(?:\w_(\d*)|\w)\.h5")


def connect_db(db_path):
    print("Connecting to database %s" %db_path)
    con = sqlite3.connect(db_path)
    return con

def close_db(con):
    con.close()


def get_filenames_from_cache(level):
    """get filenames from cache.db"""
    
    print("Getting data for level %s from cache.db" %level)
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



channels, filepaths = get_filenames_from_cache(level)

filenames = [os.path.basename(f) for c,f in zip(channels, filepaths) if c == "SO"]

regexes = [re.findall(regex, f)[0] for f in filenames]
filenames_split = [[int(i) for i in s] for s in regexes if s[2]]



#split by order, then by year-month

unique_orders = sorted(list(set([s[2] for s in filenames_split])))

labels = []
for year in range(2018, 2022):
    for month in range(1, 13):
        # print(month, year)
        labels.append("%02i/%04i" %(month, year))

obs = {}
for unique_order in unique_orders:
    
    obs[unique_order] = []

    for year in range(2018, 2023):
        for month in range(1, 13):
            # print(month, year)
            
            #find all files matching month and year
            
            orders = [1 for s in filenames_split if s[0] == year and s[1] == month and s[2] == unique_order]
            obs[unique_order].append(sum(orders))

max_obs = max([max(n) for n in obs.values()])

orders = [119, 121, 134, 136]

fig, axes = plt.subplots(figsize=(13,8), nrows=len(orders), sharex=True, constrained_layout=True)
x_pos = [i for i, _ in enumerate(labels)]

fig.suptitle("Number of observations per diffraction order")

for i, order in enumerate(orders):
    
    axes[i].grid()
    axes[i].bar(x_pos, obs[order])
    axes[i].set_ylim([0, max_obs+10])
    axes[i].text(0.01, 0.85, order, transform = axes[i].transAxes)

# plt.xlabel("Month")
axes[-1].set_xlabel("Month")

axes[-1].set_xticks(x_pos)
axes[-1].set_xticklabels(labels, rotation=90)