# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:21:45 2023

@author: iant

READ IN SQL DB, SMOOTH AND OUTPUT N INTERPOLATED POINTS TO NEW DB FOR THE WEBSITE



"""

import os
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tools.sql.sql import sql_db

from tools.file.paths import paths
from tools.general.progress_bar import progress

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


CLEAR_DB = True
# CLEAR_DB = False

SMOOTH_STEPS = 99


db_path = os.path.join(paths["DATASTORE_ROOT_DIRECTORY"], "reports", "exports", "heaters_temp", "heaters_temp.db")
# db_path = os.path.join(r"C:\Users\iant\Documents\DATA\db", "heaters_temp.db")


with sql_db(db_path) as db:
    # TODO: get last date from existing db to append only the latest values

    # get first and last entries
    print("Querying database for start and end dates")
    first_ts = db.query("SELECT ts FROM heaters_temp ORDER BY ts ASC LIMIT 1")[0][0]
    last_ts = db.query("SELECT ts FROM heaters_temp ORDER BY ts DESC LIMIT 1")[0][0]

    # make date at start of first month
    start_month = first_ts.date().replace(day=1)
    # make date at start of month after the last month
    end_month = last_ts.date().replace(day=1) + relativedelta(months=+1)

    months = []
    month = start_month
    while month < end_month:
        month += relativedelta(months=+1)
        months.append(month)

    # start_month = datetime(2023, 10, 1).date()
    # end_month = datetime(2016, 6, 1).date()

    ts = []
    lno_nominal = []

    print("Getting monthly data from database from %s to %s" % (months[0], months[-1]))
    # loop through months, getting timestamp and temperatures
    for month in progress(months):
        # print(month)
        like_str = "'%04i-%02i-%%'" % (month.year, month.month)
        rows = db.query("SELECT ts, lno_nominal FROM heaters_temp WHERE ts LIKE %s" % like_str)

        ts.extend([row[0] for row in rows])
        lno_nominal.extend([row[1] for row in rows])

# convert to arrays
lno_nominal = np.asarray(lno_nominal)
ts = np.asarray(ts)

# smooth data then select entries every N values
lno_smooth = savgol_filter(lno_nominal, SMOOTH_STEPS, 1)
lno_small = lno_smooth[::SMOOTH_STEPS]
ts_small = ts[::SMOOTH_STEPS]

# write out to new sqlite db
print("Writing to new database file")
with sql_db("red_heaters_temp.db") as new_db:
    if CLEAR_DB:
        print("Clearing existing database if it exists")
        new_db.query("DROP TABLE IF EXISTS temps")
        new_db.query("CREATE TABLE temps (id INTEGER PRIMARY KEY AUTOINCREMENT, ts datetime NOT NULL, lno decimal NOT NULL)")

    # loop through each line, adding temperatures to db
    for i, (ts, lno) in enumerate(progress(list(zip(ts_small, lno_small)))):
        # if np.mod(i, 1000) == 0:
        #     print("%i/%i" % (i, ts_small.size))
        new_db.query(["INSERT INTO temps (ts, lno) VALUES (?, ?)", (ts, lno)])

plt.figure(figsize=(20, 6), constrained_layout=True)
plt.title("NOMAD LNO Temperature")
plt.plot(ts_small, lno_small)
plt.grid()
plt.xlabel("Date")
plt.ylabel("Temperature of LNO (nominal)")
plt.savefig("nomad_temperature_full_mission.png")
