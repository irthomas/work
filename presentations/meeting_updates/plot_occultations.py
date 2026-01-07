# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:55:48 2024

@author: iant

PLOT SOLAR OCCULTATION COVERAGE FOR EACH MARTIAN YEAR FROM THE OBS DATABASE
"""


import os
import decimal
from datetime import datetime
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from tools.general.get_mars_year_ls import get_mars_year_ls


def connect_db(db_path):
    print("Connecting to database %s" % db_path)
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return con


def close_db(con):
    con.close()


def query(con, input_query):
    # print(input_query)
    cur = con.cursor()

    c = cur.execute(input_query)
    con.commit()
    output = c.fetchall()
    return output


def read_table(con, table_name):
    query_string = "SELECT * FROM %s" % table_name
    table = query(con, query_string)

    new_table_data = []
    for row in table:
        new_table_data.append([float(element) if type(element) == decimal.Decimal else element for element in row])

    return new_table_data


# get data from planning db
con = connect_db(r"C:/Users/iant/Documents/PROGRAMS/nomad_obs/planning.db")

rows = read_table(con, "occultations")

close_db(con)

d = {"my": [], "ls": [], "lat": [], "lon": [], "lst": [], "my_ls": []}
for row in rows:
    if row[1] == "NOMAD":
        dt = row[6]

        # ignore merged occultations
        if dt < datetime(2001, 1, 1):
            continue

        my, ls = get_mars_year_ls(dt)

        d["my"].append(my)
        d["ls"].append(ls)
        d["lat"].append(row[13])
        d["lon"].append(row[10])
        d["lst"].append(row[15])

        d["my_ls"].append(my * 360.0 + ls)

for key in d.keys():
    d[key] = np.asarray(d[key])


unique_mys = list(set(d["my"]))

for unique_my in unique_mys:
    my_ixs = np.where(d["my"] == unique_my)[0]

    ls_my = d["ls"][my_ixs]
    lat_my = d["lat"][my_ixs]

    plt.scatter(ls_my, lat_my, alpha=0.1, label="MY%02i" % unique_my)

plt.grid()
plt.title("NOMAD Solar Occultation Coverage")
plt.xlabel("Ls")
plt.ylabel("Latitude")
leg = plt.legend(loc="lower right")
plt.ylim((-91, 91))
plt.xlim((-1, 361))

for lh in leg.legendHandles:
    lh.set_alpha(1)
