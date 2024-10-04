# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:49:01 2024

@author: iant

CONVERT LIUZZI 2019 MY AND LS TO LAT/LON/LST FOR KAITO
"""

import os
import decimal
from datetime import datetime
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from tools.general.get_mars_year_ls import get_mars_year_ls


liuzzi_filepaths = ["Figure_5_ice_abundance_NH.dat", "Figure_5_ice_abundance_SH.dat", "Figure_5_dust_abundance_NH.dat", "Figure_5_dust_abundance_SH.dat"]


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

# get data from liuzzi repo files
for filepath in liuzzi_filepaths:

    if "_NH" in filepath:
        region = "N"

        # make new dict only for north
        d_plan = {}
        good_ix = [i for i, v in enumerate(d["lat"]) if v > 0.0]
        for key in d.keys():
            d_plan[key] = np.asarray([d[key][i] for i in good_ix])

    if "_SH" in filepath:
        region = "S"

        # make new dict only for south
        d_plan = {}
        good_ix = [i for i, v in enumerate(d["lat"]) if v < 0.0]
        for key in d.keys():
            d_plan[key] = np.asarray([d[key][i] for i in good_ix])

    d_liu = {"my": [], "ls": [], "my_ls": [], "values": []}

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == "#":
            continue

        line_split = line.split(",")

        my = float(line_split[0].strip())
        ls = float(line_split[1].strip())
        values = [float(s) for s in line_split[2:]]

        if np.all(np.isnan(values)):
            continue

        d_liu["my"].append(my)
        d_liu["ls"].append(ls)

        my_ls = my * 360.0 + ls

        d_liu["my_ls"].append(my_ls)
        d_liu["values"].append(np.asarray(values))

    for key in d_liu.keys():
        d_liu[key] = np.asarray(d_liu[key])

    # compare each liuzzi my ls to the planning db
    d_out = {"dls": [], "my_liu": [], "ls_liu": [], "my_ls_liu": [], "values_liu": [], "my_plan": [],
             "ls_plan": [], "my_ls_plan": [], "lat_plan": [], "lon_plan": [], "lst_plan": []}

    lines_out = ["# MY, Ls, Latitude, Longitude, Local solar time, Values [kg/kg*10-6] on an altitude grid from 0.5 to 110 km with a spacing of 0.5 km\n"]
    for i in np.arange(len(d_liu["my_ls"])):
        closest_ix = np.abs(d_liu["my_ls"][i] - d_plan["my_ls"]).argmin()

        dls = d_liu["my_ls"][i] - d_plan["my_ls"][closest_ix]

        if np.abs(dls) < 0.1:
            d_out["my_liu"].append(d_liu["my"][i])
            d_out["ls_liu"].append(d_liu["ls"][i])
            d_out["my_ls_liu"].append(d_liu["my_ls"][i])
            d_out["values_liu"].append(d_liu["values"][i])
            d_out["my_plan"].append(d_plan["my"][closest_ix])
            d_out["ls_plan"].append(d_plan["ls"][closest_ix])
            d_out["my_ls_plan"].append(d_plan["my_ls"][closest_ix])
            d_out["lat_plan"].append(d_plan["lat"][closest_ix])
            d_out["lon_plan"].append(d_plan["lon"][closest_ix])
            d_out["lst_plan"].append(d_plan["lst"][closest_ix])
            d_out["dls"].append(dls)

    # plt.scatter(d_out["my_ls_liu"], d_out["dls"])

    for i in np.arange(len(d_out["my_ls_liu"])):
        values = ", ".join([str(i) for i in d_out["values_liu"][i]])
        line = "%0.0f, %0.4f, %0.2f, %0.2f, %0.3f, %s\n" % (d_out["my_liu"][i], d_out["ls_liu"][i],
                                                            d_out["lat_plan"][i], d_out["lon_plan"][i], d_out["lst_plan"][i], values)
        lines_out.append(line)

    with open("%s_extra.dat" % os.path.splitext(filepath)[0], "w") as f:
        f.writelines(lines_out)

    # plt.figure()
    plt.scatter(d_out["my_ls_liu"], d_out["lat_plan"])
