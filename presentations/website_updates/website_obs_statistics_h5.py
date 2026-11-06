# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:03:24 2026

@author: iant

MAKE H5 FILE OF ORDER AND OTHER STATISTICS FOR WEBSITE

$order = 121;
$url = 'https://wsgi.aeronomie.be/runner/nomad_opendap/order_statistics.h5.json?orders.'.$order;
$json = file_get_contents($url);
$data = json_decode($json, true);

"""

import os
import sqlite3
import re
import matplotlib.pyplot as plt
import h5py
import numpy as np
from datetime import datetime, timedelta

# from tools.file.paths import paths
# from tools.file.list_files import list_files


SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"

# START_YEAR = 2018
# END_YEAR = 2023
# ORDERS_TO_PLOT = [119, 121, 129, 132, 134, 136]
# ORDERS_TO_PLOT = range(119, 137)

level = "hdf5_level_1p0a"


def get_filenames_from_cache(level):
    """get filenames from cache.db"""

    def connect_db(db_path):
        print("Connecting to database %s" % db_path)
        con = sqlite3.connect(db_path)
        return con

    def close_db(con):
        con.close()

    print("Getting data for level %s from cache.db" % level)
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


# get filenames inc .h5
channels, filenames = get_filenames_from_cache(level)

with h5py.File("order_statistics.h5", "w") as h5:

    for channel in ["so", "lno"]:
        if channel == "so":
            regex = re.compile(r"(....)(..).._......_1p0a_SO_._(?:\w_(\d*)|\w|\w\w)\.h5")
        if channel == "lno":
            regex = re.compile(r"(....)(..).._......_1p0a_LNO_._D._(?:\w_(\d*)|\w|\w\w)\.h5")

        # for so extract orders
        channel_ixs = [i for i, s in enumerate(channels) if s == channel.upper()]

        # make dictionary of each order measured in each observation
        obs_d = {}
        all_orders = []
        for channel_ix in channel_ixs:
            filename = filenames[channel_ix]

            filename_split = filename.split("_")

            if len(filename_split) == 7:
                order = int(filename_split[6].replace(".h5", ""))
                prefix = filename[0:15]

                # f0 = filename_split[0]
                # f1 = filename_split[1]
                # dt = datetime(int(f0[0:4]), int(f0[4:6]), int(f0[6:8]), int(f1[0:2]), int(f1[2:4]), int(f1[4:6]))

                if prefix not in obs_d.keys():
                    obs_d[prefix] = []

                if order not in all_orders:
                    all_orders.append(order)

                obs_d[prefix].append(order)

        # sort ascending diffraction orders
        for key in obs_d.keys():
            obs_d[key] = sorted(obs_d[key])

        # count orders measured per month
        first_obs = sorted(list(obs_d.keys()))[0]
        last_obs = sorted(list(obs_d.keys()))[-1]

        # first_month = first_obs[0:4] + "-" + first_obs[4:6]
        # last_month = last_obs[0:4] + "-" + last_obs[4:6]

        first_month = datetime(int(first_obs[0:4]), int(first_obs[4:6]), 1)
        last_month = datetime(int(last_obs[0:4]), int(last_obs[4:6]), 1)

        day_dts = [first_month + timedelta(days=i) for i in range((last_month - first_month).days + 1)]
        dates = sorted(list(set(["%04i-%02i" % (day.year, day.month) for day in day_dts])))

        orders_days = {i: {day: 0 for day in dates} for i in all_orders}

        for prefix in obs_d.keys():
            day = "%04i-%02i" % (int(prefix[0:4]), int(prefix[4:6]))
            for order in obs_d[prefix]:
                orders_days[order][day] += 1

        # remove if sparsely used
        sparse_orders = []
        MIN_N_OBS = 75
        for order in orders_days.keys():
            n_obs = np.sum(list(orders_days[order].values()))
            if n_obs < MIN_N_OBS:
                sparse_orders.append(order)

        for sparse_order in sparse_orders:
            del orders_days[sparse_order]

        channel_group = h5.create_group("%s_orders" % channel)
        channel_group.create_dataset("dates", len(orders_days[169].keys()), "S10", list(orders_days[169].keys()))
        sup_group = channel_group.create_group("orders_n_obs")
        for order in orders_days.keys():
            group = sup_group.create_group(str(order))
            group.create_dataset("n_obs", shape=len(orders_days[order].values()), data=np.asarray(list(orders_days[order].values())), dtype=int)
