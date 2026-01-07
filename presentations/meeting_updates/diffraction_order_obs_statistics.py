# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:06:51 2025

@author: iant

SO/LNO observation statistics

TODO:
    Include special fullscans when counting order statistics

"""
import sys
sys.path.append(r"C:\Users\iant\Documents\PROGRAMS\nomad_obs")  # noqa

import sqlite3
import decimal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from datetime import datetime, timedelta


planning_db_path = r"C:/Users/iant/Documents/PROGRAMS/nomad_obs/planning.db"


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


def get_planning_db_dict(table_name):
    con = connect_db(planning_db_path)

    if table_name == "occultations":
        occultation_data = read_table(con, table_name)

        # convert to dictionaries
        d = {"obs_id": [],
             "prime_instrument": [],
             "orbit_number": [],
             "mtp_number": [],
             "occultation_type": [],
             "utc_start_time": [],
             "utc_transition_time": [],
             "utc_end_time": [],
             "duration": [],
             "start_longitude": [],
             "transition_longitude": [],
             "end_longitude": [],
             "start_latitude": [],
             "transition_latitude": [],
             "end_latitude": [],
             "transition_local_time": [],
             "orbit_type": [],
             "ir_observation_name": [],
             "ir_description": [],
             "uvis_description": [],
             "orbit_comment": [],
             }

        for row in occultation_data:
            if row[1] == "NOMAD":
                for i, k in enumerate(d.keys()):
                    d[k].append(row[i])

    if table_name == "nadirs":
        nadir_data = read_table(con, table_name)

        # convert to dictionaries
        d = {"obs_id": [],
             "orbit_number": [],
             "mtp_number": [],
             "nadir_type": [],
             "utc_start_time": [],
             "utc_centre_time": [],
             "utc_end_time": [],
             "duration": [],
             "start_longitude": [],
             "centre_longitude": [],
             "end_longitude": [],
             "start_latitude": [],
             "centre_latitude": [],
             "end_latitude": [],
             "centre_incidence_angle": [],
             "centre_local_time": [],
             "orbit_type": [],
             "ir_observation_name": [],
             "ir_description": [],
             "uvis_description": [],
             "orbit_comment": [],
             }

        for row in nadir_data:
            if row[3] == "Dayside":
                for i, k in enumerate(d.keys()):
                    d[k].append(row[i])

    return d


for channel in ["so"]:

    if channel == "so":
        from nomad_obs.observation_names import occultationObservationDict as observation_dict
        # y_lim = 12
        obs_type = "occultations"
    # elif channel == "lno":
    #     from nomad_obs.observation_names import nadirObservationDict as observation_dict
    #     y_lim = 25
    #     obs_type = "nadirs"

    obs_dict = get_planning_db_dict(obs_type)

    start_dt = datetime(2018, 1, 1)

    unique_ir_observation_names = sorted([s for s in list(set(obs_dict["ir_observation_name"])) if s])

    search_orders = [127, 128, 129]
    search_alt = "low"

    hists = []

    for search_order in search_orders:

        search_ir_observation_names = []

        for unique_ir_observation_name in unique_ir_observation_names:

            # convert high-low to all
            if ";" in unique_ir_observation_name:
                split = unique_ir_observation_name.split(";")

                orders_name = split[0].strip()
                orders_name2 = split[1].strip()

                orders = observation_dict[orders_name][0]
                orders2 = observation_dict[orders_name2][0]

                if search_alt == "low" and search_order in orders:
                    search_ir_observation_names.append(unique_ir_observation_name)
                elif search_alt == "high" and search_order in orders2:
                    search_ir_observation_names.append(unique_ir_observation_name)

            else:
                orders_name = unique_ir_observation_name
                orders_name2 = ""

                orders = observation_dict[unique_ir_observation_name][0]
                orders2 = []

                if search_order in orders:
                    search_ir_observation_names.append(unique_ir_observation_name)

        # get list of observations matching the observation name
        for search_ir_observation_name in search_ir_observation_names:
            ixs = [i for i, obs_name in enumerate(obs_dict["ir_observation_name"]) if obs_name == search_ir_observation_name]
            print(search_order, ":", search_ir_observation_name, "(", len(ixs), ") times")

        ixs = np.asarray([i for i, obs_name in enumerate(obs_dict["ir_observation_name"]) if obs_name in search_ir_observation_names])
        dts = [obs_dict["utc_start_time"][ix] for ix in ixs]

        dts_s = [(dt - start_dt).total_seconds() for dt in dts]

        hist, bin_edges = np.histogram(dts_s, bins=int(dts_s[-1]/604800))

        dt_ticks = [datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
                    datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1), datetime(2025, 1, 1), datetime(2026, 1, 1)]
        dt_s_ticks = [(dt - start_dt).total_seconds() for dt in dt_ticks]
        bin_ticks = [np.abs(dt_tick - np.asarray(dt_s_ticks)).argmin() for dt_tick in dt_s_ticks]

        hists.append(hist)

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    for i, hist in enumerate(hists):
        if i == 0:
            ax1.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], label="Order %i" % search_orders[i])
            hist_cumul = hist
        else:
            ax1.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], bottom=hist_cumul, label="Order %i" % search_orders[i])
            hist_cumul += hist
    ax1.set(xticks=dt_s_ticks, xticklabels=[datetime.strftime(dt_tick, "%Y %b %d") for dt_tick in dt_ticks])

    ax1.legend()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Number of times measured per week")
    ax1.grid()
    ax1.set_title("Number of times the HCl orders are measured per week, excluding fullscans")

    # search_ir_observation_names = [name for name in unique_ir_observation_names if search_order in observation_dict[name][0]]

    # obs_times = [str(s)[0:7] for s in occ_dict["utc_start_time"]]

    # unique_times = sorted(list(set(obs_times)))

    # month_d = {}
    # for unique_time in unique_times:
    #     # get number of obs in each MTP
    #     month_ixs = [i for i, t in enumerate(obs_times) if t == unique_time and occ_dict["ir_observation_name"][i]]

    #     # count number of indices, make blank entry for each obs name
    #     month_d[unique_time] = {"n_obs": len(month_ixs), "obs": {unique_ir_observation_name: 0 for unique_ir_observation_name in unique_ir_observation_names}}

    #     for ix in month_ixs:
    #         obs_name = occ_dict["ir_observation_name"][ix]
    #         month_d[unique_time]["obs"][obs_name] += 100. / len(month_ixs)

    # # sort obs names based on the last N months
    # end_unique_times = unique_times[-N_MONTHS:]

    # end_obs_freqs = []
    # for unique_ir_observation_name in unique_ir_observation_names:
    #     obs_freqs = [month_d[t]["obs"][unique_ir_observation_name] for t in end_unique_times]
    #     end_obs_freqs.append(sum(obs_freqs))

    # sort_ixs = np.argsort(end_obs_freqs)[::-1]

    # sorted_unique_ir_observation_names = [unique_ir_observation_names[i] for i in sort_ixs]
