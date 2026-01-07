# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:50:29 2024

@author: iant
"""

import numpy as np
from pds4_tools import read
import matplotlib.pyplot as plt
from urllib.error import HTTPError
import requests
from datetime import datetime, timedelta

# get_url_list = True
get_url_list = False

# from psa_utils import tap
# pds = tap.PsaTap(tap_url="https://vo-pds-ppi.igpp.ucla.edu/tap")

# query = "SELECT granule_uid FROM maven_sep_calibrate_spec.epn_core TABLESAMPLE(10)"
# result = tap_service.search(query)
# print(result)


# import pyvo as vo
# tap_service = vo.dal.TAPService("https://vo-pds-ppi.igpp.ucla.edu/tap")
# # resultset = service.search("SELECT TOP 1 * FROM maven_sep_calibrate_spec.epn_core")


def url_exists(url):

    response = requests.get(url)
    if response.status_code == 200:
        return True
    else:
        return False


year = 2023
month = 11
day = 1
r = 1


error = True


def get_url(year, month, day, start_r):

    r = start_r

    while error:

        data_url = "https://pds-ppi.igpp.ucla.edu/data/maven-sep-calibrated/data/spec/%04i/%02i/mvn_sep_l2_s1-cal-svy-full_%04i%02i%02i_v04_r%02i.xml" \
            % (year, month, year, month, day, r)

        if url_exists(data_url):
            return data_url
        else:
            r += 1

        if r > 15:
            print("Error")
            return ""


def get_urls(year, month, day, start_r, n_days):
    datetimes = [datetime(year=year, month=month, day=day) + timedelta(days=i) for i in range(n_days)]

    urls = []
    for dt in datetimes:
        print(dt)
        url = get_url(dt.year, dt.month, dt.day, r)
        if url == "":
            break
        else:
            urls.append(url)
    return urls


if get_url_list:
    urls = get_urls(year, month, day, r, 120)
    with open("maven_sep_urls.txt", "w") as f:
        for url in urls:
            f.write("%s\n" % url)
else:
    with open("maven_sep_urls.txt", "r") as f:
        lines = f.readlines()
        urls = [line for line in lines]


sep_d = {}
for data_url in urls:

    sep_d[data_url] = {}

    try:
        structs = read(data_url).structures
    except Exception as e:
        print(e)
        continue

    for struct in structs:
        sep_d[data_url][struct.id] = np.asarray(struct.data)

t = []
counts = []
for key in sep_d.keys():
    if "time_unix" in sep_d[key].keys():
        t.extend(sep_d[key]["time_unix"])
        counts.extend(sep_d[key]["f_elec_flux_tot"])
    # t = d["time_unix"] - d["time_unix"][1]


t_dt = [datetime.utcfromtimestamp(i) if not np.isnan(i) else datetime(2023, 12, 1) for i in t]
# plt.plot(t/3600.0, d["f_elec_flux_tot"])
# # time = struct["time_unix"]

plt.figure()
plt.plot(t_dt, counts)
plt.xlabel("Unix time")
plt.ylabel("f_elec_flux_tot")
plt.title("MAVEN-SEP f_elec_flux_tot")
