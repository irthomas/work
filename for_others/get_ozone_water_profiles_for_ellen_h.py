# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:19:06 2024

@author: iant

CHECK PICCIALLI PUBLISHED OZONE DATA AND COMPARE TO AOKI PUBLISHED WATER RESULTS
https://doi.org/10.18758/71021079
https://doi.org/10.18758/71021072

USE ONION-PEELING RESULTS FOR OZONE
USE 2022 WATER VAPOUR RESULTS (ONLY LOW ALTITUDES WITH ORDERS 134/136)

"""
import scipy
import h5py
import zipfile
# import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# choose retrieval type and path to file containing Piccialli 2023 ozone retrievals
O3_FILE_PATH = r"C:\Users\iant\Downloads\op_retrievals_0905202.h5"
# FILE_PATH = r"C:\Users\iant\Downloads\foem_retrievals_0509202.h5"
# FILE_PATH = r"C:\Users\iant\Downloads\dop_retrievals_0905202.h5"

# select path to zip file containing Aoki 2022 water retrievals
DIR_PATH_H2O = r"C:\Users\iant\Downloads\archive.zip"

zip_ = zipfile.ZipFile(DIR_PATH_H2O)
filelist = sorted([s.split("/")[1].strip() for s in zip_.namelist() if s.split("/")[1].strip() != ""])

"""get water data"""
# filter zip filelist for July 2021 observations only
filelist = [s for s in filelist if s[0:6] == "202107"]

# loop through files, adding data to an input dictionary
d_h2o_in = {}
for filename in filelist:

    with zip_.open("archive/%s" % filename) as f:
        lines = f.readlines()
        d_h2o_in[filename] = {"dt": [], "z": [], "h2o": [], "lat": [], "ls": []}
        for line in lines[1:]:
            line_split = line.decode().split(",")
            d_h2o_in[filename]["z"].append(float(line_split[0]))
            d_h2o_in[filename]["h2o"].append(float(line_split[1]))
            d_h2o_in[filename]["lat"].append(float(line_split[6]))
            d_h2o_in[filename]["ls"].append(float(line_split[5]))

            dt = datetime(int(filename[0:4]), int(filename[4:6]), int(filename[6:8]), int(filename[9:11]))
            d_h2o_in[filename]["dt"].append(dt)

# condense data into a dictionary for plotting
d_h2o = {"zs": [], "h2o": [], "Lat": [], "Ls": [], "dts": []}

for filename in d_h2o_in.keys():
    d_h2o["zs"].extend(d_h2o_in[filename]["z"])
    d_h2o["h2o"].extend(d_h2o_in[filename]["h2o"])
    d_h2o["Lat"].extend(d_h2o_in[filename]["lat"])
    d_h2o["Ls"].extend(d_h2o_in[filename]["ls"])
    d_h2o["dts"].extend(d_h2o_in[filename]["dt"])

# convert to numpy arrays
for key in d_h2o.keys():
    if key not in ["dts"]:
        d_h2o[key] = np.asarray(d_h2o[key])


"""get ozone data"""
dset_paths = ["Filtering/chi2_no_O3", "Filtering/chi2_O3",
              "Science/Nd_O3_Err", "Science/Nd_O3",
              "Geometry/Ls", "Geometry/Lat", "Geometry/LST", "Geometry/Day", "Geometry/Month", "Geometry/Year"]

# get the correct hdf5 dataset name
if r"\op_" in O3_FILE_PATH:
    zname = "z_OP"
elif r"\foem_" in O3_FILE_PATH:
    zname = "z_FOEM"
elif r"\dop_" in O3_FILE_PATH:
    zname = "z_ASIMUT"
dset_paths.append("Science/%s" % zname)

# read in the ozone data from h5 file
d = {}
with h5py.File(O3_FILE_PATH, "r") as h5f:

    for dset_path in dset_paths:
        dset_name = dset_path.split("/")[-1]

        d[dset_name] = h5f[dset_path][...]

# make datetimes from year/month/day
dts = [datetime(int(year.decode()), int(month.decode()), int(day.decode()), 1) for year, month, day in zip(d["Year"], d["Month"], d["Day"])]
zs = d[zname]

# specify indices for plotting a subset of the ozone data, here the July 2021 SEP event
# sep_event = [6041, 6149] # July 2021 limited range
sep_event = [5963, 6157]  # all July 2021


"""sort the data into altitude and Ls bins using scipy"""

"""ozone"""
fig1, (ax1a, ax1b) = plt.subplots(figsize=(10, 10), nrows=2, sharex=True, constrained_layout=True)
fig2, (ax2a, ax2b) = plt.subplots(figsize=(10, 10), nrows=2, sharex=True, constrained_layout=True)

ixs = np.where(d["Lat"][:, sep_event[0]:sep_event[1]].flatten() > 0.0)[0]

x = d["Ls"][:, sep_event[0]:sep_event[1]].flatten()[ixs]
y = zs[:, sep_event[0]:sep_event[1]].flatten()[ixs]
z = d["Nd_O3"][:, sep_event[0]:sep_event[1]].flatten()[ixs]
title = "Ozone abundance northern hemisphere (Piccialli 2023 OP method unfiltered)"

grid1 = scipy.stats.binned_statistic_2d(y, x, z, bins=[30, 20])
grid1.statistic[0, :] = np.nan

ax1a.set_title(title)
im = ax1a.imshow(grid1.statistic, aspect="auto", origin="lower", vmin=0.0, vmax=1.0e9, extent=[dts[sep_event[0]], dts[sep_event[1]], 0, 60], cmap="jet")
cbar = plt.colorbar(im)
cbar.set_label("Ozone abundance", rotation=270, labelpad=10)
ax1a.grid()

x = d["Ls"][:, sep_event[0]:sep_event[1]].flatten()[~ixs]
y = zs[:, sep_event[0]:sep_event[1]].flatten()[~ixs]
z = d["Nd_O3"][:, sep_event[0]:sep_event[1]].flatten()[~ixs]
title = "Ozone abundance southern hemisphere (Piccialli 2023 OP method unfiltered)"

grid2 = scipy.stats.binned_statistic_2d(y, x, z, bins=[30, 20])
grid2.statistic[0, :] = np.nan

ax2a.set_title(title)
im = ax2a.imshow(grid2.statistic, aspect="auto", origin="lower", vmin=0.0, vmax=1.0e9, extent=[dts[sep_event[0]], dts[sep_event[1]], 0, 60], cmap="jet")
cbar = plt.colorbar(im)
cbar.set_label("Ozone abundance", rotation=270, labelpad=10)
ax2a.grid()

"""water"""
ixs = np.where(d_h2o["Lat"] > 0.0)[0]


x = d_h2o["Ls"][ixs]
y = d_h2o["zs"][ixs]
z = d_h2o["h2o"][ixs]
title = "Water abundance northern hemisphere (Aoki 2022)"

grid3 = scipy.stats.binned_statistic_2d(y, x, z, bins=[30, 20])

ax1b.set_title(title)
im = ax1b.imshow(grid3.statistic, aspect="auto", origin="lower", vmin=0.0, vmax=20.0, extent=[d_h2o["dts"][0], d_h2o["dts"][-1], 0, 40], cmap="jet")
cbar = plt.colorbar(im)
cbar.set_label("Water vmr (ppm)", rotation=270, labelpad=10)
ax1b.grid()

x = d_h2o["Ls"][~ixs]
y = d_h2o["zs"][~ixs]
z = d_h2o["h2o"][~ixs]
title = "Water abundance southern hemisphere (Aoki 2022)"

grid4 = scipy.stats.binned_statistic_2d(y, x, z, bins=[30, 20])

ax2b.set_title(title)
im = ax2b.imshow(grid4.statistic, aspect="auto", origin="lower", vmin=0.0, vmax=20.0, extent=[d_h2o["dts"][0], d_h2o["dts"][-1], 0, 40], cmap="jet")
cbar = plt.colorbar(im)
cbar.set_label("Water vmr (ppm)", rotation=270, labelpad=10)
ax2b.grid()

fig1.savefig("ozone_water_northern_hemisphere.png")
fig2.savefig("ozone_water_southern_hemisphere.png")
