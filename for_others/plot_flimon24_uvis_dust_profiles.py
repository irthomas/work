# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:21:19 2025

@author: iant

READ IN AN PLOT FLIMON ET AL 2025 DATA

https://data.aeronomie.be/dataset/uvis-aerosols-climatology
"""

# import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import binned_statistic_2d

# select path to downloaded file
filepath = "C:/Users/iant/Downloads/uvis_paper.txt"

# choose northern or southern hemisphere

# title = "Southern Hemisphere"
# lat_range = [-90, -30]

# title = "Northern Hemisphere"
# lat_range = [30, 90]

title = "Mid Latitudes"
lat_range = [-30, 30]

# end of user inputs


def get_data(filepath):
    """get data and save to a dictionary"""

    with open(filepath, "r") as f:
        lines = f.readlines()

    d = {}

    # loop through lines in file
    for i, line in enumerate(lines):
        if line[0:4] == "Name":
            i_start = i

        if i == i_start + 1:
            split = line.replace("\n", "").split(",")
            h5 = split[0]

            # get datetime from hdf5 file name
            year = int(h5[0:4])
            month = int(h5[4:6])
            day = int(h5[6:8])
            hour = int(h5[9:11])
            minute = int(h5[11:13])

            dt = datetime(year, month, day, hour, minute)

            # get Martian Year from datetime
            my_bounds = {34: [datetime(2018, 1, 1), datetime(2019, 3, 23)],
                         35: [datetime(2019, 3, 23), datetime(2021, 2, 7, 10)],
                         36: [datetime(2021, 2, 7, 10), datetime(2022, 12, 27)]}

            # determine martian year, when found save to dictionary
            for my, (my_start, my_end) in my_bounds.items():
                if my_start <= dt and my_end > dt:

                    d[h5] = {
                        "dt": dt,
                        "my": my,
                        "lat": float(split[1]),
                        "lon": float(split[2]),
                        "ls": float(split[3]),
                        "lst": float(split[4]),
                        "alts": [],
                        "exts": [],
                        "reffs": [],
                    }

        if i >= i_start + 3:
            # get data until next header found in file
            split = line.replace("\n", "").split(",")
            d[h5]["alts"].append(float(split[0]))
            d[h5]["exts"].append(float(split[1]))
            d[h5]["reffs"].append(float(split[3]))

    return d


# only load data from text file if not yet loaded
if "d" not in globals():
    d = get_data(filepath)


# loop through dictionary, saving all relevant data points
alts = []
reffs = []
dts = []
mys = []
lss = []

for h5 in list(sorted(d.keys())):

    # if latitude in desired range, store data
    if d[h5]["lat"] > lat_range[0] and d[h5]["lat"] < lat_range[1]:

        alts.extend(d[h5]["alts"])
        reffs.extend(d[h5]["reffs"])
        dts.extend(np.repeat(d[h5]["dt"], len(d[h5]["alts"])))
        mys.extend(np.repeat(d[h5]["my"], len(d[h5]["alts"])))
        lss.extend(np.repeat(d[h5]["ls"], len(d[h5]["alts"])))

# convert to arrays
alts = np.asarray(alts)
reffs = np.asarray(reffs)
# datetime needs to changed to seconds past start time
dts_s = np.asarray([(dt - dts[0]).total_seconds() for dt in dts])

# do 2D mean binning by datetime seconds and altitude
binned = binned_statistic_2d(dts_s, alts, reffs, statistic="mean", bins=(80, 100))

# extents for plotting
extents = (dts_s[0], dts_s[-1], min(binned[2]), max(binned[2]))

fig1, ax1 = plt.subplots(figsize=(8, 3), constrained_layout=True)

# plot data
im = ax1.imshow(binned[0].T, origin="lower", extent=extents, aspect="auto")
cbar = plt.colorbar(im)
cbar.set_label("Effective radius (um)", rotation=270, labelpad=10)

# find indices for plotting the x axis tick labels
# convert martian year and ls into a single number
mylss = np.asarray([my*360 + ls for (my, ls) in zip(mys, lss)])

# select martian years and ls tick labels
myls_ticks = [(34, 180), (34, 270), (35, 0), (35, 90), (35, 180), (35, 270), (36, 0), (36, 90), (36, 180), (36, 270), (37, 0)]

# get indices to map desired x tick labels onto data point my-ls values
dt_ixs = [np.abs(mylss - (myls_tick[0]*360 + myls_tick[1])).argmin() for myls_tick in myls_ticks]

# set x tick labels
ax1.set(xticks=[dts_s[i] for i in dt_ixs], xticklabels=["MY%i\n%0.0f" % (myls_tick[0], myls_tick[1]) for myls_tick in myls_ticks])
ax1.set_ylabel("Altitude (km)")
ax1.set_title("Particle Effective Radius (%s)" % title)

# set altitude limit to 0-90km
ax1.set_ylim(0, 90)

# save figure
fig1.savefig("Flimon_et_al_reff_%s.png" % title.lower().replace(" ", "_"))
