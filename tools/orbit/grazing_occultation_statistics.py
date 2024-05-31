# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:57:42 2024

@author: iant

GRAZING OCCULTATION STATISTICS
"""


from matplotlib.ticker import FormatStrFormatter
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl


from tools.general.get_mars_year_ls import get_mars_year_ls
from tools.plotting.colours import get_colours

from tools.file.hdf5_functions import make_filelist, open_hdf5_file


normal_occultation_duration = 803  # seconds, from other analysis
n_years_of_mission = 6.0

regex = re.compile("20......_.*_SO_G_.*")
file_level = "hdf5_level_1p0a"


def get_grazing_data(regex, file_level):
    _, h5s, _ = make_filelist(regex, file_level, open_files=False)

    h5_prefixes = []

    d = {}

    for i, h5 in enumerate(h5s):

        h5_split = h5.split("_")
        h5_prefix = h5[0:15] + "_" + h5_split[-2]

        if h5_prefix in h5_prefixes:
            continue

        else:
            h5_prefixes.append(h5_prefix)

        h5f = open_hdf5_file(h5)

        alts = h5f["Geometry/Point0/TangentAltAreoid"][:, 0]
        lats = h5f["Geometry/Point0/Lat"][:, 0]
        indbin = h5f["Channel/IndBin"][...]

        # only get data for 0-120km region
        ixs = np.where((alts < 120.0) & (indbin == 1) & (alts > -998.))[0]

        # TC20 is for ingress and egress together - only count if ingress
        if h5_split[-2] == "I":
            duration = float(h5f["Telecommand20/SODurationTime"][...])
        else:
            duration = 0.0

        dt = datetime.strptime(h5[0:15], "%Y%m%d_%H%M%S")
        my, ls = get_mars_year_ls(dt)

        d[h5_prefix] = {"alts": alts[ixs], "lats": lats[ixs], "duration": duration, "my": np.repeat(my, len(ixs)), "ls": np.repeat(ls, len(ixs))}

    return d


if "d" not in globals():
    d = get_grazing_data(regex, file_level)

# colours = get_colours(len(d.keys()))
# for i, h5_prefix in enumerate(d.keys()):
#     plt.plot(d[h5_prefix]["lats"], d[h5_prefix]["alts"], label=h5_prefix, color=colours[i])

# plt.xlabel("Latitude (degrees)")
# plt.ylabel("Tangent altitude (km)")
# plt.grid()
# plt.legend()

cmap = plt.cm.jet
bounds = np.linspace(0, 70, 7*2+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


alt_limits = [5.0, 10., 20., 40., 50., 75., 100.]
alt_durations = []
alt_noccs = []

for alt_limit in alt_limits:

    mys = []
    lss = []
    lats = []
    alts = []
    durations = []

    for i, h5_prefix in enumerate(sorted(list(d.keys()))):

        alt = d[h5_prefix]["alts"]

        if np.any(alt < alt_limit):

            mys.extend(d[h5_prefix]["my"])
            lss.extend(d[h5_prefix]["ls"])
            lats.extend(d[h5_prefix]["lats"])
            alts.extend(alt)
            durations.append(d[h5_prefix]["duration"])

    mys = np.asarray(mys)
    lss = np.asarray(lss)
    lats = np.asarray(lats)
    alts = np.asarray(alts)

    hours = np.sum(durations)/3600.

    n_occs = np.sum(durations)/normal_occultation_duration / n_years_of_mission

    # print(alt_limit, hours)
    alt_durations.append(hours)
    alt_noccs.append(n_occs)

    # plot Ls but discontinuous, each occultation evenly spread horizontally
    # for i, my in enumerate(sorted(list(set(mys)))):

    #     fig1, ax1 = plt.subplots(figsize=(15, 6), constrained_layout=True)

    #     ixs = np.where(mys == my)[0]

    #     ls_x = sorted(list(set(lss[ixs])))
    #     ls_ix = list(range(len(ls_x)))

    #     lss_ix = np.zeros_like(lss[ixs])
    #     for value, index in zip(ls_x, ls_ix):
    #         lss_ix[lss[ixs] == value] = index

    #     scat = ax1.scatter(lss_ix, lats[ixs], c=alts[ixs], cmap=cmap, norm=norm)

    #     ax1.set_title("Grazing Occultations in MY%i" % my)
    #     ax1.set_ylabel("Latitude (degrees)")
    #     ax1.grid()
    #     ax1.set_ylim([-90, 90])

    #     ax1.set_xticks(ls_ix, labels=["%0.2f" % i for i in ls_x], rotation=90)

    #     ax1.set_xlabel("Ls (degrees); note that the axis is discontinuous")
    #     cbar = fig1.colorbar(scat)
    #     cbar.set_label("Tangent Altitude (km)", rotation=270, labelpad=20)

    #     fig1.savefig("grazing_occultations_alt_limit_%ikm_my%i.png" % (alt_limit, my))


plt.subplots()
plt.scatter([0.0]+alt_limits, [0.0]+alt_durations)
plt.xlabel("Tangent altitude limit (km)")
plt.ylabel("Detector operating hours used (hours)")
plt.title("Detector operating hours used for grazing occultations")
plt.xlim(left=-5)
plt.ylim(bottom=-5)
plt.grid()

plt.subplots()
plt.scatter([0.0]+alt_limits, [0.0]+alt_noccs)
plt.xlabel("Tangent altitude limit (km)")
plt.ylabel("Mean number of occultations lost per year")
plt.title("Detector operating time, in occultations, 'lost' to grazing occultations")
plt.xlim(left=-5)
plt.ylim(bottom=-5)
plt.grid()


# # plot Ls on the x axis (doesn't work well, all the lines are plotted on top of one another)
# # for i, my in enumerate(sorted(list(set(mys)))):
# for i, my in enumerate([35]):
#     fig1, ax1 = plt.subplots(figsize=(15, 6))
#     ixs = np.where(mys == my)[0]
#     scat = ax1.scatter(lss[ixs], lats[ixs], c=alts[ixs], cmap=cmap, norm=norm, s=2)
#     ax1.set_title("Grazing Occultations in MY%i" % my)
#     ax1.set_xlabel("Ls (degrees)")
#     ax1.set_ylabel("Latitude (degrees)")
#     ax1.grid()
#     ax1.set_xlim([-1, 361])
#     ax1.set_ylim([-90, 90])
#     cbar = fig1.colorbar(scat)
#     cbar.set_label("Tangent Altitude (km)", rotation=270, labelpad=20)
