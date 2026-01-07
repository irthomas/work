# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:10:42 2025

@author: iant

PLOT THE OFF-POINTING ANGLE BETWEEN SO AND THE CENTRE OF THE SOLAR DISK FOR OCCULTATIONS BEFORE AND AFTER THE GYROLESS UPDATE

TO DO:
    PLOT EACH H5 PREFIX ONLY ONCE
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tools.file.hdf5_functions import make_filelist2
from tools.general.progress_bar import progress


file_level = "hdf5_level_1p0a"
path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
# path = r"D:\data\hdf5"

centre_bin = 124


"""Select a date before and after the gyroless update (2 Nov 2024)"""
regexes = [
    [re.compile("2024(08..|09..|10..|11[0-1].)_.*_SO_A_[IE].*"), "Before update"],
    [re.compile("(2025(02[1-3].|03..)|2024(11[2-3].|12..))_.*_SO_A_[IE].*"), "After update"],
    # re.compile("20241..._.*_SO_A_[IE].*"),
]


fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(13, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax1.set_title("SO Off-Pointing Angle Mean")
ax2.set_title("SO Off-Pointing Angle Stdev")
ax3.set_title("SO Off-Pointing Angle Stdev Histogram")

ax1.set_ylabel("Mean angle between boresight and Sun (arcminutes)")
ax1.set_xlabel("Observation date")
ax2.set_ylabel("Standard deviation of angle between boresight and Sun (arcminutes)")
ax2.set_xlabel("Observation date")
ax3.set_ylabel("Number of standard deviations")
ax3.set_xlabel("Binned angle between boresight and Sun (arcminutes)")

d_means_all = []
d_stds_all = []

for regex, text in regexes:

    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=path)

    h5_prefixes = []
    d1 = {"h5s": [], "ixs": [], "offs": []}
    for file_ix, (h5, h5_f) in enumerate(progress(list(zip(h5s, h5_fs)))):

        h5_prefix = h5[0:27]

        # skip if another order of same occultation has been measured already
        if h5_prefix in h5_prefixes:
            continue
        h5_prefixes.append(h5_prefix)

        bins = h5_f["Science/Bins"][:, 0]

        bin_ixs = np.where(bins == centre_bin)[0]
        # print(h5, bins[0:20])

        offp = h5_f["Geometry/FOVSunCentreAngle"][bin_ixs, 0]

        d1["h5s"].extend([h5 for i in range(len(offp))])
        d1["ixs"].extend([i for i in range(len(offp))])
        d1["offs"].extend(list(offp))

    for key in ["ixs", "offs"]:
        d1[key] = np.asarray(d1[key])

    unique_h5s = list(sorted(set(d1["h5s"])))

    # plt.figure()
    # plt.title("SO Off-Pointing Angle: %s" % regex.pattern.split("_")[0])

    d_means = []
    d_stds = []

    for unique_h5 in unique_h5s:
        ixs = np.asarray([i for i, h5 in enumerate(d1["h5s"]) if h5 == unique_h5])
        off_angle = d1["offs"][ixs]

        off_angle_mean = np.mean(off_angle)
        off_angle_std = np.std(off_angle)

        dt = datetime.strptime(unique_h5[0:15], "%Y%m%d_%H%M%S")

        d_stds.append([dt, off_angle_std])
        d_means.append([dt, off_angle_mean])

        # plt.plot(off_angle, label=unique_h5)

    # plt.legend()

    d_stds_all.extend(d_stds)
    d_means_all.extend(d_means)

    d_means = np.asarray(d_means)
    d_stds = np.asarray(d_stds)

    ax1.scatter(d_means[:, 0], d_means[:, 1], alpha=0.5, label=text)

    H, bins = np.histogram(d_stds[:, 1], bins=30)
    ax3.bar(bins[:-1], H, width=np.diff(bins)*0.9, alpha=0.5, label=text)

d_stds_all = np.asarray(d_stds_all)
d_means_all = np.asarray(d_means_all)
sc = ax2.scatter(d_stds_all[:, 0], d_stds_all[:, 1], alpha=0.5, c=d_means_all[:, 1])
cbar = fig2.colorbar(sc)
cbar.set_label("Mean off-pointing angle", rotation=270, labelpad=20)

ax1.grid()
ax2.grid()
ax3.grid()
ax1.legend()
ax3.legend()


fig1.savefig("so_off_pointing_angle_mean.png")
fig2.savefig("so_off_pointing_angle_std.png")
fig3.savefig("so_off_pointing_angle_hist.png")
