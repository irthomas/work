# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 10:16:11 2025

@author: iant

PLOT PHOBOS DISTANCE ANGLE DATA FOR PAPER
"""


import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tools.file.hdf5_functions import make_filelist2

# highest cal level of Phobos data
file_level = "hdf5_level_0p3a"
regex = re.compile(".*_LNO_1_P.*")

# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"


def get_phobos_geom_data(regex, file_level):
    """find all files matching regex and get data from them"""
    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=data_path)

    d = {}
    for i, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        # just take 1 diffraction order for each observation, whichever is first
        h5_prefix = h5[0:15]

        # if the datetime prefix has not yet been found, add to dictionary. If another order of this observation has already been measured, skip it
        if h5_prefix not in d.keys():

            top_bins = h5_f["Science"]["Bins"][:, 0]
            unique_bins = sorted(list(set(top_bins)))

            centre_bin_ix = int(len(unique_bins)/2)
            centre_bin = unique_bins[centre_bin_ix]

            print(unique_bins, centre_bin)

            # just use phase from 2nd bin
            bin_ixs = np.where(top_bins == centre_bin)[0]

            ets = h5_f["Geometry/ObservationEphemerisTime"][bin_ixs, 0]
            obs_alts = h5_f["Geometry/ObsAlt"][bin_ixs, 0]
            phase_angles = h5_f["Geometry/Point0/PhaseAngle"][bin_ixs, 0]
            # order = h5_f["Channel"]["DiffractionOrder"][0]

            d[h5_prefix] = {"obs_alts": obs_alts, "phase_angles": phase_angles, "ets": ets}

    return d


if "d" not in globals():
    d = get_phobos_geom_data(regex, file_level)

# plot phase angle vs observation elapsed time
plt.figure(figsize=(10, 6), constrained_layout=True)
for h5_prefix in d.keys():

    # find where not -999
    good_ixs = np.where(d[h5_prefix]["phase_angles"] > -998.0)[0]

    # if more than 5 good points in the observation
    if good_ixs.shape[0] > 5:

        # if not changing too much e.g. not a FOV scan
        if np.max(np.diff(good_ixs)) < 5:

            # if first TGO-Phobos distance is not the smallest value (normal obs start further away with minimum in the centre)
            if d[h5_prefix]["obs_alts"][good_ixs[0]] != np.min(d[h5_prefix]["obs_alts"][good_ixs]):

                # elapsed time
                x = d[h5_prefix]["ets"][good_ixs]-d[h5_prefix]["ets"][good_ixs[0]]
                y = d[h5_prefix]["obs_alts"][good_ixs]
                z = d[h5_prefix]["phase_angles"][good_ixs]

                x2 = np.arange(min(x), max(x), 2)
                y2 = np.interp(x2, x, y)
                z2 = np.interp(x2, x, z)

                scat = plt.scatter(x2, y2, c=z2, vmin=0, vmax=70, cmap="viridis_r", alpha=0.1, edgecolor="none")
                # print(min(d[h5_prefix]["phase_angles"][good_ixs]), max(d[h5_prefix]["phase_angles"][good_ixs]))

plt.title("Distance between TGO and Phobos during each observation")
plt.xlabel("Observation time (seconds)")
plt.ylabel("TGO-Phobos distance (km)")
plt.grid()

cbar = plt.colorbar(scat, alpha=1.0)
cbar.set_label("Sun-Phobos-TGO phase angle (degrees)", rotation=270, labelpad=10)
cbar.solids.set(alpha=1)

plt.savefig("lno_phobos_distance_vs_phase_angle.png")

# plot phase vs mission timeline
plt.figure(figsize=(12, 4), constrained_layout=True)

# first_et = 0.0
for h5_prefix in sorted(d.keys()):

    good_ixs = np.where(d[h5_prefix]["phase_angles"] > -998.0)[0]

    if good_ixs.shape[0] > 5:

        if np.max(np.diff(good_ixs)) < 5:

            if d[h5_prefix]["obs_alts"][good_ixs[0]] != np.min(d[h5_prefix]["obs_alts"][good_ixs]):

                # if first_et == 0.0:
                #     first_et = d[h5_prefix]["ets"][good_ixs[0]]

                # x = d[h5_prefix]["ets"][good_ixs[0]] - first_et
                x = datetime.strptime(h5_prefix, "%Y%m%d_%H%M%S")
                y = np.min(d[h5_prefix]["phase_angles"][good_ixs])

                scat = plt.scatter(x, y, color="k")
                # print(min(d[h5_prefix]["phase_angles"][good_ixs]), max(d[h5_prefix]["phase_angles"][good_ixs]))

plt.title("Phobos illumination phase angle variations over time")
plt.xlabel("UTC datetime")
plt.ylabel("Minimum phase angle during observation (degrees)")
plt.grid()

plt.savefig("lno_phobos_phase_angle_vs_time.png")
