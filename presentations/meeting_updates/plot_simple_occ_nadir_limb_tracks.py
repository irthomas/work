# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 10:06:39 2025

@author: iant

SIMPLE SCRIPT TO PLOT LIMB / OCCULTATION / NADIR TRACKS FOR PRESENTATIONS, DIRECTLY FROM H5 FILES


"""


import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist2
from tools.plotting.colours import get_colours
from tools.datasets.tes_albedo import get_TES_albedo_map

obs_types = [
    "Infrared Nadir",
    "UV-Vis Nadir",
    "Infrared Solar Occultation",
    "UV-Vis Solar Occultation",
    "UV-Vis Limb",
]


# colour_d = {
#     134: "r", 136: "r",
#     167: "b", 168: "b", 169: "b", 121: "lightblue",
#     189: "g", 190: "g",
#     193: "k", 196: "k", 132: "k", 133: "k",
#     149: "y"}

colour_d = {
    "Infrared Nadir": "red",
    "UV-Vis Nadir": "purple",
    "Infrared Solar Occultation": "green",
    "UV-Vis Solar Occultation": "blue",
    "UV-Vis Limb": "black",
}


def get_data(regex, file_level, obs_type):
    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for i, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        d[h5] = {}

        lats = h5_f["Geometry/Point0/Lat"][...]
        d[h5]["lats"] = lats
        lons = h5_f["Geometry/Point0/Lon"][...]
        d[h5]["lons"] = lons

        # et = h5_f["Geometry/ObservationEphemerisTime"][...]

        # dur = np.max(et) - np.min(et)

        # print(h5, lons.shape, dur, lons.shape[0]/dur, dur/lons.shape[0])

        # if "Infrared" in obs_type:
        #     order = h5_f["Channel"]["DiffractionOrder"][0]
        #     d[h5]["order"] = order

        # if "Nadir" in obs_type:
        #     szas = h5_f["Geometry/Point0/SunSZA"][...]
        #     d[h5]["szas"] = szas

        # elif "Solar Occultation" in obs_type:
        #     alts = h5_f["Geometry/Point0/TangentAltAreoid"][...]
        #     d[h5]["alts"] = alts

        h5_f.close()

    return d


d = {}
if "Infrared Nadir" in obs_types:
    # LNO nadir
    regex = re.compile("202507.._......_.*_LNO_1_D._.*")
    file_level = "hdf5_level_1p0a"
    obs_type = "Infrared Nadir"

    d["Infrared Nadir"] = get_data(regex, file_level, obs_type)

if "UV-Vis Nadir" in obs_types:
    # UVIS nadir
    regex = re.compile("202507.._......_.*_UVIS_D")
    file_level = "hdf5_level_1p0a"
    obs_type = "UV-Vis Nadir"

    d["UV-Vis Nadir"] = get_data(regex, file_level, obs_type)

if "Infrared Solar Occultation" in obs_types:
    # SO occ
    regex = re.compile("202507.._......_.*_SO_[HLA]_[IE]_.*")
    file_level = "hdf5_level_1p0a"
    obs_type = "Infrared Solar Occultation"

    d["Infrared Solar Occultation"] = get_data(regex, file_level, obs_type)

if "UV-Vis Solar Occultation" in obs_types:
    # UVIS occ
    regex = re.compile("202507.._......_.*_UVIS_[IE]")
    file_level = "hdf5_level_1p0a"
    obs_type = "UV-Vis Solar Occultation"

    d["UV-Vis Solar Occultation"] = get_data(regex, file_level, obs_type)

if "UV-Vis Limb" in obs_types:
    # UVIS limb
    regex = re.compile("202507.._......_.*_UVIS_[LO]")
    file_level = "hdf5_level_1p0a"
    obs_type = "UV-Vis Limb"

    d["UV-Vis Limb"] = get_data(regex, file_level, obs_type)

albedoMap, albedoMapExtents = get_TES_albedo_map()

albedoMap = albedoMap[80:(1440-80), :]
albedoMapExtents = [-180, 180, -80, 80]


combs = [
    ["UV-Vis Nadir"],
    ["UV-Vis Nadir", "Infrared Nadir"],
    ["UV-Vis Nadir", "Infrared Nadir", "Infrared Solar Occultation"],
    ["UV-Vis Nadir", "Infrared Nadir", "Infrared Solar Occultation", "UV-Vis Solar Occultation"],
    ["UV-Vis Nadir", "Infrared Nadir", "Infrared Solar Occultation", "UV-Vis Solar Occultation", "UV-Vis Limb"],
]

for comb in combs:

    fig1, ax1 = plt.subplots(figsize=(15, 8), constrained_layout=True)

    ax1.set_title("All %s NOMAD Observations: July 2025" % (comb[-1]))

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim((-180, 180))
    ax1.set_ylim((-90, 90))
    ax1.grid()

    albedoPlot = ax1.imshow(albedoMap, extent=albedoMapExtents, vmin=0.1, vmax=0.32, cmap="gist_earth")

    for ot in comb:

        i = 0

        for h5 in d[ot].keys():

            colour = colour_d[ot]

            if ot == "Infrared Solar Occultation":
                ax1.scatter(d[ot][h5]["lons"][::5], d[ot][h5]["lats"][::5], color=colour, alpha=0.05)
            elif ot == "UV-Vis Limb":
                ax1.scatter(d[ot][h5]["lons"][::5], d[ot][h5]["lats"][::5], color=colour, alpha=0.7)
            else:
                ax1.scatter(d[ot][h5]["lons"], d[ot][h5]["lats"], color=colour, alpha=0.05)

            i += d[ot][h5]["lons"].shape[0]

        print(ot, i)

    plt.savefig("%s_tracks_july25.png" % comb[-1].replace(" ", "_").lower())
