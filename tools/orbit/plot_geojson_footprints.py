# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:49:29 2023

@author: iant


READ AND PLOT GEOJSON
"""

import matplotlib.patches as mpatches
import os
import re

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import glob


from tools.file.paths import paths
# from tools.plotting.colours import get_colours
# from tools.datasets.tes_albedo import get_TES_albedo_map
from tools.datasets.mola_topo import get_MOLA_topo_map

# plot MOLA high res map as background?
# PLOT_MOLA = True
PLOT_MOLA = False

# choose colour for each order
order_colours = {132: "red", 133: "darkred", 136: "salmon", 168: "green", 189: "lightblue", 190: "blue", 193: "darkblue"}

# select directory
footprint_version = "v01"

footprint_root_dir = os.path.join(paths["DATASTORE_ROOT_DIRECTORY"], "footprint", footprint_version)  # W drive
# footprint_root_dir = os.path.join(paths["FOOTPRINT_DIRECTORY"], footprint_version)  # D drive
# footprint_root_dir = os.path.join(r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD", "footprint", footprint_version)  # WSL local


print("Getting file list")
geojson_paths = sorted(glob.glob(footprint_root_dir + os.sep + "**" + os.sep + "*.geojson", recursive=True))
print("%i files found" % len(geojson_paths))

# get filenames from paths
geojsons = [os.path.basename(s) for s in geojson_paths]


# filter by regex
# regex = re.compile("2023...._......_...._LNO_1_D._..._00....geojson")
# regex = re.compile("20230802_......_...._LNO_1_D._..._00....geojson")
regex = re.compile("20180[45].._......_...._UVIS_D_00....geojson")
# regex = re.compile(".*")

# channel = "lno"
channel = "uvis"

# get indices of matching files from regex filtering
matching_ixs = [i for i, geojson in enumerate(geojsons) if re.search(regex.pattern, geojson)]
print("%i footprints matching regex" % (len(matching_ixs)))

footprints_list = [s for i, s in enumerate(geojson_paths) if i in matching_ixs]

if PLOT_MOLA:
    topo_map, topo_map_extents = get_MOLA_topo_map(18.0, 226.0)


# plot lat/lon map with orbits on top
fig1, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
# albedo_plot = ax1.imshow(albedo_map, extent=albedo_map_extents, vmin=0.1, vmax=0.4)

if PLOT_MOLA:
    topo_plot = ax1.imshow(topo_map, extent=topo_map_extents, cmap="gist_ncar", alpha=0.7)
    ax1.set_title("MOLA albedo with %s surface footprints" % channel.upper())
else:
    ax1.set_title("%s surface footprints" % channel.upper())

ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_xlim((-181, 181))
ax1.set_ylim((-90, 90))
ax1.grid()
# fig1.tight_layout()


for geojson, footprint_path in zip([geojsons[i] for i in matching_ixs], footprints_list):

    # get diffraction order, assign colour
    if channel in ["so", "lno"]:
        order = int(geojson.split("_")[-2])
        colour = order_colours[order]
    elif channel in ["uvis"]:
        colour = "black"

    df_places = gpd.read_file(footprint_path)

    # plot footprints in the file
    lines = df_places.plot(ax=ax1, color=colour, alpha=0.7)

    # if so/lno with orders, make a patch for each order for the legend
    if channel in ["so", "lno"]:
        patches = [mpatches.Patch(color=v, label="Order %i" % k) for k, v in order_colours.items()]
    # if uvis, just use one colour
    elif channel in ["uvis"]:
        patches = [mpatches.Patch(color="black", label="UVIS nadir")]

# add order patches to the legend
ax1.legend(handles=patches)
