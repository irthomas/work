# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:49:29 2023

@author: iant


READ AND PLOT GEOJSON
"""

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


# PLOT_MOLA = True
PLOT_MOLA = False

order_colours = {132:"red", 133:"darkred", 136:"salmon", 168:"green", 189:"lightblue", 190:"blue", 193:"darkblue"}

footprint_version = "v01"

# footprint_root_dir = os.path.join(paths["DATASTORE_ROOT_DIRECTORY"], "footprint", footprint_version) #W drive
footprint_root_dir = os.path.join(paths["FOOTPRINT_DIRECTORY"], footprint_version) #D drive

print("Getting file list")
geojson_paths = sorted(glob.glob(footprint_root_dir + os.sep + "**" + os.sep + "*.geojson", recursive=True))
print("%i files found" %len(geojson_paths))

geojsons = [os.path.basename(s) for s in geojson_paths]
 

# regex = re.compile("2023...._......_...._LNO_1_D._..._00....geojson")
regex = re.compile("20230802_......_...._LNO_1_D._..._00....geojson")
# regex = re.compile(".*")


#get indices of matching files
matching_ixs = [i for i, geojson in enumerate(geojsons) if re.search(regex.pattern, geojson)]
print("%i footprints matching regex" %(len(matching_ixs)))

footprints_list = [s for i,s in enumerate(geojson_paths) if i in matching_ixs]






# colours = get_colours(80) #orders 120-200




if PLOT_MOLA:
    topo_map, topo_map_extents = get_MOLA_topo_map(18.0, 226.0)



# plot lat/lon map with orbits on top
fig1, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
# albedo_plot = ax1.imshow(albedo_map, extent=albedo_map_extents, vmin=0.1, vmax=0.4)

if PLOT_MOLA:
    topo_plot = ax1.imshow(topo_map, extent=topo_map_extents, cmap="gist_ncar", alpha=0.7)

ax1.set_title("MOLA albedo with LNO surface ice footprints")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_xlim((-180, 180))
ax1.set_ylim((-90, 90))
ax1.grid()
# fig1.tight_layout()






for geojson, footprint_path in zip([geojsons[i] for i in matching_ixs], footprints_list):
    
    #get order
    order = int(geojson.split("_")[-2])
    
    colour = order_colours[order]
    
    df_places = gpd.read_file(footprint_path)
    #if any points off-nadir
    # if np.min([np.min(df_places["geometry"][i].exterior.coords.xy[0]) for i in range(df_places["geometry"].shape[0])]) > -990.0:

    #don't plot wrap arounds
    lon_stds = [np.std(df_places["geometry"][i].exterior.coords.xy[0])for i in range(df_places["geometry"].shape[0])]
    
    #cutoff = 10.0
    good_ixs = [i for i,f in enumerate(lon_stds) if f < 10.0]
    
    df_good = []
    
    for i in good_ixs:
        df_good.append(df_places.iloc[i])
    else:
        print(geojson)
    
    df_good = gpd.GeoDataFrame(df_good)
    lines = df_good.plot(ax=ax1, color=colour)
    
    
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=v, label="Order %i" %k) for k,v in order_colours.items()]

ax1.legend(handles=patches)
    


