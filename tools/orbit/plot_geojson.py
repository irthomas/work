# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:49:29 2023

@author: iant


READ AND PLOT GEOJSON
"""

import os
import re

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt



from tools.file.paths import paths
from tools.plotting.colours import get_colours

from other.pipeline.nomad_ops.core.footprints.db_functions import get_h5s_from_cache


# regex = re.compile("2023010[1-5]_......_...._LNO_1_D._193")
# regex = re.compile("2023010[1-5]_......_...._LNO_1_D._.*")
# regex = re.compile("20230524_025254_...._LNO_1_D._.*")

regex = re.compile("2018...._......_...._LNO_1_D._189")


h5_path = os.path.join(paths["DATASTORE_ROOT_DIRECTORY"], "hdf5", )
h5_level = "hdf5_level_1p0a"

footprint_version = "v01"

footprint_root_dir = os.path.join(paths["DATASTORE_ROOT_DIRECTORY"], "footprint", footprint_version)



h5s_all, h5_paths_all = get_h5s_from_cache(h5_path, h5_level)


#get indices of matching files
matching_ixs = [i for i, h5 in enumerate(h5s_all) if re.search(regex.pattern, h5)]



#now get list of directories containing the matching files
footprints_list = []
for i in matching_ixs:
    h5_path = h5_paths_all[i]
    
    path_split = os.path.normpath(h5_path).split(os.path.sep)
    ymd = os.path.join(*path_split[-4:-1])
    
    #get list of footprints in directory
    footprint_dir = os.path.join(footprint_root_dir, ymd)
    footprints_in_dir = os.listdir(footprint_dir)              
    
    matching_footprints = [fp for fp in footprints_in_dir if h5s_all[i].replace(".h5", "") in fp]
    footprints_list.extend([os.path.join(footprint_dir, fp) for fp in matching_footprints])

print("%i footprints matching regex" %(len(footprints_list)))





colours = get_colours(80) #orders 120-200

# file_path = os.path.normcase(r"C:/Users/iant/Documents/DATA/footprints/v01/2023/01/01/20230101_034107_1p0a_LNO_1_DP_167_00000.geojson")

fig, ax = plt.subplots(figsize=(15, 12), constrained_layout=True)
ax.grid()
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
fig.suptitle(regex.pattern)

for footprint_path in footprints_list:
    #get order
    basename = os.path.basename(footprint_path).split(".")[0]
    split = basename.split("_")
    order = int(split[-2])
    
    df_places = gpd.read_file(footprint_path)
    if np.min([np.min(df_places["geometry"][i].exterior.coords.xy[0]) for i in range(df_places["geometry"].shape[0])]) > -990.0:

        df_places.plot(ax=ax, color=colours[order-120])