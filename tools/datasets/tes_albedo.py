# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:06:48 2020

@author: iant

LNO - TES ALBEDO FUNCTIONS





"""
import os
import numpy as np
from PIL import Image

from tools.file.paths import paths



def get_TES_albedo_map():
    """read in TES file"""
    
    im = Image.open(os.path.join(paths["REFERENCE_DIRECTORY"],"Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"))
    albedoMap = np.array(im)
    albedoMapExtents = [-180,180,-90,90]
    
    return albedoMap, albedoMapExtents



def get_albedo(lons_in, lats_in, albedo_map):
    """find TES albedo"""

    lonIndexFloat = np.asarray([int(np.round((180.0 + lon) * 8.0)) for lon in lons_in])
    latIndexFloat = np.asarray([int(np.round((90.0 - lat) * 8.0)) for lat in lats_in])
    lonIndexFloat[lonIndexFloat==2880] = 0
    latIndexFloat[latIndexFloat==1440] = 0
    albedos_out = np.asfarray([albedo_map[lat, lon] for lon, lat in zip(lonIndexFloat, latIndexFloat)])
    return albedos_out


#code for plotting

# FIG_X = 12
# FIG_Y = 10
# albedoMap, albedoMapExtents = get_TES_albedo_map()

# #cut off top and bottom like NOMAD. 8px per degree => 16 deg cutoff = 128 points
# albedoMap = albedoMap[128:(1440-128), :]
# albedoMapExtents = [-180, 180, -74, 74]
# fig1, ax1 = plt.subplots(figsize=(FIG_X+5, FIG_Y+2))
# albedoPlot = ax1.imshow(albedoMap, extent=albedoMapExtents, vmin=0.05, vmax=0.5)

# ax1.set_title("MGS/TES Albedo Global Mosaic")
# ax1.set_xlabel("Longitude")
# ax1.set_ylabel("Latitude")
# ax1.set_xlim((-180, 180))
# ax1.set_ylim((-90, 90))
# cb1 = fig1.colorbar(albedoPlot)
# cb1.set_label("MGS/TES albedo, scaled to reflectance factor", rotation=270, labelpad=10)
# ax1.grid()
# fig1.tight_layout()
