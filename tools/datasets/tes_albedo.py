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
