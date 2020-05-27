# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:20 2020

@author: iant

LNO radiance factor mean values
"""
import matplotlib.pyplot as plt
import numpy as np


from tools.file.paths import paths, FIG_X, FIG_Y
from tools.sql.obs_database import obs_database
from tools.plotting.colours import get_colours


diffraction_order = 134
file_level = "hdf5_level_1p0a"


search_query = "SELECT * from %s WHERE diffraction_order == %i " %(file_level, diffraction_order)


database_name = "lno_nadir_%s" %file_level
db_obj = obs_database(database_name, silent=True)
query_output = db_obj.query(search_query)
db_obj.close()


fig0, ax0 = plt.subplots(figsize=(FIG_X, FIG_Y))

lon = [column[12] for column in query_output]
lat = [column[13] for column in query_output]
y = [column[17] for column in query_output]

#colours = get_colours(50)

sca = ax0.scatter(lon, lat, c=y)
ax0.set_xlabel("Longitude")
ax0.set_ylabel("Latitude")
ax0.set_xlim((-180, 180))
ax0.set_ylim((-90, 90))
plt.colorbar(sca)
ax0.legend()
ax0.grid()
