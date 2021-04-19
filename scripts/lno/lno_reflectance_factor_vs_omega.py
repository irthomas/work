# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:20 2020

@author: iant

Get LNO reflectance factor mean continuum values; plot lat lon maps and compare with TES albedo

"""
import matplotlib.pyplot as plt
import numpy as np
#import scipy.stats

from tools.file.paths import FIG_X, FIG_Y
from tools.sql.obs_database import obs_database
#from tools.plotting.colours import get_colours
from tools.datasets.tes_albedo import get_TES_albedo_map, get_albedo

SAVE_FIGS = False
#SAVE_FIGS = True

#diffraction_order = 190
diffraction_order = 169
file_level = "hdf5_level_1p0a"


search_query = "SELECT * from %s WHERE diffraction_order == %i " %(file_level, diffraction_order)

database_name = "lno_nadir_%s" %file_level
db_obj = obs_database(database_name, silent=True)
query_output = db_obj.query(search_query)
db_obj.close()


albedoMap, albedoMapExtents = get_TES_albedo_map()



lon = np.asfarray([column[12] for column in query_output if column[12]>-999])
lat = np.asfarray([column[13] for column in query_output if column[12]>-999])
incidence_angle = np.asfarray([column[15] for column in query_output if column[12]>-999])
y = np.asfarray([column[17] for column in query_output if column[12]>-999])

max_incidence_angle = 15
good_indices = np.where(incidence_angle < max_incidence_angle)[0]


y_albedo = get_albedo(lon, lat, albedoMap)

fig, ax = plt.subplots(figsize=(FIG_X-3, FIG_Y))
ax.scatter(y_albedo[good_indices], y[good_indices], color="purple", s=5)
albedo_range = np.linspace(min(y_albedo[good_indices]), max(y_albedo[good_indices]), num=10)
linear_fit = np.polyfit(y_albedo[good_indices], y[good_indices], 1)
linear_range = np.polyval(linear_fit, albedo_range)
ax.plot(albedo_range, linear_range, "r")
ax.grid()
ax.set_xlabel("TES bolometric albedo")
ax.set_ylabel("LNO reflectance factor")
ax.set_title(search_query)
ax.set_xlim((0.05, 0.33))
ax.set_ylim((0.0, 0.6))
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("LNO_reflectance_factor_order_%i_vs_TES_albedo" %diffraction_order)




