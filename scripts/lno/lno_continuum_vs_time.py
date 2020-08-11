# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:20 2020

@author: iant

Get LNO reflectance factor mean continuum values; plot lat lon maps and compare the TES albedo

"""
import matplotlib.pyplot as plt
import numpy as np
#import scipy.stats
from datetime import datetime, timedelta

from tools.file.paths import FIG_X, FIG_Y
from tools.sql.obs_database import obs_database
#from tools.plotting.colours import get_colours
from tools.datasets.tes_albedo import get_TES_albedo_map, get_albedo
from tools.plotting.lno_groundtrack_functions import make_query_from_search_dict
#from tools.spice.datetime_functions import utc2et


#SAVE_FIGS = False
SAVE_FIGS = True

#diffraction_order = 190
diffraction_order = 169
#diffraction_order = 134
file_level = "hdf5_level_1p0a"


min_tes_albedo = 0.25



search_dict ={
        134:{"incidence_angle":[0,60]},
        169:{"incidence_angle":[0,60]},
        190:{"incidence_angle":[0,60]},
}


search_query = make_query_from_search_dict(search_dict, file_level, diffraction_order)


database_name = "lno_nadir_%s" %file_level
db_obj = obs_database(database_name, silent=True)
query_output = db_obj.query(search_query)
db_obj.close()


albedoMap, albedoMapExtents = get_TES_albedo_map()



lon_in = np.asfarray([column[12] for column in query_output if column[12]>-999])
lat_in = np.asfarray([column[13] for column in query_output if column[12]>-999])
#incidence_angle = np.asfarray([column[15] for column in query_output if column[12]>-999])
y_in = np.asfarray([column[17] for column in query_output if column[12]>-999])
datetimes_in = [column[8] for column in query_output if column[12]>-999]

ets_in = [date_time-datetimes_in[0] for date_time in datetimes_in]

y_albedo_in = get_albedo(lon_in, lat_in, albedoMap)

good_indices = np.where(y_albedo_in > min_tes_albedo)[0]

lons = lon_in[good_indices]
lats = lat_in[good_indices]
ys = y_in[good_indices]
et_seconds = np.asfarray([ets_in[i].total_seconds() for i in good_indices])
ets = [ets_in[i] for i in good_indices]
y_albedos = y_albedo_in[good_indices]

bins = np.linspace(np.min(et_seconds), np.max(et_seconds), num=100)
bin_indices = np.digitize(et_seconds, bins)
unique_bin_indices = list(set(bin_indices))


"""plot LNO albedo vs TES albedo, colour points according to time"""
#fig, ax = plt.subplots(figsize=(FIG_X-3, FIG_Y))
#ax.scatter(y_albedos, ys, c=et_seconds, cmap="gnuplot", alpha=0.5, linewidths=0, s=20)
#ax.grid()
#ax.set_xlabel("TES bolometric albedo")
#ax.set_ylabel("LNO reflectance factor")
#ax.set_title(search_query)
##ax.set_xlim((0.05, 0.33))
#ax.set_ylim((0.0, 0.6))
##fig.tight_layout()
#if SAVE_FIGS:
#    fig.savefig("LNO_reflectance_factor_order_%i_vs_TES_albedo" %diffraction_order)


"""bin in time, then plot mean LNO continuum vs time"""
fig, ax = plt.subplots(figsize=(FIG_X+3, FIG_Y))
for unique_bin_index in unique_bin_indices:
    bin_values = ys[bin_indices == unique_bin_index]
    et_bin_values = [ets[i] for i,v in enumerate(bin_indices) if v == unique_bin_index]
    if len(bin_values)>5:
        
        mean_bin_values = np.mean(bin_values)
        std_bin_values = np.std(bin_values)
        mean_datetime = (sum(et_bin_values, timedelta(0))/len(et_bin_values))+datetimes_in[0]
        if std_bin_values < 0.03:
            ax.scatter(mean_datetime, mean_bin_values)
            ax.errorbar(mean_datetime, mean_bin_values, yerr=std_bin_values)
            
ax.set_xlabel("Time at centre of bin")
ax.set_ylabel("LNO Mean binned continuum albedo")
ax.set_title("LNO order %i: continuum albedo changes during the mission\nfor points where TES albedo > %0.2f" %(diffraction_order, min_tes_albedo))
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("LNO_order_%i_continuum_albedo_change_vs_time.png" %diffraction_order)
