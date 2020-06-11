# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:20 2020

@author: iant

LNO radiance factor mean values
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from PIL import Image
import os

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.sql.obs_database import obs_database
#from tools.plotting.colours import get_colours

SAVE_FIGS = True

#diffraction_order = 190
diffraction_order = 169
file_level = "hdf5_level_1p0a"


search_query = "SELECT * from %s WHERE diffraction_order == %i " %(file_level, diffraction_order)

database_name = "lno_nadir_%s" %file_level
db_obj = obs_database(database_name, silent=True)
query_output = db_obj.query(search_query)
db_obj.close()



##read in TES file
im = Image.open(os.path.join(paths["REFERENCE_DIRECTORY"],"Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"))
albedoMap = np.array(im)
albedoMapExtents = [-180,180,-90,90]

#find TES albedo
def getAlbedo(lons_in, lats_in, albedo_map):
    lonIndexFloat = np.asarray([int(np.round((180.0 + lon) * 8.0)) for lon in lons_in])
    latIndexFloat = np.asarray([int(np.round((90.0 - lat) * 8.0)) for lat in lats_in])
    lonIndexFloat[lonIndexFloat==2880] = 0
    latIndexFloat[latIndexFloat==1440] = 0
    albedos_out = np.asfarray([albedo_map[lat, lon] for lon, lat in zip(lonIndexFloat, latIndexFloat)])
    return albedos_out




lon = np.asfarray([column[12] for column in query_output if column[12]>-999])
lat = np.asfarray([column[13] for column in query_output if column[12]>-999])
incidence_angle = np.asfarray([column[15] for column in query_output if column[12]>-999])
y = np.asfarray([column[17] for column in query_output if column[12]>-999])

max_incidence_angle = 15
good_indices = np.where(incidence_angle < max_incidence_angle)[0]


y_albedo = getAlbedo(lon, lat, albedoMap)

fig, ax = plt.subplots(figsize=(FIG_X-3, FIG_Y))
ax.scatter(y_albedo[good_indices], y[good_indices], color="purple", s=5)
albedo_range = np.linspace(min(y_albedo[good_indices]), max(y_albedo[good_indices]), num=10)
linear_fit = np.polyfit(y_albedo[good_indices], y[good_indices], 2)
linear_range = np.polyval(linear_fit, albedo_range)
ax.plot(albedo_range, linear_range, "r")
ax.grid()
ax.set_xlabel("TES bolometric albedo")
ax.set_ylabel("LNO reflectance factor")
ax.set_title(search_query)
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("LNO_reflectance_factor_order_%i_vs_TES_albedo" %diffraction_order)


stop()

degree_binning = 3
bins = [int(360/degree_binning), int(180/degree_binning)]

H, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(lon, lat, y, statistic='mean', bins=bins)


fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))




#plot TES only
albedoPlot = ax1.imshow(albedoMap, extent=albedoMapExtents)

ax1.set_title("MGS/TES Albedo Global Mosaic")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_xlim((-180, 180))
ax1.set_ylim((-90, 90))
cb1 = fig1.colorbar(albedoPlot)
cb1.set_label("MGS/TES Albedo", rotation=270, labelpad=10)
ax1.grid()
if SAVE_FIGS:
    fig1.savefig("MGS_TES_albedo_global_mosaic.png")

#plot LNO only
#plot = ax2.imshow(np.flipud(H.T), vmin=0.04, vmax=0.15, alpha=1.0, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])
plot = ax2.imshow(np.flipud(H.T), alpha=1.0, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])

ax2.set_title("LNO order %i data, %ix%i degree binning\n%s" %(diffraction_order, degree_binning, degree_binning, search_query))
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_xlim((-180, 180))
ax2.set_ylim((-90, 90))
cb2 = fig2.colorbar(plot)
cb2.set_label("Mean continuum radiance factor", rotation=270, labelpad=10)
ax2.grid()
if SAVE_FIGS:
    fig2.savefig("LNO_order_%i_mean_continuum_%ix%i_binning.png" %(diffraction_order, degree_binning, degree_binning))



#plot LNO over TES
albedoPlot2 = ax3.imshow(albedoMap, extent = [-180,180,-90,90], cmap="binary")
plot = ax3.imshow(np.flipud(H.T), vmin=0.04, vmax=0.15, alpha=0.7, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])

ax3.set_title("LNO order %i data, %ix%i degree binning\n%s" %(diffraction_order, degree_binning, degree_binning, search_query))
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.set_xlim((-180, 180))
ax3.set_ylim((-90, 90))
cb3 = fig3.colorbar(plot)
cb3.set_label("Mean continuum radiance factor", rotation=270, labelpad=10)
ax3.grid()
if SAVE_FIGS:
    fig3.savefig("TES_and_LNO_order_%i_mean_continuum.png" %(diffraction_order))


