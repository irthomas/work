# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:20 2020

@author: iant

Get LNO reflectance factor mean continuum values; plot lat lon maps and compare the TES albedo

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from tools.file.paths import FIG_X, FIG_Y
from tools.sql.obs_database import obs_database
#from tools.plotting.colours import get_colours
from tools.datasets.tes_albedo import get_TES_albedo_map, get_albedo

# SAVE_FIGS = False
SAVE_FIGS = True

diffraction_orders = [167,168,169,170]
order_text = "167-170"
file_level = "hdf5_level_1p0a"

# degree_binning = 2
degree_binning = 1


search_query = "SELECT * from %s WHERE " %file_level + " OR ".join(["diffraction_order == %i" %i for i in diffraction_orders])


database_name = "lno_nadir_%s" %file_level
db_obj = obs_database(database_name, silent=True)
query_output = db_obj.query(search_query)
db_obj.close()


albedoMap, albedoMapExtents = get_TES_albedo_map()

#cut off top and bottom like NOMAD. 8px per degree => 16 deg cutoff = 128 points
albedoMap = albedoMap[128:(1440-128), :]
albedoMapExtents = [-180, 180, -74, 74]


"""conversion TES to OMEGA order 169
TES = 0.1, OMEGA = 0.108
TES = 0.3, OMEGA = 0.462

slope = (0.462-0.108)/(0.3-0.1)
"""

def tes_to_omega_169(tes):
    """TES to OMEGA albedo: omega = omega/TES * TES"""
    tes_xy = [0.1, 0.3]
    omega_xy = [0.11, 0.42]
    omega = (omega_xy[1]-omega_xy[0])/(tes_xy[1]-tes_xy[0]) * tes  + (omega_xy[1] - (omega_xy[1]-omega_xy[0])/(tes_xy[1] - tes_xy[0]) * tes_xy[1])
    return omega

albedoMap = tes_to_omega_169(albedoMap) #convert TES from bolometric to ref factor


lon = np.asfarray([column[12] for column in query_output if column[12]>-999])
lat = np.asfarray([column[13] for column in query_output if column[12]>-999])
#incidence_angle = np.asfarray([column[15] for column in query_output if column[12]>-999])
y = np.asfarray([column[17] for column in query_output if column[12]>-999])





bins = [int(360/degree_binning), int(180/degree_binning)]

H, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(lon, lat, y, statistic='mean', bins=bins)

#remove nans by getting mean of non-nan 
nan_indices = np.where(~np.isfinite(H))

H_new = np.copy(H)
for i,j in zip(nan_indices[0], nan_indices[1]):
    surrounding_points = []
    for points in [[i-1,j], [i+1, j], [i, j-1], [i, j+1]]:
        try:
            surrounding_points.append(H[points[0], points[1]])
        except IndexError:
            continue
    surrounding_points = np.asfarray(surrounding_points)
    mean = np.mean(surrounding_points[np.isfinite(surrounding_points)])
    H_new[i,j] = mean

fig1, ax1 = plt.subplots(figsize=(FIG_X+5, FIG_Y+2))
fig2, ax2 = plt.subplots(figsize=(FIG_X+5, FIG_Y+2))
# fig3, ax3 = plt.subplots(figsize=(FIG_X+5, FIG_Y+2))




#plot TES only
albedoPlot = ax1.imshow(albedoMap, extent=albedoMapExtents, vmin=0.05, vmax=0.5)

ax1.set_title("MGS/TES Albedo Global Mosaic")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_xlim((-180, 180))
ax1.set_ylim((-90, 90))
cb1 = fig1.colorbar(albedoPlot)
cb1.set_label("MGS/TES albedo, scaled to reflectance factor", rotation=270, labelpad=10)
ax1.grid()
fig1.tight_layout()
if SAVE_FIGS:
    fig1.savefig("MGS_TES_albedo_global_mosaic.png", dpi=300)
#plot LNO only
#plot = ax2.imshow(np.flipud(H.T), vmin=0.04, vmax=0.15, alpha=1.0, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])
plot = ax2.imshow(np.flipud(H.T), alpha=1.0, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], vmin=0.05, vmax=0.5)
plot = ax2.imshow(np.flipud(H_new.T), alpha=1.0, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], vmin=0.05, vmax=0.5)

ax2.set_title("LNO nadir reflectance factor for diffraction orders %s, %ix%i degree binning" %(order_text, degree_binning, degree_binning))
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_xlim((-180, 180))
ax2.set_ylim((-90, 90))
cb2 = fig2.colorbar(plot)
cb2.set_label("Mean continuum reflectance factor", rotation=270, labelpad=10)
ax2.grid()
fig2.tight_layout()
if SAVE_FIGS:
    fig2.savefig("LNO_order_%s_mean_continuum_%ix%i_binning.png" %(order_text, degree_binning, degree_binning), dpi=300)



# #plot LNO over TES
# albedoPlot2 = ax3.imshow(albedoMap, extent = [-180,180,-90,90], cmap="binary")
# plot = ax3.imshow(np.flipud(H.T), vmin=0.04, vmax=0.15, alpha=0.7, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])

# ax3.set_title("LNO order %s data, %ix%i degree binning" %(order_text, degree_binning, degree_binning))
# ax3.set_xlabel("Longitude")
# ax3.set_ylabel("Latitude")
# ax3.set_xlim((-180, 180))
# ax3.set_ylim((-90, 90))
# cb3 = fig3.colorbar(plot)
# cb3.set_label("Mean continuum reflectance factor", rotation=270, labelpad=10)
# ax3.grid()
# if SAVE_FIGS:
#     fig3.savefig("TES_and_LNO_order_%s_mean_continuum.png" %(order_text), dpi=300)


