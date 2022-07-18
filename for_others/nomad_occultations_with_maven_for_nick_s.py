# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:25:47 2022

@author: iant

PLOT GROUNDTRACKS SOUTHERN HEMISPHERE 2022 05 27 T04 24 35 FOR NICK IUVS
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from tools.file.hdf5_functions import open_hdf5_file

from tools.orbit.plot_groundtracks_from_db import plot_files_tracks
from tools.orbit.plot_groundtracks_from_db import plot_3d_files_tracks

from tools.plotting.colours import get_colours
# search_tuple = ("files", {"utc_start_time":[datetime(2022, 5, 27), datetime(2022, 5, 28)]})
# h5s = plot_3d_files_tracks(search_tuple, title="All SO observations 27th May 2022")


# # stop()
# # get info from files
# h5_file_times = []

# plt.figure(figsize=(15, 10))
# for h5 in h5s:
    
#     h5_file_time = h5[:15]
    
#     if h5_file_time not in h5_file_times:
#         h5_file_times.append(h5_file_time)
#     else:
#         continue
    
#     h5_f = open_hdf5_file(h5)

#     lat = h5_f["Geometry/Point0/Lat"][:, 0]
#     lon = h5_f["Geometry/Point0/Lon"][:, 0]
#     alt = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
    
#     dt_strs = h5_f["Geometry/ObservationDateTime"][:, 0]
    
#     h5_f.close()
    
#     if lat[0] < 0.0:
#         scatter = plt.scatter(lon, lat, c=alt)
        
#     plt.text(lon[0]+5, lat[0]+0.3, dt_strs[0].decode())

# plt.grid()
# plt.xlabel("Longtitude")
# plt.ylabel("Latitude")
# plt.title("All southern hemisphere SO observations 27th May 2022")
# cbar = plt.colorbar(scatter)
# cbar.set_label("Tangent altitude (km)", rotation=270, labelpad=10)





# search_tuple = ("files", {"utc_start_time":[datetime(2022, 5, 27, 4, 3, 30, 0), datetime(2022, 5, 27, 5, 30, 0)]})
# h5s = plot_3d_files_tracks(search_tuple, title="NOMAD observations 27th May 2022")


# get info from files
h5_file_times = []

plt.figure()

h5s = ["20220527_043302_1p0a_UVIS_E"]
for h5 in h5s:
    
    h5_file_time = h5[:15]
    
    if h5_file_time not in h5_file_times:
        h5_file_times.append(h5_file_time)
    else:
        continue
    
    h5_f = open_hdf5_file(h5)

    lat = h5_f["Geometry/Point0/Lat"][:, 0]
    lon = h5_f["Geometry/Point0/Lon"][:, 0]
    alt = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
    y = h5_f["Science/Y"][...]
    x = h5_f["Science/X"][0, :]
    if "UVIS" not in h5:
        bins = h5_f["Science/Bins"][:, 0]
    
    dt_strs = h5_f["Geometry/ObservationDateTime"][:, 0]
    
    h5_f.close()
    scatter = plt.scatter(lon, lat, c=alt)

plt.grid()
plt.xlabel("Longtitude")
plt.ylabel("Latitude")
plt.title("SO observation\n%s - %s" %(dt_strs[0].decode(), dt_strs[-1].decode()))
plt.colorbar(scatter)

# h5s = plot_3d_files_tracks(search_tuple, title="SO observation\n%s - %s" %(dt_strs[0].decode(), dt_strs[-1].decode()))


if "UVIS" not in h5:
    unique_bins = sorted(list(set(bins)))
    bin_ixs = np.where(bins == unique_bins[0])[0]
    pixels = np.arange(100, 320, 40)
else:
    bin_ixs = np.where((alt<76) & (alt>20))[0]
    pixels = np.arange(400, 1000, 100)
    
plt.figure()
plt.title("%s: transmittance vs altitude" %h5)
for px in pixels:
    plt.plot(alt[bin_ixs], y[bin_ixs, px], label="Pixel %i" %px)

plt.legend()
plt.grid()
plt.xlabel("Altitude (km)")
plt.ylabel("Transmittance")

plt.figure(figsize=(12, 8), constrained_layout=True)
plt.title("%s: transmittance vs altitude" %h5)
colours = get_colours(len(bin_ixs), cmap="brg")
for i in np.arange(len(bin_ixs)):
    plt.plot(x, y[bin_ixs[i], :], color=colours[i])
    plt.text(x[700-i*30], y[bin_ixs[i], 400]+0.005, "%0.1f km" %alt[bin_ixs][i], color=colours[i])

plt.grid()
plt.xlabel("Wavelength nm")
plt.ylabel("Transmittance")
plt.savefig("uvis_iuvs_southern_clouds_%s.png" %h5)