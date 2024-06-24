# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:34:18 2024

@author: iant

PLOT SIMPLE GROUNDTRACKS FROM HDF5
"""


import re
import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist, open_hdf5_file

from tools.datasets.tes_albedo import get_TES_albedo_map, get_albedo

# calibrated files
# regex = re.compile("20180421_......_.*_LNO_1_D._.*")
regex = re.compile("20180421_......_.*_UVIS_D")
file_level = "hdf5_level_1p0a"
CENTRE_ONLY = False

# UVIS moon
# regex = re.compile("20......_......_.*_UVIS_Q")
# file_level = "hdf5_level_0p3c"
# CENTRE_ONLY = True

# LNO moon
# regex = re.compile("20......_......_.*_LNO_1_P_.*")
# file_level = "hdf5_level_0p3a"
# CENTRE_ONLY = True


# get files
print("Searching for files")
h5_fs, h5s, _ = make_filelist(regex, file_level)

print("Getting data from files")
d = {}
for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

    # make dictionary entry per file
    d[h5] = {}

    if CENTRE_ONLY:
        points = ["Point0"]
    else:
        # get list of points except centre
        points = [s for s in h5_f["Geometry"].keys() if ("Point" in s and s != "Point0")]

    # get all lat lons for points
    lats = np.asarray([(h5_f["Geometry/%s/Lat" % s][:, :]) for s in points])
    lons = np.asarray([h5_f["Geometry/%s/Lon" % s][:, :] for s in points])

    d[h5]["lats"] = lats
    d[h5]["lons"] = lons

plt.figure()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid()
if CENTRE_ONLY:
    plt.title("Lat/lon maps for the centre of the FOV\nregex = %s" % regex.pattern)

for h5 in h5s:

    n_spectra = d[h5]["lons"].shape[1]

    if CENTRE_ONLY:
        lon_start = d[h5]["lons"][0, :, 0]
        lat_start = d[h5]["lats"][0, :, 0]
        lon_end = d[h5]["lons"][0, :, 1]
        lat_end = d[h5]["lats"][0, :, 1]

        lon_start = lon_start[lon_start > -998.0]
        lat_start = lat_start[lat_start > -998.0]
        lon_end = lon_end[lon_end > -998.0]
        lat_end = lat_end[lat_end > -998.0]

        p = plt.scatter(lon_start, lat_start, alpha=0.5)
        # plot the end with the same colour
        plt.scatter(lon_end, lat_end, color=p.get_facecolors()[0], alpha=0.5)

    else:
        for i in range(n_spectra):

            # add the first point to the end to complete the FOV
            lon_start = np.concatenate((d[h5]["lons"][:, i, 0], d[h5]["lons"][0:1, i, 0]))
            lat_start = np.concatenate((d[h5]["lats"][:, i, 0], d[h5]["lats"][0:1, i, 0]))
            lon_end = np.concatenate((d[h5]["lons"][:, i, 1], d[h5]["lons"][0:1, i, 1]))
            lat_end = np.concatenate((d[h5]["lats"][:, i, 1], d[h5]["lats"][0:1, i, 1]))

            # plot start and end groundtracks separately
            p = plt.plot(lon_start, lat_start)
            # plot the end with the same colour
            plt.plot(lon_end, lat_end, color=p[0].get_color())
