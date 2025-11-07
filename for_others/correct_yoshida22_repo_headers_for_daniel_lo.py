# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:48:45 2025

@author: iant

FIX YOSHIDA 2022 ET AL DATA REPOSITORY FILES WITH CORRECT GEOMETRY
"""


import os
import h5py
import glob
import numpy as np

from tools.file.hdf5_functions import get_filepath
from tools.general.get_nearest_index import get_nearest_index

# get list of repo files
filelist = glob.glob(r"C:\Users\iant\Documents\DATA\temp\yoshida22_repo\*.txt")

# loop through each one
for filepath in filelist:

    # get filename, remove .txt and append order (190)
    basename = os.path.basename(filepath)[:-4] + "_190"
    print(basename)

    # get altitude range from file
    with open(filepath, "r") as f:
        lines = f.readlines()

        # continue if there are data points in file, if not skip
        if len(lines) > 4:
            # get min/max altitudes from repo file
            alt_min = float(lines[4].split()[0])
            alt_max = float(lines[-1].split()[0])

            # open hdf5 file
            h5 = get_filepath(basename, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

            with h5py.File(h5, "r") as h5_f:

                alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]

                # get indices matching min/max altitudes
                # if only one entry, get the nearest index, otherwise get the range
                if alt_min == alt_max:
                    ixs = [get_nearest_index(alt_min, alts)]
                else:
                    ixs = np.where((alts > alt_min) & (alts < alt_max))[0]

                ls = h5_f["Geometry/LSubS"][ixs, 0]
                lat = h5_f["Geometry/Point0/Lat"][ixs, 0]
                lst = h5_f["Geometry/Point0/LST"][ixs, 0]

                # print(min(ls), max(ls), min(lat), max(lat), min(lst), max(lst))

                # replace header in lines with correct values
                lines[0] = "Ls range %0.8f %0.8f\n" % (min(ls), max(ls))
                lines[1] = "Lat range %0.8f %0.8f\n" % (min(lat), max(lat))
                lines[2] = "LST range %0.8f %0.8f\n" % (min(lst), max(lst))

            # make new filepath
            filepath_new = filepath.replace("yoshida22_repo", "yoshida22_repo_new")

            # write new filepath
            with open(filepath_new, "w") as f:
                f.writelines(lines)
