# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:12:52 2024

@author: iant

NORMAL OCCULTATION STATISTICS
"""


# from matplotlib.ticker import FormatStrFormatter
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import matplotlib as mpl


from tools.general.get_mars_year_ls import get_mars_year_ls
# from tools.plotting.colours import get_colours
from tools.general.progress_bar import progress


from tools.file.hdf5_functions import make_filelist, open_hdf5_file


regex = re.compile("2023...._.*_SO_A_[IE]_.*")
file_level = "hdf5_level_1p0a"


def get_occultation_data(regex, file_level):
    _, h5s, _ = make_filelist(regex, file_level, open_files=False)

    h5_prefixes = []

    d = {}

    for i, h5 in enumerate(progress(h5s)):

        h5_split = h5.split("_")
        h5_prefix = h5[0:15] + "_" + h5_split[-2]

        if h5_prefix in h5_prefixes:
            continue

        else:
            h5_prefixes.append(h5_prefix)

        h5f = open_hdf5_file(h5)

        alts = h5f["Geometry/Point0/TangentAltAreoid"][:, 0]
        lats = h5f["Geometry/Point0/Lat"][:, 0]
        indbin = h5f["Channel/IndBin"][...]

        # only get data for 0-120km region
        ixs = np.where((alts < 120.0) & (indbin == 1) & (alts > -998.))[0]

        duration = float(h5f["Telecommand20/SODurationTime"][...])

        dt = datetime.strptime(h5[0:15], "%Y%m%d_%H%M%S")
        my, ls = get_mars_year_ls(dt)

        d[h5_prefix] = {"alts": alts[ixs], "lats": lats[ixs], "duration": duration, "my": np.repeat(my, len(ixs)), "ls": np.repeat(ls, len(ixs))}

    return d


if "d" not in globals():
    d = get_occultation_data(regex, file_level)


durations = []

for i, h5_prefix in enumerate(sorted(list(d.keys()))):

    durations.append(d[h5_prefix]["duration"])

plt.plot(durations)

durations = np.asarray(durations)
ix_normal = np.where(durations < 1500)[0]
mean_normal = np.mean(durations[ix_normal])
print("Mean duration of normal occultation:", mean_normal, "seconds")

precooling_reduction = 160

print("Mean duration of normal occultations after precooling reduction:", mean_normal-precooling_reduction, "seconds")
