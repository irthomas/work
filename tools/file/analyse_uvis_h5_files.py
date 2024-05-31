# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:53:45 2024

@author: iant
"""


import re
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime


from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours


regex = re.compile("2023...._......_.*_UVIS_O")
file_level = "hdf5_level_1p0a"


def uvis_limbs_inertial_or_tracking(regex, file_level):
    """determine which uvis limbs are inertial and which are altitude tracking"""
    h5fs, h5s, _ = make_filelist(regex, file_level)
    # colours = get_colours(len(h5s))

    # fig, ax = plt.subplots()

    alts = []

    for file_index, (h5f, h5) in enumerate(zip(h5fs, h5s)):

        tangent_alts = np.mean(h5f["Geometry/Point0/TangentAltAreoid"][...], axis=1)

        alts.append(tangent_alts)

        # valid_indices = np.where(tangent_alts > -990.0)[0]

        # ax.plot(tangent_alts[valid_indices], label=h5, color=colours[file_index])

    # ax.legend()

    return alts


def uvis_limb_inertial_or_tracking(tangent_alts, cutoff=0.35):

    # remove negative values
    ixs = np.where(tangent_alts > 0.0)[0]
    alt2 = tangent_alts[ixs]

    # sort by absolute difference between consecutive altitudes, then take the smallest half
    ixs2 = np.sort(np.argsort(np.abs(np.diff(alt2)))[:int(len(alt2)/2)])

    diffs = np.diff(alt2[ixs2])

    # can still be jumps in indices, so diff can find high values. Repeat sorting and discarded largest half
    ixs3 = np.sort(np.argsort(np.abs(diffs))[:int(len(diffs)/2)])

    # define a coefficient to distinguish the two types
    coeff = np.std(diffs[ixs3])

    if coeff > cutoff:
        limb_type = "inertial"
    else:
        limb_type = "tracking"

    # if potentially ambiguous, plot to check
    if (coeff > 0.08) and (coeff < 0.9):
        fig1, (ax1a, ax1b) = plt.subplots(nrows=2)
        ax1a.plot(tangent_alts[ixs])
        # ax1b.plot(np.diff(alt2))

        ax1b.plot(diffs[ixs3])

        if coeff > cutoff:
            ax1a.set_title("Inertial %0.3f" % coeff)
        else:
            ax1a.set_title("Tracking %0.3f" % coeff)

    print(limb_type, coeff)
    return coeff


alts = uvis_limbs_inertial_or_tracking(regex, file_level)
for i in range(len(alts)):
    alt = alts[i]
    coeff = uvis_limb_inertial_or_tracking(alt)
    # plt.scatter(i, coeff)

# plt.close('all')
