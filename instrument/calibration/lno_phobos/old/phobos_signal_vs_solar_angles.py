# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:28:44 2025

@author: iant

PLOT PHOBOS SIGNAL VS SOLAR ANGLE(S)

MUST USE 0.3A DATA, NOT YET FOR FULLSCAN

"""

import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist2
from tools.file.hdf5_functions import open_hdf5_file

from tools.plotting.colours import get_colours

SAVE_FIGS = True
# SAVE_FIGS = False

px_range = range(120, 280)


dset_name = "EmissionAngle"
# dset_name = "IncidenceAngle"
# dset_name = "PhaseAngle"
# dset_name = "PhaseAngle"

# regex = re.compile("20240126_123236_0p3a_LNO_1_P_.*")
# regex = re.compile("20250628_......_0p3a_LNO_1_P_.*")
regex = re.compile("2025...._......_0p3a_LNO_1_P_.*")

# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"


file_level = "hdf5_level_" + regex.pattern.split("_")[2]
h5_fs, h5s, _ = make_filelist2(regex, file_level, path=data_path)  # open file, get matching filename

orders = [int(h5.split("_")[-1]) for h5 in h5s]

colours = get_colours(max(orders)-min(orders))

data_d = {order: {} for order in orders}

# plt.figure()
# plt.title(dset_name)
for i, (h5_f, h5) in enumerate(zip(h5_fs, h5s)):
    # h5_f = open_hdf5_file(h5)
    h5_f = open_hdf5_file(h5, path=r"C:\Users\iant\Documents\DATA\hdf5", silent=True)

    bins = h5_f["Science/Bins"][:, 0]  # detector row of top of each bin
    bin_height = h5_f["Channel/Binning"][0] + 1  # number of arcminutes per bin
    unique_bins = sorted(list(set(bins)))
    n_bins = len(unique_bins)

    order = int(h5_f["Channel/DiffractionOrder"][0])

    # get y, reshape to 3d
    y_all = h5_f["Science/Y"][...]
    y_all_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

    # get angle, reshape to 2d
    phase_angle = h5_f["Geometry/Point0/%s" % dset_name][:, 0]
    phase_angle[phase_angle == -999] = np.nan

    phase_angle_3d = np.reshape(phase_angle, [-1, n_bins])

    # print(h5, y_all_3d.shape)

    y_mean_px = np.mean(y_all_3d[:, :, px_range], axis=2).T  # mean of spectral pixels
    # mask = np.all(np.isnan(y_mean_px) | np.equal(y_mean_px, 0), axis=1)  # make nan row mask if using 0.2a UVIS data and unmeasured rows are stored as nans
    # y_mean_px = y_mean_px[~mask].T  # remove all rows with nans
    y_mean_px = y_mean_px.T  # no masking

    # plt.figure()
    # im = plt.imshow(y_mean_px.T)
    # cbar = plt.colorbar(im)
    # cbar.set_label("Signal (counts)", rotation=270, labelpad=10)

    # basic correction - subtract last row on detector
    y_mean_px -= np.tile(y_mean_px[:, -1], (len(unique_bins), 1)).T  # subtract the last bin
    # plt.imshow(phase_angle_3d.T)

    # plt.figure()
    # im = plt.imshow(y_mean_px.T)
    # cbar = plt.colorbar(im)
    # cbar.set_label("Signal (counts)", rotation=270, labelpad=10)

    n_spectra = phase_angle_3d.shape[0]

    data_d[order]["y"] = y_all_3d
    data_d[order]["y_mean_px"] = y_mean_px

    # plt.figure()
    plt.scatter(phase_angle_3d[3:int(n_spectra/3), 1], y_mean_px[3:int(n_spectra/3), 1], color=colours[order-min(orders)-1])
    plt.scatter(phase_angle_3d[:, 1], y_mean_px[:, 1], color=colours[order-min(orders)-1])

# for i in range(4):
#     plt.figure()
#     for order, data in data_d.items():
#         # plt.plot(data["y_mean_px"][:, i])
#         plt.plot(data["y"][:, i, :].T, alpha=0.1)
