# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:21:43 2024

@author: iant
"""

import numpy as np

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.cprint import cprint


illuminated_row_dict = {24: slice(6, 19)}


def list_miniscan_data_1p0a(regex, file_level, channel, starting_orders, aotf_steppings, binnings, path=None):

    if channel == "so":
        from instrument.nomad_so_instrument_v03 import m_aotf
    elif channel == "lno":
        from instrument.nomad_lno_instrument_v02 import m_aotf

    if path:
        h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True, path=path)
    else:
        h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True)

    matching_h5 = []
    matching_h5_prefix = []
    for h5_f, h5 in zip(h5_files, h5_filenames):  # loop through orders of that observation

        binning_ = h5_f["Channel/Binning"][0]
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]

        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        # unique_bins = sorted(list(set(bins)))

        orders = np.array([m_aotf(i) for i in unique_aotf_freqs])
        unique_orders = sorted(list(set(orders)))
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]

        h5_split = h5.split("_")
        h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}-%i-%i" % (np.min(unique_orders), np.round(aotf_freqs_step))

        if unique_orders[0] not in starting_orders:
            print("%s order %i stepping %0.1fkHz" % (h5, unique_orders[0], aotf_freqs_step))
            continue

        if aotf_freqs_step not in aotf_steppings or binning_ not in binnings:
            cprint("%s order %i stepping %0.1fkHz" % (h5, unique_orders[0], aotf_freqs_step), "y")
            continue

        cprint("%s order %i stepping %0.1fkHz" % (h5, unique_orders[0], aotf_freqs_step), "c")
        matching_h5.append(h5)
        matching_h5_prefix.append(h5_prefix)

    return matching_h5, matching_h5_prefix


# def get_miniscan_data_0p1a(regex):


#     h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True)

#     d = {}
#     for h5_f, h5 in zip(h5_files, h5_filenames):

#         h5_split = h5.split("_")
#         h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}"

#         d[h5_prefix] = {}


#         y = h5_f["Science/Y"][...] #y is 3d in 0.1A
#         aotf_freqs = h5_f["Channel/AOTFFrequency"][...]
#         unique_aotf_freqs = sorted(list(set(aotf_freqs)))

#         orders = np.array([m_aotf(i) for i in unique_aotf_freqs])
#         unique_orders = sorted(list(set(orders)))


#         aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]

#         print("Miniscan stepping = %0.1fkHz" %(aotf_freqs_step))
#         print(h5, "orders", unique_orders[0])


#         bin_ = 12

#         y_bin = y[:, bin_, :]

#         for unique_aotf_freq in unique_aotf_freqs:

#             aotf_ixs = np.where(aotf_freqs == unique_aotf_freq)[0]

#             for aotf_ix in aotf_ixs[0:1]:
#                 y_spectrum = y_bin[aotf_ix, :]

#                 d[h5_prefix][unique_aotf_freq] = {0.0:y_spectrum} #set temperature to 0

#     return d


def get_miniscan_data_1p0a(h5_filenames, channel):

    print("Getting data for %i files" % len(h5_filenames))

    if channel == "so":
        from instrument.nomad_so_instrument_v03 import m_aotf
    elif channel == "lno":
        from instrument.nomad_lno_instrument_v02 import m_aotf

    d = {}
    for h5 in h5_filenames:
        # if "SO" in h5:
        #     good_bins = np.arange(126, 131)
        # elif "LNO" in h5:
        #     good_bins = np.arange(150, 155)

        print("Getting data for %s" % h5)
        h5_f = open_hdf5_file(h5)

        # observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
        # bins = h5_f["Science/Bins"][:, 0]
        y = h5_f["Science/Y"][...]
        t = h5_f["Channel/InterpolatedTemperature"][...]
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]

        # number of bins per aotf_freq
        n_bins = np.where(np.diff(aotf_freqs) > 0)[0][0] + 1  # first index where diff is nonzero

        # reshape by bin
        y = np.reshape(y, (-1, n_bins, 320))
        t = np.reshape(t, (-1, n_bins))[:, 0]
        aotf_freqs = np.reshape(aotf_freqs, (-1, n_bins))[:, 0]
        aotf_freqs = np.array([int(np.round(i)) for i in aotf_freqs])

        illuminated_rows = illuminated_row_dict[n_bins]
        y_mean = np.mean(y[:, illuminated_rows, :], axis=1)  # mean of illuminated rows

        # number of aotf freqs
        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        n_aotf_freqs = len(unique_aotf_freqs)

        # remove incomplete aotf repetitions
        n_repetitions = int(np.floor(np.divide(y.shape[0], n_aotf_freqs)))

        y_rep = np.reshape(y_mean[0:(n_repetitions*n_aotf_freqs), :], (-1, n_aotf_freqs, 320))
        t_rep = np.reshape(t[0:(n_repetitions*n_aotf_freqs)], (-1, n_aotf_freqs)).T

        # unique_bins = sorted(list(set(bins)))
        starting_order = m_aotf(np.min(unique_aotf_freqs))
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]

        h5_split = h5.split("_")
        h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}-%i-%i" % (starting_order, np.round(aotf_freqs_step))

        # output mean y for illuminated bins (2D), temperatures (1D) and aotf freqs (1D)
        # also output truncated arrays containing n repeated aotf freqs: y_rep (3D), t_rep (2D), a_rep(1D)
        d[h5_prefix] = {"y": y_mean, "t": t, "a": aotf_freqs, "y_rep": y_rep, "t_rep": t_rep, "a_rep": unique_aotf_freqs}

    return d
