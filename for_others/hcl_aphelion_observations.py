# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:59:17 2025

@author: iant

CHECK MY38 APHELION HCL OBS
"""

import h5py
# import re
import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import get_filepath
# from tools.plotting.colours import get_colours

from tools.spectra.baseline_als import baseline_als

HDF5_PATH = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"

# hcl_lines = [2906.247, 2904.110]  # both lines approx same depth
hcl_lines = [2906.247]  # both lines approx same depth

h2o_lines = [2906.726]  # , 2904.438] # 2nd line is approx 10% depth of the first

bin_starts = {120: 0, 124: 1, 128: 2, 132: 3}

irtf_h5s = [
    "20250608_191452_1p0a_SO_A_E_129",
    "20250608_202832_1p0a_SO_A_I_129",
    "20250608_231043_1p0a_SO_A_E_129",
    "20250609_002426_1p0a_SO_A_I_129",
    "20250609_061818_1p0a_SO_A_I_129",
    "20250609_101405_1p0a_SO_A_I_129",
    "20250609_125623_1p0a_SO_A_E_129",

    "20250621_174731_1p0a_SO_A_E_129",
    "20250621_214324_1p0a_SO_A_E_129",
    "20250622_025248_1p0a_SO_A_I_129",
    "20250622_084642_1p0a_SO_A_I_129",
    "20250622_124229_1p0a_SO_A_I_129",

    "20250728_170741_1p0a_SO_A_E_129",
    "20250728_203400_1p0a_SO_A_I_129",
    "20250728_210336_1p0a_SO_A_E_129",
    "20250729_002946_1p0a_SO_A_I_129",
    "20250729_005948_1p0a_SO_A_E_129",
    "20250729_042523_1p0a_SO_A_I_129",
    "20250729_045557_1p0a_SO_A_E_129",
    "20250729_082057_1p0a_SO_A_I_129",
    "20250729_085151_1p0a_SO_A_E_129",
]

dm_h5s = {
    "20250611_235436_1p0a_SO_A_E_129": {"zero": 0, "toa": 32},
    "20250612_213158_1p0a_SO_A_E_129": {"zero": 0, "toa": 31},
    "20250613_230506_1p0a_SO_A_E_129": {"zero": 0, "toa": 26},
    "20250614_224016_1p0a_SO_A_E_129": {"zero": 0, "toa": 30},
    "20250616_234826_1p0a_SO_A_E_129": {"zero": 0, "toa": 31},
    "20250619_005625_1p0a_SO_A_E_129": {"zero": 0, "toa": 28},
    "20250623_031155_1p0a_SO_A_E_129": {"zero": 0, "toa": 20},
    "20250625_041923_1p0a_SO_A_E_129": {"zero": 0, "toa": 28},
    "20250626_015615_1p0a_SO_A_E_129": {"zero": 0, "toa": 18},
    "20250627_013056_1p0a_SO_A_E_129": {"zero": 0, "toa": 23},
    "20250627_032847_1p0a_SO_A_E_129": {"zero": 0, "toa": 24},
    "20250628_050113_1p0a_SO_A_E_129": {"zero": 0, "toa": 25},

    "20250629_023753_1p0a_SO_A_E_129": {"zero": 0, "toa": 38},  # high TOA
    "20250630_041010_1p0a_SO_A_E_129": {"zero": 0, "toa": 43},
    "20250702_031845_1p0a_SO_A_E_129": {"zero": 0, "toa": 43},
    "20250702_051634_1p0a_SO_A_E_129": {"zero": 0, "toa": 42},
    "20250704_042434_1p0a_SO_A_E_129": {"zero": 0, "toa": 46},
    "20250801_090058_1p0a_SO_A_I_129": {"zero": 22, "toa": 34},

    # "20250626_213511_1p0a_SO_A_E_129",  # No HCL expected
}


def get_h5_data(h5s):

    d = {}

    for i, h5 in enumerate(h5s):
        print("%i/%i: %s" % (i, len(h5s), h5))
        path = get_filepath(h5, path=HDF5_PATH)

        with h5py.File(path, "r") as h5_f:

            alts_all = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            ixs = np.where(alts_all < 60)[0]

            y = h5_f["Science/Y2"][ixs, :]
            yraw = h5_f["Science/YUnmodified"][ixs, :]
            bins = h5_f["Science"]["Bins"][ixs, 0]
            x = h5_f["Science/X"][ixs, :]
            alts = alts_all[ixs]
            lats = h5_f["Geometry/Point0/Lat"][ixs, 0]
            lons = h5_f["Geometry/Point0/Lon"][ixs, 0]

            d[h5] = {}

            for bin_start, bin_n in bin_starts.items():

                bin_ixs = np.where(bins == bin_start)[0]

                d[h5][bin_n] = {"ys": y[bin_ixs, :], "yraws": yraw[bin_ixs, :], "xs": x[bin_ixs, :],
                                "bins": bins[bin_ixs], "alts": alts[bin_ixs], "lats": lats[bin_ixs], "lons": lons[bin_ixs]}

    return d


dm_d = get_h5_data(dm_h5s.keys())

plt.figure(figsize=(30, 10))

for h5 in list(dm_d.keys()):

    # plt.figure(figsize=(30, 10))
    # plt.title("%s" % (h5))

    for bin_n in list(dm_d[h5].keys())[1:2]:

        # plt.figure(figsize=(30, 10))
        # plt.title("%s bin %i" % (h5, bin_n))

        # y_mean = np.mean(dm_d[h5]["ys"][:, 160:240], axis=1)

        ys = dm_d[h5][bin_n]["ys"]
        yraws = dm_d[h5][bin_n]["yraws"]
        x = np.arange(320)
        alts = dm_d[h5][bin_n]["alts"]

        toa_alt = dm_h5s[h5]["toa"]

        # recalibrate
        toa_ixs = np.where((alts < (toa_alt + 10)) & (alts > toa_alt))[0]

        if len(toa_ixs) > 0:
            # if TOA spectra are available
            y_toa = np.mean(yraws[toa_ixs, :], axis=0)
            ytranss = yraws / y_toa
        else:
            ytranss = ys

        ytranss_mean = np.mean(ytranss, axis=1)

        # find lowest index where mean trans > 0.3
        t_min_inve = alts[np.min(np.where(ytranss > 1/np.e)[0])]
        print(h5, np.mean(dm_d[h5][bin_n]["lats"]), np.mean(dm_d[h5][bin_n]["lons"]), dm_h5s[h5]["toa"], t_min_inve)

        plt.plot(alts, ytranss_mean, label=h5)
        plt.legend()

        atmos_ixs = np.where((ytranss_mean > 0.1) & (ytranss_mean < 0.5))[0]

        # print(atmos_ixs)

        # plt.plot(alts, ytranss[:, 104])
        # plt.plot(alts, ytranss[:, 110], label=h5)

        # plt.plot(alts, yraws[:, 110], linestyle="--", label=h5)

        # plt.plot(ytranss.T)

        ydiffs = np.zeros_like(ytranss)
        ydiffs2 = np.zeros_like(ytranss)

        for i, ytrans in enumerate(ytranss):

            polyfit = Polynomial.fit(x, ytrans, 9)
            polyvals = polyfit(x)
            y_diff = ytrans - polyvals

            ydiffs[i, :] = y_diff

            # if i in atmos_ixs:

            # plt.plot(dm_d[h5][bin_n]["xs"][0, :]+0.0605, y_diff)

        # y_diff_atm = np.mean(ydiffs[30:50, :], axis=0)

        # ydiffs2 = ydiffs - y_diff_atm

        # plt.imshow(ydiffs)
        # # plt.figure()
        # # plt.imshow(ydiffs2)
        # plt.plot(dm_d[h5][bin_n]["xs"][0, :]+0.0605, np.mean(ydiffs[atmos_ixs, :], axis=0))

        # for hcl_line in hcl_lines:
        #     plt.axvline(x=hcl_line, color="k", linestyle="--")
        # for h2o_line in h2o_lines:
        #     plt.axvline(x=h2o_line, color="b", linestyle="--")

    # plt.plot(alts, yraws[:, 104])
    # plt.plot(alts, yraws[:, 110], linestyle="--")

    # for y, yraw in zip(ys, yraws):
    #     if y[100] > 0.05:

    # ybs = baseline_als(yraw)
    # y_diff = yraw - ybs

    # polyfit = Polynomial.fit(x, yraw, 9)
    # polyvals = polyfit(x)
    # y_diff = yraw - polyvals
    # plt.plot(dm_d[h5][bin_n]["xs"][0, :]+0.0605, y_diff)

    # plt.plot(dm_d[h5][bin_n]["xs"][0, :]+0.0605, yraw)
    # plt.plot(yraw)

    # for hcl_line in hcl_lines:
    #     plt.axvline(x=hcl_line, color="k", linestyle="--")
    # for h2o_line in h2o_lines:
    #     plt.axvline(x=h2o_line, color="b", linestyle="--")
