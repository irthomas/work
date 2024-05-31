# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:02:52 2024

@author: iant

CHECK UVIS NIGHTSIDE SPECTRA FOR SUMANTA
"""


import re
import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist, open_hdf5_file

# regex = re.compile("202[0123]...._......_.*_UVIS_O")
regex = re.compile("20231..._......_.*_UVIS_O")


def get_uvis_data(regex):
    file_level = "hdf5_level_1p0a"

    h5_fs, h5s, _ = make_filelist(regex, file_level)

    d = {"alts": [], "szas": [], "xs": [], "ys": []}
    for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        x = h5_f["Science/X"][...]
        y = h5_f["Science/Y"][...]
        szas = np.mean(h5_f["Geometry/Point0/SunSZA"][...], axis=1)
        alt = np.mean(h5_f["Geometry/Point0/TangentAltAreoid"][...], axis=1)

        d["xs"].extend(list(x[:, 100:]))  # ignore first 100 pixels in UV
        d["ys"].extend(list(y[:, 100:]))  # ignore first 100 pixels in UV
        d["alts"].extend(list(alt))
        d["szas"].extend(list(szas))

    for key in ["alts", "szas", "xs", "ys"]:
        d[key] = np.asarray(d[key])

    return d


# only read in data once
if "uvis_dict" not in globals():
    uvis_dict = get_uvis_data(regex)

# get wavelength grid
x = uvis_dict["xs"][0, :]

# 550-588 nm for the left side of the emission and 590- 630
continuum_ixs_left = np.where((x > 550.) & (x < 588.))[0]
continuum_ixs_right = np.where((x > 590.) & (x < 630.))[0]
emission_ixs = np.abs(x - 589).argmin()

y_binned_d = {}

for lower_alt in np.arange(45., 75., 5.):
    upper_alt = lower_alt + 5.0

    alt_ixs = np.where((uvis_dict["alts"] > lower_alt) & (uvis_dict["alts"] < upper_alt))[0]

    # plot all spectra individually, grouped into altitude bins
    # plt.figure()
    # plt.title("%i - %i km (%i spectra)" % (lower_alt, upper_alt, len(alt_ixs)))

    y_norms = []
    for alt_ix in alt_ixs:
        sza = uvis_dict["szas"][alt_ix]

        if sza > 100:
            y = uvis_dict["ys"][alt_ix, :]
            # plt.plot(x, y, color=[sza/180, sza/180, sza/180], alpha=1.0)

            # print(lower_alt, np.mean(y[:300]), sza)

            polyfit = np.polyfit(x, y, 2)
            polyval = np.polyval(polyfit, x)
            # plt.plot(x, polyval)

            y_norm = y - polyval

            y_norms.append(y_norm)

            # plt.plot(x, y_norm, alpha=0.1)

            mean_left = np.mean(y_norm[continuum_ixs_left])
            mean_right = np.mean(y_norm[continuum_ixs_right])

            std_left = np.std(y_norm[continuum_ixs_left])
            std_right = np.std(y_norm[continuum_ixs_right])

            emission = y_norm[emission_ixs]

            # plt.scatter([569, 610], [mean_left, mean_right])
            # plt.scatter([569, 610], [mean_left-std_left, mean_right-std_right])
            # plt.scatter([569, 610], [mean_left+std_left, mean_right+std_right])

            # plt.plot([569, 589, 610], [mean_left, emission, mean_right])

    y_norms = np.asfarray(y_norms)
    y_binned_d[lower_alt] = {"y_norms": y_norms, "mean": np.mean(y_norms, axis=0), "std": np.std(y_norms, axis=0)}


plt.figure()
for lower_alt in np.arange(45., 75., 5.):

    y_mean = y_binned_d[lower_alt]["mean"]

    plt.plot(y_mean, alpha=0.3, label=lower_alt)

    mean_left = np.mean(y_mean[continuum_ixs_left])
    mean_right = np.mean(y_mean[continuum_ixs_right])

    std_left = np.std(y_mean[continuum_ixs_left])
    std_right = np.std(y_mean[continuum_ixs_right])

    emission = y_mean[emission_ixs]

    plt.scatter([569, 589, 610], [mean_left, emission, mean_right])

plt.legend()
