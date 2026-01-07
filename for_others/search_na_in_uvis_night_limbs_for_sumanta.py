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

regex = re.compile("20(2[012]....|230...)_......_.*_UVIS_O")
# regex = re.compile("20231..._......_.*_UVIS_O")


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
    # for lower_alt in [55.]:
    upper_alt = lower_alt + 5.0

    alt_ixs = np.where((uvis_dict["alts"] > lower_alt) & (uvis_dict["alts"] < upper_alt))[0]

    # plot all spectra individually, grouped into altitude bins
    # plt.figure()
    # plt.title("%i - %i km (%i spectra)" % (lower_alt, upper_alt, len(alt_ixs)))
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Radiance")
    # plt.grid()

    y_norms = []
    for alt_ix in alt_ixs:
        sza = uvis_dict["szas"][alt_ix]

        if sza > 110:
            y = uvis_dict["ys"][alt_ix, :]
            colour = (sza-90)/120
            # plt.plot(x, y, color=[colour, colour, colour], alpha=0.4)
            # plt.plot(x, y, alpha=0.3)

            # print(lower_alt, np.mean(y[:300]), sza)

            polyfit = np.polyfit(x, y, 2)
            polyval = np.polyval(polyfit, x)
            # plt.plot(x, polyval, color=[colour, colour, colour], alpha=1.0)

            y_norm = y - polyval

            if np.mean(y) > 0.0:
                y_norms.append(y_norm)

            # plt.plot(x, y_norm, color=[colour, colour, colour], alpha=0.4)

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


# check spectral calibration
plt.figure()
plt.title("Spectral calibration check: which pixel index corresponds to the Na emission line (589nm)")
plt.xlabel("Spectrum number, for spectra taken from all files")
plt.ylabel("Wavelength of pixel (nm)")
for i in [765, 766]:
    plt.plot(uvis_dict["xs"][:, i], label="Pixel %i" % (i+100))
plt.legend()
plt.grid()


# #investigate one altitude
# lower_alt = 55
# y_norms = y_binned_d[lower_alt]["y_norms"]
# plt.figure()
# plt.plot(y_norms.T, alpha=0.3)
# plt.title("All spectra in the %i-%i altitude")


plt.figure()
# for i, lower_alt in enumerate([55.]):
for i, lower_alt in enumerate(np.arange(45., 75., 5.)):

    y_mean = y_binned_d[lower_alt]["mean"]

    offset = i * 1e-6

    plt.plot(x, y_mean + offset, alpha=0.7, label="%i-%ikm" % (lower_alt, lower_alt+5), color="C%i" % i)
    plt.axhline(offset, alpha=1.0, color="C%i" % i)

    mean_left = np.mean(y_mean[continuum_ixs_left])
    mean_right = np.mean(y_mean[continuum_ixs_right])

    std_left = np.std(y_mean[continuum_ixs_left])
    std_right = np.std(y_mean[continuum_ixs_right])

    emission = y_mean[emission_ixs]

    # plt.scatter([569, 589, 610], [mean_left, emission, mean_right])
    # plt.axvline(x[continuum_ixs_left[0]], color="k")
    # plt.axvline(x[continuum_ixs_left[-1]], color="k")
    # plt.axvline(x[continuum_ixs_right[0]], color="k")
    # plt.axvline(x[continuum_ixs_right[-1]], color="k")

plt.title("Mean of all spectra for different latitude bins, offset for clarity")
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance")
plt.grid()
