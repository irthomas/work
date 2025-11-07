# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:22:04 2021

@author: iant

PLOT SPECTRA IN A FILE
"""
import re
import numpy as np

import matplotlib.pyplot as plt
# import os
# import h5py
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

from matplotlib.backends.backend_pdf import PdfPages

from tools.file.hdf5_functions import make_filelist2

plt.rcParams.update({'font.size': 8})

# apply_filter = False
apply_filter = True
std_cutoff = 2.0

# read in h5 file
# regex = re.compile("20210603_010708_1p0a_UVIS_O")
# regex = re.compile("20210724_013356_1p0a_UVIS_O")
# regex = re.compile("20210725_050456_1p0a_UVIS_O")

# regex = re.compile("20220109_150820_1p0a_UVIS_O|20220112_135320_1p0a_UVIS_O")

# regex = re.compile("20251003_044802_0p3c_UVIS_X") # Atlas comet
# regex = re.compile("(20210603_010708|20210725_050456|20210724_013356)_...._UVIS_O")


# regex = re.compile("20210603_010708_0p3b_UVIS_O")
# regex = re.compile("20210724_013356_0p3b_UVIS_O")
# regex = re.compile("20210725_050456_0p3b_UVIS_O")
regex = re.compile("20251003_044802_0p3b_UVIS_X")


if "0p3b" in regex.pattern:
    ff = True
elif "0p3c" in regex.pattern:
    ff = False


file_level = "hdf5_level_%s" % regex.pattern.split("_")[2]

hdf5_files, hdf5_filenames, hdf5_paths = make_filelist2(regex, file_level, path=r"C:\Users\iant\Documents\DATA\hdf5")

for hdf5_file, hdf5_filename in zip(hdf5_files, hdf5_filenames):

    observation_type = regex.pattern.split("_")[-1]
    ielo = observation_type in ["I", "E", "L", "O"]
    pqx = observation_type in ["P", "Q", "X"]

    y = hdf5_file["Science/Y"][...]
    x = hdf5_file["Science/X"][0, :]

    n_spectra = y.shape[0]

    if ielo:

        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][...]

    if not pqx:
        lats = hdf5_file["Geometry/Point0/Lat"][...]
        lons = hdf5_file["Geometry/Point0/Lon"][...]
    dates = hdf5_file["Geometry/ObservationDateTime"][...]

    with PdfPages("%s_spectra%s.pdf" % (hdf5_filename, {True: "_mean_gf", False: ""}[apply_filter])) as pdf:  # open pdf

        frame_sum = np.zeros(n_spectra)
        for i in range(n_spectra):
            plt.figure(figsize=(9, 3), constrained_layout=True)

            if ielo:
                plt.title("%s: i=%i, %0.1f-%0.1fkm, (%0.1fN, %0.1fE)" % (dates[i, 0].decode(), i, alts[i, 0], alts[i, 1], lats[i, 0], lons[i, 0]))
            else:
                plt.title("%s: i=%i" % (hdf5_filename, i))

            if ff:
                frame = y[i, :, :]
                if apply_filter:
                    mean = np.nanmean(frame)
                    std = np.nanstd(frame)
                    frame[frame > (mean+std*std_cutoff)] = mean
                    frame[frame < (mean-std*std_cutoff)] = mean
                    frame = gaussian_filter(frame, 1)

                frame_sum[i] = np.sum(frame[70:120, 500:1000]-frame[20:70, 500:1000])
                # plt.imshow(frame, vmin=(mean - std * 0.5), vmax=(mean + std * 0.5))
                # plt.imshow(frame, vmin=-25., vmax=25.)
                plt.imshow(frame)
                plt.colorbar()

            else:
                plt.plot(x, y[i, :], label="Science/Y")

                sav_gol = savgol_filter(y[i, :], 39, 2)
                plt.plot(x, sav_gol, label="Smoothed")

                plt.grid()
                plt.legend(loc="lower right")

                frame_sum[i] = np.sum(sav_gol[500:1000])

            pdf.savefig()
            plt.close()

    plt.figure()
    plt.plot(frame_sum)
