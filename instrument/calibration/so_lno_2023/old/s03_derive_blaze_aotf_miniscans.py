# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:13:26 2022

@author: iant


OLD - SUPERSEDED BY S02_SIMULATE_CORRECTED_MINISCANS.PY (SOLAR LINE DICT NOT USED ANY MORE)

INVESTIGATE BLAZE FUNCTION VS TEMPERATURE FROM MINISCANS
USE THIS TO CHECK THE CALIBRATION AND FITS, NOT FOR CONVERTING FILES
USE correct_miniscan_diagonals.py TO MAKE DIAGONALLY CORRECTED H5 FILES
THE FUNCTIONS HERE ARE STILL USED, DO NOT DELETE

"""

# import sys
import os
import re
import numpy as np
# from scipy.interpolate import RegularGridInterpolator
import h5py

import matplotlib.pyplot as plt
# from scipy.interpolate import splrep, splev, BSpline

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.cprint import cprint
# from tools.spectra.running_mean import running_mean_1d

from instrument.nomad_so_instrument_v03 import aotf_peak_nu, lt22_waven
from instrument.nomad_lno_instrument_v02 import nu0_aotf, nu_mp

from instrument.calibration.so_lno_2023.make_hr_array import make_hr_array


# inflight

# channel = "SO"
channel = "LNO"


file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel)

regex = re.compile("20200201.*_%s_.*_CM" % channel)
# regex = re.compile("20191002.*_%s_.*_CM" %channel)

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


HR_SCALER = 10.

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")


if channel == "SO":
    from instrument.nomad_so_instrument_v03 import m_aotf
elif channel == "LNO":
    from instrument.nomad_lno_instrument_v02 import m_aotf


if __name__ == "__main__":
    if channel == "SO":
        # aotf_steppings = [8.0]
        aotf_steppings = [4.0]
        binnings = [0]
        # starting_orders = [188]
        starting_orders = [191]
        # dictionary of fft_cutoff for each aotf_stepping
        fft_cutoff_dict = {
            1: 4,
            2: 15,
            4: 15,
            8: 40,
        }

    elif channel == "LNO":
        # aotf_steppings = [8.0]
        aotf_steppings = [4.0, 8.0]
        binnings = [0]
        starting_orders = [194]
        # dictionary of fft_cutoff for each aotf_stepping
        fft_cutoff_dict = {}  # don't apply FFT - no ringing


# solar line dict
# solar_line_dict = {
# "20181206_171850-191-4":{"left":np.arange(215, 222), "centre":229, "right":np.arange(234, 245)}, #6.4C
# "20211105_155547-191-4":{"left":np.arange(208, 215), "centre":222, "right":np.arange(227, 238)}, #-2.1C
# "20230112_084925-191-4":{"left":np.arange(201, 208), "centre":215, "right":np.arange(220, 231)}, #-8.3C

# "20181206_171850":{"left":np.arange(144, 148), "centre":149, "right":np.arange(153, 157)},
# "20211105_155547":{"left":np.arange(137, 141), "centre":142, "right":np.arange(146, 150)},

# "20181206_171850":{"left":np.arange(201, 205), "centre":206, "right":np.arange(208, 212)},
# "20211105_155547":{"left":np.arange(194, 198), "centre":200, "right":np.arange(201, 205)},

# "20190416_020948-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
# "20210717_072315-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
# "20220120_125011-194-1":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},

# "20201010_113533-188-8":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},
# "20210201_111011-188-8":{"left":np.arange(209, 213), "centre":218, "right":np.arange(220, 224)},
# "20210523_001053-188-8":{"left":np.arange(205, 209), "centre":212, "right":np.arange(216, 220)},
# "20221011_132104-188-8":{"left":np.arange(209, 213), "centre":215, "right":np.arange(218, 222)},

# LNO
# "20191002_000902-176-8":{"left":np.arange(196, 200), "centre":210, "right":np.arange(220, 224), "centres":[510, 850, 950, 2100, 2760, 2390]}, #-5.4C
# "20200812_135659-176-8":{"left":np.arange(196, 200), "centre":210, "right":np.arange(220, 224), "row":39}, #-3.6C
# "20210606_021551-176-8":{"left":np.arange(196, 200), "centre":206, "right":np.arange(220, 224), "row":39}, #-9.5C

# "20220619_140101-176-4":{"centres":[206, 235, 272]},

# "20190408_032951-194-1":{"centres":[]},
# "20210724_201241-194-1":{"centres":[]},

# "20181021_071529-194-4":{"centres":[129, 139, 178, 195, 236]},
# # "20181121_133754-194-2":{"centres":[]},
# "20190212_145904-194-8":{"centres":[128, 139, 157, 194]},
# "20190408_040458-194-4":{"centres":[125, 134, 184, 191]},
# "20200201_001633-194-4":{"centres":[124, 134, 183, 190]},
# "20200827_133646-194-8":{"centres":[242, 257]},

# "20191002_000902-192-8":{"left":np.arange(203, 207), "centre":214, "right":np.arange(221, 224)},
# }


# get data from h5 files?
# list_files = True
list_files = False

if "d" not in globals():
    list_files = True

# find new solar lines
plot_miniscans = True
# plot_miniscans = False


plot_fft = True
# plot_fft = False

plot_blaze = True
# plot_blaze = False

plot_absorptions = True
# plot_absorptions = False

# plot_aotf = True
plot_aotf = False


# find absorptions in 2d grid
# find local minima


"""get data"""
if __name__ == "__main__":
    if file_level == "hdf5_level_1p0a":
        if list_files:
            h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, starting_orders, aotf_steppings, binnings)
            if "d" not in globals():
                d = get_miniscan_data_1p0a(h5_filenames)
            if h5_prefixes != list(d.keys()):
                d = get_miniscan_data_1p0a(h5_filenames)

    if channel == "SO":
        d2 = remove_oscillations(d, fft_cutoff_dict, cut_off=["inner"], plot=plot_fft)

    elif channel == "LNO":
        # no FFT ringing correction
        d2 = {}
        for h5_prefix in h5_prefixes:

            n_reps = d[h5_prefix]["y_rep"].shape[0]

            # plot LR array spectra for different repetitions
            plt.figure()
            plt.title("Miniscan spectra for different repetitions")
            for rep_ix in range(n_reps):
                plt.plot(d[h5_prefix]["y_rep"][rep_ix, 128, :], label=rep_ix)
            for rep_ix in range(n_reps):
                plt.plot(d[h5_prefix]["y_rep"][rep_ix, :, 160], label=rep_ix)
            plt.legend()

            d2[h5_prefix] = {"aotf": d[h5_prefix]["a_rep"], "t": np.mean(d[h5_prefix]["t_rep"])}  # don't do anything

            # bad pixel correction
            for rep_ix in range(n_reps):
                miniscan_array = d[h5_prefix]["y_rep"][rep_ix, :, :]  # get 2d array for 1st repetition in file
                # miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)
                d2[h5_prefix]["array%i" % (rep_ix)] = miniscan_array
            d2[h5_prefix]["nreps"] = n_reps

    # """plot miniscan arrays"""
    if plot_miniscans:
        for h5_prefix in h5_prefixes:
            temperature = d2[h5_prefix]["t"]
            # array = d2[h5_prefix]["array_raw"]
            array_corrected = d2[h5_prefix]["array1"]

            # plt.figure(figsize=(8, 5), constrained_layout=True)
            # plt.title("Miniscan: %s, %0.2fC" %(h5_prefix, temperature))
            # plt.imshow(array)
            # plt.xlabel("Pixel number")
            # plt.ylabel("Frame index (AOTF frequency)")

            plt.figure(figsize=(8, 5), constrained_layout=True)
            plt.title("Miniscan corrected array: %s, %0.2fC" % (h5_prefix, temperature))
            plt.imshow(array_corrected)
            plt.xlabel("Pixel number")
            plt.ylabel("Frame index (AOTF frequency)")

            # plt.figure(figsize=(8, 5), constrained_layout=True)
            # plt.title("Miniscan difference: %s, %0.2fC" %(h5_prefix, temperature))
            # plt.imshow(array - array_corrected)
            # plt.xlabel("Pixel number")
            # plt.ylabel("Frame index (AOTF frequency)")

    """make HR arrays"""
    for h5_prefix in h5_prefixes:

        n_reps = d2[h5_prefix]["nreps"]
        # HR array spectra for all repetitions
        aotfs = d2[h5_prefix]["aotf"]
        for rep in range(n_reps):

            # interpolate onto high res grid
            array = d2[h5_prefix]["array%i" % rep]
            array_hr, aotf_hr = make_hr_array(array, aotfs, HR_SCALER)
            d2[h5_prefix]["array%i_hr" % rep] = array_hr

        d2[h5_prefix]["aotf_hr"] = aotf_hr

        t_rep = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
        d2[h5_prefix]["t"] = t_rep

    """Correct diagonals"""
    for h5_prefix in h5_prefixes:
        for rep in range(d2[h5_prefix]["nreps"]):
            # calc blaze diagonals

            t = d2[h5_prefix]["t"][rep]
            px_ixs = np.arange(d2[h5_prefix]["array%i_hr" % rep].shape[1])

            px_peaks, aotf_nus = find_peak_aotf_pixel(t, d2[h5_prefix]["aotf_hr"], px_ixs)
            px_peaks = np.asarray(px_peaks) * int(HR_SCALER)
            aotf_nus = np.asarray(aotf_nus)
            blaze_diagonal_ixs_all = get_diagonal_blaze_indices(px_peaks, px_ixs)

            # make diagonally corrected array
            diagonals = []
            diagonals_aotf = []

            for row in range(d2[h5_prefix]["array%i_hr" % rep].shape[0]-5):
                # find closest diagonal pixel number (in first column)
                closest_ix = np.argmin(np.abs(blaze_diagonal_ixs_all[:, 0] - row))
                row_offset = blaze_diagonal_ixs_all[closest_ix, 0] - row

                # apply offset to diagonal indices
                blaze_diagonal_ixs = (blaze_diagonal_ixs_all[closest_ix, :] - row_offset)

                if np.all(blaze_diagonal_ixs < d2[h5_prefix]["array%i_hr" % rep].shape[0]):
                    diagonals.append(d2[h5_prefix]["array%i_hr" % rep][blaze_diagonal_ixs, px_ixs])
                    diagonals_aotf.append(d2[h5_prefix]["aotf_hr"][blaze_diagonal_ixs])

            diagonals = np.asarray(diagonals)
            diagonals_aotf = np.asarray(diagonals_aotf)
            d2[h5_prefix]["array_diag%i_hr" % rep] = diagonals
            d2[h5_prefix]["aotf_diag%i_hr" % rep] = diagonals_aotf

    """Save figures and files"""
    for h5_prefix in h5_prefixes:
        # save diagonally-correct array and aot freqs to hdf5
        with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" % h5_prefix), "w") as f:
            for rep in range(d2[h5_prefix]["nreps"]):
                f.create_dataset("array%02i" % rep, dtype=np.float32, data=d2[h5_prefix]["array_diag%i_hr" % rep],
                                 compression="gzip", shuffle=True)
            f.create_dataset("aotf", dtype=np.float32, data=d2[h5_prefix]["aotf_diag%i_hr" % rep],
                             compression="gzip", shuffle=True)

        # save miniscan png
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.title(h5_prefix)
        plt.imshow(d2[h5_prefix]["array_diag%i_hr" % rep], aspect="auto")
        # plt.savefig(os.path.join(MINISCAN_PATH, channel, "%s.png" %h5_prefix))
        plt.close()

    # for h5_prefix in h5_prefixes:
    #     fig1, (ax1a, ax1b) = plt.subplots(nrows=2)
    #     fig1.suptitle("Diagonals")
    #     for rep in range(d2[h5_prefix]["nreps"])[0:1]:

    #         diagonals = d2[h5_prefix]["array_diag%i_hr" %rep]

    #         stripe = diagonals[int(78*HR_SCALER), int(170*HR_SCALER):int(210*HR_SCALER)]
    #         plt.figure(); plt.plot(stripe)

    #         left = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(180*HR_SCALER)]
    #         right = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(200*HR_SCALER)]

    #         left_corr = left# * np.mean(right) / np.mean(left)
    #         left_right = left#np.mean([right, left_corr], axis=0)

    #         centre1 = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(189*HR_SCALER)]
    #         centre_corr1 = centre1# * np.mean(left_right[0:10]) / np.mean(centre1[0:10])

    #         # plt.plot(left_corr)
    #         # plt.plot(right)
    #         # ax1a.plot(centre_corr1, color="C%i" %rep)
    #         # plt.plot(centre_corr2)
    #         # plt.plot(centre_corr3)
    #         # ax1a.plot(left_right, color="C%i" %rep)

    #         centre_corr1_sg = savgol_filter(centre_corr1, 7, 1)
    #         left_right_sg = savgol_filter(left_right, 7, 1)

    #         ax1a.plot(centre_corr1_sg, label=rep, color="C%i" %rep)
    #         ax1a.plot(left_right_sg, label=rep, color="C%i" %rep)

    #     # plt.figure()
    #         # ax1b.plot(centre_corr1 / left_right, label=rep)
    #         ax1b.plot(centre_corr1_sg / left_right_sg, label=rep, color="C%i" %rep)
    #         # plt.plot(centre_corr2 / left_right)
    #         # plt.plot(centre_corr3 / left_right)
    #     ax1a.legend()
    #     ax1b.legend()

    # stop()
    for h5_prefix in h5_prefixes:
        if plot_miniscans:

            # get blaze diagonal indices and set array values to nan for plotting
            plt.figure()
            for blaze_diagonal_ixs in blaze_diagonal_ixs_all:
                plt.plot(array_hr[blaze_diagonal_ixs, np.arange(array_hr.shape[1])])
                array_hr[blaze_diagonal_ixs, np.arange(array_hr.shape[1])] = np.nan

            plt.figure(figsize=(8, 5), constrained_layout=True)
            plt.title("Miniscan HR array: %s, %0.2fC" % (h5_prefix, temperature))
            plt.imshow(array_hr)
            plt.xlabel("Pixel number")
            plt.ylabel("Frame index (AOTF frequency)")

    #     # band depths
    #     depths = []
    #     for centre in solar_line_dict[h5_prefix]["centres"]:
    #         centre_px = int(centre * HR_SCALER)
    #         ixs_left = np.arange(centre_px - int(6 * HR_SCALER), centre_px - int(5 * HR_SCALER))
    #         ixs_right = np.arange(centre_px + int(5 * HR_SCALER), centre_px + int(6 * HR_SCALER))

    #         depth = []
    #         for line in diagonals:
    #             depth.append(band_depth(line, centre_px, ixs_left, ixs_right, ax=None))
    #         depths.append(depth)

    #     depths = np.asarray(depths)

    #     plt.figure()
    #     plt.plot(depths.T)

    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonals[:, (centre-1):(centre+1)], axis=1), label=centre)
    #     # plt.legend()

    #     # a = np.mean(diagonals[:, 1820:1860], axis=1)

    #     # diagonals = np.array

    #     diagonals_norm = diagonals / np.repeat(np.max(diagonals[:, :], axis=1), array_hr.shape[1]).reshape((-1, array_hr.shape[1]))

    #     plt.figure(figsize=(8, 5), constrained_layout=True)
    #     plt.title("Miniscan diagonally-corrected array: %s, %0.2fC" %(h5_prefix, temperature))
    #     plt.imshow(diagonals_norm, aspect="auto")
    #     plt.xlabel("Diagonally corrrected pixel number")
    #     plt.ylabel("Diagonally corrrected frame index (AOTF frequency)")

    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonals_norm[:, (centre-1):(centre+1)], axis=1), label=centre)
    #     # plt.legend()

    #     diagonal_median = np.median(diagonals_norm, axis=0)

    #     diagonal_div = diagonals_norm / diagonal_median

    #     # plt.figure()
    #     # plt.imshow(diagonal_div, aspect="auto")

    #     # plt.figure()
    #     # for i in range(diagonals_norm.shape[0]):
    #     #     plt.plot(diagonals_norm[i, :], alpha=0.1)
    #     # plt.plot(diagonal_median, "k")

    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonal_div[:, (centre-10):(centre+10)], axis=1), label=centre)
    #     # plt.legend()

    # # if plot_blaze:
    # #     row_offsets = [-2, -1, 0, 1, 2]
    # #     extrapolated_blazes = make_blaze_functions(d2, row_offsets, array_name="array", plot=plot_miniscans)
    # #     mean_blaze = plot_blazes(d2, extrapolated_blazes)

    #     # #save mean blaze function to file
    #     # with open("blaze_%s.tsv" %channel.lower(), "w") as f:
    #     #     f.write("Pixel number\tBlaze function\n")
    #     #     for i, px_blaze in enumerate(mean_blaze):
    #     #         f.write("%i\t%0.4f\n" %(i, px_blaze))

    # #method 1 - normal band depth calc
    # # if plot_aotf:
    # #     plot_aotf(d2)

    # #method 2 - vertical stripe band depth calc

    # # for h5_prefix in h5_prefixes:
    # #     pixel_number = solar_line_dict[h5_prefix]["centre"]
    # #     np.savetxt("raw_aotf_%s.tsv" %h5_prefix, d[h5_prefix]["y_rep"][0, :, pixel_number])
    # # # stop()

    # # for h5_prefix in h5_prefixes[0:1]:

    # #     pixel_number = solar_line_dict[h5_prefix]["centre"]
    # #     band_centre = d[h5_prefix]["y_rep"][0, :, pixel_number]
    # #     band_left = d[h5_prefix]["y_rep"][0, :, pixel_number-10]
    # #     band_right = d[h5_prefix]["y_rep"][0, :, pixel_number+10]

    # #     # row_centre =

    # #     plt.figure()
    # #     plt.plot(band_centre)
    # #     plt.plot(band_left)
    # #     plt.plot(band_right)
