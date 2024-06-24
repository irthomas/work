# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:29:14 2024

@author: iant
"""


import numpy as np
import matplotlib.pyplot as plt


from instrument.nomad_so_instrument_v03 import aotf_peak_nu, lt22_waven
from instrument.nomad_lno_instrument_v02 import nu0_aotf, nu_mp


def find_peak_aotf_pixel(t, aotf_freqs, px_ixs, channel):
    """get pixel number corresponding to the peak of the AOTF in each spectrum"""

    if channel == "so":
        from instrument.nomad_so_instrument_v03 import m_aotf
    elif channel == "lno":
        from instrument.nomad_lno_instrument_v02 import m_aotf

    # spectral calibration to find peak aotf and pixel wavenumbers
    orders = [m_aotf(i) for i in aotf_freqs]
    if channel == "so":
        aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
        px_nus = [lt22_waven(i, t) for i in orders]
    elif channel == "lno":
        aotf_nus = [nu0_aotf(i) for i in aotf_freqs]
        px_nus = [nu_mp(i, px_ixs, t) for i in orders]

    # pixel position of peak AOTF in each frame
    px_peaks = []
    for aotf_nu, px_nu in zip(aotf_nus, px_nus):
        px_peak = (np.abs(px_nu - aotf_nu)).argmin()
        px_peaks.append(px_peak)

    return px_peaks, aotf_nus


def get_diagonal_blaze_indices(px_peaks, px_ixs):

    # make array of blaze functions, one blaze for each row offset value
    blazes = []
    px_ix = 0
    blaze = []
    for array_row_ix, px_peak in enumerate(px_peaks):

        if px_peak < px_ix:  # start new list when order changes
            blazes.append(blaze)
            blaze = []

        px_ix = px_peak

        blaze.append([px_ix, array_row_ix])  # save pixel indices, column and row

    blaze_diagonal_ixs = []
    for blaze in blazes:
        blaze = np.array(blaze)

        # extrapolate to get pixel indices for whole detector and interpolate between pixels
        polyfit = np.polyfit(blaze[:, 0], blaze[:, 1], 1)
        px_extrap = np.array([int(np.round(i)) for i in np.polyval(polyfit, px_ixs)])

        if np.any(px_extrap < 0.0):
            continue

        blaze_diagonal_ixs.append(px_extrap)

    return np.asarray(blaze_diagonal_ixs)


# def make_blaze_functions(d2, row_offsets, array_name="array", plot=False):
#     """inputs:
#     oscillation corrected/uncorrected miniscan array
#     row_offsets: find blaze functions for rows above and below the peak blaze
#     These have the same shape but a lower intensity which is corrected for by the normalisation
#     array_name: the name of the dictionary key in the miniscan array"""

#     extrapolated_blazes = {}

#     for h5_ix, h5_prefix in enumerate(d2.keys()):
#         extrapolated_blazes[h5_prefix] = []

#         px_peaks, _ = find_peak_aotf_pixel(h5_prefix) #get pixel number where AOTF peaks for every spectrum

#         array = d2[h5_prefix][array_name] #get data

#         for offset in row_offsets:
#             #make array of blaze functions, one blaze for each row offset value
#             blazes = []
#             px_ix = 0
#             blaze = []
#             for array_row_ix, px_peak in enumerate(px_peaks):

#                 if px_peak + offset < px_ix: #start new list when order changes
#                     blazes.append(blaze)
#                     blaze = []

#                 px_ix = px_peak + offset

#                 blaze.append([px_ix, array_row_ix]) #save pixel indices, column and row


#             #for plotting only
#             if offset == 0 and plot:
#                 array_peaks = array.copy()

#             for blaze in blazes:
#                 blaze = np.array(blaze)

#                 #extrapolate to get pixel indices for whole detector and interpolate between pixels
#                 polyfit = np.polyfit(blaze[:, 0], blaze[:, 1], 1)
#                 px_range = np.arange(320)
#                 px_extrap = np.array([int(np.round(i)) for i in np.polyval(polyfit, px_range)])

#                 if np.any(px_extrap < 0.0):
#                     continue

#                 #not interpolated or extrapolated
#                 # px_row = array[blaze[:, 1], blaze[:, 0]]

#                 #interpolated/extrapolated
#                 px_row = array[px_extrap, px_range]

#                 extrapolated_blazes[h5_prefix].append(px_row/np.max(px_row))

#                 if offset == 0 and plot:

#                     """plot on miniscan array where pixel nu = aotf nu i.e. the diagonals"""
#                     for px_row_ix, px_column_ix in zip(px_range, px_extrap):
#                         array_peaks[px_column_ix, px_row_ix] = -999

#                 # for row, px_peak in enumerate(px_peaks):
#                 #     array_peaks[row, px_peak] = -999


#             if offset == 0 and plot:
#                 plt.figure(figsize=(8, 5), constrained_layout=True)
#                 plt.title("Miniscan corrected array")
#                 plt.imshow(array_peaks)

#     return extrapolated_blazes


# def plot_blazes(d2, extrapolated_blazes):
#     fig, ax = plt.subplots()
#     for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
#         blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
#         ax.plot(blazes.T, color="C%i" %h5_ix, alpha=0.1) #linestyle=linestyles[h5_ix], alpha=0.2)


#     blaze_all = []
#     for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
#         blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
#         blazes_median = np.median(blazes, axis=0)

#         t = d2[h5_prefix]["t"]

#         #smooth the median and plot
#         blazes_median_rm = running_mean_1d(blazes_median, 9)
#         ax.plot(blazes_median_rm, color="C%i" %h5_ix, label="%s, %0.2fC" %(h5_prefix, t))
#         blaze_all.append(blazes_median_rm)

#     ax.legend()
#     ax.grid()
#     ax.set_title("Derived blaze functions")
#     ax.set_xlabel("Pixel number")
#     ax.set_ylabel("Normalised blaze function")

#     mean_blaze = np.mean(np.array(blaze_all), axis=0)

#     return mean_blaze / np.max(mean_blaze)


# def band_depth(line, centre_px, ixs_left, ixs_right, ax=None):
#     """solar line band depth"""

#     abs_pxs = np.concatenate((ixs_left, ixs_right))
#     abs_vals = line[abs_pxs]

#     cont_pxs = np.arange(ixs_left[0], ixs_right[-1] + 1)
#     cont_vals = np.polyval(np.polyfit(abs_pxs, abs_vals, 2), cont_pxs)

#     abs_vals = line[cont_pxs] / cont_vals
#     abs_depth = 1.0 - abs_vals[centre_px - ixs_left[0]]

#     if ax:
#         ax.plot(cont_pxs, abs_vals)

#     return abs_depth


# def make_aotf_functions(d2, array_name="array", plot=plot_absorptions):
#     """get aotf function from depth of a solar line"""

#     aotfs = {}
#     for h5_prefix in d2.keys():


#         t = d2[h5_prefix]["t"]

#         aotf_freqs = d2[h5_prefix]["aotf"]

#         if channel == "so":
#             aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
#         elif channel == "lno":
#             aotf_nus = [nu0_aotf(i) for i in aotf_freqs]

#         array = d2[h5_prefix][array_name]

#         if plot:
#             fig1, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True)
#             ax1a.set_title(h5_prefix)
#             ax1a.grid()
#             ax1b.grid()

#         ixs_left = solar_line_dict[h5_prefix]["left"]
#         ixs_right = solar_line_dict[h5_prefix]["right"]
#         centre_px = solar_line_dict[h5_prefix]["centre"]

#         #absorption depth variations
#         abs_depths = []
#         for row_ix in np.arange(array.shape[0]):
#             line = array[row_ix, :]

#             if plot:
#                 abs_depth = band_depth(line, centre_px, ixs_left, ixs_right, ax=ax1b)
#                 if abs_depth > 0.1:
#                     cont_pxs = np.arange(ixs_left[0] - 3, ixs_right[-1] + 4)
#                     ax1a.plot(cont_pxs, line[cont_pxs])

#             else:
#                 abs_depth = band_depth(line, centre_px, ixs_left, ixs_right)
#             abs_depths.append(abs_depth)

#             # plt.plot(line)
#             # plt.plot(cont_pxs, cont_vals)

#         aotfs[h5_prefix] = {"aotf_freqs":aotf_freqs, "aotf_nus":aotf_nus, "abs_depths":np.array(abs_depths)}

#     return aotfs


# def plot_aotf(d2, array_name="array"):
#     aotfs = make_aotf_functions(d2, array_name=array_name)

#     plt.figure()
#     for h5_prefix in aotfs.keys():

#         t = d2[h5_prefix]["t"]

#         aotf_nus = aotfs[h5_prefix]["aotf_nus"]
#         aotf_func = aotfs[h5_prefix]["abs_depths"]

#         plt.plot(aotf_nus, aotf_func/np.max(aotf_func), label="%s, %0.2fC" %(h5_prefix, t))
#         plt.xlabel("Wavenumber cm-1")
#         plt.ylabel("Normalised AOTF Function")
#         plt.title("AOTF Functions")

#         with open("aotf_%s.tsv" %h5_prefix, "w") as f:
#             f.write("Wavenumber\tAOTF function\n")
#             for aotf_nu, aotf_f in zip(aotf_nus, aotf_func):
#                 f.write("%0.4f\t%0.4f\n" %(aotf_nu, aotf_f))


#     plt.grid()
#     plt.legend()
