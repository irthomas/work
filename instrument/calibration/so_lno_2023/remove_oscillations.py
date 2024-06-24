# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:23:12 2024

@author: iant
"""


import numpy as np
import matplotlib.pyplot as plt


def remove_oscillations(d, channel, plot=False):
    """make dictionary of miniscan arrays before and after oscillation removal
    input: dictionary of raw miniscan data,
    fft_cutoff: index to start setting fft to zero (symmetrical from centre)
    cut_off_inner: whether to set the inner indices to zero (removes high res oscillations)
    or outer indices to zero (removes large features)"""

    # dictionary of fft_cutoff for each aotf_stepping value
    # this is used to remove oscillations from the spectra by cutting off some parts of the fft'ed spectrum
    if channel == "so":
        fft_cutoff_dict = {
            1: 4,
            2: 15,
            4: 15,
            8: 40,
        }
        cut_off = ["inner"]
    elif channel == "lno":
        fft_cutoff_dict = {
            1: 0,
            2: 0,
            4: 0,
            8: 0,
        }  # don't apply FFT to LNO - no ringing
        cut_off = []

    d2 = {}
    for h5_prefix in d.keys():

        stepping = int(h5_prefix.split("-")[-1])
        fft_cutoff = fft_cutoff_dict[stepping]

        # miniscan_array = np.zeros((len(d[h5_prefix].keys()), 320))
        # for i, aotf_freq in enumerate(d[h5_prefix].keys()):

        #     for temperature in list(d[h5_prefix][aotf_freq].keys())[0:1]:
        #         miniscan_array[i, :] = d[h5_prefix][aotf_freq][temperature] #get 2d array for 1st temperature in file

        # miniscan_array = d[h5_prefix]["y_rep"][0, :, :]  #get 2d array for 1st repetition in file

        n_reps = d[h5_prefix]["y_rep"].shape[0]
        d2[h5_prefix] = {}
        d2[h5_prefix]["nreps"] = n_reps
        d2[h5_prefix]["aotf"] = d[h5_prefix]["a_rep"]
        d2[h5_prefix]["t"] = np.mean(d[h5_prefix]["t_rep"])

        # bad pixel correction
        for rep_ix in range(n_reps):

            miniscan_array = d[h5_prefix]["y_rep"][rep_ix, :, :]  # get 2d array for 1st repetition in file

            if "inner" in cut_off or "outer" in cut_off:

                # simple bad pixel correction
                if channel == "so":
                    miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)

                fft = np.fft.fft2(miniscan_array)

                if plot:
                    fig, ax = plt.subplots()
                    ax.set_title(h5_prefix)
                    ax.plot(fft.real[:, 200])
                    # ax.plot(fft.imag[:, 200])

                if "inner" in cut_off:
                    fft.real[fft_cutoff:(256-fft_cutoff), :] = 0.0
                    fft.imag[fft_cutoff:(256-fft_cutoff), :] = 0.0

                if "outer" in cut_off:
                    fft.real[0:fft_cutoff, :] = 0.0
                    fft.real[(256-fft_cutoff):, :] = 0.0
                    fft.imag[0:fft_cutoff, :] = 0.0
                    fft.imag[(256-fft_cutoff):, :] = 0.0

                if plot:
                    ax.plot(fft.real[:, 200], linestyle=":")
                    # ax.plot(fft.imag[:, 200])

                ifft = np.fft.ifft2(fft).real

                d2[h5_prefix]["array%i" % (rep_ix)] = ifft

            else:  # if no cutoff given, don't do anything
                d2[h5_prefix]["array%i" % (rep_ix)] = miniscan_array

    return d2


"""vertical slices"""
# plt.figure(figsize=(8, 5), constrained_layout=True)
# for file_ix, h5_prefix in enumerate(d2.keys()):
#     aotf_freqs =  [f for f in d[h5_prefix].keys()]
#     miniscan_array = d2[h5_prefix]["array"]
#     for line_ix, line in enumerate([180, 200, 220]):
#         if file_ix == 0:
#             label = "Pixel number %i" %line
#         else:
#             label = ""
#         plt.plot(aotf_freqs, miniscan_array[:, line]+line_ix*100000, label=label, color="C%i" %line_ix)

# for k, v in A_aotf.items():
#     if v in aotf_freqs:
#         plt.axvline(x=v, color="k", linestyle="dashed")

# plt.legend()
# plt.xlabel("AOTF frequency (kHz)")
# plt.ylabel("Signal on detector")
# plt.grid()
# plt.savefig("miniscan_vertical_slice.png")


"""horizontal slices"""
# plt.figure(figsize=(8, 5), constrained_layout=True)
# h5_prefix = list(d2.keys())[-1]
# aotf_freqs =  [f for f in d[h5_prefix].keys()]
# miniscan_array = d2[h5_prefix]["array"]
# for line_ix, line in enumerate([135, 137, 139, 141, 143]):
#     label = "AOTF frequency %0.1f" %aotf_freqs[line]
#     plt.plot(np.arange(320), miniscan_array[line, :], label=label, color="C%i" %line_ix)
# plt.legend()
# plt.xlabel("Pixel number")
# plt.ylabel("Signal on detector")
# plt.grid()
# plt.savefig("miniscan_horizontal_slice.png")
