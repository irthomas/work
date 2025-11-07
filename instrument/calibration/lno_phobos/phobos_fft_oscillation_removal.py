# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:25:30 2025

@author: iant

REMOVE OSCILLATIONS FROM PHOBOS MEAN ROWS IN EACH FRAME USING FFT

y_frame_corr comes from phobos_deimos_raw_signal_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def two_sine(x, A1, f1, phi1, A2, f2, phi2, offset):
    """
    Model function: sum of two sine waves.
    A1, f1, phi1 = amplitude, frequency, phase of first sine
    A2, f2, phi2 = amplitude, frequency, phase of second sine
    offset = vertical offset
    """
    return (A1 * np.sin(2 * np.pi * f1 * x + phi1) +
            A2 * np.sin(2 * np.pi * f2 * x + phi2) +
            offset)


def fit_two_sines(x, y, guess):
    """ fit y(x) with a sum of two sine waves."""

    popt, pcov = curve_fit(two_sine, x, y, p0=guess, maxfev=10000)
    return popt, pcov


guess = [3.36515935e+04, -2.90702034e-04,  1.60813581e+00,  2.80874060e+02,
         5.36361973e-01, -3.65375303e+00, -3.36420385e+04]


y_frame_fitted = np.zeros_like(y_frame_corr)
x = np.arange(y_frame_corr.shape[1])

# for i in range(y_frame_fitted.shape[0]):
#     a_in = y_frame_corr[i, :]

# try:
#     popt, pcov = fit_two_sines(x, a_in, guess)
# except RuntimeError:
#     a_out = np.zeros(y_frame_fitted.shape[0])

# a_out = two_sine(x, *popt)
# y_frame_fitted[i, :] = a_in - a_out

# if i == 63:

#     # # Compare fit with data
#     plt.figure()
#     plt.plot(x, a, "k", label="data")
#     plt.plot(x, two_sine(x, *guess), label="guess")
#     plt.plot(x, two_sine(x, *popt), "r-", label="fit")
#     plt.legend()


# plt.figure()
# plt.imshow(y_frame_corr)

# plt.figure()
# plt.imshow(y_frame_fitted)

# i = 40
# i = 63
# i = 80

y_frame_fft = np.zeros_like(y_frame_corr)

fs = 24

for i in range(y_frame_fft.shape[0]):
    a_in = y_frame_corr[i, :]

    amp_cutoff = 18.0 * len(a_in)

    # Compute FFT
    fft_vals = np.fft.fft(a_in)
    fft_freqs = np.fft.fftfreq(len(a_in), 1/fs)

    fft_vals_mod = fft_vals.copy()

    magnitude = np.abs(fft_vals)
    # magnitude[0] = 0  # ignore DC
    bad_ixs = np.where(magnitude > amp_cutoff)[0]

    fft_vals_mod[bad_ixs] = 0

    # magnitude = np.abs(fft_vals_mod)
    # # magnitude[0] = 0  # ignore DC
    # strongest_idx = np.argmax(magnitude)

    # fft_vals_mod[strongest_idx] = 0.0

    ifft = np.fft.ifft(fft_vals_mod).real

    # Only take the positive half of frequencies
    pos_mask = fft_freqs >= 0
    fft_freqs = fft_freqs
    fft_vals = np.abs(fft_vals) / len(a_in)  # Normalize
    fft_vals_mod = np.abs(fft_vals_mod) / len(a_in)  # Normalize

    y_frame_fft[i, :] = ifft

    if i in [-1]:

        # Plot time domain signal
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x, a_in)
        plt.plot(x, ifft)
        plt.title("Time Domain Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # Plot frequency domain (FFT)
        plt.subplot(1, 2, 2)
        plt.scatter(fft_freqs, fft_vals)
        plt.scatter(fft_freqs, fft_vals_mod)
        plt.title("Frequency Domain (FFT)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")

        plt.tight_layout()

# plt.figure(figsize=(8, 5), constrained_layout=True)
# im1 = plt.imshow(y_frame_corr.T, aspect="auto")
# plt.title("Offset corrected signal on detector from early observation")
# plt.ylabel("Binned detector row index (spatial direction)")
# plt.xlabel("Detector frame index (temporal direction)")

# cbar1 = plt.colorbar(im1)
# cbar1.set_label("Signal on each binned detector row", rotation=270, labelpad=10)

# plt.savefig("lno_phobos_raw_signal_offset_corrected.png")

# plt.figure(figsize=(8, 5), constrained_layout=True)
# im2 = plt.imshow(y_frame_fft.T[6:18, :], aspect="auto", extent=(-0.5, len(y_frame_fft)-0.5, 18-0.5, 6-0.5))
# plt.title("Raw signal on detector after oscillation correction")
# plt.ylabel("Binned detector row index (spatial direction)")
# plt.xlabel("Detector frame index (temporal direction)")

# cbar2 = plt.colorbar(im2)
# cbar2.set_label("Signal on each binned detector row", rotation=270, labelpad=10)

# plt.savefig("lno_phobos_raw_signal_osc_corrected.png")

# frame_mean_corr = np.mean(y_frame_corr, axis=0)
# frame_mean_fft = np.mean(y_frame_fft, axis=0)


mean_mean_corr = np.mean(frame_mean_corr[~np.isin(np.arange(y_frame_corr.shape[1]), [10, 11, 12, 13])])
mean_mean_fft = np.mean(frame_mean_fft[~np.isin(np.arange(y_frame_corr.shape[1]), [10, 11, 12, 13])])
std_mean_corr = np.std(frame_mean_corr[~np.isin(np.arange(y_frame_corr.shape[1]), [10, 11, 12, 13])])
std_mean_fft = np.std(frame_mean_fft[~np.isin(np.arange(y_frame_corr.shape[1]), [10, 11, 12, 13])])

plt.figure(figsize=(9, 5), constrained_layout=True)
plt.plot(frame_mean_corr - mean_mean_corr, label="Offset-corrected signal", marker="o")
plt.plot(frame_mean_fft - mean_mean_fft, label="Oscillation-corrected signal", marker="o")
plt.title("Mean detector counts when averaging all detector frames from early observation")
plt.xlabel("Binned detector row index (spatial direction)")
plt.ylabel("Mean signal on each binned detector row")
plt.grid()

plt.axhline(y=+std_mean_corr, color="C0", linestyle="--", label="Stdev of offset-corrected non-illuminated rows")
plt.axhline(y=-std_mean_corr, color="C0", linestyle="--")
plt.axhline(y=+std_mean_fft, color="C1", linestyle="--", label="Stdev of oscillation-corrected non-illuminated rows")
plt.axhline(y=-std_mean_fft, color="C1", linestyle="--")


plt.legend()

plt.savefig("lno_phobos_raw_mean_signal_osc_corrected.png")
