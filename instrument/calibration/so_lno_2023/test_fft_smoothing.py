# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:25:54 2024

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.spectra.savitzky_golay import savitzky_golay

nrows, ncols = miniscan_array.shape

cut_off = ["inner"]
direction = "v"

ix = 200

plt.figure(figsize=(20, 10))
plt.plot(miniscan_array[:, ix], "k")
# plt.plot(miniscan_array[ix, :], "k")

smooth = savitzky_golay(miniscan_array[:, ix], 9, 2)
plt.plot(smooth)

stop()

# fft_cutoff = 30
for fft_cutoff in np.arange(19, 161, 20):

    # if direction == "h":
    #     plt.plot(fft.real[:, 200], label="FFT before correction")
    # elif direction == "v":
    #     plt.plot(fft.real[200, :], label="FFT before correction")

    fft = np.fft.fft2(miniscan_array.copy())

    if "inner" in cut_off:
        if direction == "h":
            fft.real[fft_cutoff:(nrows-fft_cutoff), :] = 0.0
            fft.imag[fft_cutoff:(nrows-fft_cutoff), :] = 0.0
        elif direction == "v":
            fft.real[:, fft_cutoff:(ncols-fft_cutoff)] = 0.0
            fft.imag[:, fft_cutoff:(ncols-fft_cutoff)] = 0.0

    if "outer" in cut_off:
        if direction == "h":
            fft.real[0:fft_cutoff, :] = 0.0
            fft.real[(nrows-fft_cutoff):, :] = 0.0
            fft.imag[0:fft_cutoff, :] = 0.0
            fft.imag[(nrows-fft_cutoff):, :] = 0.0
        elif direction == "v":
            fft.real[:, 0:fft_cutoff] = 0.0
            fft.real[:, (ncols-fft_cutoff):] = 0.0
            fft.imag[:, 0:fft_cutoff] = 0.0
            fft.imag[:, (ncols-fft_cutoff):] = 0.0

    # if direction == "h":
    #     plt.plot(fft.real[:, 200], linestyle=":", label="FFT after correction")
    # elif direction == "v":
    #     plt.plot(fft.real[200, :], linestyle=":", label="FFT after correction")

    ifft = np.fft.ifft2(fft).real

    plt.plot(ifft[:, ix]*10, label=fft_cutoff, alpha=0.5)
    plt.plot(ifft[ix, :], label=fft_cutoff, alpha=0.5)

plt.legend()
plt.grid()
