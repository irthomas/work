# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:09:59 2025

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def scale_spectrum(params, spectrum):
    [scaler, offset] = params

    return spectrum * scaler + offset


def min_solar(params, args):

    [spectrum, solar_spectrum] = args

    chisq_px = (scale_spectrum(params, solar_spectrum) - spectrum)**2
    chisq = np.sum(chisq_px)
    # print(params, chisq)
    return float(chisq)


def fit_solar_spectrum(spectrum, first_guess, solar_spectrum):
    # least squares doesn't work here?
    # res = minimize(min_solar, first_guess, args=[spectrum, solar_spectrum], method="Nelder-Mead", bounds=((0.0, 1000.0), (-999.0, 999.0)))
    res = minimize(min_solar, first_guess, args=[spectrum, solar_spectrum], method="Nelder-Mead")

    scaled_spectrum = scale_spectrum(res.x, solar_spectrum)
    # set offset to zero
    corr_spectrum = scale_spectrum([res.x[0], 0], solar_spectrum)
    return scaled_spectrum, corr_spectrum, res.x


def fit_spectra(y_3d, solar_spectrum, plot=[[]]):
    """correct offsets in input 3d array of raw spectra with a fit to the solar spectrum, replacing data by best fit solar spectrum
    y_3d has dimensions frames x rows x wavelengths and plot is list of [row x frame] index lists"""

    nframes = y_3d.shape[0]
    nrows = y_3d.shape[1]

    fitted_spectra = np.zeros_like(y_3d)
    corr_spectra = np.zeros_like(y_3d)
    fitted_params = np.zeros((nframes, nrows, 2))

    for row in range(nrows):
        for frame in range(nframes):

            spectrum = y_3d[frame, row, :]

            # now fit the solar spectrum function
            first_guess = [float(np.max(spectrum)), float(np.mean(spectrum[0:50]))]

            # params are: scaler, offset
            scaled_spectrum, corr_spectrum, params = fit_solar_spectrum(spectrum, first_guess, solar_spectrum)

            # stop()

            fitted_spectra[frame, row, :] = scaled_spectrum
            corr_spectra[frame, row, :] = corr_spectrum
            fitted_params[frame, row, :] = params

            if len(plot) > 1 and len(plot[0]) > 1:
                for plot_ix in plot:
                    if row == plot_ix[0] and frame == plot_ix[1]:

                        fig, ax = plt.subplots()
                        ax.set_title("Offset correction check, row=%i, frame=%i" % (row, frame))
                        ax.plot(y_3d[frame, row, :], alpha=0.5)
                        ax.plot(fitted_spectra[frame, row, :], alpha=0.5)
                        ax.plot(corr_spectra[frame, row, :], alpha=0.5)

    return fitted_spectra, corr_spectra, fitted_params
