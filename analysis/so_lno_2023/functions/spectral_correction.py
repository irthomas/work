# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:07:00 2026

@author: iant
"""

import numpy as np
from scipy.optimize import curve_fit


def get_consecutive_indices(list_of_indices, n_consecutive_indices=1):
    """from a list of indices, make a list of lists where each list contains only consecutive indices. If there are less than n consecutive values, ignore"""
    indices_list = []
    sub_list = []
    prev_n = -1

    for n in list_of_indices:
        if prev_n + 1 != n:            # end of previous subList and beginning of next
            if sub_list:              # if subList already has elements
                if len(sub_list) > n_consecutive_indices:
                    indices_list.append(sub_list)
                sub_list = []
        sub_list.append(n)
        prev_n = n

    if len(sub_list) > n_consecutive_indices:
        indices_list.append(sub_list)
    return indices_list


def fit_gaussian_absorption(x_in, y_in, error=False):
    """fit inverted gaussian to absorption band. Continuum already normalised"""
    """optional argument to calculate chi sq error"""

    def func(x, a, b, c, d):
        return 1.0 - a * np.exp(-((x - b) / c)**2.0) + d

    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    try:
        popt, pcov = curve_fit(func, x_centred, y_in, p0=[0.05, 0.01, 0.12, 0.0])
    except RuntimeError:  # curve fit failed to find solution
        if error:
            return 0.0, 0.0, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0

    if error:
        y_fit = func(x_centred, *popt)
        chi_squared = np.sum(((y_in - y_fit) / y_fit)**2)  # divide by yfit to normalise large and small absorption bands

    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
    y_hr = func(x_hr - x_mean, *popt)

    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()

    x_min_position = x_hr[min_index]

    # print(popt)

    if error:
        return x_hr, y_hr, x_min_position, chi_squared
    else:
        return x_hr, y_hr, x_min_position


def find_spectra_minima(x, y_norm, min_sig=0.9, stds_sig=3, ax=None):
    """return cm-1 of all absorption lines matching detection criteria"""

    logger_msg = ""

    minimum_signal_for_absorption = min_sig
    n_stds_for_absorption = stds_sig

    # find pixel containing minimum value in subset of real data
    std_corrected_spectrum = np.std(y_norm)
    mean_corrected_spectrum = np.mean(y_norm)

    # plot different stds on the plot
    if ax:
        for std_scalar in np.arange(0.0, n_stds_for_absorption + 2.0, 1.0):
            ax.axhline(y=mean_corrected_spectrum - std_corrected_spectrum * std_scalar, c="k", linestyle=":", alpha=0.2)
        ax.axhline(y=mean_corrected_spectrum - std_corrected_spectrum * n_stds_for_absorption, c="k", linestyle="--")
        ax.axhline(y=min_sig, c="k", linestyle=":")

    abs_points = np.where((y_norm < (mean_corrected_spectrum - std_corrected_spectrum * n_stds_for_absorption)
                           ) & (y_norm < minimum_signal_for_absorption))[0]

    if len(abs_points) == 0:
        logger_msg += "No nadir absorptions found with sufficient signal and depth. "
        return [], [], logger_msg

    # find pixel indices containing absorptions in nadir data
    # split indices for different absorptions into different lists
    indices_all = get_consecutive_indices(abs_points, n_consecutive_indices=0)

    indices_all_extra = []
    # add extra points to left and right of found indices
    for indices in indices_all:
        if len(indices) > 0:
            if (indices[0] - 2) > 0 and (indices[-1] + 2) < (len(y_norm) - 1):
                indices_all_extra.append([indices[0] - 2] + [indices[0] - 1] + indices + [indices[-1] + 1] + [indices[-1] + 2])

    if len(indices_all_extra) == 0:
        logger_msg += "No absorptions found with sufficient depth. "
#        logger_msg += "Minimum incidence angle is %0.1f" %(min_incidence_angle)
        return [], [], logger_msg
    else:
        logger_msg += "Using %i absorption bands for analysis. " % (len(indices_all_extra))

    nu_obs_minima = []
    chi_sq_all = []
    for extra_indices in indices_all_extra:

        # plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, spectrum_minimum, chi_sq = fit_gaussian_absorption(x[extra_indices], y_norm[extra_indices], error=True)
        if chi_sq == 0:
            logger_msg += "Curve fit failed. "
        else:

            if ax:
                ax.scatter(x[extra_indices], y_norm[extra_indices], c="k", s=10)
                ax.plot(x_absorption, y_absorption, "k--")
                ax.axvline(x=spectrum_minimum, c="g")

                nu_obs_minima.append(spectrum_minimum)

                chi_sq_all.append(chi_sq)

    return np.asarray(nu_obs_minima), np.asarray(chi_sq_all), logger_msg
