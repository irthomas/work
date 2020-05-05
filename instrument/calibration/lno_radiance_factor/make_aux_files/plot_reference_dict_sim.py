# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:21:11 2020

@author: iant
"""
import numpy as np

from fit_gaussian_absorption import fit_gaussian_absorption
from get_consecutive_indices import get_consecutive_indices



def plot_reference_dict_sim(ax, reference_dict):
    """plot reference line dictionary and find absorption line positions"""
 
    if reference_dict["solar_or_molecular"] == "":
        return [],[],[]
    
    colour = {"Solar":"c", "Molecular":"b"}[reference_dict["solar_or_molecular"]]

    #define spectral range
    nu_hr = reference_dict["nu_hr"]
    
    #plot convolved high res solar spectrum to lower resolution. Scaled to avoid swamping figure
    ax.plot(nu_hr, reference_dict["solar"], "b--")
    ax.plot(nu_hr, reference_dict["molecular"], "c--")

    #search reference spectra for solar / molecular absorptions
    n_stds_for_reference_absorption = reference_dict["ref_abs_stds"]
    std_reference_spectrum = np.std(reference_dict["reference_hr"])

    ax.axhline(y=1.0-std_reference_spectrum*n_stds_for_reference_absorption, c=colour)

    reference_abs_points = np.where(reference_dict["reference_hr"] < (1.0-std_reference_spectrum * n_stds_for_reference_absorption))[0]

    if len(reference_abs_points) == 0:
        print("Reference absorption not deep enough for detection. Change nadir dict")
        return [], [], []

    #find pixel indices containing absorptions in molecular/solar data
    #split indices for different absorptions into different lists
    reference_indices_all = get_consecutive_indices(reference_abs_points)

    #here, don't add extra points to left and right of found indices
    reference_indices_all_extra = []
    for indices in reference_indices_all:
        if len(indices)>0:
            reference_indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1])
    
    
    true_wavenumber_minima = []
    for reference_indices in reference_indices_all_extra:
                    
#        plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, reference_spectrum_minimum, chi_sq = fit_gaussian_absorption(nu_hr[reference_indices], reference_dict["reference_hr"][reference_indices], error=True)
        ax.plot(x_absorption, y_absorption, "y")
        ax.axvline(x=reference_spectrum_minimum, c="y")

        true_wavenumber_minima.append(reference_spectrum_minimum)

    
    return nu_hr, reference_dict["reference_hr"], true_wavenumber_minima


