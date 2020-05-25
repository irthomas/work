# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:02:09 2020

@author: iant
"""
import numpy as np

from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.get_consecutive_indices import get_consecutive_indices
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.baseline_als import baseline_als
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.fit_gaussian_absorption import fit_gaussian_absorption



def find_ref_spectra_minima(ax, reference_dict):
    """return cm-1 of all solar/molecular reference lines matching detection criteria"""
    
    
    logger_msg = ""

    n_stds_for_reference_absorption = reference_dict["stds_ref"]
    
    ref_nu = reference_dict["nu_hr"]
    ref_spectra = reference_dict["reference_hr"]
    
    std_ref_spectrum = np.std(np.asfarray(ref_spectra))
    #plot different stds on the plot
    for std_scalar in np.arange(1.0, n_stds_for_reference_absorption+2.0, 1.0):
        ax.axhline(y=1.0-std_ref_spectrum*std_scalar, c="k", linestyle=":", alpha=0.2)
        
    
    ax.axhline(y=1.0-std_ref_spectrum*n_stds_for_reference_absorption, c="k", linestyle="--")

    true_wavenumber_minima = []

    for ref_spectrum in ref_spectra:
    
        reference_abs_points = np.where(ref_spectrum < (1.0-std_ref_spectrum * n_stds_for_reference_absorption))[0]
    
        if len(reference_abs_points) == 0:
            logger_msg += "Reference absorption not deep enough for detection. y"
            return [], logger_msg
    
        #find pixel indices containing absorptions in hitran/solar data
        #split indices for different absorptions into different lists
        reference_indices_all = get_consecutive_indices(reference_abs_points)
    
        #add extra points to left and right of found indices
        reference_indices_all_extra = []
        for indices in reference_indices_all:
            if len(indices)>0:
                reference_indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1])
        
        
        for reference_indices in reference_indices_all_extra:
    #        plot gaussian and find wavenumber at minimum
            x_absorption, y_absorption, reference_spectrum_minimum, chi_sq_fit = fit_gaussian_absorption(ref_nu[reference_indices], ref_spectrum[reference_indices], error=True)
            ax.plot(x_absorption, y_absorption, "k")
            ax.axvline(x=reference_spectrum_minimum, c="k")
    
            true_wavenumber_minima.append(reference_spectrum_minimum)

    
    return true_wavenumber_minima, ""






def find_nadir_spectra_minima(ax, reference_dict, x, obs_spectrum):
    """return cm-1 of all mean nadir spectrum absorption lines matching detection criteria"""
    

    logger_msg = ""
    
    minimum_signal_for_absorption = reference_dict["min_sig"]
    n_stds_for_absorption = reference_dict["stds_sig"]


    #find pixel containing minimum value in subset of real data
    obs_continuum = baseline_als(obs_spectrum)
    obs_absorption = obs_spectrum / obs_continuum

    std_corrected_spectrum = np.std(obs_absorption)
    mean_corrected_spectrum = np.mean(obs_absorption)

    #plot different stds on the plot
    for std_scalar in np.arange(0.0, n_stds_for_absorption+2.0, 1.0):
        ax.axhline(y=mean_corrected_spectrum - std_corrected_spectrum * std_scalar, c="k", linestyle=":", alpha=0.2)
    ax.axhline(y=mean_corrected_spectrum - std_corrected_spectrum * n_stds_for_absorption, c="k", linestyle="--")



    abs_points = np.where((obs_absorption < (mean_corrected_spectrum - std_corrected_spectrum * n_stds_for_absorption)) & (obs_spectrum > minimum_signal_for_absorption))[0]

    if len(abs_points) == 0:
        logger_msg += "No nadir absorptions found with sufficient signal and depth. "
        return [],[], logger_msg

        
    #find pixel indices containing absorptions in nadir data
    #split indices for different absorptions into different lists
    indices_all = get_consecutive_indices(abs_points)

    indices_all_extra = []
    #add extra points to left and right of found indices
    for indices in indices_all:
        if len(indices) > 0:
            if (indices[0] - 2) > 0 and (indices[-1] + 2) < (len(obs_spectrum) - 1):
                indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1] + [indices[-1]+2])
    
    if len(indices_all_extra) == 0:
        logger_msg += "No absorptions found with sufficient depth. "
#        logger_msg += "Minimum incidence angle is %0.1f" %(min_incidence_angle)
        return [],[], logger_msg
    else:
        logger_msg += "Using %i absorption bands for analysis. " %(len(indices_all_extra))




    nu_obs_minima = []
    chi_sq_all = []
    for extra_indices in indices_all_extra:

        #plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, spectrum_minimum, chi_sq = fit_gaussian_absorption(x[extra_indices], obs_absorption[extra_indices], error=True)
        if chi_sq == 0:
            logger_msg += "Curve fit failed. "
        else:

            ax.scatter(x[extra_indices], obs_absorption[extra_indices], c="k", s=10)
            ax.plot(x_absorption, y_absorption, "k--")
            ax.axvline(x=spectrum_minimum, c="g")
            
            nu_obs_minima.append(spectrum_minimum)
            
            chi_sq_all.append(chi_sq)

        
    return nu_obs_minima, chi_sq_all, logger_msg
                   
