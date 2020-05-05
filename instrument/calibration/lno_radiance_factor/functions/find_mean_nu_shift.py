# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:30:30 2020

@author: iant
"""

def find_mean_nu_shift(nadir_lines_nu, ref_lines_nu, chi_sq_fits):
    """compare wavenumbers of nadir and reference absorption lines
    and calculate mean spectral shift"""
    
    import numpy as np

    logger_msg = ""

    nu_shifts = []
    chi_sq_matching = []

    for nu_obs_minimum, chi_sq in zip(nadir_lines_nu, chi_sq_fits): #loop through found nadir absorption minima
        found = False
        for nu_ref_minimum in ref_lines_nu: #loop through found hitran absorption minima
            if nu_ref_minimum - 0.3 < nu_obs_minimum < nu_ref_minimum + 0.3: #if absorption is within 1.0cm-1 then consider it found
                found = True
                nu_shift = nu_obs_minimum - nu_ref_minimum
                nu_shifts.append(nu_shift)
                chi_sq_matching.append(chi_sq)
                logger_msg += "line found (shift=%0.3fcm-1); " %nu_shift
        if not found:
            logger_msg += "Warning: matching line not found for line %0.3f; " %nu_obs_minimum
    
    mean_nu_shift = np.mean(nu_shifts) #get mean shift

    logger_msg += "mean shift = %0.3f. " %mean_nu_shift
#    logger.info(logger_info)
    logger_msg += "%i/%i nadir lines matched to ref lines" %(len(nu_shifts), len(nadir_lines_nu))

    return mean_nu_shift, chi_sq_matching, logger_msg


