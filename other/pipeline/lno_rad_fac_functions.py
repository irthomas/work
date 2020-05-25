# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:26:03 2020

@author: iant

FUNCTIONS FOR LNO RADIANCE FACTOR CALIBRATION
"""

#TESTING=True
TESTING=False

import numpy as np
import os
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
import h5py
import logging


"""set paths to calibration files"""
if not TESTING:
    from nomad_ops.config import PFM_AUXILIARY_FILES
else:
    PFM_AUXILIARY_FILES = r"C:\Users\iant\Documents\DATA\pfm_auxiliary_files"

RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration")

LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME = "LNO_Radiometric_Calibration_Table_v03"
LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v03"


logger = logging.getLogger( __name__ )


##diffraction order: [nadir mean signal cutoff, minimum signal for absorption, n stds for absorption, n stds for reference spectrum absorption]
NADIR_DICT = {

#    118:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    120:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    126:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    130:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    133:[4.0, 2.0, 2.0, 2.0, "H2O"],
#    142:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    151:[4.0, 2.0, 2.0, 2.0, "CO2"],
#    156:[4.0, 2.0, 2.0, 2.0, "CO2"],
        
        
        
    160:[4.0, 2.0, 2.0, 2.0, "CO2"],
    162:[4.0, 2.0, 2.0, 2.0, "CO2"],
    163:[4.0, 2.0, 2.0, 2.0, "CO2"],

    167:[4.0, 2.0, 2.0, 2.0, "H2O"],
    168:[4.0, 2.0, 2.0, 2.0, "H2O"],
    169:[4.0, 2.0, 2.0, 2.0, "H2O"],
    189:[4.0, 2.0, 1.0, 1.0, "CO"],
    194:[4.0, 2.0, 0.45, 4.0, "Solar"],
    196:[4.0, 2.0, 0.45, 4.0, "Solar"],
    #can't do 197+ due to no solar spectrum
}




def baseline_als(y, lam=250.0, p=0.95, niter=10):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z



def fit_gaussian_absorption(x_in, y_in):

    def func(x, a, b, c, d):
        return 1.0 - a * np.exp(-((x - b)/c)**2.0) + d
    
    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    try:
        popt, pcov = curve_fit(func, x_centred, y_in, p0=[0.1, 0.02, 0.25, 0.0])
    except RuntimeError:
        logger.warning("Curve fit failed")
        return [[0], [0], 0, 0]
    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
    y_hr = func(x_hr - x_mean, *popt)
    
    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    
    x_min_position = x_hr[min_index]
    
    #find 1-sigma error
#    std = np.sqrt(np.diag(pcov))
    #find simple chisq error
    y_fit = func(x_centred, *popt)
    chi_sq_fit = np.sum((y_in - y_fit)**2 / y_fit)
    
    return x_hr, y_hr, x_min_position, chi_sq_fit



def getConsecutiveIndices(list_of_indices, n_consecutive_indices=1):
    """from a list of indices, make a list of lists where each list contains only consecutive indices. If there are less than n consecutive values, ignore"""
    b = []
    subList = []
    prev_n = -1
    
    for n in list_of_indices:
        if prev_n+1 != n:            # end of previous subList and beginning of next
            if subList:              # if subList already has elements
                if len(subList)>n_consecutive_indices:
                    b.append(subList)
                subList = []
        subList.append(n)
        prev_n = n
    
    if len(subList)>n_consecutive_indices:
        b.append(subList)
    return b




def getReferenceSpectra(diffractionOrder, ax):
    """get high res solar spectra and hitran spectra of chosen molecule"""
    #define spectral range
    hr_spectra = np.loadtxt(os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration", "order_%i.txt" %diffractionOrder), delimiter=",")
    nu_hr = hr_spectra[:, 0]
    
    solar_to_atmos_scaling_factor = (1.0-np.min(hr_spectra[:, 1]))/(1.0-np.min(hr_spectra[:, 2]))
    
    normalised_solar_spectrum = hr_spectra[:, 1]
    normalised_atmos_spectrum = hr_spectra[:, 2] * solar_to_atmos_scaling_factor - (solar_to_atmos_scaling_factor - 1.0)

    #plot convolved high res solar spectrum to lower resolution. Scaled to avoid swamping figure
    ax.plot(nu_hr, normalised_solar_spectrum, "b--")
    ax.plot(nu_hr, normalised_atmos_spectrum, "c--")

    #search reference spectra for solar / molecular absorptions
    molecule = NADIR_DICT[diffractionOrder][4]
    n_stds_for_reference_absorption = NADIR_DICT[diffractionOrder][3]

    if molecule == "Solar":
        normalised_reference_spectrum = normalised_solar_spectrum
    else:
        normalised_reference_spectrum = normalised_atmos_spectrum
    std_reference_spectrum = np.std(normalised_reference_spectrum)

    if molecule == "Solar":
        ax.axhline(y=1.0-std_reference_spectrum*n_stds_for_reference_absorption, c="b")
    else:
        ax.axhline(y=1.0-std_reference_spectrum*n_stds_for_reference_absorption, c="c")


    reference_abs_points = np.where(normalised_reference_spectrum < (1.0-std_reference_spectrum * n_stds_for_reference_absorption))[0]

    if len(reference_abs_points) == 0:
        logger.error("Reference absorption not deep enough for detection. Change nadir dict")
        return []

    #find pixel indices containing absorptions in hitran/solar data
    #split indices for different absorptions into different lists
    reference_indices_all = getConsecutiveIndices(reference_abs_points)

    #add extra points to left and right of found indices
    reference_indices_all_extra = []
    for indices in reference_indices_all:
        if len(indices)>0:
            reference_indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1])
    
    
    true_wavenumber_minima = []
    for reference_indices in reference_indices_all_extra:
#        plot quadratic and find wavenumber at minimum
#        coeffs = np.polyfit(nu_hr[reference_indices], normalised_reference_spectrum[reference_indices], 2)
#        ax.plot(nu_hr[reference_indices], np.polyval(coeffs, nu_hr[reference_indices]), "b")
#        reference_spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
#        ax.axvline(x=reference_spectrum_minimum, c="b")
                    
#        plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, reference_spectrum_minimum, chi_sq_fit = fit_gaussian_absorption(nu_hr[reference_indices], normalised_reference_spectrum[reference_indices])
        ax.plot(x_absorption, y_absorption, "y")
        ax.axvline(x=reference_spectrum_minimum, c="y")

        true_wavenumber_minima.append(reference_spectrum_minimum)

    
    return nu_hr, normalised_reference_spectrum, true_wavenumber_minima






def checkNadirSpectra(xIn, yBinnedNorm, diffractionOrder, hdf5FileIn, axa, axb):
    
    nadir_mean_signal_cutoff = NADIR_DICT[diffractionOrder][0]
    minimum_signal_for_absorption = NADIR_DICT[diffractionOrder][1]
    n_stds_for_absorption = NADIR_DICT[diffractionOrder][2]


    yBinnedNorm[np.isnan(yBinnedNorm)] = 0.0 #replace nans
    yBinnedNormMean = np.nanmean(yBinnedNorm[:, 160:240], axis=1)
    #find max value
    yBinnedNormMeanMax = np.max(yBinnedNormMean)
    #take spectra where mean value is greater than 3/4 of max mean value
    validIndices = np.where(yBinnedNormMean > (0.75 * np.max(yBinnedNormMean)))[0]

    for validIndex in validIndices:
        axa.plot(xIn, yBinnedNorm[validIndex, :], alpha=0.3)
    #plot mean spectrum
    mean_spectrum = np.mean(yBinnedNorm[validIndices, :], axis=0)
    axa.plot(xIn, mean_spectrum, "k")
    axa.axhline(y=nadir_mean_signal_cutoff, c="k", alpha=0.7)
    axa.axhline(y=minimum_signal_for_absorption, c="k", alpha=0.7)
    
    #plot baseline corrected spectra
    mean_spectrum_baseline = baseline_als(mean_spectrum) #find continuum of mean spectrum
    axa.plot(xIn, mean_spectrum_baseline, "k--")
    
    mean_corrected_spectrum = mean_spectrum / mean_spectrum_baseline
    axb.plot(xIn[30:320], mean_corrected_spectrum[30:320], "k")
    #do quadratic fit to find true absorption minima
    std_corrected_spectrum = np.std(mean_corrected_spectrum)
    abs_points = np.where((mean_corrected_spectrum < (1.0 - std_corrected_spectrum * n_stds_for_absorption)) & (mean_spectrum > minimum_signal_for_absorption))[0]
#    axb.scatter(xIn[abs_points], mean_corrected_spectrum[abs_points], c="r", s=10)

    axb.axhline(y=1.0-std_corrected_spectrum*n_stds_for_absorption, c="k")

    if "IncidenceAngle" in hdf5FileIn["Geometry/Point0"].keys():
        min_incidence_angle = np.min(hdf5FileIn["Geometry/Point0/IncidenceAngle"][...])
    else:
        min_incidence_angle = -999


    if yBinnedNormMeanMax < nadir_mean_signal_cutoff:
        logger_msg = ""
        logger_msg += "Minimum incidence angle is %0.1f. " %(min_incidence_angle)
        logger_msg += "Signal too low to use (values from %0.1f to %0.1f). Radiance factor calibration not implemented" %(np.min(np.nanmean(yBinnedNorm, axis=1)), np.max(np.nanmean(yBinnedNorm, axis=1)))
        logger.warning(logger_msg)
        return [[0],[0],[0], min_incidence_angle]


    if len(abs_points) == 0:
        logger_msg = ""
        logger_msg += "No absorptions found with sufficient depth. "
        logger_msg += "Minimum incidence angle is %0.1f" %(min_incidence_angle)
        logger.warning(logger_msg)
        return [[0],[0],[0], min_incidence_angle]

        
    #find pixel indices containing absorptions in nadir data
    #split indices for different absorptions into different lists
    indices_all = getConsecutiveIndices(abs_points)

    indices_all_extra = []
    #add extra point to left and right of found indices
    for indices in indices_all:
        if len(indices)>0:
            if (indices[0]-2)>0 and (indices[-1]+2)<319:
                indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1] + [indices[-1]+2])
    
    if len(indices_all_extra) == 0:
        logger_msg = ""
        logger_msg += "No absorptions found with sufficient depth. "
        logger_msg += "Minimum incidence angle is %0.1f" %(min_incidence_angle)
        logger.warning(logger_msg)
        return [[0],[0],[0], min_incidence_angle]
    else:
        logger.info("Using %i absorption bands for analysis" %(len(indices_all_extra)))




    observation_wavenumber_minima = []
    chi_sq_fit_all = []
    for extra_indices in indices_all_extra:
#        plot quadratic and find wavenumber at minimum
#        coeffs = np.polyfit(xIn[extra_indices], mean_corrected_spectrum[extra_indices], 2)
#        axb.plot(xIn[extra_indices], np.polyval(coeffs, xIn[extra_indices]), "g")
#        spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
#        axb.axvline(x=spectrum_minimum, c="g")
    
#        plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, spectrum_minimum, chi_sq_fit = fit_gaussian_absorption(xIn[extra_indices], mean_corrected_spectrum[extra_indices])
        if chi_sq_fit == 0:
            logger.warning("Curve fit failed")
        else:

            axb.scatter(xIn[extra_indices], mean_corrected_spectrum[extra_indices], c="k", s=10)
            axb.plot(x_absorption, y_absorption, "r--")
            axb.axvline(x=spectrum_minimum, c="r")
            
            observation_wavenumber_minima.append(spectrum_minimum)
            
            chi_sq_fit_all.append(chi_sq_fit)

        
    return observation_wavenumber_minima, validIndices, chi_sq_fit_all, min_incidence_angle
                   

def correctSpectralShift(xIn, observation_wavenumber_minima, true_wavenumber_minima, chi_sq_fit_all):

    #find mean wavenumber shift
    logger_info = ""
    wavenumber_shifts = []
    chi_sq_fit_matching = []
    for observation_wavenumber_minimum, chi_sq_fit in zip(observation_wavenumber_minima, chi_sq_fit_all): #loop through found nadir absorption minima
        found = False
        for true_wavenumber_minimum in true_wavenumber_minima: #loop through found hitran absorption minima
            if true_wavenumber_minimum - 0.3 < observation_wavenumber_minimum < true_wavenumber_minimum + 0.3: #if absorption is within 1.0cm-1 then consider it found
                found = True
                wavenumber_shift = observation_wavenumber_minimum - true_wavenumber_minimum
                wavenumber_shifts.append(wavenumber_shift)
                chi_sq_fit_matching.append(chi_sq_fit)
                logger_info += "line found (shift=%0.3fcm-1); " %wavenumber_shift
        if not found:
            logger_info += "Warning: matching line not found for line %0.3f; " %observation_wavenumber_minimum
    
    mean_shift = np.mean(wavenumber_shifts) #get mean shift
    observation_wavenumbers = xIn - mean_shift #realign observation wavenumbers to match hitran

    logger_info += "mean shift = %0.3f" %mean_shift
    logger.info(logger_info)
    logger.info("%i/%i matching absorption bands found" %(len(wavenumber_shifts), len(observation_wavenumber_minima)))

    return observation_wavenumbers, chi_sq_fit_matching





def getCorrectedSolarSpectrum(diffractionOrder, observation_wavenumbers, observationTemperature):

    #read in data from radiance factor calibration table - do I/F calibration
#    logger.info("Opening radiance factor calibration file %s.h5 for reading" %LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
    radiometric_calibration_table = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
    with h5py.File("%s.h5" % radiometric_calibration_table, "r") as radianceFactorFile:
        
        if "%i" %diffractionOrder in radianceFactorFile.keys():
        
            #read in coefficients and wavenumber grid
            wavenumber_grid_in = radianceFactorFile["%i" %diffractionOrder+"/wavenumber_grid"][...]
            coefficient_grid_in = radianceFactorFile["%i" %diffractionOrder+"/coefficients"][...].T
            
    #find coefficients at wavenumbers matching real observation
    corrected_solar_spectrum = []
    for observation_wavenumber in observation_wavenumbers:
        index = np.abs(observation_wavenumber - wavenumber_grid_in).argmin()
        
        coefficients = coefficient_grid_in[index, :]
        correct_solar_counts = np.polyval(coefficients, observationTemperature)
        corrected_solar_spectrum.append(correct_solar_counts)
    corrected_solar_spectrum = np.asfarray(corrected_solar_spectrum)
    
    return corrected_solar_spectrum
