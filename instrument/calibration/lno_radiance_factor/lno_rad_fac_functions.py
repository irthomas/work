# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:26:03 2020

@author: iant

FUNCTIONS FOR LNO RADIANCE FACTOR CALIBRATION
LNO CALIBRATION IN 3 STEPS:
    1. SIMULATE SOLAR LINES AND MOLECULAR SPECTRA FOR ALL ORDERS, DECIDE WHICH IS MOST PROMINENT
    2. BUILD SOLAR FULLSCAN CALIBRATION TABLE FROM ORDERS WHERE SOLAR LINE IS DETECTED
    3. USE CAL TABLE TO MAKE SOLAR REF SPECTRUM, USE SOLAR/MOLECULAR LINES TO CALIBRATE NADIR DATA

"""

#TESTING=True
TESTING=False

import numpy as np
import os
import datetime
#from scipy.optimize import curve_fit
#from scipy import interpolate
#import matplotlib.pyplot as plt
import h5py
import logging

from tools.file.paths import paths, SYSTEM
#from instrument.nomad_lno_instrument import nu_mp
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.general.get_consecutive_indices import get_consecutive_indices
from tools.general.get_nearest_datetime import get_nearest_datetime
from tools.sql.heaters_temp import get_temperature_range

#from tools.spectra.solar_spectrum_lno import 
#from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import absorption_line_dict


"""set paths to calibration files"""
if SYSTEM == "Windows":
    PFM_AUXILIARY_FILES = paths["PFM_AUXILIARY_FILES"]
else:
    from nomad_ops.config import PFM_AUXILIARY_FILES

#input files
RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration")
RADIOMETRIC_CALIBRATION_ORDERS = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, "lno_radiance_factor_order_data")

#table to output
#LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME = "LNO_Radiometric_Calibration_Table_v03"
LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v03"

logger = logging.getLogger( __name__ )



def get_diffraction_orders(aotf_frequencies):
    """get orders from aotf"""
    aotf_order_coefficients = np.array([3.9186850E-09, 6.3020400E-03, 1.3321030E+01])

    diffraction_orders_calculated = [np.int(np.round(np.polyval(aotf_order_coefficients, aotf_frequency))) for aotf_frequency in aotf_frequencies]
    #set darks to zero
    diffraction_orders = np.asfarray([diffraction_order if diffraction_order > 50 else 0 for diffraction_order in diffraction_orders_calculated])
    return diffraction_orders





def get_nearest_temperature_measurement(hdf5_file, frame_index):
    """get LNO nominal temperature readout closest to chosen frame in hdf5_file"""
    utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0].decode()
    utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, 0].decode()
    utc_start_datetime = datetime.datetime.strptime(utc_start_time, "%Y %b %d %H:%M:%S.%f")
    utc_end_datetime = datetime.datetime.strptime(utc_end_time, "%Y %b %d %H:%M:%S.%f")
    temperatures = get_temperature_range(utc_start_datetime, utc_end_datetime)

    utc_obs_time = hdf5_file["Geometry/ObservationDateTime"][frame_index, 0].decode()
    utc_obs_datetime = datetime.datetime.strptime(utc_obs_time, "%Y %b %d %H:%M:%S.%f")

    obs_temperature_index = get_nearest_datetime([i[0] for i in temperatures], utc_obs_datetime)
    #get LNO nominal temperature
    measurement_temperature = float(temperatures[obs_temperature_index][2])

    return measurement_temperature




def get_reference_dict(diffraction_order, rad_fact_order_dict):
    """get dict of high res smoothed reference spectra from files and additional information"""
    hr_spectra = np.loadtxt(os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration", "lno_radiance_factor_order_data", "order_%i.txt" %diffraction_order), delimiter=",")
    
    reference_dict = {}
    reference_dict["nu_hr"] = hr_spectra[:, 0]
    reference_dict["solar"] = hr_spectra[:, 1]
    reference_dict["molecular"] = hr_spectra[:, 2]
    
    if "solar_molecular" in rad_fact_order_dict.keys():
        solar_molecular = rad_fact_order_dict["solar_molecular"]
        reference_dict["solar_molecular"] = solar_molecular
    else:
        solar_molecular = ""
        reference_dict["solar_molecular"] = ""
        
#    reference_dict["molecule"] = BEST_ABSORPTION_DICT[diffraction_order][1]
#    detection_criteria = BEST_ABSORPTION_DICT[diffraction_order][2]
#    reference_dict["obs_mean_cutoff"] = detection_criteria[0]
#    reference_dict["min_signal"] = detection_criteria[1]
#    reference_dict["obs_abs_stds"] = detection_criteria[2]
#    reference_dict["ref_abs_stds"] = detection_criteria[3]
    
    if solar_molecular == "molecular":
        reference_dict["reference_hr"] = reference_dict["molecular"]
    elif solar_molecular == "solar":
        reference_dict["reference_hr"] = reference_dict["solar"]
    else:
        reference_dict["reference_hr"] = np.array(0.0)

    return reference_dict





def plot_reference_sim(ax, diffraction_order, reference_dict):
    """get high res smoothed solar and hitran spectra of chosen molecule"""

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
        logger.error("Reference absorption not deep enough for detection. Change nadir dict")
        return []

    #find pixel indices containing absorptions in hitran/solar data
    #split indices for different absorptions into different lists
    reference_indices_all = get_consecutive_indices(reference_abs_points)

    #here, don't add extra points to left and right of found indices
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
        x_absorption, y_absorption, reference_spectrum_minimum, chi_sq = fit_gaussian_absorption(nu_hr[reference_indices], reference_dict["reference_hr"][reference_indices], error=True)
        ax.plot(x_absorption, y_absorption, "y")
        ax.axvline(x=reference_spectrum_minimum, c="y")

        true_wavenumber_minima.append(reference_spectrum_minimum)

    
    return nu_hr, reference_dict["reference_hr"], true_wavenumber_minima


#fig, ax = plt.subplots()
#diffraction_order = 115
#reference_dict = get_reference_dict(diffraction_order)
#plot_reference_sim(ax, diffraction_order, reference_dict)


#x_in, y_in, incidence_angle in dictionary
def check_nadir_spectra(axa, axb, reference_dict, obs_dict):

    
    obs_mean_signal_cutoff = reference_dict["obs_mean_cutoff"]
    minimum_signal_for_absorption = reference_dict["min_signal"]
    n_stds_for_absorption = reference_dict["obs_abs_stds"]

    x = obs_dict["x"]
    y = obs_dict["y"]
    min_incidence_angle = np.nanmin(obs_dict["incidence_angle"])

    y[np.isnan(y)] = 0.0 #replace nans
    y_mean = np.nanmean(y[:, 160:240], axis=1)
    #find max value
    y_mean_max = np.max(y_mean)


    if y_mean_max < obs_mean_signal_cutoff:
        logger_msg = ""
        logger_msg += "Minimum incidence angle is %0.1f. " %(min_incidence_angle)
        logger_msg += "Signal too low to use (values from %0.1f to %0.1f). Radiance factor calibration not implemented" %(np.min(np.nanmean(y, axis=1)), np.max(np.nanmean(y, axis=1)))
        logger.warning(logger_msg)
        return [[0],[0],[0], min_incidence_angle]



    #take spectra where mean value is greater than 3/4 of max mean value
    validIndices = np.where(y_mean > (0.75 * y_mean_max))[0]

    #plot spectra
    for validIndex in validIndices:
        axa.plot(x, y[validIndex, :], alpha=0.3)
    #plot mean spectrum
    mean_spectrum = np.mean(y[validIndices, :], axis=0)
    axa.plot(x, mean_spectrum, "k")
    axa.axhline(y=obs_mean_signal_cutoff, c="k", alpha=0.7)
    axa.axhline(y=minimum_signal_for_absorption, c="k", alpha=0.7)
    
    #plot baseline corrected spectra
    mean_spectrum_baseline = baseline_als(mean_spectrum) #find continuum of mean spectrum
    axa.plot(x, mean_spectrum_baseline, "k--")
    
    mean_corrected_spectrum = mean_spectrum / mean_spectrum_baseline
    axb.plot(x[30:320], mean_corrected_spectrum[30:320], "k")
    #do quadratic fit to find true absorption minima
    std_corrected_spectrum = np.std(mean_corrected_spectrum)
    abs_points = np.where((mean_corrected_spectrum < (1.0 - std_corrected_spectrum * n_stds_for_absorption)) & (mean_spectrum > minimum_signal_for_absorption))[0]
#    axb.scatter(xIn[abs_points], mean_corrected_spectrum[abs_points], c="r", s=10)

    axb.axhline(y=1.0-std_corrected_spectrum*n_stds_for_absorption, c="k")

#    if "IncidenceAngle" in hdf5FileIn["Geometry/Point0"].keys():
#        min_incidence_angle = np.nanmin(hdf5FileIn["Geometry/Point0/IncidenceAngle"][...])
#    else:
#        min_incidence_angle = -999
    



    if len(abs_points) == 0:
        logger_msg = ""
        logger_msg += "No absorptions found with sufficient depth. "
        logger_msg += "Minimum incidence angle is %0.1f" %(min_incidence_angle)
        logger.warning(logger_msg)
        return [[0],[0],[0], min_incidence_angle]

        
    #find pixel indices containing absorptions in nadir data
    #split indices for different absorptions into different lists
    indices_all = get_consecutive_indices(abs_points)

    indices_all_extra = []
    #add extra points to left and right of found indices
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




    nu_obs_minima = []
    chi_sq_all = []
    for extra_indices in indices_all_extra:
#        plot quadratic and find wavenumber at minimum
#        coeffs = np.polyfit(xIn[extra_indices], mean_corrected_spectrum[extra_indices], 2)
#        axb.plot(xIn[extra_indices], np.polyval(coeffs, xIn[extra_indices]), "g")
#        spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
#        axb.axvline(x=spectrum_minimum, c="g")
    
#        plot gaussian and find wavenumber at minimum
        x_absorption, y_absorption, spectrum_minimum, chi_sq = fit_gaussian_absorption(x[extra_indices], mean_corrected_spectrum[extra_indices])
        if chi_sq == 0:
            logger.warning("Curve fit failed")
        else:

            axb.scatter(x[extra_indices], mean_corrected_spectrum[extra_indices], c="k", s=10)
            axb.plot(x_absorption, y_absorption, "r--")
            axb.axvline(x=spectrum_minimum, c="r")
            
            nu_obs_minima.append(spectrum_minimum)
            
            chi_sq_all.append(chi_sq)

        
    return nu_obs_minima, validIndices, chi_sq_all
                   



def correct_nu_obs(obs_dict, nu_obs_minima, nu_ref_minima, chi_sq_all):

    #find mean wavenumber shift
    logger_info = ""
    nu_shifts = []
    chi_sq_matching = []
    for nu_obs_minimum, chi_sq in zip(nu_obs_minima, chi_sq_all): #loop through found nadir absorption minima
        found = False
        for nu_ref_minimum in nu_ref_minima: #loop through found hitran absorption minima
            if nu_ref_minimum - 0.3 < nu_obs_minimum < nu_ref_minimum + 0.3: #if absorption is within 1.0cm-1 then consider it found
                found = True
                nu_shift = nu_obs_minimum - nu_ref_minimum
                nu_shifts.append(nu_shift)
                chi_sq_matching.append(chi_sq)
                logger_info += "line found (shift=%0.3fcm-1); " %nu_shift
        if not found:
            logger_info += "Warning: matching line not found for line %0.3f; " %nu_obs_minimum
    
    mean_shift = np.mean(nu_shifts) #get mean shift
    obs_dict["x_corrected"] = obs_dict["x"] - mean_shift #realign observation wavenumbers to match hitran

    logger_info += "mean shift = %0.3f" %mean_shift
    logger.info(logger_info)
    logger.info("%i/%i matching absorption bands found" %(len(nu_shifts), len(nu_obs_minima)))

    return obs_dict, chi_sq_matching





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
