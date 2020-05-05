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
#import logging

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
LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v04"



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
        reference_dict["reference_hr"] = [reference_dict["molecular"]]
        reference_dict["molecule"] = rad_fact_order_dict["molecule"]
        

    elif solar_molecular == "solar":
        reference_dict["reference_hr"] = [reference_dict["solar"]]
        reference_dict["molecule"] = ""
 
    elif solar_molecular == "both":
        reference_dict["reference_hr"] = [reference_dict["solar"], reference_dict["molecular"]]
        reference_dict["molecule"] = rad_fact_order_dict["molecule"]

    else:
        reference_dict["reference_hr"] = [np.array(0.0)]
        reference_dict["molecule"] = ""

    if solar_molecular != "":
        #add other keys to dictionary
        for key_name in ["mean_sig","min_sig","stds_sig","stds_ref"]:
            reference_dict[key_name] = rad_fact_order_dict[key_name]
    else:
        for key_name in ["mean_sig","min_sig","stds_sig","stds_ref"]:
            reference_dict[key_name] = 0.0

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
        print("Reference absorption not deep enough for detection. Change nadir dict")
        return [], [], []

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



def find_ref_spectra_minima(ax, reference_dict):
    """return cm-1 of all solar/molecular reference lines matching detection criteria"""
    
    logger_msg = ""

    n_stds_for_reference_absorption = reference_dict["stds_ref"]
    ref_nu = reference_dict["nu_hr"]
    ref_spectra = reference_dict["reference_hr"]
    
    std_ref_spectrum = np.std(np.asfarray(ref_spectra))
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



def find_nadir_spectra_minima(ax, reference_dict, x, obs_spectrum, obs_absorption):

    logger_msg = ""
    
    minimum_signal_for_absorption = reference_dict["min_sig"]
    n_stds_for_absorption = reference_dict["stds_sig"]


    #find pixel containing minimum value in subset of real data
    obs_continuum = baseline_als(obs_spectrum)
    obs_absorption = obs_spectrum / obs_continuum

    std_corrected_spectrum = np.std(obs_absorption)
    ax.axhline(y=1.0-std_corrected_spectrum*n_stds_for_absorption, c="k", linestyle="--")



    abs_points = np.where((obs_absorption < (1.0 - std_corrected_spectrum * n_stds_for_absorption)) & (obs_spectrum > minimum_signal_for_absorption))[0]

    if len(abs_points) == 0:
        logger_msg += "No nadir absorptions found with sufficient signal and depth. "
        return [],[], logger_msg

        
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
                   



def find_nu_shift(nadir_lines_nu, ref_lines_nu, chi_sq_fits):

    #find mean wavenumber shift
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





def make_synth_solar_spectrum(diffraction_order, observation_wavenumbers, observation_temperature):

    #read in data from radiance factor calibration table - do I/F calibration
#    logger.info("Opening radiance factor calibration file %s.h5 for reading" %LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
    radiometric_calibration_table = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
    with h5py.File("%s.h5" % radiometric_calibration_table, "r") as radianceFactorFile:
        
        if "%i" %diffraction_order in radianceFactorFile.keys():
        
            #read in coefficients and wavenumber grid
            wavenumber_grid_in = radianceFactorFile["%i" %diffraction_order+"/wavenumber_grid"][...]
            coefficient_grid_in = radianceFactorFile["%i" %diffraction_order+"/coefficients"][...].T
            
    #find coefficients at wavenumbers matching real observation
    corrected_solar_spectrum = np.zeros_like(observation_wavenumbers)
    for pixel_index, observation_wavenumber in enumerate(observation_wavenumbers):
        index = np.abs(observation_wavenumber - wavenumber_grid_in).argmin()
        
        coefficients = coefficient_grid_in[index, :]
        correct_solar_counts = np.polyval(coefficients, observation_temperature)
        corrected_solar_spectrum[pixel_index] = correct_solar_counts
    
    return corrected_solar_spectrum




def calculate_radiance_factor(y, synth_solar_spectrum, sun_mars_distance_au):
    """calculate radiance factor from i and f, accounting for sun to mars distance"""
    
    n_spectra = len(y[:, 0])
    
    #TODO: add conversion factor to account for solar incidence angle
    #TODO: this needs checking. No nadir or so FOV in calculation!
    rSun = 695510.0 # radius of Sun in km
    dSun = sun_mars_distance_au * 1.496e+8 #1AU to km
    angle_solar = np.pi * (rSun / dSun) **2 / 2.0 #why /2.0?
    #do I/F using shifted observation wavenumber scale
    y_rad_fac = y / np.tile(synth_solar_spectrum, [n_spectra, 1]) / angle_solar
        
    return y_rad_fac
