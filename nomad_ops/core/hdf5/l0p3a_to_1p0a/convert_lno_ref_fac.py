# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:16:27 2020

@author: iant


LNO CALIBRATION:
    1. SIMULATE SOLAR LINES AND MOLECULAR SPECTRA FOR ALL ORDERS, DECIDE WHICH IS MOST PROMINENT
    2. BUILD SOLAR FULLSCAN CALIBRATION TABLE FROM ORDERS WHERE SOLAR LINE IS DETECTED
    3. USE CAL TABLE TO MAKE SOLAR REF SPECTRUM, USE SOLAR/MOLECULAR LINES TO CALIBRATE NADIR DATA


GET NADIR DATA FROM SQL DATABASE AND CALIBRATE WITH REFLECTANCE FACTOR AUX FILE
PLOT SPECTRAL CALIBRATION FITS TO LINES IN THE ORDER AND CHECK

# currently working for orders 160, 162, 163, 167, 168, 169, 189, 194, 196
# work on order 134. 


FILENAME CONVENTION:
    
OLD:
    20200430_220801_1p0a_LNO_1_D_168

NEW: 
    _LNO_1_DP_168 (IF FILE PASSES)
    _LNO_1_DF_168 (IF FILE FAILS)

    Y = Radiance
    YReflectanceFactor = Reflectance factor
    YReflectanceFactorError = error on reflectance factor
    YSimple = Radiance simple conversion
    
    
"""

import logging
#import sys
import os
import numpy as np
#import h5py
#import re
import matplotlib.pyplot as plt
import platform
#import datetime
#from scipy import interpolate


from nomad_ops.core.hdf5.l0p3a_to_1p0a.lno_ref_fac_orders import ref_fact_orders_dict

from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.baseline_als import baseline_als
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.get_y_normalised import get_y_normalised
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.make_synth_solar_spectrum import make_synth_solar_spectrum
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.calculate_reflectance_factor import calculate_reflectance_factor
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.find_mean_nu_shift import find_mean_nu_shift
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.find_absorption_lines import find_ref_spectra_minima, find_nadir_spectra_minima
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.make_reference_line_dict import make_reference_line_dict
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.output_filename import output_filename
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.get_min_mean_max_of_field import get_min_mean_max_of_field
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.prepare_nadir_fig_tree import prepare_nadir_fig_tree
from nomad_ops.core.hdf5.l0p3a_to_1p0a.curvature.curvature_functions import get_temperature_corrected_mean_curve, read_hdf5_to_dict
from nomad_ops.core.hdf5.l0p3a_to_1p0a.guillaume.lno_guillaume_cal import convert_ref_fac_guillaume


from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, \
    LNO_REFLECTANCE_FACTOR_CALIBRATION_TABLE_NAME, RADIOMETRIC_CALIBRATION_CURVATURE_FILES, \
    RADIOMETRIC_CALIBRATION_ORDERS, PFM_AUXILIARY_FILES, FIG_X, FIG_Y, SYSTEM, GOOD_PIXELS

#if SYSTEM == "Windows":
#    logging.basicConfig(level=logging.INFO)

__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian . thomas @ aeronomie .be"



logger = logging.getLogger( __name__ )


    
def convert_lno_ref_fac(hdf5_filename, hdf5_file, errorType):
    
    diffraction_order = int(hdf5_filename.split("_")[-1])
    
    x = hdf5_file["Science/X"][0, :]
    y = get_y_normalised(hdf5_file)
    y_mean_nadir = np.mean(y, axis=1)
    t = hdf5_file["Channel/MeasurementTemperature"][0]

    #get sun-mars distance
    sun_mars_distance_au = hdf5_file["Geometry/DistToSun"][0,0] #in AU. Take first value in file only

    #get solar incidence angles from all points in FOV, find mean for each spectrum
    incidence_angles = np.zeros((len(y_mean_nadir), 5))
    for point in range(5):
        incidence_angles[:, point] = np.mean(hdf5_file["Geometry/Point%i/IncidenceAngle" %point][...], axis=1)
    mean_incidence_angles_deg = np.mean(incidence_angles, axis=1)

    #get info from rad fac order dictionary
    ref_fact_order_dict = ref_fact_orders_dict[diffraction_order]
    



    fig2 = plt.figure(figsize=(FIG_X, FIG_Y))
    #axes:
    #raw nadir / synth solar counts
    #reference solar/molecular spectra
    #nadir continuum removed
    #ref_fac
    #ref_fac_normalised
    gs = fig2.add_gridspec(3,2)
    ax2a = fig2.add_subplot(gs[0, 0])
    ax2b = fig2.add_subplot(gs[1, 0], sharex=ax2a)
    ax2c = fig2.add_subplot(gs[2, 0], sharex=ax2a)
    ax2d = fig2.add_subplot(gs[0:2, 1], sharex=ax2a)
    ax2e = fig2.add_subplot(gs[2, 1], sharex=ax2a)
    
    ax2a2 = ax2a.twinx()
    ax2d2 = ax2d.twinx()
    # ax2e2 = ax2e.twinx()

    #if no solar or nadir lines - set cutoff to be 75% of max value
    if ref_fact_order_dict["solar_molecular"] == "":
        mean_nadir_signal_cutoff = np.max(y_mean_nadir) * 0.75
    else:
        mean_nadir_signal_cutoff = ref_fact_order_dict["mean_sig"]

    #first, check how many spectra pass the cutoff test
    validIndices = np.where(y_mean_nadir > mean_nadir_signal_cutoff)[0]
    
#    validIndices = np.zeros(len(y_mean_nadir), dtype=bool)
#    for frameIndex, mean_nadir in enumerate(y_mean_nadir):
#        if mean_nadir > mean_nadir_signal_cutoff:
#            validIndices[frameIndex] = True

            
    if sum(validIndices) == 0:
        logger.info("%s: nadir mean signal too small. Max = %0.2f", hdf5_filename, np.max(y_mean_nadir))
        if SYSTEM == "Windows":
            "%s: nadir mean signal too small. Max = %0.2f" %(hdf5_filename, np.max(y_mean_nadir))
        error = True
        ax2a.annotate("Warning: nadir signal too small for fit", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)

        #reduce cutoff to 75% of max value and try again
        validIndices = np.where(y_mean_nadir > np.max(y_mean_nadir) * 0.75)[0]

        #continue with all spectra, but flag error
#        validIndices = np.ones(len(y_mean_nadir), dtype=bool)
    else:
        logger.info("%s: %i spectra above nadir mean signal", hdf5_filename, sum(validIndices))

    #plot
    for validIndex in validIndices:
        ax2a.plot(x, y[validIndex, :], "grey", alpha=0.3)#, label="%i %0.1f" %(frameIndex, mean_nadir))

    obs_spectrum = np.mean(y[validIndices, :], axis=0)
    ax2a.plot(x[0], obs_spectrum[0], "b", label="Solar") #dummy for legend only
    ax2a.plot(x, obs_spectrum, "k", label="Nadir") #plot good signal averaged spectrum


    #find pixel containing minimum value in subset of real data
    obs_continuum = baseline_als(obs_spectrum)
    obs_absorption = obs_spectrum / obs_continuum
            
    ax2a.plot(x, obs_continuum, "k--")
    ax2c.plot(x, obs_absorption, "k")
    
    solar_line = ref_fact_order_dict["solar_line"]
    solar_molecular = ref_fact_order_dict["solar_molecular"]
    
    
    hr_simulation_filepath = os.path.join(RADIOMETRIC_CALIBRATION_ORDERS, "order_%i.txt" %diffraction_order)

    reference_dict = make_reference_line_dict(ref_fact_order_dict, hr_simulation_filepath)
    nu_solar_hr = reference_dict["nu_hr"]
    solar_spectrum_hr = reference_dict["solar"]
    molecular_spectrum_hr = reference_dict["molecular"]
    molecule = reference_dict["molecule"]

    ax2a.axhline(y=reference_dict["mean_sig"], c="k", linestyle="--", alpha=0.7)
    ax2a.axhline(y=reference_dict["min_sig"], c="k", alpha=0.7)

    ax2b.plot(nu_solar_hr, solar_spectrum_hr, "b", label="Solar")
    ax2b.plot(nu_solar_hr, molecular_spectrum_hr, "r", label="Molecular %s" %molecule)
    
    chi_sq_matching = []

    if solar_molecular != "": #fit to solar and/or molecular lines
        
        #get wavenumbers of minimua of solar/molecular line reference spectra
        ref_lines_nu, logger_msg = find_ref_spectra_minima(ax2b, reference_dict)
        logger.info(logger_msg)
        #find lines in mean nadir spectrum and check it satisfies the criteria
        nadir_lines_nu, chi_sq_fits, logger_msg = find_nadir_spectra_minima(ax2c, reference_dict, x[GOOD_PIXELS], obs_spectrum[GOOD_PIXELS])
        logger.info(logger_msg)

        #if no lines found in nadir data
        if len(nadir_lines_nu) == 0:
            nadir_lines_fit = False
            x_obs = x
            
        else:
            #compare positions of nadir and reference lines and calculate mean spectral shift
            mean_nu_shift, chi_sq_matching, logger_msg = find_mean_nu_shift(nadir_lines_nu, ref_lines_nu, chi_sq_fits)
            if len(chi_sq_matching) == 0:
                logger_msg += "No match between %i nadir lines and %i reference lines" %(len(nadir_lines_nu), len(ref_lines_nu))
                nadir_lines_fit = False
                x_obs = x
            else:
                nadir_lines_fit = True
                x_obs = x - mean_nu_shift #realign observation wavenumbers to match reference lines

            logger.info(logger_msg)
        
    
    else:
        nadir_lines_fit = False
        x_obs = x

    #make solar reference spectrum from LNO fullscans
    ref_fac_aux_filepath = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_REFLECTANCE_FACTOR_CALIBRATION_TABLE_NAME)
    synth_solar_spectrum = make_synth_solar_spectrum(diffraction_order, x, t, ref_fac_aux_filepath)
    ax2a2.plot(x, synth_solar_spectrum, "b")

    #calculate reflectance factor for all nadir spectra
    y_ref_fac, synth_solar_spectrum_tiled, solar_to_nadir_scaling_factor_tiled = calculate_reflectance_factor(y, synth_solar_spectrum, sun_mars_distance_au, mean_incidence_angles_deg)
    
    """get mean temperature corrected curvature from hdf5 aux file"""
    #get data from hdf5 dict
    curvature_dict = read_hdf5_to_dict(os.path.join(RADIOMETRIC_CALIBRATION_CURVATURE_FILES, "lno_reflectance_factor_curvature_order_%i") %diffraction_order)[0]

    mean_curve_shifted = get_temperature_corrected_mean_curve(t, curvature_dict)
    y_ref_fac_flat = y_ref_fac / mean_curve_shifted
    
    for valid_index in validIndices:
        ax2d.plot(x_obs, y_ref_fac_flat[valid_index, :], "grey", alpha=0.3)
            
    mean_ref_fac = np.mean(y_ref_fac[validIndices, :], axis=0)
    mean_ref_fac_flat = np.mean(y_ref_fac_flat[validIndices, :], axis=0)
    
    #remove wavey continuum from mean reflectance factor spectrum
    mean_ref_fac_continuum = baseline_als(mean_ref_fac, lam=500.0)
    ref_fac_normalised = mean_ref_fac / mean_ref_fac_continuum * np.mean(mean_ref_fac[50:310]) #scale to mean ref fac value in centre of detector

    ###add Guillaume calibration###
    ref_fac_baseline_removed = convert_ref_fac_guillaume(hdf5_file, mean_incidence_angles_deg)
    mean_ref_fac_baseline_removed = np.mean(ref_fac_baseline_removed[validIndices, :], axis=0)

    
    ax2d.plot(x_obs, mean_ref_fac, "k", label="Mean YReflectanceFactor")
    ax2d.plot(x_obs, mean_ref_fac_continuum, "k--", label="Continuum Fit")
    ax2d2.plot(x_obs, mean_curve_shifted, "g--", label="Temperature-corrected Mean Curve")

    ax2e.plot(x_obs, ref_fac_normalised, "k--", label="Ref fac (continuum removed & scaled to mean)")
    ax2e.plot(x_obs, mean_ref_fac_flat, "g", label="Temperature-dependent curvature corrected")
    ax2e.plot(x_obs, mean_ref_fac_baseline_removed, "b", label="Guillaume calibration")

    #format plot
    ax2c.set_xlabel("Wavenumber cm-1")
    ax2e.set_xlabel("Wavenumber cm-1")

    ax2a.set_ylabel("Counts\nper px per second")
    ax2b.set_ylabel("Reference\nspectra")
    ax2c.set_ylabel("Nadir\ncontinuum removed")
    ax2d.set_ylabel("Reflectance\nfactor")
    ax2e.set_ylabel("Mean reflectance factor")
    
    ax2a.set_ylim(bottom=0)
    ax2a2.set_ylim(bottom=0)
    
    ax2c.set_ylim((min(obs_absorption[GOOD_PIXELS])-0.1, max(obs_absorption[GOOD_PIXELS])+0.1))
    ax2d.set_ylim([min(mean_ref_fac[10:310])-0.05, max(mean_ref_fac[10:310])+0.05])
    ax2e.set_ylim([min(ref_fac_normalised[10:310])-0.15, max(ref_fac_normalised[10:310])+0.15])
    # ax2e2.set_ylim([min(mean_ref_fac_flat[50:310])-0.05, max(mean_ref_fac_flat[50:310])+0.05])
#    ax2e.set_ylim(bottom=0)

    ax2c.set_xlim([np.floor(min(x)), np.ceil(max(x))])
    ax2c.set_xlim([np.floor(min(x)), np.ceil(max(x))])
    ticks = ax2c.get_xticks()
    ax2a.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2a2.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2b.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2c.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2d.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2d2.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2e.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    # ax2e2.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))

    ax2a.grid()
    ax2b.grid()
    ax2c.grid()
    ax2d.grid()
    ax2e.grid()

    ax2a.legend(loc="upper right")
    ax2b.legend(loc="lower right")
    ax2d.legend(loc="lower right")
    ax2e.legend(loc="lower right")
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    #print warnings:
    if not solar_line:
        ax2a.annotate("Warning: no solar reference line correction", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)
    if not nadir_lines_fit:
        ax2c.annotate("Warning: no nadir absorption line correction", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)





    ref_fac_cal_dict = {}
    ref_fac_cal_dict["Science/SolarSpectrum"] = {"data":synth_solar_spectrum_tiled, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Science/SolarToNadirScalingFactor"] = {"data":solar_to_nadir_scaling_factor_tiled, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Geometry/MeanIncidenceAngle"] = {"data":mean_incidence_angles_deg, "dtype":np.float32, "compression":True}

    ref_fac_cal_dict["Science/X"] = {"data":x_obs, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Science/YReflectanceFactor"] = {"data":y_ref_fac, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Science/MeanCurveShifted"] = {"data":mean_curve_shifted, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Science/YReflectanceFactorFlat"] = {"data":y_ref_fac_flat, "dtype":np.float32, "compression":True}
    ref_fac_cal_dict["Criteria/LineFit/NumberOfLinesFit"] = {"data":len(chi_sq_matching), "dtype":np.int16, "compression":False}
    ref_fac_cal_dict["Criteria/LineFit/ChiSqError"] = {"data":chi_sq_matching, "dtype":np.float32, "compression":True}

    ref_fac_cal_dict["Science/YReflectanceFactorBaselineRemoved"] = {"data":ref_fac_baseline_removed, "dtype":np.float32, "compression":True}

    #write calibration and error references    
    if solar_line and nadir_lines_fit:
        calib_ref = "Reflectance factor calibration fit to solar reference and nadir absorption lines"
        error_ref = ""
        error = False
    elif nadir_lines_fit:
        calib_ref = "Reflectance factor calibration fit to nadir absorption lines only (no solar lines)"
        error_ref = ""
        error = True
    elif solar_line:
        calib_ref = "Reflectance factor calibration fit to solar reference only (no nadir absorption lines)"
        error_ref = ""
        error = True
    else:
        calib_ref = "Reflectance factor calibration did not fit solar reference or nadir absorption lines"
        error_ref = ""
        error = True
    
    #use error to update criteria
    ref_fac_cal_dict["Criteria/LineFit/Error"] = {"data":error, "dtype":np.bool, "compression":False}

    #use error to update figure title to show correct filename
    hdf5_filename_new = output_filename(hdf5_filename, error)
    #manually update level in filename
    hdf5_filename_new = hdf5_filename_new.replace("0p3a", "1p0a")
    
    
    
    #add extra geometry info to title
    lons = get_min_mean_max_of_field(hdf5_file, "Geometry/Point0/Lon")
    lats = get_min_mean_max_of_field(hdf5_file, "Geometry/Point0/Lat")
    ls = get_min_mean_max_of_field(hdf5_file, "Geometry/LSubS")
    incid = get_min_mean_max_of_field(hdf5_file, "Geometry/Point0/IncidenceAngle")
    
    if len(chi_sq_matching) == 0: #change from emtpy list to zero for adding to figure title
        chi_sq_string = "N/A"
    else:
        chi_sq_string = "%0.2f (%i lines fit)" %(np.mean(chi_sq_matching), len(chi_sq_matching))
    
    title = "%s: absorption line fit quality: %s\nL$_s$: %0.3f$^\circ$; mean longitude: %0.2f$^\circ$E; latitude range: %0.1f to %0.1f$^\circ$; solar incidence angle: min=%0.1f$^\circ$; max=%0.1f$^\circ$" \
        %(hdf5_filename_new, chi_sq_string, ls[1], lons[1], lats[0], lats[2], incid[0], incid[2])
    
    fig2.suptitle(title)
    
    thumbnail_path = prepare_nadir_fig_tree("%s_ref_fac.png" %hdf5_filename_new)
    logger.info("Saving thumbnail: %s_ref_fac.png", hdf5_filename_new)
    fig2.savefig(thumbnail_path), 
    
    if platform.system() != "Windows":
        plt.close(fig2)
    
        
    ref_fac_refs = {"calib_ref":calib_ref, "error_ref":error_ref, "error":error}
    
    return ref_fac_cal_dict, ref_fac_refs
        
