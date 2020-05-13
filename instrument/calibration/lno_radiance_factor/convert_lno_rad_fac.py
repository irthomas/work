# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:16:27 2020

@author: iant


LNO CALIBRATION:
    1. SIMULATE SOLAR LINES AND MOLECULAR SPECTRA FOR ALL ORDERS, DECIDE WHICH IS MOST PROMINENT
    2. BUILD SOLAR FULLSCAN CALIBRATION TABLE FROM ORDERS WHERE SOLAR LINE IS DETECTED
    3. USE CAL TABLE TO MAKE SOLAR REF SPECTRUM, USE SOLAR/MOLECULAR LINES TO CALIBRATE NADIR DATA


GET NADIR DATA FROM SQL DATABASE AND CALIBRATE WITH RADIANCE FACTOR AUX FILE
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
    YRadianceFactor = Radiance factor
    YRadianceFactorError = error on radiance factor
    YSimple = Radiance simple conversion
    
    
"""

import sys
import os
import numpy as np
#import h5py
#import re
import matplotlib.pyplot as plt
#import datetime
#from scipy import interpolate

#from tools.sql.obs_database import obs_database 
#from tools.sql.make_obs_dict import make_obs_dict
#from tools.plotting.colours import get_colours
from tools.spectra.baseline_als import baseline_als
from tools.spectra.get_y_normalised import get_y_normalised
#from tools.general.get_minima_maxima import get_local_minima
#from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import FIG_X, FIG_Y

from instrument.calibration.lno_radiance_factor.functions.lno_rad_fac_orders import rad_fact_orders_dict
from instrument.calibration.lno_radiance_factor.functions.make_synth_solar_spectrum import make_synth_solar_spectrum
from instrument.calibration.lno_radiance_factor.functions.calculate_radiance_factor import calculate_radiance_factor
from instrument.calibration.lno_radiance_factor.functions.find_mean_nu_shift import find_mean_nu_shift
from instrument.calibration.lno_radiance_factor.functions.find_absorption_lines import find_ref_spectra_minima, find_nadir_spectra_minima
from instrument.calibration.lno_radiance_factor.functions.make_reference_line_dict import make_reference_line_dict

from instrument.calibration.lno_radiance_factor.config import RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME, PFM_AUXILIARY_FILES, THUMBNAIL_DIRECTORY



    
def convert_lno_rad_fac(hdf5_filename, hdf5_file, errorType):
    
#    hdf5_file = open_hdf5_file(hdf5_filename)
    
    diffraction_order = int(hdf5_filename.replace(".h5","").split("_")[-1])
    
    x = hdf5_file["Science/X"][0, :]
    y = get_y_normalised(hdf5_file)
    y_mean_nadir = np.mean(y, axis=1)
    t = hdf5_file["Channel/MeasurementTemperature"][0]

    #get sun-mars distance
    sun_mars_distance = hdf5_file["Geometry/DistToSun"][0,0] #in AU. Take first value in file only

    #get info from rad fac order dictionary
    rad_fact_order_dict = rad_fact_orders_dict[diffraction_order]
    mean_nadir_signal_cutoff = rad_fact_order_dict["mean_sig"]
    



    fig2 = plt.figure(figsize=(FIG_X+7, FIG_Y+4))
    fig2.suptitle(hdf5_filename)
    gs = fig2.add_gridspec(3,2)
    ax2a = fig2.add_subplot(gs[0, 0])
    ax2b = fig2.add_subplot(gs[1, 0], sharex=ax2a)
    ax2c = fig2.add_subplot(gs[2, 0], sharex=ax2a)
    ax2d = fig2.add_subplot(gs[0:2, 1], sharex=ax2a)
    ax2e = fig2.add_subplot(gs[2, 1], sharex=ax2a)
    
    ax2a2 = ax2a.twinx()
    #raw nadir / synth solar counts
    #reference solar/molecular spectra
    #nadir continuum removed
    #rad_fac
    #rad_fac_normalised

    
    validIndices = np.zeros(len(y_mean_nadir), dtype=bool)

    for frameIndex, (spectrum, mean_nadir) in enumerate(zip(y, y_mean_nadir)):
        if mean_nadir > mean_nadir_signal_cutoff:
            ax2a.plot(x, spectrum, "grey", alpha=0.3)#, label="%i %0.1f" %(frameIndex, mean_nadir))
            validIndices[frameIndex] = True
        else:
            validIndices[frameIndex] = False
            
    if sum(validIndices) == 0:
        print("%s: nadir mean signal too small. Max = %0.2f" %(hdf5_filename, np.max(y_mean_nadir)))
        sys.exit()
    else:
        print("%s: %i spectra above nadir mean signal" %(hdf5_filename, sum(validIndices)))
    obs_spectrum = np.mean(y[validIndices, :], axis=0)
    ax2a.plot(x[0], obs_spectrum[0], "b", label="Solar") #dummy for legend only
    ax2a.plot(x, obs_spectrum, "k", label="Nadir")


    #find pixel containing minimum value in subset of real data
    obs_continuum = baseline_als(obs_spectrum)
    obs_absorption = obs_spectrum / obs_continuum
            
    ax2a.plot(x, obs_continuum, "k--")
    ax2c.plot(x, obs_absorption, "k")
    
    solar_line = rad_fact_order_dict["solar_line"]
    solar_molecular = rad_fact_order_dict["solar_molecular"]
    
    
    hr_simulation_filepath = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration", "lno_radiance_factor_order_data", "order_%i.txt" %diffraction_order)

    reference_dict = make_reference_line_dict(rad_fact_order_dict, hr_simulation_filepath)
    nu_solar_hr = reference_dict["nu_hr"]
    solar_spectrum_hr = reference_dict["solar"]
    molecular_spectrum_hr = reference_dict["molecular"]
    molecule = reference_dict["molecule"]

    ax2a.axhline(y=reference_dict["mean_sig"], c="k", linestyle="--", alpha=0.7)
    ax2a.axhline(y=reference_dict["min_sig"], c="k", alpha=0.7)

    ax2b.plot(nu_solar_hr, solar_spectrum_hr, "b", label="Solar")
    ax2b.plot(nu_solar_hr, molecular_spectrum_hr, "r", label="Molecular %s" %molecule)
    ax2b.legend()
    
    chi_sq_matching = []

    if solar_molecular != "": #fit to solar and/or molecular lines
        
        #get wavenumbers of minimua of solar/molecular line reference spectra
        ref_lines_nu, logger_msg = find_ref_spectra_minima(ax2b, reference_dict)
        print(logger_msg)
        #find lines in mean nadir spectrum and check it satisfies the criteria
        nadir_lines_nu, chi_sq_fits, logger_msg = find_nadir_spectra_minima(ax2c, reference_dict, x, obs_spectrum, obs_absorption)
        print(logger_msg)

        #if no lines found in nadir data
        if len(nadir_lines_nu) == 0:
            nadir_lines_fit = False
            x_obs = x
            
        else:
            #compare positions of nadir and reference lines and calculate mean spectral shift
            mean_nu_shift, chi_sq_matching, logger_msg = find_mean_nu_shift(nadir_lines_nu, ref_lines_nu, chi_sq_fits)
            if len(chi_sq_matching) == 0:
                logger_msg += "No match between %i nadir lines and %i reference lines" %(len(nadir_lines_nu), len(ref_lines_nu))
                print(logger_msg)
                nadir_lines_fit = False
                x_obs = x
            else:
                print(logger_msg)
                nadir_lines_fit = True
                x_obs = x - mean_nu_shift #realign observation wavenumbers to match reference lines
        
    
    else:
        nadir_lines_fit = False
        x_obs = x

    #make solar reference spectrum from LNO fullscans
    rad_fac_aux_filepath = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
    synth_solar_spectrum = make_synth_solar_spectrum(diffraction_order, x, t, rad_fac_aux_filepath)
    ax2a2.plot(x, synth_solar_spectrum, "b")

    #calculate radiance factor for all nadir spectra
    y_rad_fac = calculate_radiance_factor(y, synth_solar_spectrum, sun_mars_distance)
    
    for spectrum_index, valid_index in enumerate(validIndices):
        if valid_index:
            ax2d.plot(x_obs, y_rad_fac[spectrum_index, :], "grey", alpha=0.3)
            
    mean_rad_fac = np.mean(y_rad_fac[validIndices, :], axis=0)
    
    #remove wavey continuum from mean radiance factor spectrum
    mean_rad_fac_continuum = baseline_als(mean_rad_fac, lam=500.0)
    rad_fac_normalised = mean_rad_fac / mean_rad_fac_continuum
    
    ax2d.plot(x_obs, mean_rad_fac, "k")
    ax2d.plot(x_obs, mean_rad_fac_continuum, "k--")
    ax2e.plot(x_obs, rad_fac_normalised, "k")

    #format plot
    ax2c.set_xlabel("Wavenumber cm-1")
    ax2e.set_xlabel("Wavenumber cm-1")

    ax2a.set_ylabel("Counts\nper px per second")
    ax2b.set_ylabel("Reference\nspectra")
    ax2c.set_ylabel("Nadir\ncontinuum removed")
    ax2d.set_ylabel("Radiance\nfactor")
    ax2e.set_ylabel("Continuum removed\nradiance factor")
    
    ax2a.set_ylim(bottom=0)
    ax2a2.set_ylim(bottom=0)
    ax2d.set_ylim([min(mean_rad_fac[10:310])-0.05, max(mean_rad_fac[10:310])+0.05])
    ax2e.set_ylim(bottom=0)

    ax2c.set_xlim([np.floor(min(x)), np.ceil(max(x))])
    ax2c.set_xlim([np.floor(min(x)), np.ceil(max(x))])
    ticks = ax2c.get_xticks()
    ax2a.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2a2.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2b.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2c.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2d.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2e.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))

    ax2a.grid()
    ax2b.grid()
    ax2c.grid()
    ax2d.grid()
    ax2e.grid()

    ax2a.legend()
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    #print warnings:
    if not solar_line:
        ax2a.annotate("Warning: no solar reference line correction", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)
    if not nadir_lines_fit:
        ax2c.annotate("Warning: no nadir absorption line correction", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)

    fig2.savefig(os.path.join(THUMBNAIL_DIRECTORY, "%s_rad_fac.png" %hdf5_filename))


    rad_fac_cal_dict = {}
    rad_fac_cal_dict["Science/X"] = {"data":x_obs, "dtype":np.float32, "compression":True}
    rad_fac_cal_dict["Science/YRadFac"] = {"data":y_rad_fac, "dtype":np.float32, "compression":True}
    rad_fac_cal_dict["Criteria/LineFit/NumberOfLinesFit"] = {"data":len(chi_sq_matching), "dtype":np.int16, "compression":False}
    rad_fac_cal_dict["Criteria/LineFit/ChiSqError"] = {"data":chi_sq_matching, "dtype":np.float32, "compression":True}


    #write calibration and error references    
    if solar_line and nadir_lines_fit:
        calib_ref = "Radiance factor calibration fit to solar reference and nadir absorption lines"
        error_ref = ""
        error = False
    elif nadir_lines_fit:
        calib_ref = "Radiance factor calibration fit to nadir absorption lines only (no solar lines)"
        error_ref = ""
        error = True
    elif solar_line:
        calib_ref = "Radiance factor calibration fit to solar reference only (no nadir absorption lines)"
        error_ref = ""
        error = True
    else:
        calib_ref = "Radiance factor calibration did not fit solar reference or nadir absorption lines"
        error_ref = ""
        error = True
        
    rad_fac_refs = {"calib_ref":calib_ref, "error_ref":error_ref, "error":error}
    
    return rad_fac_cal_dict, rad_fac_refs
        
