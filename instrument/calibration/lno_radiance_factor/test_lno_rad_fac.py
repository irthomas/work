# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:16:27 2020

@author: iant


GET NADIR DATA FROM SQL DATABASE AND CALIBRATE WITH RADIANCE FACTOR AUX FILE
PLOT SPECTRAL CALIBRATION FITS TO LINES IN THE ORDER AND CHECK

"""

import sys
#import os
import numpy as np
#import h5py
#import re
import matplotlib.pyplot as plt
#import datetime
#from scipy import interpolate

from tools.sql.obs_database import obs_database 
from tools.sql.make_obs_dict import make_obs_dict
from tools.plotting.colours import get_colours
from tools.spectra.baseline_als import baseline_als
from tools.spectra.get_y_normalised import get_y_normalised
#from tools.general.get_minima_maxima import get_local_minima
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import FIG_X, FIG_Y#, paths

from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import rad_fact_orders_dict
from instrument.calibration.lno_radiance_factor.lno_rad_fac_functions import \
get_reference_dict, make_synth_solar_spectrum, calculate_radiance_factor, find_ref_spectra_minima, find_nadir_spectra_minima, find_nu_shift


plot_track = False

#from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
#from tools.file.hdf5_functions_v04 import makeFileList

file_level = "hdf5_level_0p3a"
database_name = "lno_nadir_%s" %file_level

# currently working for orders 160, 162, 163, 167, 168, 169, 189, 194, 196
# work on order 134. 

search_dict ={
        134:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-30,15], "latitude":[-15,5], "longitude":[127,147]},

        168:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,-2], "latitude":[-15,5], "longitude":[127,147]},
            
        188:{"n_orders":[0,4]},
        189:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[1,2], "latitude":[-10,10], "longitude":[-10,10]},
        
        193:{"n_orders":[0,4], "incidence_angle":[0,10], "temperature":[-5,5]},#, "latitude":[-90,90], "longitude":[-180,180]},
}



def make_query_from_search_dict(search_dict, table_name, diffraction_order):

    search_query = "SELECT * from %s WHERE diffraction_order == %i " %(table_name, diffraction_order)
    
    for key, value in search_dict[diffraction_order].items():
        search_query += "AND %s > %i AND %s < %i " %(key, value[0], key, value[1])
    
    return search_query





"""get nadir data from observations of a region"""
for diffraction_order in [134]:
#for diffraction_order in [168]:

    #search database for parameters
    search_query = make_query_from_search_dict(search_dict, file_level, diffraction_order)
    
    db_obj = obs_database(database_name)
    query_output = db_obj.query(search_query)
    db_obj.close()
    
    #test new calibrations on a nadir observation
    obs_data_dict = make_obs_dict("lno", query_output, filenames_only=True)

    #get filenames matching search parameters
    hdf5_filenames = obs_data_dict["filename"]
    
    colours = get_colours(len(hdf5_filenames))
    
    if plot_track:
        fig0, ax0 = plt.subplots(figsize=(FIG_X, FIG_Y))
    
        #draw rectangle on search area
        rectangle = np.asarray([
            [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][0]], \
            [search_dict[diffraction_order]["longitude"][1], search_dict[diffraction_order]["latitude"][0]], \
            [search_dict[diffraction_order]["longitude"][1], search_dict[diffraction_order]["latitude"][1]], \
            [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][1]], \
            [search_dict[diffraction_order]["longitude"][0], search_dict[diffraction_order]["latitude"][0]], \
        ])
        ax0.plot(rectangle[:, 0], rectangle[:, 1], "k")
    
    
    
    for hdf5_filename in hdf5_filenames:
        
        hdf5_file = open_hdf5_file(hdf5_filename)
        
        x = hdf5_file["Science/X"][0, :]
        y = get_y_normalised(hdf5_file)
        y_mean_nadir = np.mean(y, axis=1)
        t = hdf5_file["Channel/MeasurementTemperature"][0]
        lat = hdf5_file["Geometry/Point0/Lat"][:, 0]
        lon = hdf5_file["Geometry/Point0/Lon"][:, 1]

        #get sun-mars distance
        sun_mars_distance = hdf5_file["Geometry/DistToSun"][0,0] #in AU. Take first value in file only

        #get info from rad fac order dictionary
        rad_fact_order_dict = rad_fact_orders_dict[diffraction_order]
        mean_nadir_signal_cutoff = rad_fact_order_dict["mean_sig"]
        
    
        
    
        if plot_track:
            ax0.set_title(search_query)
            ax0.scatter(lon, lat, label=hdf5_filename)
            ax0.set_xlabel("Longitude")
            ax0.set_ylabel("Latitude")


    
#        fig2, (ax2a, ax2b, ax2c, ax2d, ax2e) = plt.subplots(nrows=5, figsize=(FIG_X, FIG_Y+4.5), sharex=True)
        fig2 = plt.figure(figsize=(FIG_X+7, FIG_Y+4))
        fig2.suptitle(hdf5_filename)
        gs = fig2.add_gridspec(3,2)
        ax2a = fig2.add_subplot(gs[0, 0])
        ax2b = fig2.add_subplot(gs[1, 0], sharex=ax2a)
        ax2c = fig2.add_subplot(gs[2, 0], sharex=ax2a)
        ax2d = fig2.add_subplot(gs[0:2, 1], sharex=ax2a)
        ax2e = fig2.add_subplot(gs[2, 1], sharex=ax2a)
        
#        ax2a.set_title(hdf5_filename)
        ax2a2 = ax2a.twinx()
        #raw nadir / synth solar counts
        #reference solar/molecular spectra
        #nadir continuum removed
        #rad_fac
        #rad_fac_normalised

        
        validIndices = np.zeros(len(lat), dtype=bool)

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
    
        reference_dict = get_reference_dict(diffraction_order, rad_fact_order_dict)
        nu_solar_hr = reference_dict["nu_hr"]
        solar_spectrum_hr = reference_dict["solar"]
        molecular_spectrum_hr = reference_dict["molecular"]
        molecule = reference_dict["molecule"]

        ax2a.axhline(y=reference_dict["mean_sig"], c="k", linestyle="--", alpha=0.7)
        ax2a.axhline(y=reference_dict["min_sig"], c="k", alpha=0.7)

        ax2b.plot(nu_solar_hr, solar_spectrum_hr, "b", label="Solar")
        ax2b.plot(nu_solar_hr, molecular_spectrum_hr, "r", label="Molecular %s" %molecule)
        ax2b.legend()
        
    
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
                mean_nu_shift, chi_sq_matching, logger_msg = find_nu_shift(nadir_lines_nu, ref_lines_nu, chi_sq_fits)
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
        synth_solar_spectrum = make_synth_solar_spectrum(diffraction_order, x, t)
        ax2a2.plot(x, synth_solar_spectrum, "b")

        
        y_rad_fac = calculate_radiance_factor(y, synth_solar_spectrum, sun_mars_distance)
        
        for spectrum_index, valid_index in enumerate(validIndices):
            if valid_index:
                ax2d.plot(x_obs, y_rad_fac[spectrum_index, :], "grey", alpha=0.3)
                
        mean_rad_fac = np.mean(y_rad_fac[validIndices, :], axis=0)
        ax2d.plot(x_obs, mean_rad_fac, "k")
        
        mean_rad_fac_continuum = baseline_als(mean_rad_fac, lam=500.0)
        rad_fac_normalised = mean_rad_fac / mean_rad_fac_continuum
        
        ax2d.plot(x_obs, mean_rad_fac_continuum, "k--")
        ax2e.plot(x_obs, rad_fac_normalised, "k")
    

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


    if plot_track:
        ax0.legend()
        ax0.grid()


