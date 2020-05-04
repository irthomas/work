# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:16:27 2020

@author: iant


GET NADIR DATA FROM SQL DATABASE AND CALIBRATE WITH RADIANCE FACTOR AUX FILE
PLOT SPECTRAL CALIBRATION FITS TO LINES IN THE ORDER AND CHECK

"""

#import sys
import os
import numpy as np
import h5py
#import re
import matplotlib.pyplot as plt
#import datetime
#from scipy import interpolate

from tools.sql.obs_database import obs_database 
from tools.sql.make_obs_dict import make_obs_dict
from tools.plotting.colours import get_colours
from tools.spectra.baseline_als import baseline_als
from tools.spectra.get_y_normalised import get_y_normalised
from tools.general.get_minima_maxima import get_local_minima
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.paths import paths, FIG_X, FIG_Y

from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import rad_fact_orders_dict
from instrument.calibration.lno_radiance_factor.lno_rad_fac_functions import get_reference_dict, make_synth_solar_spectrum



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
        t = hdf5_file["Channel/MeasurementTemperature"]
        lat = hdf5_file["Geometry/Point0/Lat"][:, 0]
        lon = hdf5_file["Geometry/Point0/Lon"][:, 1]

        #get info from rad fac order dictionary
        rad_fact_order_dict = rad_fact_orders_dict[diffraction_order]
        mean_nadir_signal_cutoff = rad_fact_order_dict["mean_sig"]
        
    
        
    
        ax0.set_title(search_query)
        ax0.scatter(lon, lat, label=hdf5_filename)
        ax0.set_xlabel("Longitude")
        ax0.set_ylabel("Latitude")


    
        fig2, (ax2a, ax2b, ax2c, ax2d) = plt.subplots(nrows=4, figsize=(FIG_X, FIG_Y), sharex=True)
        
        validIndices = np.zeros(len(lat), dtype=bool)

        for frameIndex, (spectrum, mean_nadir) in enumerate(zip(y, y_mean_nadir)):
            if mean_nadir > mean_nadir_signal_cutoff:
                ax2a.plot(x, spectrum, alpha=0.3, label="%i %0.1f" %(frameIndex, mean_nadir))
                validIndices[frameIndex] = True
            else:
                validIndices[frameIndex] = False
                
        if sum(validIndices) == 0:
            print("%s: nadir mean signal too small. Max = %0.2f" %(hdf5_filename, np.max(y_mean_nadir)))
        else:
            print("%s: %i spectra above nadir mean signal" %(hdf5_filename, sum(validIndices)))
        #ax0.legend()
        obs_spectrum = np.mean(y[validIndices, :], axis=0)
        ax2a.plot(x, obs_spectrum, "k")
        ax2a.set_xlabel("Wavenumber cm-1 (approx)")
        ax2a.set_ylabel("Counts per px per second")
        ax2a.grid()
        ax2b.grid()
    

        #find pixel containing minimum value in subset of real data
        obs_continuum = baseline_als(obs_spectrum)
        obs_absorption = obs_spectrum / obs_continuum
        
        
        ax2b.plot(x, obs_absorption)
        
        solar_line = rad_fact_order_dict["solar_line"]
        solar_molecular = rad_fact_order_dict["solar_molecular"]
    
        reference_dict = get_reference_dict(diffraction_order, rad_fact_order_dict)
        nu_solar_hr = reference_dict["nu_hr"]
        solar_spectrum_hr = reference_dict["solar"]
        molecular_spectrum_hr = reference_dict["molecular"]
        molecule = reference_dict["molecule"]
        
    
        if solar_line: #find solar line to calibrate nadir obs
    
            solar_line_max_transmittance = rad_fact_order_dict["trans_solar"]
            solar_line_nu_start = rad_fact_order_dict["nu_range"][0]
            solar_line_nu_end = rad_fact_order_dict["nu_range"][1]
            
            ax2c.plot(nu_solar_hr, solar_spectrum_hr+0.1, "b--", label="Solar")
            ax2c.plot(nu_solar_hr, molecular_spectrum_hr+0.1, "r--", label="Molecule %s" %molecule)
            ax2c.legend()
            
            if solar_molecular == "solar":

                ax2c.axhline(y=solar_line_max_transmittance, c="k", linestyle="--")
                ax2c.axvline(x=solar_line_nu_start, c="k", linestyle="--", alpha=0.5)
                ax2c.axvline(x=solar_line_nu_end, c="k", linestyle="--", alpha=0.5)

                solar_minimum_indices_all = get_local_minima(solar_spectrum_hr)
                
                solar_minimum_indices = [solar_minimum_index for solar_minimum_index in solar_minimum_indices_all \
                                       if solar_spectrum_hr[solar_minimum_index] < solar_line_max_transmittance \
                                       and solar_line_nu_start < nu_solar_hr[solar_minimum_index] < solar_line_nu_end]
                solar_minima_nu = [nu_solar_hr[solar_minimum_index] for solar_minimum_index in solar_minimum_indices]
    
                """find exact wavenumber of solar line minima in nu range (should only be 1)"""
                for solar_minimum_index, solar_minimum_nu in zip(solar_minimum_indices, solar_minima_nu):
                    #get n points on either side
                    n_points = 300
                    solar_line_indices = range(np.max((0, solar_minimum_index - n_points)), np.min((len(nu_solar_hr), solar_minimum_index + n_points)))
                    ax2c.scatter(nu_solar_hr[solar_line_indices], solar_spectrum_hr[solar_line_indices], color="b")


        synth_solar_spectrum = make_synth_solar_spectrum(diffraction_order, x, t)
        ax2b.plot(x, synth_solar_spectrum)
        
        #TODO: add conversion factor to account for solar incidence angle
        #TODO: this needs checking. No nadir or so FOV in calculation!
        rSun = 695510.0 # radius of Sun in km
        dSun = sun_mars_distance * 1.496e+8 #1AU to km
        angleSolar = np.pi * (rSun / dSun) **2 / 2.0 #why /2.0?
        #do I/F using shifted observation wavenumber scale
        YRadFac = y / np.tile(synth_solar_spectrum, [nSpectra, 1]) / angleSolar
        
    
    ax0.legend()
    ax0.grid()



#plt.figure()
#observation_min_pixel = findAbsorptionMininumIndex(observation_absorption_spectrum, plot=True)
observation_min_pixel = findAbsorptionMininumIndex(observation_absorption_spectrum)
observation_min_pixel = observation_min_pixel + continuum_pixels[0]
observation_min_wavenumber = nu_mp(diffractionOrder, observation_min_pixel, observationTemperature)

#calculate wavenumber error            
observation_delta_wavenumber = solar_line_wavenumber - observation_min_wavenumber
print("observation_delta_wavenumber=", observation_delta_wavenumber)

#shift wavenumber scale to match solar line
observation_wavenumbers = nu_mp(diffractionOrder, pixels, observationTemperature) + observation_delta_wavenumber

#plot shifted
ax0.plot(observation_wavenumbers, observation_spectrum, "k--")
ax0.axvline(x=solar_line_wavenumber)

ax0.set_title("LNO averaged nadir observation (Gale Crater)")
ax0.set_xlabel("Wavenumbers (cm-1)")
ax0.set_ylabel("Counts")
ax0.legend()




"""make interpolated spectrum and calibrate observation as I/F"""

#read in coefficients and wavenumber grid
with h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "r") as hdf5File:
    wavenumber_grid_in = hdf5File["%i" %diffractionOrder+"/wavenumber_grid"][...]
    coefficient_grid_in = hdf5File["%i" %diffractionOrder+"/coefficients"][...].T


#find coefficients at wavenumbers matching real observation
corrected_solar_spectrum = []
for observation_wavenumber in observation_wavenumbers:
    index = np.abs(observation_wavenumber - wavenumber_grid_in).argmin()
    
    coefficients = coefficient_grid_in[index, :]
    correct_solar_counts = np.polyval(coefficients, observationTemperature)
    corrected_solar_spectrum.append(correct_solar_counts)
corrected_solar_spectrum = np.asfarray(corrected_solar_spectrum)

ax1.plot(observation_wavenumbers, corrected_solar_spectrum)

#add conversion factor to account for solar incidence angle
rSun = 695510. #km
dSun = 215.7e6 #for 20180611 obs 227.9e6 #km

#find 1 arcmin on sun in km
#d1arcmin = dSun * np.tan((1.0 / 60.0) * (np.pi/180.0))

angleSolar = np.pi * (rSun / dSun) **2
#ratio_fov_full_sun = (np.pi * rSun**2) / (d1arcmin * d1arcmin*4.0)
#SOLSPEC file is Earth TOA irradiance (no /sr )
RADIANCE_TO_IRRADIANCE = angleSolar #check factor 2.0 is good
#RADIANCE_TO_IRRADIANCE = 1.0

conversion_factor = 1.0 / RADIANCE_TO_IRRADIANCE

#do I/F using shifted observation wavenumber scale
observation_i_f = observation_spectrum / corrected_solar_spectrum * conversion_factor

plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(observation_wavenumbers, observation_i_f)
plt.title("Nadir calibrated spectra order %i" %diffractionOrder)
plt.xlabel("Wavenumbers (cm-1)")
plt.ylabel("Radiance factor ratio")
