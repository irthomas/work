# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:00:02 2020


MAKE SOLAR FULLSCAN CAL FILE
"""


import os
import numpy as np
import h5py
import re
import matplotlib.pyplot as plt
import datetime
from scipy import interpolate


from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions_v04 import getFile, makeFileList

from tools.spectra.baseline_als import baseline_als
#from tools.spectra.fit_polynomial import fit_polynomial
#from tools.general.get_nearest_index import get_nearest_index
from tools.plotting.colours import get_colours

from instrument.nomad_lno_instrument import nu_mp, A_aotf, temperature_p0
from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import BEST_ABSORPTION_DICT
from instrument.calibration.lno_radiance_factor.lno_rad_fac_functions import get_reference_dict
from instrument.heaters_temp import get_temperature_range
from tools.spectra.nu_hr_grid import nu_hr_grid


FORMAT_STR_SECONDS = "%Y %b %d %H:%M:%S.%f"


regex = re.compile("(20161121_233000|20180702_112352|20181101_213226|20190314_021825|20190609_011514|20191207_051654)_0p1a_LNO_1")
fileLevel = "hdf5_level_0p1a"
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)


def get_diffraction_orders(aotf_frequencies):
    """get orders from aotf"""
    aotf_order_coefficients = np.array([3.9186850E-09, 6.3020400E-03, 1.3321030E+01])

    diffraction_orders_calculated = [np.int(np.round(np.polyval(aotf_order_coefficients, aotf_frequency))) for aotf_frequency in aotf_frequencies]
    #set darks to zero
    diffraction_orders = np.asfarray([diffraction_order if diffraction_order > 50 else 0 for diffraction_order in diffraction_orders_calculated])
    return diffraction_orders

def get_nearest_datetime(datetime_list, search_datetime):
    time_diff = np.abs([date - search_datetime for date in datetime_list])
    return time_diff.argmin(0)


    




#read in solar spectra from order files
#list most prominent line wavenumbers
#find most prominant line in each fullscan order


##dictionary of approximate solar line positions (to begin calculation)
#solarLineDict = {
#118:2669.8,
#120:2715.5,
#121:2733.2,
#126:2837.8,
#130:2943.7,
#133:3012.0,
#142:3209.4,
#151:3414.4,
#156:3520.4,
#160:3615.1,
#162:3650.9,
#163:3688.0,
#164:3693.7,
#166:3750.0,
#167:3767.0,
#168:3787.9,
#169:3812.5,
#173:3902.4,
#174:3934.1,
#178:4021.3,
#179:4042.7,
#180:4069.5,
#182:4101.5,
#184:4156.9,
#189:4276.1,
#194:4383.2,
#195:4402.6,
#196:4422.0,
#}

#best orders
#bestOrders = [118, 120, 142, 151, 156, 162, 166, 167, 178, 189, 194]
#bestOrders = solarLineDict.keys()


"""plot solar lines in solar fullscan data for orders contains strong solar lines"""
temperature_range = np.arange(-20., 15., 0.1)
pixels = np.arange(320.0)

colours = get_colours(len(temperature_range), "plasma")


#output_title = "LNO_Radiance_Factor_Calibration_Table"
#hdf5File = h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "w")
    
for diffraction_order in BEST_ABSORPTION_DICT.keys():
    
    reference_dict = get_reference_dict(diffraction_order)
    
    #plot in nu
    fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2), sharex=True)
    fig1.suptitle("Diffraction Order %i" %diffraction_order)
    #plot in px
    fig2, (ax2a, ax2b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
    fig2.suptitle("Diffraction Order %i" %diffraction_order)
#
#    fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))
#    fig3.suptitle("Diffraction Order %i" %diffractionOrder)
    
    nu_solar_hr = reference_dict["nu_hr"]
    solar_spectrum_hr = reference_dict["solar"]
    px_hr = np.linspace(0.0, 320.0, num=len(nu_solar_hr ))
    nu_px_p0 = nu_mp(diffraction_order, pixels, temperature_p0)
    nu_px_p0_hr = nu_mp(diffraction_order, px_hr, temperature_p0)

    ax1a.plot(nu_solar_hr , solar_spectrum_hr)
    
    solar_minimum_index = np.where(np.min(solar_spectrum_hr)==solar_spectrum_hr)[0][0]
    solar_minimum_nu = nu_solar_hr[solar_minimum_index]
    
    """find exact wavenumber of solar line"""
    
    #get n points on either side
    n_points = 300
    solar_line_indices = range(solar_minimum_index - n_points, solar_minimum_index + n_points)
    ax1a.scatter(nu_solar_hr[solar_line_indices], solar_spectrum_hr[solar_line_indices], color="b")
    
    #find all local minima in this range
    minimum_index = (np.diff(np.sign(np.diff(solar_spectrum_hr[solar_line_indices]))) > 0).nonzero()[0] + 1
    #check if 1 minima only was found
    if len(minimum_index) == 1:
        #plot minimum
        ax1b.axvline(x=solar_minimum_nu, c="k", linestyle="--")
        
        #convert to pixel (p0=0) and plot on pixel grid
        #find subpixel containing solar line when p0=0
        px_p0_hr_index = (np.abs(nu_px_p0_hr - solar_minimum_nu)).argmin()
        solar_minimum_px_p0 = px_hr[px_p0_hr_index]
        print("Solar line pixel (p0=0)= ", solar_minimum_px_p0)
        
        ax2b.axvline(x=solar_minimum_px_p0, c="k", linestyle="--")

    else:
        print("Error: %i minima found for order %i" %(len(minimum_index), diffraction_order))


    calibration_dict = {}

    """get fullscan data"""
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        
        y_raw = hdf5_file["Science/Y"][...]
        aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
        
        #get temperature from TGO readout instead of channel
        utc_start_time = hdf5_file["Geometry/ObservationDateTime"][0, 0].decode()
        utc_end_time = hdf5_file["Geometry/ObservationDateTime"][-1, 0].decode()
        utc_start_datetime = datetime.datetime.strptime(utc_start_time, "%Y %b %d %H:%M:%S.%f")
        utc_end_datetime = datetime.datetime.strptime(utc_end_time, "%Y %b %d %H:%M:%S.%f")
        temperatures = get_temperature_range(utc_start_datetime, utc_end_datetime)
            
        diffraction_orders = get_diffraction_orders(aotf_frequencies)
        frame_indices = list(np.where(diffraction_orders == diffraction_order)[0])
 

        frame_index = frame_indices[0] #just take first frame of solar fullscan
        utc_obs_time = hdf5_file["Geometry/ObservationDateTime"][frame_index, 0].decode()
        utc_obs_datetime = datetime.datetime.strptime(utc_obs_time, "%Y %b %d %H:%M:%S.%f")

        obs_temperature_index = get_nearest_datetime([i[0] for i in temperatures], utc_obs_datetime)
        #get LNO nominal temperature
        measurement_temperature = float(temperatures[obs_temperature_index][2])

        nu_px = nu_mp(diffraction_order, pixels, measurement_temperature)

        #for plotting colour
        temperature_index = (np.abs(temperature_range - measurement_temperature)).argmin()
        
        #cut unused / repeated orders from file (01pa only)
        #solar spectrum - take centre line only
        y_spectrum = y_raw[frame_index, 11, :]
        aotf_frequency = aotf_frequencies[frame_index]
    
        integration_time_raw = hdf5_file["Channel/IntegrationTime"][0]
        number_of_accumulations_raw = hdf5_file["Channel/NumberOfAccumulations"][0]
        
        integration_time = np.float(integration_time_raw) / 1.0e3 #microseconds to seconds
        number_of_accumulations = np.float(number_of_accumulations_raw)/2.0 #assume LNO nadir background subtraction is on
        n_px_rows = 1.0
        
        measurement_seconds = integration_time * number_of_accumulations
        #normalise to 1s integration time per pixel
        spectrum_counts = y_spectrum / measurement_seconds / n_px_rows
        label = "%s order %0.0fkHz %0.1fC" %(hdf5_filename[:15], aotf_frequency, measurement_temperature)

        #remove baseline
        y_baseline = baseline_als(y_spectrum) #find continuum of mean spectrum
        y_corrected = y_spectrum / y_baseline



        """find centres of solar lines"""
        #step1: find nearest pixels that contain the solar line
        solarLinePixelIndex1 = (np.abs(nu_px - solar_minimum_nu)).argmin()
        nPoints = 3
        solarLinePixelIndices1 = range(max([0, solarLinePixelIndex1-nPoints]), min([320, solarLinePixelIndex1+nPoints+1]))
    

        #step2: find pixel containing minimum points
        solarLinePixelIndex2 = (np.abs(y_corrected[solarLinePixelIndices1] - np.min(y_corrected[solarLinePixelIndices1]))).argmin() + solarLinePixelIndices1[0]

        #step3: get pixel indices on either side of minimum
        nPoints = 7
        solarLinePixelIndices = range(max([0, solarLinePixelIndex2-nPoints]), min([320, solarLinePixelIndex2+nPoints+1]))

        #do gaussian fit to find minimum
        #first in wavenumber
        nu_pixels_solar_line_position_hr, solarLineAbsorptionHr, spectrum_minimum_nu = fit_gaussian_absorption(nu_px[solarLinePixelIndices], y_corrected[solarLinePixelIndices])
        #then in pixel space
        pixels_solar_line_position_hr, solar_line_abs_pixel_hr, minimumPixel = fit_gaussian_absorption(pixels[solarLinePixelIndices], y_corrected[solarLinePixelIndices])

        #calculate wavenumber error            
        delta_wavenumber = solar_minimum_nu - spectrum_minimum_nu
        print("measurement_temperature=", measurement_temperature)
        print("delta_wavenumber=", delta_wavenumber)

        #shift wavenumber scale to match solar line
        nu_obs = nu_px + delta_wavenumber
        solarLineWavenumbersHr = nu_pixels_solar_line_position_hr + delta_wavenumber
        ax1a.plot(nu_obs, y_corrected, color=colours[temperature_index], label="%0.1fC" %measurement_temperature)

        ax1a.scatter(nu_obs[solarLinePixelIndices], y_corrected[solarLinePixelIndices], color=colours[temperature_index])
        ax1a.plot(solarLineWavenumbersHr, solarLineAbsorptionHr, color=colours[temperature_index], linestyle="--")
        ax1a.axvline(x=spectrum_minimum_nu, color=colours[temperature_index], linestyle=":")
        ax1a.axvline(x=spectrum_minimum_nu + delta_wavenumber, color=colours[temperature_index])
#        ax2a.axvline(x=minimumPixel, color=colours[temperature_index])
        print(measurement_temperature, minimumPixel, spectrum_minimum_nu)
        
#        ax2a.plot(pixels, y_corrected, color=colours[temperature_index], label="%0.1fC" %measurement_temperature)
        
        
        """make solar spectra and wavenumbers on high resolution grids"""
        #interpolation method
        #add points before and after spectrum to avoid messy fft edges
        wavenumbers_hr = np.linspace(nu_obs[0], nu_obs[-1], num=6400)
        spectrum_counts_hr = np.interp(wavenumbers_hr, nu_obs, spectrum_counts)

        

#        ax3.plot(wavenumbers, spectrum_counts, label=label, color=colours[fileIndex])
        ax3.plot(wavenumbers_hr, spectrum_counts_hr, "--", color=colours[temperature_index], alpha=0.5)
        
        
        calibration_dict["%s" %hdf5_filename] = {
                "aotf_frequency":aotfFrequencySelected,
#                "integrationTimeRaw":integrationTimeRaw,
                "integration_time":integrationTime,
#                "numberOfAccumulationsRaw":numberOfAccumulationsRaw,
                "number_of_accumulations":numberOfAccumulations,
    #            "binningRaw":binningRaw,
    #            "binning":binning,
                "measurementSeconds":measurementSeconds,
                "measurementPixels":measurementPixels,
                "spectrum_counts":spectrum_counts,
                "measurementTemperature":measurementTemperature,
                "wavenumbers_hr":wavenumbers_hr,
                "spectrum_counts_hr":spectrum_counts_hr,
                }


    ax3.set_xlabel("Wavenumbers (cm-1)")
    ax3.set_ylabel("Counts")
#        ax3.legend()

    
          
    ax1a.legend()
    ticks = ax1a.get_xticks()
    ax1a.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax1b.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax1a.grid()
    ax1b.grid()

    fig1.savefig(os.path.join(BASE_DIRECTORY, "lno_solar_fullscan_order_%i_wavenumber_fit.png" %diffractionOrder))
    fig2.savefig(os.path.join(BASE_DIRECTORY, "lno_solar_fullscan_order_%i_pixel.png" %diffractionOrder))
    fig3.savefig(os.path.join(BASE_DIRECTORY, "lno_solar_fullscan_order_%i_counts.png" %diffractionOrder))
    
    fig1.close()
    fig2.close()
    fig3.close()

    
    

    
    #get coefficients for all wavenumbers
    POLYNOMIAL_DEGREE = 2
    
    #find min/max wavenumber of any temperature
    first_wavenumber = np.max([calDict[obsName]["wavenumbers_hr"][0] for obsName in calDict.keys()])
    last_wavenumber = np.min([calDict[obsName]["wavenumbers_hr"][-1] for obsName in calDict.keys()])
    
    wavenumber_grid = np.linspace(first_wavenumber, last_wavenumber, num=6720)
    temperature_grid_unsorted = np.asfarray([calDict[obsName]["measurementTemperature"] for obsName in calDict.keys()])
    
    
    
    #get data from dictionary
    spectra_interpolated = []
    for obsName in calDict.keys():
        waven = calDict[obsName]["wavenumbers_hr"]
        spectrum = calDict[obsName]["spectrum_counts_hr"]
        spectrum_interpolated = np.interp(wavenumber_grid, waven, spectrum)
        spectra_interpolated.append(spectrum_interpolated)
    spectra_grid_unsorted = np.asfarray(spectra_interpolated)
    
    #sort by temperature
    sort_indices = np.argsort(temperature_grid_unsorted)
    spectra_grid = spectra_grid_unsorted[sort_indices, :]
    temperature_grid = temperature_grid_unsorted[sort_indices]
    
    
    #plot solar fullscan relationship between temperature and counts
    cmap = plt.get_cmap('jet')
    colours2 = [cmap(i) for i in np.arange(len(wavenumber_grid))/len(wavenumber_grid)]
    
    
    plt.figure();
    coefficientsAll = []
    for wavenumberIndex in range(len(wavenumber_grid)):
    
        coefficients = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex], POLYNOMIAL_DEGREE)
        coefficientsAll.append(coefficients)
        if wavenumberIndex in range(40, 6700, 200):
            quadratic_fit = np.polyval(coefficients, temperature_grid)
            linear_coefficients_normalised = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), POLYNOMIAL_DEGREE)
            linear_fit_normalised = np.polyval(linear_coefficients_normalised, temperature_grid)
            plt.scatter(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), label="Wavenumber %0.1f" %wavenumber_grid[wavenumberIndex], color=colours2[wavenumberIndex])
            plt.plot(temperature_grid, linear_fit_normalised, "--", color=colours2[wavenumberIndex])
    
    coefficientsAll = np.asfarray(coefficientsAll).T
    
    plt.legend()
    plt.xlabel("Instrument temperature (Celsius)")
    plt.ylabel("Counts for given pixel (normalised to peak)")
    plt.title("Interpolating spectra w.r.t. temperature")
    plt.savefig(os.path.join(BASE_DIRECTORY, "lno_solar_fullscan_order_%i_interpolation.png" %diffractionOrder))
    plt.close()


    #write to hdf5 aux file
    
        
#    for obsName in calDict.keys():
#        groupName = "%i/%0.1f" %(diffractionOrder, calDict[obsName]["measurementTemperature"])
#        hdf5File[groupName+"/wavenumbers"] = calDict[obsName]["wavenumbers_hr"]
#        hdf5File[groupName+"/counts"] = calDict[obsName]["spectrum_counts_hr"]
    hdf5File["%i" %diffractionOrder+"/wavenumber_grid"] = wavenumber_grid
    hdf5File["%i" %diffractionOrder+"/spectra_grid"] = spectra_grid
    hdf5File["%i" %diffractionOrder+"/temperature_grid"] = temperature_grid
    hdf5File["%i" %diffractionOrder+"/coefficients"] = coefficientsAll
    
    
#    hdf5File.close()


#now test with nadir data

"""get nadir data from observations of a region"""
#test new calibrations on a mean nadir observation
from database_functions_v01 import obsDB, makeObsDict
dbName = "lno_0p3a"
db_obj = obsDB(dbName)
#CURIOSITY = -4.5895, 137.4417
if diffractionOrder in [168]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND n_orders < 4 AND incidence_angle < 10 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 2000/400
elif diffractionOrder in [189]:
#    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND n_orders < 4 AND incidence_angle < 20 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 10 AND latitude > -10 AND longitude < 10 AND longitude > -10 AND n_orders < 4 AND incidence_angle < 10 AND temperature_tgo > 1 AND temperature_tgo < 2 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 3000/400
elif diffractionOrder in [188]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE n_orders < 4 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 3000/400
elif diffractionOrder in [193]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE n_orders < 4 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 3000/400
obsDict = makeObsDict("lno", searchQueryOutput)
db_obj.close()
plt.figure()
plt.scatter(obsDict["longitude"], obsDict["latitude"])

fig0, ax0 = plt.subplots(figsize=(FIG_X, FIG_Y))
validIndices = np.zeros(len(obsDict["x"]), dtype=bool)
for frameIndex, (x, y) in enumerate(zip(obsDict["x"], obsDict["y"])):
    if np.mean(y) > SIGNAL_CUTOFF:
        ax0.plot(x, y, alpha=0.3, label="%i %0.1f" %(frameIndex, np.mean(y)))
        validIndices[frameIndex] = True
    else:
        validIndices[frameIndex] = False
#ax0.legend()
observation_spectrum = np.mean(np.asfarray(obsDict["y"])[validIndices, :], axis=0)
xMean = np.mean(np.asfarray(obsDict["x"])[validIndices, :], axis=0)
ax0.plot(xMean, observation_spectrum, "k")

#shift xMean to match solar line
observationTemperature = obsDict["temperature"][0]
continuum_pixels, solar_line_wavenumber = getSolarLinePosition(observationTemperature, diffractionOrder, solarLineNumber)



#find pixel containing minimum value in subset of real data
observation_continuum = baseline_als(observation_spectrum, 250.0, 0.95)

ax0.plot(xMean, observation_continuum)

observation_absorption_spectrum = spectrum_counts[continuum_pixels] / spectrum_continuum[continuum_pixels]

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

