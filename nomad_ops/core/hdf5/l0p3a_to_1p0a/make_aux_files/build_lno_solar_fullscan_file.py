# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:00:02 2020


MAKE SOLAR FULLSCAN CAL FILE
"""

import sys
import os
import numpy as np
import h5py
import re
import matplotlib.pyplot as plt
#import datetime
#from scipy import interpolate


from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions_v04 import makeFileList

from tools.spectra.baseline_als import baseline_als
#from tools.spectra.fit_polynomial import fit_polynomial
#from tools.general.get_nearest_index import get_nearest_index
from tools.plotting.colours import get_colours

from instrument.nomad_lno_instrument import nu_mp, temperature_p0
from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import rad_fact_orders_dict
from instrument.calibration.lno_radiance_factor.lno_rad_fac_functions import get_reference_dict, get_diffraction_orders, get_nearest_temperature_measurement

#from tools.spectra.nu_hr_grid import nu_hr_grid
from tools.general.get_minima_maxima import get_local_minima

FORMAT_STR_SECONDS = "%Y %b %d %H:%M:%S.%f"
Y_OFFSET = 0.05

MAKE_AUX_FILE = True
#MAKE_AUX_FILE = False

regex = re.compile("(20161121_233000|20180702_112352|20181101_213226|20190314_021825|20190609_011514|20191207_051654)_0p1a_LNO_1")
fileLevel = "hdf5_level_0p1a"
hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)





"""plot solar lines in solar fullscan data for orders contains strong solar lines"""
temperature_range = np.arange(-20., 15., 0.1)
pixels = np.arange(320.0)

colours = get_colours(len(temperature_range), "plasma")


output_title = "LNO_Radiance_Factor_Calibration_Table"
if MAKE_AUX_FILE:
    hdf5_file_out = h5py.File(os.path.join(paths["BASE_DIRECTORY"], output_title+".h5"), "w")
    
#for diffraction_order in [116]:
for diffraction_order in rad_fact_orders_dict.keys():
    
    error = False
    
    rad_fact_order_dict = rad_fact_orders_dict[diffraction_order]
    
    reference_dict = get_reference_dict(diffraction_order, rad_fact_order_dict)
    solar_molecular = reference_dict["solar_molecular"]
    solar_line = rad_fact_order_dict["solar_line"]
        
    #plot in nu
    fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2))
    if solar_line:
        fig1.suptitle("Diffraction order %i - solar line calibration" %diffraction_order)
    else:
        fig1.suptitle("Diffraction order %i" %diffraction_order)
    #plot in px
    fig2, (ax2a, ax2b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2))
    if solar_line:
        fig2.suptitle("Diffraction order %i - solar line calibration" %diffraction_order)
    else:
        fig2.suptitle("Diffraction order %i" %diffraction_order)

    nu_solar_hr = reference_dict["nu_hr"]
    solar_spectrum_hr = reference_dict["solar"]
    px_hr = np.linspace(0.0, 320.0, num=len(nu_solar_hr ))
    nu_px_p0 = nu_mp(diffraction_order, pixels, temperature_p0)
    nu_px_p0_hr = nu_mp(diffraction_order, px_hr, temperature_p0)

    ax1b.plot(nu_solar_hr, solar_spectrum_hr, "k")
    
    
    if solar_line:
        print("order %i, calibrating with %s lines" %(diffraction_order, solar_molecular))
        
        solar_line_max_transmittance = rad_fact_order_dict["trans_solar"]
        solar_line_nu_start = rad_fact_order_dict["nu_range"][0]
        solar_line_nu_end = rad_fact_order_dict["nu_range"][1]
        
        ax1b.axhline(y=solar_line_max_transmittance, c="k", linestyle="--")
        ax2b.axvline(x=solar_line_nu_start, c="k", linestyle="--", alpha=0.5)
        ax2b.axvline(x=solar_line_nu_end, c="k", linestyle="--", alpha=0.5)
        
        
        solar_minimum_indices_all = get_local_minima(solar_spectrum_hr)
        
        solar_minimum_indices = [solar_minimum_index for solar_minimum_index in solar_minimum_indices_all \
                               if solar_spectrum_hr[solar_minimum_index] < solar_line_max_transmittance \
                               and solar_line_nu_start < nu_solar_hr[solar_minimum_index] < solar_line_nu_end]
        
        if len(solar_minimum_indices) == 0:
            print("Error: no solar lines found matching criteria")
            error = True
            
        if not error:
        
    #        solar_minimum_index = np.where(np.min(solar_spectrum_hr)==solar_spectrum_hr)[0][0]
            solar_minima_nu = [nu_solar_hr[solar_minimum_index] for solar_minimum_index in solar_minimum_indices]
            
            """find exact wavenumber of solar line"""
            for solar_minimum_index, solar_minimum_nu in zip(solar_minimum_indices, solar_minima_nu):
                #get n points on either side
                n_points = 300
                solar_line_indices = range(np.max((0, solar_minimum_index - n_points)), np.min((len(nu_solar_hr), solar_minimum_index + n_points)))
                ax1b.scatter(nu_solar_hr[solar_line_indices], solar_spectrum_hr[solar_line_indices], color="b")
                
                #find all local minima in this range
                minimum_index = get_local_minima(solar_spectrum_hr[solar_line_indices])
                #check if 1 minima only was found i.e. that two solar lines are not too close together
                if len(minimum_index) == 1:
                    #plot minimum
                    ax1b.axvline(x=solar_minimum_nu, c="k", linestyle="--")
                    ax2b.axvline(x=solar_minimum_nu, c="k", linestyle="--")
    
                    ax1b.annotate("T=%0.3f" %solar_spectrum_hr[solar_minimum_index], xy=(solar_minimum_nu + 0.5, solar_spectrum_hr[solar_minimum_index]))
                    
                    #convert to pixel (p0=0) and plot on pixel grid
                    #find subpixel containing solar line when p0=0
                    px_p0_hr_index = (np.abs(nu_px_p0_hr - solar_minimum_nu)).argmin()
                    solar_minimum_px_p0 = px_hr[px_p0_hr_index]
    #                print("Solar line pixel (p0=0)= ", solar_minimum_px_p0)
                    
                    ax1a.axvline(x=solar_minimum_px_p0, c="k", linestyle="--")
            
                else:
                    print("Error: %i solar line minima found for order %i" %(len(minimum_index), diffraction_order))
                    error = True


    if error:
        #stop on error
        sys.exit()

    calibration_dict = {}

    """get fullscan data"""
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        
        y_raw = hdf5_file["Science/Y"][...]
        aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]

        diffraction_orders = get_diffraction_orders(aotf_frequencies)
        frame_indices = list(np.where(diffraction_orders == diffraction_order)[0])
        frame_index = frame_indices[0] #just take first frame of solar fullscan
        
        #get temperature from TGO readout instead of channel
        measurement_temperature = get_nearest_temperature_measurement(hdf5_file, frame_index)

        nu_px = nu_mp(diffraction_order, pixels, measurement_temperature)

        #for plotting colour
        temperature_index = (np.abs(temperature_range - measurement_temperature)).argmin()
        
        #cut unused / repeated orders from file (01pa only)
#        #solar spectrum - take centre line only
#        y_spectrum = y_raw[frame_index, 11, :]

        #solar spectrum - take mean of centre lines
        y_spectrum = np.mean(y_raw[frame_index, 6:16, :], axis=0)
        
        #remove bad pixel
        y_spectrum[40] = np.mean((y_spectrum[39], y_spectrum[41]))
        
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
        y_baseline = baseline_als(spectrum_counts) #find continuum of mean spectrum
        y_corrected = spectrum_counts / y_baseline
        ax1a.plot(pixels, y_corrected, color=colours[temperature_index], label="%0.1fC" %measurement_temperature)
        ax2a.plot(pixels, spectrum_counts, "--", color=colours[temperature_index], alpha=0.7, label="%0.1fC" %measurement_temperature)

        if solar_line:
            
            """find centres of solar lines. Spectral cal is approximate so first find solar line in data using approx cal"""
            #step1: find nearest pixel number where calculated px nu = real solar line nu
            solar_line_pixel_index1 = (np.abs(nu_px - solar_minimum_nu)).argmin()
            nPoints = 3
            #get indices of pixels on either side
            solar_line_pixel_indices1 = range(max([0, solar_line_pixel_index1-nPoints]), min([320, solar_line_pixel_index1+nPoints+1]))
        
    
            #step2: within this approximate range, find pixel number containing the minimum signal
            solar_line_pixel_index2 = (np.abs(y_corrected[solar_line_pixel_indices1] - np.min(y_corrected[solar_line_pixel_indices1]))).argmin() + solar_line_pixel_indices1[0]
            #step3: get pixel indices on either side of this minimum
            nPoints = 7
            solar_line_pixel_indices = range(max([0, solar_line_pixel_index2-nPoints]), min([320, solar_line_pixel_index2+nPoints+1]))

            #now that the pixel range containing the solar line has been found, 
            #do gaussian fit to find minimum
            #first in wavenumber
            spectrum_solar_line_fit_nu_hr, solar_line_fit_hr, spectrum_minimum_nu = fit_gaussian_absorption(nu_px[solar_line_pixel_indices], y_corrected[solar_line_pixel_indices])
            #then in pixel space
            spectrum_solar_line_fit_px_hr, solar_line_fit_px_hr, spectrum_minimum_px = fit_gaussian_absorption(pixels[solar_line_pixel_indices], y_corrected[solar_line_pixel_indices])




            #calculate wavenumber error            
            delta_wavenumber = solar_minimum_nu - spectrum_minimum_nu
#            print("measurement_temperature=", measurement_temperature)
            print("delta_wavenumber=", delta_wavenumber)
    
            #shift wavenumber scale to match solar line
            nu_obs = nu_px + delta_wavenumber
            ax1b.plot(nu_obs, y_corrected + Y_OFFSET, color=colours[temperature_index], label="%0.1fC" %measurement_temperature)
            ax1b.scatter(nu_obs[solar_line_pixel_indices], y_corrected[solar_line_pixel_indices] + Y_OFFSET, color=colours[temperature_index])
    
            #plot solar line gaussian fit, after correcting for wavenumber shift        
            solar_line_fit_nu_hr = spectrum_solar_line_fit_nu_hr + delta_wavenumber
            ax1b.plot(solar_line_fit_nu_hr, solar_line_fit_hr + Y_OFFSET, color=colours[temperature_index], linestyle="--")
    
            #plot vertical lines before and after wavenumber shift
            ax1b.axvline(x=spectrum_minimum_nu, color=colours[temperature_index], linestyle=":")
            ax1b.axvline(x=spectrum_minimum_nu + delta_wavenumber, color=colours[temperature_index])
#            print(measurement_temperature, spectrum_minimum_px, spectrum_minimum_nu)
        
            #plot on pixel grid
            ax1a.scatter(pixels[solar_line_pixel_indices], y_corrected[solar_line_pixel_indices], color=colours[temperature_index])
            ax1a.plot(spectrum_solar_line_fit_px_hr, solar_line_fit_px_hr, color=colours[temperature_index], linestyle="--")
            ax1a.axvline(x=spectrum_minimum_px, color=colours[temperature_index])
        
        else:
            #don't shift spectrum
            nu_obs = nu_px
            ax1b.plot(nu_obs, y_corrected + Y_OFFSET, color=colours[temperature_index], label="%0.1fC" %measurement_temperature)
            if file_index == 0:
                ax1b.annotate("Warning: no solar line correction", xycoords='axes fraction', xy=(0.05, 0.05), fontsize=16)
            
        """make solar spectra and wavenumbers on high resolution grids"""
        #interpolate solar spectrum using simple linear onto 20x grid
        interpolated_nu_hr = np.linspace(nu_obs[0], nu_obs[-1], num=6400)
        interpolated_counts_hr = np.interp(interpolated_nu_hr, nu_obs, spectrum_counts)

        ax2b.plot(interpolated_nu_hr, interpolated_counts_hr, "--", color=colours[temperature_index], alpha=0.7)
        
        
        calibration_dict["%s" %hdf5_filename] = {
                "aotf_frequency":aotf_frequency,
                "integration_time_raw":integration_time_raw,
                "integration_time":integration_time,
                "number_of_accumulations_raw":number_of_accumulations_raw,
                "number_of_accumulations":number_of_accumulations,
    #            "binning_raw":binning_raw,
    #            "binning":binning,
                "measurement_seconds":measurement_seconds,
                "pixels":pixels,
                "spectrum_counts":spectrum_counts,
                "measurement_temperature":measurement_temperature,
                "interpolated_nu_hr":interpolated_nu_hr,
                "interpolated_counts_hr":interpolated_counts_hr,
                }
            


    ax1a.set_title("Solar line fit pixels")
    ax1b.set_title("Solar line fit wavenumbers")
    ax1a.legend()
    ticks = ax1a.get_xticks()
    ax1a.set_xticks(np.arange(ticks[0], ticks[-1], 20.0))
    ticks = ax1b.get_xticks()
    ax1b.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax1a.grid()
    ax1b.grid()


    ax2a.set_xlabel("Pixels")
    ax2a.set_ylabel("Counts per second per px")
    ax2b.set_xlabel("Wavenumbers (cm-1)")
    ax2b.set_ylabel("Counts per second per px")
    ax2a.set_ylim(bottom=0.0)
    ax2b.set_ylim(bottom=0.0)
    ax2a.legend()
    ticks = ax2a.get_xticks()
    ax2a.set_xticks(np.arange(ticks[0], ticks[-1], 20.0))
    ticks = ax2b.get_xticks()
    ax2b.set_xticks(np.arange(ticks[0], ticks[-1], 2.0))
    ax2a.grid()
    ax2b.grid()

    
          

    fig1.savefig(os.path.join(paths["BASE_DIRECTORY"], "lno_solar_fullscan_order_%i_baseline.png" %diffraction_order))
    fig2.savefig(os.path.join(paths["BASE_DIRECTORY"], "lno_solar_fullscan_order_%i_counts.png" %diffraction_order))
#    fig3.savefig(os.path.join(paths["BASE_DIRECTORY"], "lno_solar_fullscan_order_%i_counts.png" %diffraction_order))
    
    
    plt.close(fig1)
    plt.close(fig2)

    
#    if solar_line and not error:

    
    #get coefficients for all wavenumbers
    POLYNOMIAL_DEGREE = 2
    
    #find min/max wavenumber of any temperature
    first_nu_hr = np.max([calibration_dict[hdf5_filename]["interpolated_nu_hr"][0] for hdf5_filename in calibration_dict.keys()])
    last_nu_hr = np.min([calibration_dict[hdf5_filename]["interpolated_nu_hr"][-1] for hdf5_filename in calibration_dict.keys()])
    
    #make new wavenumber grid covering all temperatures
    wavenumber_grid = np.linspace(first_nu_hr, last_nu_hr, num=6720)
    temperature_grid_unsorted = np.asfarray([calibration_dict[hdf5_filename]["measurement_temperature"] for hdf5_filename in calibration_dict.keys()])
    
    
    
    #get data from dictionary
    spectra_interpolated = []
    for obsName in calibration_dict.keys():
        waven = calibration_dict[obsName]["interpolated_nu_hr"]
        spectrum = calibration_dict[obsName]["interpolated_counts_hr"]
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
    
        
    fig3, ax3 = plt.subplots(figsize=(FIG_X+3, FIG_Y+3))
    if solar_line:
        fig3.suptitle("Interpolating spectra w.r.t. temperature order %i" %diffraction_order)
    else:
        fig3.suptitle("Interpolating spectra w.r.t. temperature order %i - no solar line" %diffraction_order)

    coefficientsAll = []
    for wavenumberIndex in range(len(wavenumber_grid)):
    
        coefficients = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex], POLYNOMIAL_DEGREE)
        coefficientsAll.append(coefficients)
        if wavenumberIndex in range(40, 6700, 400):
            quadratic_fit = np.polyval(coefficients, temperature_grid)
            linear_coefficients_normalised = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), POLYNOMIAL_DEGREE)
            linear_fit_normalised = np.polyval(linear_coefficients_normalised, temperature_grid)
            ax3.scatter(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), label="Wavenumber %0.1f" %wavenumber_grid[wavenumberIndex], color=colours2[wavenumberIndex])
            ax3.plot(temperature_grid, linear_fit_normalised, "--", color=colours2[wavenumberIndex])
    
    coefficientsAll = np.asfarray(coefficientsAll).T
    
    ax3.set_xlabel("Instrument temperature (Celsius)")
    ax3.set_ylabel("Counts for given pixel (normalised to peak)")
    ax3.legend()
    fig3.savefig(os.path.join(paths["BASE_DIRECTORY"], "lno_solar_fullscan_order_%i_interpolation.png" %diffraction_order))
    plt.close(fig3)


    #write to hdf5 aux file

    if MAKE_AUX_FILE:
        hdf5_file_out["%i" %diffraction_order+"/wavenumber_grid"] = wavenumber_grid
        hdf5_file_out["%i" %diffraction_order+"/spectra_grid"] = spectra_grid
        hdf5_file_out["%i" %diffraction_order+"/temperature_grid"] = temperature_grid
        hdf5_file_out["%i" %diffraction_order+"/coefficients"] = coefficientsAll
        
        if not solar_line:
            hdf5_file_out["%i" %diffraction_order+"/warning"] = 1.0
            
hdf5_file_out.close()
    


