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
from datetime import datetime
from scipy import interpolate
from fit_absorption_band_v01 import fit_gaussian_absorption

from hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, LOCAL_DIRECTORY, FIG_X, FIG_Y, makeFileList
from plot_solar_line_simulations_lno import smoothHighRes#, nu_mp, t_p0


SMOOTHING_LEVEL = 600 #must be even number

MAKE_CALIBRATION_TABLE = True
#MAKE_CALIBRATION_TABLE = False



regex = re.compile("(20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1|20191207_051654_0p1a_LNO_1)")
fileLevel = "hdf5_level_0p1a"

hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)


#new values from order 118 solar line
#Q0=-10.159
#Q1=-0.8688
#new values from mean gradient/offset of best orders: 142, 151, 156, 162, 166, 167, 178, 189, 194
Q0=-10.13785778
Q1=-0.829174444
#old values
#Q0=-6.267734
#Q1=-7.299039e-1
Q2=0.0
def t_p0(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0



#updated for LNO
F0=22.478113
F2=3.774791e-8
F1=5.508335e-4
def nu_mp(m, p, t, p0=None, F0=F0, F1=F1, F2=F2):
    """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
    if p0 == None:
        p0 = t_p0(t)
    f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
    return f



def baseline_als(y, lam=250.0, p=0.95, niter=10):
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


def getDiffractionOrders(aotfFrequencies):
    """get orders from aotf"""
    AOTFOrderCoefficients = np.array([3.9186850E-09, 6.3020400E-03, 1.3321030E+01])

    diffractionOrdersCalculated = [np.int(np.round(np.polyval(AOTFOrderCoefficients, aotfFrequency))) for aotfFrequency in aotfFrequencies]
    #set darks to zero
    diffractionOrders = np.asfarray([diffractionOrder if diffractionOrder > 50 else 0 for diffractionOrder in diffractionOrdersCalculated])
    return diffractionOrders


def fft_zerofilling(row, filling_amount):
    """apply fft, zero fill by a multiplier then reverse fft to give very high resolution spectrum"""
    n_pixels = len(row)
    
    rowrfft = np.fft.rfft(row, len(row))
    rowzeros = np.zeros(n_pixels * filling_amount, dtype=np.complex)
    rowfft = np.concatenate((rowrfft, rowzeros))
    row_hr = np.fft.irfft(rowfft).real #get real component for reversed fft
    row_hr *= len(row_hr)/len(row) #need to scale by number of extra points

    pixels_hr = np.linspace(0, n_pixels, num=len(row_hr))    
    return pixels_hr, row_hr



CHANNEL = "lno"
column_number = {"so":1, "lno":2}[CHANNEL]



#def prepExternalTemperatureReadings(column_number):
#    """read in TGO channel temperatures from file (only do once)"""
#    utc_datetimes = []
#    temperatures = []
#    
#    filenames = ["heaters_temp_2016-04-01T000153_to_2017-03-31T235852.csv", "heaters_temp_2018-03-24T000131_to_2019-12-25T235954.csv"]
#    for filename in filenames:
#        with open(os.path.join(LOCAL_DIRECTORY, "reference_files", filename)) as f:
#            lines = f.readlines()
#                
#        for line in lines[1:]:
#            split_line = line.split(",")
#            utc_datetimes.append(datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
#            temperatures.append(split_line[column_number])
#    
#    return np.asarray(utc_datetimes), np.asfarray(temperatures)
#    
#if "EXTERNAL_TEMPERATURE_DATETIMES" not in globals():
#    print("Reading in TGO temperatures")
#    EXTERNAL_TEMPERATURE_DATETIMES, EXTERNAL_TEMPERATURES = prepExternalTemperatureReadings(column_number) #1=SO nominal, 2=LNO nominal
#
#
#
#def getExternalTemperatureReadings(utc_string, column_number): #input format 2015 Mar 18 22:41:03.916651
#    """get TGO readout temperatures. Input SPICE style datetime, output in Celsius
#    column 1 = SO baseplate nominal, 2 = LNO baseplate nominal"""
#    utc_datetime = datetime.strptime(utc_string[:20].decode(), "%Y %b %d %H:%M:%S")
#    
#    external_temperature_datetimes = EXTERNAL_TEMPERATURE_DATETIMES
#    external_temperatures = EXTERNAL_TEMPERATURES
#    
#    closestIndex = np.abs(external_temperature_datetimes - utc_datetime).argmin()
#    closest_time_delta = np.min(np.abs(external_temperature_datetimes[closestIndex] - utc_datetime).total_seconds())
#    if closest_time_delta > 60 * 5:
#        print("Error: time delta %0.1f too high" %closest_time_delta)
#        print(external_temperature_datetimes[closestIndex])
#        print(utc_datetime)
#    else:
#        closestTemperature = np.float(external_temperatures[closestIndex])
#    
#    return closestTemperature
    


#dictionary of approximate solar line positions (to begin calculation)
solarLineDict = {
118:2669.8,
120:2715.5,
121:2733.2,
126:2837.8,
130:2943.7,
133:3012.0,
142:3209.4,
151:3414.4,
156:3520.4,
160:3615.1,
162:3650.9,
163:3688.0,
164:3693.7,
166:3750.0,
167:3767.0,
168:3787.9,
169:3812.5,
173:3902.4,
174:3934.1,
178:4021.3,
179:4042.7,
180:4069.5,
182:4101.5,
184:4156.9,
189:4276.1,
194:4383.2,
195:4402.6,
196:4422.0,
}

#best orders
#bestOrders = [118, 120, 142, 151, 156, 162, 166, 167, 178, 189, 194]
bestOrders = solarLineDict.keys()


"""plot solar lines in solar fullscan data for orders contains strong solar lines"""
temperatureRange = np.arange(-20., 15., 0.1)
cmap = plt.get_cmap('plasma')
colours = [cmap(i) for i in np.arange(len(temperatureRange))/len(temperatureRange)]


selectedDiffractionOrders = bestOrders
#selectedDiffractionOrders = range(168, 169)
#selectedDiffractionOrders = range(189, 190)


if MAKE_CALIBRATION_TABLE:

    outputTitle = "LNO_Radiance_Factor_Calibration_Table"
    hdf5File = h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "w")
    
    for diffractionOrder in selectedDiffractionOrders:
    
        fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2), sharex=True)
        fig1.suptitle("Diffraction Order %i" %diffractionOrder)
        fig2, (ax2a, ax2b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
        fig2.suptitle("Diffraction Order %i" %diffractionOrder)
    
        fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))
        fig3.suptitle("Diffraction Order %i" %diffractionOrder)
    
        """get high res solar spectra and convolve to LNO-like resolution"""
        #define spectral range
        pixels = np.arange(320.0)
        pixels_hr = np.arange(0.0, 320.0, 0.01)
        nu_pixels_no_shift = nu_mp(diffractionOrder, pixels, 0.0, p0=0.0) #spectral calibration without temperature shift
        nu_pixels_no_shift_hr = nu_mp(diffractionOrder, pixels_hr, 0.0, p0=0.0) #spectral calibration without temperature shift
        nu_hr_min = np.min(nu_pixels_no_shift)
        nu_hr_max = np.max(nu_pixels_no_shift)
        dnu = 0.001
        Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
        nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
        dnu = nu_hr[1]-nu_hr[0]
        
        
        #get solar spectrum (only to check if absorption exists)
        solspecFile = os.path.join(BASE_DIRECTORY, "reference_files", "nomad_solar_spectrum_solspec.txt")
        with open(solspecFile, "r") as f:
            nu_solar = []
            I0_solar = []
            for line in f:
                nu, I0 = [float(val) for val in line.split()]
                if nu < nu_hr_min - 2.:
                    continue
                if nu > nu_hr_max + 2.:
                    break
                nu_solar.append(nu)
                I0_solar.append(I0)
        f_solar = interpolate.interp1d(nu_solar, I0_solar)
        I0_solar_hr = f_solar(nu_hr)
        
        
        #convolve high res solar spectrum to lower resolution. Normalise to 1
        solar_spectrum = smoothHighRes(I0_solar_hr, window_len=(SMOOTHING_LEVEL-1))
    #    normalised_solar_spectrum = ((solar_spectrum-np.min(solar_spectrum)) / (np.max(solar_spectrum)-np.min(solar_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]# / 5.0 + 0.8
        normalised_solar_spectrum = (solar_spectrum / (np.max(solar_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]
        ax1b.plot(nu_hr, normalised_solar_spectrum, "k")
    
        """find exact wavenumber of solar line"""
        #get approx wavenumber
        approximateWavenumber = solarLineDict[diffractionOrder]
        
        #get n points on either side
        wavenumberIndex = (np.abs(nu_hr - approximateWavenumber)).argmin()
        nPoints = 300
        solarLineIndices = range(wavenumberIndex-nPoints, wavenumberIndex+nPoints)
        ax1b.scatter(nu_hr[solarLineIndices], normalised_solar_spectrum[solarLineIndices], color="k")
        
        
        #find local minima in this range
        minimumIndex = (np.diff(np.sign(np.diff(normalised_solar_spectrum[solarLineIndices]))) > 0).nonzero()[0] + 1
        #check if 1 minima only was found
        if len(minimumIndex) == 1:
            #get exact minimum
            solarLineWavenumber = nu_hr[solarLineIndices[minimumIndex[0]]]
            ax1b.axvline(x=solarLineWavenumber, c="k", linestyle="--")
            
            #convert to pixel (p0=0) and plot on pixel grid
            #find subpixel containing solar line when p0=0
            pixelHrIndex = (np.abs(nu_pixels_no_shift_hr - solarLineWavenumber)).argmin()
            solarLinePixelNoShift = pixels_hr[pixelHrIndex]
            print("Solar line pixel (p0=0)= ", solarLinePixelNoShift)
            
            ax2b.axvline(x=solarLinePixelNoShift, c="k", linestyle="--")
    
        else:
            print("Error: %i minima found for order %i" %(len(minimumIndex), diffractionOrder))
    
        calDict = {}
    
        """get fullscan data"""
        for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
            
            yRaw = hdf5_file["Science/Y"][...]
            aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
                
            diffractionOrders = getDiffractionOrders(aotfFrequency)
            frameIndices = list(np.where(diffractionOrders == diffractionOrder)[0])
        
            frameIndex = frameIndices[0] #just take first frame of solar fullscan
        
            #get temperature from TGO readout instead of channel
            utcStartTime = hdf5_file["Geometry/ObservationDateTime"][frameIndex, 0]
            measurementTemperature = getExternalTemperatureReadings(utcStartTime, 2)
            #for plotting colour
            temperatureIndex = (np.abs(temperatureRange - measurementTemperature)).argmin()
            
            #cut unused / repeated orders from file (01pa only)
            ySelected = yRaw[frameIndex, :, :]
            aotfFrequencySelected = aotfFrequency[frameIndex]
        
            integrationTimeRaw = hdf5_file["Channel/IntegrationTime"][0]
        #    binningRaw = hdf5_file["Channel/Binning"][0]
            numberOfAccumulationsRaw = hdf5_file["Channel/NumberOfAccumulations"][0]
            nBins = ySelected.shape[0]
            nPixels = ySelected.shape[1]
            
            integrationTime = np.float(integrationTimeRaw) / 1.0e3 #microseconds to seconds
            numberOfAccumulations = np.float(numberOfAccumulationsRaw)/2.0 #assume LNO nadir background subtraction is on
        #    binning = np.float(binningRaw) + 1.0 #binning starts at zero
        
            #solar spectrum - take centre line only
            yBinned = ySelected[11, :]
            measurementPixels = 1.0
            
            measurementSeconds = integrationTime * numberOfAccumulations
            #normalise to 1s integration time per pixel
            spectrum_counts = yBinned / measurementSeconds / measurementPixels
            label = "%s order %0.0fkHz %0.1fC" %(hdf5_filename[:15], aotfFrequencySelected, measurementTemperature)
    
            #remove baseline
            yBaseline = baseline_als(yBinned) #find continuum of mean spectrum
            yCorrected = yBinned / yBaseline
    
    
            nu_pixels = nu_mp(diffractionOrder, pixels, measurementTemperature)
    
            """find centres of solar lines"""
            #step1: find nearest pixels that contain the solar line
            solarLinePixelIndex1 = (np.abs(nu_pixels - solarLineWavenumber)).argmin()
            nPoints = 3
            solarLinePixelIndices1 = range(max([0, solarLinePixelIndex1-nPoints]), min([320, solarLinePixelIndex1+nPoints+1]))
        
    
            #step2: find pixel containing minimum points
            solarLinePixelIndex2 = (np.abs(yCorrected[solarLinePixelIndices1] - np.min(yCorrected[solarLinePixelIndices1]))).argmin() + solarLinePixelIndices1[0]
    
            #step3: get pixel indices on either side of minimum
            nPoints = 7
            solarLinePixelIndices = range(max([0, solarLinePixelIndex2-nPoints]), min([320, solarLinePixelIndex2+nPoints+1]))
    
            #do gaussian fit to find minimum
            #first in wavenumber
            nu_pixels_solar_line_position_hr, solarLineAbsorptionHr, minimumWavenumber = fit_gaussian_absorption(nu_pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices])
            #then in pixel space
            pixels_solar_line_position_hr, solar_line_abs_pixel_hr, minimumPixel = fit_gaussian_absorption(pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices])
    
            #calculate wavenumber error            
            delta_wavenumber = solarLineWavenumber - minimumWavenumber
            print("measurementTemperature=", measurementTemperature)
            print("delta_wavenumber=", delta_wavenumber)
    
            #shift wavenumber scale to match solar line
            wavenumbers = nu_pixels + delta_wavenumber
            solarLineWavenumbersHr = nu_pixels_solar_line_position_hr + delta_wavenumber
            ax1a.plot(wavenumbers, yCorrected, color=colours[temperatureIndex], label="%0.1fC" %measurementTemperature)
    
            ax1a.scatter(wavenumbers[solarLinePixelIndices], yCorrected[solarLinePixelIndices], color=colours[temperatureIndex])
            ax1a.plot(solarLineWavenumbersHr, solarLineAbsorptionHr, color=colours[temperatureIndex], linestyle="--")
            ax1a.axvline(x=minimumWavenumber, color=colours[temperatureIndex], linestyle=":")
            ax1a.axvline(x=minimumWavenumber+delta_wavenumber, color=colours[temperatureIndex])
            ax2a.axvline(x=minimumPixel, color=colours[temperatureIndex])
            print(measurementTemperature, minimumPixel, minimumWavenumber)
            
            ax2a.plot(pixels, yCorrected, color=colours[temperatureIndex], label="%0.1fC" %measurementTemperature)
            
            
            """make solar spectra and wavenumbers on high resolution grids"""
    #        #fft method
    #        #add points before and after spectrum to avoid messy fft edges
    #        spectrum_counts_ex = np.concatenate((np.tile(spectrum_counts[0], 60), spectrum_counts, np.tile(spectrum_counts[-1], 60)))
    #        pixels_hr, spectrum_counts_hr = fft_zerofilling(spectrum_counts_ex, 10)
    #        
    #        #remove extra points after fft
    #        pixels_hr = pixels_hr[60*21:len(pixels_hr)-60*21] - pixels_hr[60*21]
    #        spectrum_counts_hr = spectrum_counts_hr[60*21:len(spectrum_counts_hr)-60*21]
    #        wavenumbers_hr = nu_mp(diffractionOrder, pixels_hr, measurementTemperature) + delta_wavenumber
    
    
            #interpolation method
            #add points before and after spectrum to avoid messy fft edges
            wavenumbers_hr = np.linspace(wavenumbers[0], wavenumbers[-1], num=6400)
            spectrum_counts_hr = np.interp(wavenumbers_hr, wavenumbers, spectrum_counts)
    
            
    
    #        ax3.plot(wavenumbers, spectrum_counts, label=label, color=colours[fileIndex])
            ax3.plot(wavenumbers_hr, spectrum_counts_hr, "--", color=colours[temperatureIndex], alpha=0.5)
            
            
            calDict["%s" %hdf5_filename] = {
                    "aotfFrequencySelected":aotfFrequencySelected,
                    "integrationTimeRaw":integrationTimeRaw,
                    "integrationTime":integrationTime,
                    "numberOfAccumulationsRaw":numberOfAccumulationsRaw,
                    "numberOfAccumulations":numberOfAccumulations,
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
        
    
    hdf5File.close()


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

