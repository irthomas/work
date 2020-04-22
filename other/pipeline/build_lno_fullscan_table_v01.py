# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:58:30 2019

@author: iant

MAKE SOLAR FULLSCAN CAL FILE
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import savgol_filter

from hdf5_functions_v04 import BASE_DIRECTORY, LOCAL_DIRECTORY, FIG_X, FIG_Y, makeFileList
from plot_solar_line_simulations_lno import getPixelSolarSpectrum, getConvolvedSolarSpectrum, nu_mp, t_p0


regex = re.compile("(20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1)")
fileLevel = "hdf5_level_0p1a"


hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)


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


def getSolarLinePosition(measurementTemperature, diffractionOrder, solarLineNumber):
    """find pixel numbers containing absorption"""
    #order, [temperatures], [pixel starts], pixel widths, wavenumber
    dict1 = {
            168:{
                0:[[-18.3, -10.8, -5.2, -2.9, 3.9], [114, 119, 122, 124, 130], 13, 3787.871601008524],
                1:[[-18.3, -10.8, -5.2, -2.9, 3.9], [77, 84, 90, 91, 98], 8, 3784.3935170320838],
                },
            188:{
#                0:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 13, 4242],
#                1:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 8, 4250.776050788092],
                },
            189:{
                0:[[-18.3, -10.8, -5.2, -2.9, 3.9], [252, 259, 262, 263, 269], 13, 4276.132702609243],
                },
            190:{
#                0:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 13, 4276.132702609243],
#                1:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 8, 4282],
                },
            193:{
#                0:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 13, 4276.132702609243],
                1:[[-18.3, -10.8, -5.2, -2.9, 3.9], [100, 101, 102, 103, 104], 8, 4364.37],
                },
            }
    
    a = dict1[diffractionOrder][solarLineNumber][0]
    b = dict1[diffractionOrder][solarLineNumber][1]
    continuum_length = dict1[diffractionOrder][solarLineNumber][2]
    solar_line_wavenumber = dict1[diffractionOrder][solarLineNumber][3]

    
    continuum_start = int(np.round(np.polyval(np.polyfit(a, b, 1), measurementTemperature)))
    continuum_end = int(np.round(continuum_start + continuum_length))
    continuum_pixels = np.arange(continuum_start, continuum_end)
    return continuum_pixels, solar_line_wavenumber

def findAbsorptionMininumIndex(absorption, findType="^2", zeroFilling=10, plot=False):
    """input spectrum containing absorption. Output location of minimum"""
    
    pixels_hr, absorption_hr = fft_zerofilling(absorption, zeroFilling)
    
    minimum_pixel = np.where(np.min(absorption) == absorption)[0]
    pixel_minima = np.arange(minimum_pixel-3, minimum_pixel+4, 1)
    pixels = np.arange(pixel_minima[0], pixel_minima[-1], 1./zeroFilling)
    absorption_minima = np.polyval(np.polyfit(pixel_minima, absorption[pixel_minima], 2), pixels)
    
    if plot:
        plt.plot(absorption)
        plt.plot(pixels_hr, absorption_hr)
        plt.plot(pixels, absorption_minima)
        
    
    
    #pixel containing minimum value in real data:
    if findType == "fft":
        spectrum_min_pixel = pixels_hr[np.where(absorption_hr == min(absorption_hr))[0][0]]
    elif findType == "^2":
        spectrum_min_pixel = pixels[np.where(np.min(absorption_minima) == absorption_minima)[0][0]]

    return spectrum_min_pixel





def prepExternalTemperatureReadings(column_number):
    """read in TGO channel temperatures from file (only do once)"""
    
    with open(os.path.join(LOCAL_DIRECTORY, "reference_files", "heaters_temp_2018-03-24T000131_to_2019-08-24T080358.csv")) as f:
        lines = f.readlines()
            
    utc_datetimes = []
    temperatures = []
    for line in lines[1:]:
        split_line = line.split(",")
        utc_datetimes.append(datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
        temperatures.append(split_line[column_number])
    
    return np.asarray(utc_datetimes), np.asfarray(temperatures)



if "EXTERNAL_TEMPERATURE_DATETIMES" not in globals():
    print("Reading in TGO temperatures")
    EXTERNAL_TEMPERATURE_DATETIMES, EXTERNAL_TEMPERATURES = prepExternalTemperatureReadings(2) #2=LNO nominal



def getExternalTemperatureReadings(utc_string): #input format 2015 Mar 18 22:41:03.916651
    """get TGO readout temperatures. Input SPICE style datetime, output in Celsius
    column 1 = SO baseplate nominal, 2 = LNO baseplate nominal"""



    
    utc_datetime = datetime.strptime(utc_string[:20].decode(), "%Y %b %d %H:%M:%S")
    
    external_temperature_datetimes = EXTERNAL_TEMPERATURE_DATETIMES
    external_temperatures = EXTERNAL_TEMPERATURES
    
    closestIndex = np.abs(external_temperature_datetimes - utc_datetime).argmin()
    closest_time_delta = np.min(np.abs(external_temperature_datetimes[closestIndex] - utc_datetime).total_seconds())
    if closest_time_delta > 60 * 5:
        print("Error: time delta %0.1f too high" %closest_time_delta)
        print(external_temperature_datetimes[closestIndex])
        print(utc_datetime)
    else:
        closestTemperature = np.float(external_temperatures[closestIndex])
    
    return closestTemperature
    




def baseline_als(y, lam, p, niter=10):
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




#diffractionOrder = 168
#solarLineNumber = 0
diffractionOrder = 189
solarLineNumber = 0
#diffractionOrder = 193
#solarLineNumber = 1
pixels = np.arange(320)







fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))

calDict = {}


cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(len(hdf5Filenames))/len(hdf5Filenames)]


for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    
    yRaw = hdf5_file["Science/Y"][...]
    aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
        
    diffractionOrders = getDiffractionOrders(aotfFrequency)
    frameIndices = list(np.where(diffractionOrders == diffractionOrder)[0])

    frameIndex = frameIndices[0] #just take first frame of solar fullscan

    #get temperature
    utcStartTime = hdf5_file["Geometry/ObservationDateTime"][frameIndex, 0]
    measurementTemperature = getExternalTemperatureReadings(utcStartTime)
    
    #cut unused / repeated orders from file (01pa only)
    if "0p1a" in hdf5_filename:
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
    
    """shift spectral calibration to match solar line"""
    #get expected solar line wavenumber and pixel range
    continuum_pixels, solar_line_wavenumber = getSolarLinePosition(measurementTemperature, diffractionOrder, solarLineNumber)
    ax1.axvline(x=solar_line_wavenumber)
    
    spectrum_continuum = baseline_als(spectrum_counts, 250.0, 0.95)
    
    #find pixel containing minimum value in subset of real data
    absorption_spectrum = spectrum_counts[continuum_pixels] / spectrum_continuum[continuum_pixels]
#    plt.figure()
#    spectrum_min_pixel = findAbsorptionMininumIndex(absorption_spectrum, plot=True)
    spectrum_min_pixel = findAbsorptionMininumIndex(absorption_spectrum)
    spectrum_min_pixel = spectrum_min_pixel + continuum_pixels[0]
    spectrum_min_wavenumber = nu_mp(diffractionOrder, spectrum_min_pixel, measurementTemperature)

    #calculate wavenumber error            
    delta_wavenumber = solar_line_wavenumber - spectrum_min_wavenumber
    print("measurementTemperature=", measurementTemperature)
    print("delta_wavenumber=", delta_wavenumber)
    
    #shift wavenumber scale to match solar line
    wavenumbers = nu_mp(diffractionOrder, pixels, measurementTemperature) + delta_wavenumber


    #make solar spectra on high resolution grid, interpolate to observation wavenumber grid
    #TODO: replace by better method?
    #add points before and after spectrum to avoid messy fft edges
    spectrum_counts_ex = np.concatenate((np.tile(spectrum_counts[0], 60), spectrum_counts, np.tile(spectrum_counts[-1], 60)))
    pixels_hr, spectrum_counts_hr = fft_zerofilling(spectrum_counts_ex, 10)
    
    #remove extra points after fft
    pixels_hr = pixels_hr[60*21:len(pixels_hr)-60*21] - pixels_hr[60*21]
    spectrum_counts_hr = spectrum_counts_hr[60*21:len(spectrum_counts_hr)-60*21]
    wavenumbers_hr = nu_mp(diffractionOrder, pixels_hr, measurementTemperature) + delta_wavenumber
#        spectrum_counts_interpolated = np.interp(observation_wavenumbers, wavenumbers_hr, spectrum_counts_hr)



    ax1.plot(wavenumbers, spectrum_counts, label=label, color=colours[fileIndex])
    ax1.plot(wavenumbers_hr, spectrum_counts_hr, "--", color=colours[fileIndex], alpha=0.5)
#        ax1.plot(observation_wavenumbers, spectrum_counts_interpolated, label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])


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




ax1.set_title("Counts")
ax1.set_xlabel("Wavenumbers (cm-1)")
ax1.set_ylabel("Counts")
ax1.legend()






#get coefficients for all wavenumbers
POLYNOMIAL_DEGREE = 2

first_wavenumber = np.max([calDict[obsName]["wavenumbers_hr"][0] for obsName in calDict.keys()])
last_wavenumber = np.min([calDict[obsName]["wavenumbers_hr"][-1] for obsName in calDict.keys()])

wavenumber_grid = np.linspace(first_wavenumber, last_wavenumber, num=6720)
temperature_grid_unsorted = np.asfarray([calDict[obsName]["measurementTemperature"] for obsName in calDict.keys()])




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
colours = [cmap(i) for i in np.arange(len(wavenumber_grid))/len(wavenumber_grid)]


plt.figure();
coefficientsAll = []
for wavenumberIndex in range(len(wavenumber_grid)):

    coefficients = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex], POLYNOMIAL_DEGREE)
    coefficientsAll.append(coefficients)
    if wavenumberIndex in range(40, 6700, 200):
        quadratic_fit = np.polyval(coefficients, temperature_grid)
        linear_coefficients_normalised = np.polyfit(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), POLYNOMIAL_DEGREE)
        linear_fit_normalised = np.polyval(linear_coefficients_normalised, temperature_grid)
        plt.scatter(temperature_grid, spectra_grid[:, wavenumberIndex]/np.max(spectra_grid[:, wavenumberIndex]), label="Wavenumber %0.1f" %wavenumber_grid[wavenumberIndex], color=colours[wavenumberIndex])
        plt.plot(temperature_grid, linear_fit_normalised, "--", color=colours[wavenumberIndex])

coefficientsAll = np.asfarray(coefficientsAll).T

plt.legend()
plt.xlabel("Instrument temperature (Celsius)")
plt.ylabel("Counts for given pixel (normalised to peak)")
plt.title("Interpolating spectra w.r.t. temperature")


#write to hdf5 aux file
#write test file
import h5py

outputTitle = "LNO_Radiance_Factor_Calibration_Table"

with h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "w") as hdf5File:
    
    for obsName in calDict.keys():
        groupName = "%i/%0.1f" %(diffractionOrder, calDict[obsName]["measurementTemperature"])
        hdf5File[groupName+"/wavenumbers"] = calDict[obsName]["wavenumbers_hr"]
        hdf5File[groupName+"/counts"] = calDict[obsName]["spectrum_counts_hr"]
    hdf5File["%i" %diffractionOrder+"/wavenumber_grid"] = wavenumber_grid
    hdf5File["%i" %diffractionOrder+"/spectra_grid"] = spectra_grid
    hdf5File["%i" %diffractionOrder+"/coefficients"] = coefficientsAll





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




