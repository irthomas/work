# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:57:57 2019

@author: iant


BLACKBODY VS SOLAR CALIBRATION
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import savgol_filter

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList
from plot_solar_line_simulations_lno import getPixelSolarSpectrum, getConvolvedSolarSpectrum, nu_mp, t_p0


#solar fullscans and bb ground cals
#regex = re.compile("(20150426_030851_0p1a_LNO_1|20150426_054602_0p1a_LNO_1|20150427_010422_0p1a_LNO_1|20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1)")
regex = re.compile("(20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1)")
#regex = re.compile("(20150426_030851_0p1a_LNO_1|20150426_054602_0p1a_LNO_1|20150427_010422_0p1a_LNO_1|20161121_233000_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1)")
#regex = re.compile("20150426_030851_0p1a_LNO_1")
fileLevel = "hdf5_level_0p1a"


hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)

rSun = 695510. #km
dSun = 215.7e6 #for 20180611 obs 227.9e6 #km

#find 1 arcmin on sun in km
d1arcmin = dSun * np.tan((1.0 / 60.0) * (np.pi/180.0))

angleSolar = np.pi * (rSun / dSun) **2
ratio_fov_full_sun = (np.pi * rSun**2) / (d1arcmin * d1arcmin*4.0)
#SOLSPEC file is Earth TOA irradiance (no /sr )
RADIANCE_TO_IRRADIANCE = angleSolar * ratio_fov_full_sun * 2.0 #check factor 2.0 is good
#RADIANCE_TO_IRRADIANCE = 1.0




def getDiffractionOrders(aotfFrequencies):
    """get orders from aotf"""
    AOTFOrderCoefficients = np.array([3.9186850E-09, 6.3020400E-03, 1.3321030E+01])

    diffractionOrdersCalculated = [np.int(np.round(np.polyval(AOTFOrderCoefficients, aotfFrequency))) for aotfFrequency in aotfFrequencies]
    #set darks to zero
    diffractionOrders = np.asfarray([diffractionOrder if diffractionOrder > 50 else 0 for diffractionOrder in diffractionOrdersCalculated])
    return diffractionOrders


def cslWindow(x_scale):
    data_in = np.loadtxt("reference_files"+os.sep+"sapphire_window.csv", skiprows=1, delimiter=",")
    wavenumbers_in =  10000. / data_in[:,0]
    transmission_in = data_in[:,1] / 100.0
    transmission = np.interp(x_scale, wavenumbers_in[::-1], transmission_in[::-1])
    return transmission



def planck(xscale, temp, units): #planck function W/cm2/sr/spectral unit. Include CSL window for blackbody spectra!

    if units=="microns" or units=="um" or units=="wavel":
        c1=1.191042e8
        c2=1.4387752e4
        
        bb_radiance = c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0) / 1.0e4 # m2 to cm2
        csl_window_transmission = cslWindow(10000.0 / xscale) #convert to cm-1
        
        return bb_radiance * csl_window_transmission

    elif units=="wavenumbers" or units=="cm-1" or units=="waven":
        c1=1.191042e-5
        c2=1.4387752
        
        bb_radiance = ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2
        csl_window_transmission = cslWindow(xscale) #already in cm-1

        return bb_radiance * csl_window_transmission

    elif units=="solar radiance cm-1" or units=="solar cm-1":
        SOLAR_SPECTRUM = np.loadtxt("reference_files"+os.sep+"nomad_solar_spectrum_solspec.txt")
        try:
            solarRad = np.zeros(len(xscale))
        except:
            xscale = [xscale]
            solarRad = np.zeros(len(xscale))
            
        wavenumberInStart = SOLAR_SPECTRUM[0,0]
        wavenumberDelta = 0.005
                
        print("Finding solar radiances in ACE file")
        for pixelIndex,xValue in enumerate(xscale):
            index = np.int((xValue - wavenumberInStart)/wavenumberDelta)
            if index == 0:
                print("Warning: wavenumber out of range of solar file (start of file). wavenumber = %0.1f" %(xValue))
            if index == len(SOLAR_SPECTRUM[:,0]):
                print("Warning: wavenumber out of range of solar file (end of file). wavenumber = %0.1f" %(xValue))
            solarRad[pixelIndex] = SOLAR_SPECTRUM[index,1] / RADIANCE_TO_IRRADIANCE
        return solarRad
            
        
    else:
        print("Error: Unknown units given")




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
                0:[[-17.6, -9.9, -5.8, -3.6, 5.2], [114, 119, 122, 124, 130], 13, 3787.871601008524],
                1:[[-17.6, -9.9, -5.8, -3.6, 5.2], [77, 84, 90, 91, 98], 8, 3784.3935170320838],
                },
            188:{
#                0:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 13, 4242],
#                1:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 8, 4250.776050788092],
                },
            189:{
#                0:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 13, 4250.776050788092],
                1:[[-17.6, -9.9, -5.8, -3.6, 5.2], [252, 259, 262, 263, 269], 13, 4276.132702609243],
                },
            190:{
#                0:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 13, 4276.132702609243],
#                1:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 8, 4282],
                },
            193:{
#                0:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 13, 4276.132702609243],
                1:[[-17.6, -9.9, -5.8, -3.6, 5.2], [100, 101, 102, 103, 104], 8, 4364.37],
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

def findAbsorptionMininumIndex(spectrum, zeroFilling=10):
    """input spectrum containing absorption. Output location of minimum"""
    
    pixels = np.arange(len(spectrum))
    spectrum_continuum = np.polyval(np.polyfit([pixels[0], pixels[-1]], [spectrum[0], spectrum[-1]], 1), pixels) #interpolate across
    
    absorption = spectrum / spectrum_continuum
    pixels_hr, absorption_hr = fft_zerofilling(absorption, zeroFilling)
    
    #pixel containing minimum value in real data:
    spectrum_min_pixel = pixels_hr[np.where(absorption_hr == min(absorption_hr))[0][0]]

    return spectrum_min_pixel




#diffractionOrder = 168
#solarLineNumber = 0
diffractionOrder = 189
solarLineNumber = 1
#diffractionOrder = 193
#solarLineNumber = 1
pixels = np.arange(320)



def prepExternalTemperatureReadings(column_number):
    """read in TGO channel temperatures from file (only do once)"""
    
    with open(os.path.join(BASE_DIRECTORY, "reference_files", "heaters_temp_2018-03-24T000131_to_2019-08-24T080358.csv")) as f:
        lines = f.readlines()
            
    utc_datetimes = []
    temperatures = []
    for line in lines[1:]:
        split_line = line.split(",")
        utc_datetimes.append(datetime.strptime(split_line[0].split(".")[0], "%Y-%m-%dT%H:%M:%S"))
        temperatures.append(split_line[column_number])
    
    return np.asarray(utc_datetimes), np.asfarray(temperatures)
    


def getExternalTemperatureReadings(utc_string, column_number): #input format 2015 Mar 18 22:41:03.916651
    """get TGO readout temperatures. Input SPICE style datetime, output in Celsius
    column 1 = SO baseplate nominal, 2 = LNO baseplate nominal"""

    if "EXTERNAL_TEMPERATURE_DATETIMES" not in globals():
        print("Reading in TGO temperatures")
        EXTERNAL_TEMPERATURE_DATETIMES, EXTERNAL_TEMPERATURES = prepExternalTemperatureReadings(2) #2=LNO nominal


    
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




"""get nadir data from observations of a region"""
#test new calibrations on a mean nadir observation
from database_functions_v01 import obsDB, makeObsDict
dbName = "lno_0p3a"
db_obj = obsDB(dbName)
#CURIOSITY = -4.5895, 137.4417
if diffractionOrder in [168]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND n_orders < 4 AND incidence_angle < 10 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 2000
elif diffractionOrder in [189]:
#    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND n_orders < 4 AND incidence_angle < 20 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 10 AND latitude > -10 AND longitude < 10 AND longitude > -10 AND n_orders < 4 AND incidence_angle < 10 AND temperature_tgo > 1 AND temperature_tgo < 2 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 8.0
elif diffractionOrder in [188]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE n_orders < 4 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 3000
elif diffractionOrder in [193]:
    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE n_orders < 4 AND diffraction_order == %i" %diffractionOrder)
    SIGNAL_CUTOFF = 3000
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
        print(np.mean(y))
        validIndices[frameIndex] = False
#ax0.legend()
observation_spectrum = np.mean(np.asfarray(obsDict["y"])[validIndices, :], axis=0)
xMean = np.mean(np.asfarray(obsDict["x"])[validIndices, :], axis=0)
ax0.plot(xMean, observation_spectrum, "k")

#shift xMean to match solar line
observationTemperature = obsDict["temperature"][0]
continuum_pixels, solar_line_wavenumber = getSolarLinePosition(observationTemperature, diffractionOrder, solarLineNumber)

#find pixel containing minimum value in subset of real data
observation_absorption_spectrum = observation_spectrum[continuum_pixels]
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

#stop()

"""get convolved solar spectrum"""
solspecFile = os.path.join(BASE_DIRECTORY, "reference_files", "nomad_solar_spectrum_solspec.txt")
convolved_solar_wavenumbers, convolved_solar_spectrum = getConvolvedSolarSpectrum(diffractionOrder, observationTemperature, solspecFile, adj_orders=0)
solar_radiance_pixels = convolved_solar_spectrum / RADIANCE_TO_IRRADIANCE


"""code to check that solar line convolution isn't affected by temperature. Find wavenumber of main absorption"""
#plt.figure()
#for diffractionOrder in [193]:
#    
#    wavenumber_minima = []
#    for measurementTemperature in [-5.0, 0.0, 5.0]:
#        convolved_solar_wavenumbers, convolved_solar_spectrum = getConvolvedSolarSpectrum(diffractionOrder, measurementTemperature, solspecFile, adj_orders=0)
#        radiance = convolved_solar_spectrum / RADIANCE_TO_IRRADIANCE #fudge to get BB and Solar to match
#        plt.plot(convolved_solar_wavenumbers, radiance, label="Order %i" %diffractionOrder)
#    
#    
#        solar_pixels_hr, solar_spectrum_hr = fft_zerofilling(radiance, 100)
#        solar_wavenumbers_hr = nu_mp(diffractionOrder, solar_pixels_hr, measurementTemperature)
#        plt.plot(solar_wavenumbers_hr, solar_spectrum_hr)
#        
#        #"pixel" containing solar line minimum
#        radiance_min_pixel = solar_pixels_hr[np.where(solar_spectrum_hr == min(solar_spectrum_hr[:]))[0][0]] #main line
#    #    radiance_min_pixel = solar_pixels_hr[np.where(solar_spectrum_hr == min(solar_spectrum_hr[:23620]))[0][0]] #smaller line
#        radiance_min_wavenumber = nu_mp(diffractionOrder, radiance_min_pixel, measurementTemperature)
#        print("radiance_min_pixel=", radiance_min_pixel)
#        print("radiance_min_wavenumber=", radiance_min_wavenumber)
#        
#        wavenumber_minima.append(radiance_min_wavenumber)
#    
#    mean_wavenumber = np.mean(wavenumber_minima)
#    print("mean_wavenumber=", mean_wavenumber)
#    for wavenumber_minimum in wavenumber_minima:
#        print(diffractionOrder, wavenumber_minimum - mean_wavenumber)
#stop()


"""code to check solar lines in other orders"""
#plt.figure()
#observationTemperature = -5.0
##for diffractionOrder in [167,168,169]:
#for diffractionOrder in [189, 193]:
#    convolved_solar_wavenumbers, convolved_solar_spectrum = getConvolvedSolarSpectrum(diffractionOrder, observationTemperature, solspecFile, adj_orders=0)
#    plt.plot(convolved_solar_wavenumbers, convolved_solar_spectrum, label="Order %i" %diffractionOrder)
#plt.legend()
#stop()


"""code to check solar lines in solar fullscan data"""



#selectedDiffractionOrders = range(119, 120)
selectedDiffractionOrders = range(168, 169)

#fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
#fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
fig3, (ax3a, ax3b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y))
fig4, (ax4a, ax4b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y))
fig5, (ax5a, ax5b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y))
fig6, (ax6a, ax6b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y))
fig7, ax7 = plt.subplots(figsize=(FIG_X, FIG_Y))

calDict = {}




for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
    
    yRaw = hdf5_file["Science/Y"][...]
    aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
    sensor1Temperature = hdf5_file["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
    
    #get temperature from TGO readout instead of channel
    if hdf5_filename[0:4] not in ["2015", "2016"]:
        utcStartTime = hdf5_file["Geometry/ObservationDateTime"][0, 0]
        measurementTemperature = getExternalTemperatureReadings(utcStartTime, 2)
        utcEndTime = hdf5_file["Geometry/ObservationDateTime"][-1, 0]
        
    else: #for ground cal use 
        measurementTemperature = np.mean(sensor1Temperature[2:10])


    diffractionOrders = getDiffractionOrders(aotfFrequency)
    frameIndices = list(np.where(diffractionOrders == diffractionOrder)[0])

    if len(frameIndices) > 2:
        frameIndices = frameIndices[:2] #just take first 2 frames

    
    #cut unused / repeated orders from file (01pa only)
    if "0p1a" in hdf5_filename:
        ySelected = yRaw[frameIndices, :, :]
        aotfFrequencySelected = aotfFrequency[frameIndices]

    integrationTimeRaw = hdf5_file["Channel/IntegrationTime"][0]
    binningRaw = hdf5_file["Channel/Binning"][0]
    numberOfAccumulationsRaw = hdf5_file["Channel/NumberOfAccumulations"][0]
    nSpectra = ySelected.shape[0]
    nBins = ySelected.shape[1]
    nPixels = ySelected.shape[2]
    
    integrationTime = np.float(integrationTimeRaw) / 1.0e3 #microseconds to seconds
    numberOfAccumulations = np.float(numberOfAccumulationsRaw)/2.0 #assume LNO nadir background subtraction is on
    binning = np.float(binningRaw) + 1.0 #binning starts at zero

    #20150426_030851_0p1a_LNO_1|20150426_054602_0p1a_LNO_1|20150427_010422_0p1a_LNO_1
    if "20150426_030851" in hdf5_filename or "20150426_054602" in hdf5_filename or "20150427_010422" in hdf5_filename:
        bbTemperature = 425.0
        planckUnits = "cm-1"
        print("bbTemperature = %0.1f" %bbTemperature)
        
        bb_radiance_pixels = planck(nu_mp(diffractionOrder, pixels, measurementTemperature), bbTemperature, planckUnits) #radiance per pixel (measurement temperature assumed to be correct)
        radiance = bb_radiance_pixels

        yBinned = np.sum(ySelected[:, :, :], axis=1)
        measurementPixels = binning * nBins

    #20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1
    if "20161121_233000_0p1a_LNO_1" in hdf5_filename or "20180702_112352_0p1a_LNO_1" in hdf5_filename or "20181101_213226_0p1a_LNO_1" in hdf5_filename or "20190314_021825_0p1a_LNO_1" in hdf5_filename or "20190609_011514_0p1a_LNO_1" in hdf5_filename:
        bbTemperature = 0.0
        planckUnits = "solar cm-1"
        print("solar spectrum")

        radiance = solar_radiance_pixels

        #solar spectrum - take centre line only
        yBinned = ySelected[:, 11, :]
        measurementPixels = 1.0
    
    measurementSeconds = integrationTime * numberOfAccumulations
    #normalise to 1s integration time per pixel
    yBinnedNorm = yBinned / measurementSeconds / measurementPixels



    if planckUnits == "cm-1":
        ax3 = ax3a
        ax4 = ax4a
        ax5 = ax5a
        ax6 = ax6a
    if planckUnits == "solar cm-1":
        ax3 = ax3b
        ax4 = ax4b
        ax5 = ax5b
        ax6 = ax6b

    for spectrumIndex, diffractionOrder in zip(range(nSpectra), diffractionOrders[frameIndices]):
        
        spectrum_counts = yBinnedNorm[spectrumIndex, :]
        label = "%s order %0.0fkHz %s %0.1fC" %(hdf5_filename[:15], aotfFrequencySelected[spectrumIndex], {"cm-1":"bb", "solar cm-1":"sun"}[planckUnits], measurementTemperature)
        
        if planckUnits == "solar cm-1":
            
#            ax4.axvline(x=solar_line_wavenumber)
            ax6.axvline(x=solar_line_wavenumber)
            
            """shift spectral calibration to match solar line"""
            continuum_pixels, solar_line_wavenumber = getSolarLinePosition(measurementTemperature, diffractionOrder, solarLineNumber)
            
            #find pixel containing minimum value in subset of real data
            absorption_spectrum = spectrum_counts[continuum_pixels]
            spectrum_min_pixel = findAbsorptionMininumIndex(absorption_spectrum)
            spectrum_min_pixel = spectrum_min_pixel + continuum_pixels[0]
            spectrum_min_wavenumber = nu_mp(diffractionOrder, spectrum_min_pixel, measurementTemperature)

            #calculate wavenumber error            
            delta_wavenumber = solar_line_wavenumber - spectrum_min_wavenumber
            print("delta_wavenumber=", delta_wavenumber)
            
            #shift wavenumber scale to match solar line
            wavenumbers = nu_mp(diffractionOrder, pixels, measurementTemperature) + delta_wavenumber


            #make solar spectra on high resolution grid, interpolate to observation wavenumber grid
            pixels_hr, spectrum_counts_hr = fft_zerofilling(spectrum_counts, 10)
            wavenumbers_hr = nu_mp(diffractionOrder, pixels_hr, measurementTemperature) + delta_wavenumber
            spectrum_counts_interpolated = np.interp(observation_wavenumbers, wavenumbers_hr, spectrum_counts_hr)



            ax3.plot(pixels, radiance , label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])
            ax5.plot(pixels, spectrum_counts , label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])
        
            ax4.plot(convolved_solar_wavenumbers, radiance , label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])
            ax6.plot(wavenumbers, spectrum_counts, label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])
#            ax6.plot(wavenumbers_hr, spectrum_counts_hr, label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])
#            ax6.plot(observation_wavenumbers, spectrum_counts_interpolated, label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])

        
#           ax7.plot(wavenumbers, obs_calibrated, label=label, linestyle={"cm-1":"--", "solar cm-1":"-"}[planckUnits])

        

            calDict["%s-%i" %(hdf5_filename, spectrumIndex)] = {
                    "planckUnits":planckUnits,
                    "aotfFrequencySelected":aotfFrequencySelected[spectrumIndex],
    #                "integrationTimeRaw":integrationTimeRaw,
    #                "integrationTime":integrationTime,
    #                "numberOfAccumulationsRaw":numberOfAccumulationsRaw,
    #                "numberOfAccumulations":numberOfAccumulations,
    #                "binningRaw":binningRaw,
    #                "binning":binning,
                    "measurementSeconds":measurementSeconds,
                    "measurementPixels":measurementPixels,
    #                "yBinned200":yBinned[spectrumIndex, 200],
    #                "yBinnedNorm200":spectrum_counts[200],
    #                "radiance200":radiance[200],
    #                "counts_per_radiance200":counts_per_radiance[200],
                    
    #                "wavenumbers":wavenumbers,
                    "spectrum_counts":spectrum_counts,
                    "measurementTemperature":measurementTemperature,
    #                "p0":p0_new,
                    "spectrum_counts_interpolated":spectrum_counts_interpolated,
                    }



#ax1.set_title("Radiometric calibration")
#ax1.set_xlabel("Pixels")
#ax1.set_ylabel("DN/pixel/second per unit radiance W/cm2/sr/cm-1")
#ax1.legend()
#
#ax2.set_title("Radiometric calibration")
#ax2.set_xlabel("Wavenumbers (cm-1)")
#ax2.set_ylabel("DN/pixel/second per unit radiance W/cm2/sr/cm-1")
#ax2.legend()

ax3a.set_title("Radiance Spectrum")
ax3b.set_xlabel("Pixels")
ax3a.set_ylabel("Radiance W/cm2/sr/cm-1")
ax3b.set_ylabel("Radiance W/cm2/sr/cm-1")
ax3a.legend()
ax3b.legend()

ax4a.set_title("Radiance Spectrum")
ax4b.set_xlabel("Wavenumbers (cm-1)")
ax4a.set_ylabel("Radiance W/cm2/sr/cm-1")
ax4b.set_ylabel("Radiance W/cm2/sr/cm-1")
ax4a.legend()
ax4b.legend()

ax5a.set_title("Counts")
ax5b.set_xlabel("Pixels")
ax5a.set_ylabel("Counts")
ax5b.set_ylabel("Counts")
ax5a.legend()
ax5b.legend()

ax6a.set_title("Counts")
ax6b.set_xlabel("Wavenumbers (cm-1)")
ax6a.set_ylabel("Counts")
ax6b.set_ylabel("Counts")
ax6a.legend()
ax6b.legend()

ax7.set_title("Nadir Spectra")
ax7.set_xlabel("Wavenumbers (cm-1)")
ax7.set_ylabel("Radiance W/cm2/sr/cm-1")
ax7.legend()




#now interpolate counts between temperatures in wavenumber
#solar scans have already been interpolated onto observation wavenumber grid

all_solar_counts = np.asfarray([calDict[obsName]["spectrum_counts_interpolated"] for obsName in calDict.keys()])
all_temperatures = np.asfarray([calDict[obsName]["measurementTemperature"] for obsName in calDict.keys()])

sort_indices = np.argsort(all_temperatures)
all_solar_counts_sorted = all_solar_counts[sort_indices, :]
all_temperatures_sorted = all_temperatures[sort_indices]


#plot linear relationships
cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(len(pixels))/len(pixels)]

POLYNOMIAL_DEGREE = 2
CHOSEN_PIXELS = np.arange(20, 300, 1)

plt.figure();
linearCoeffientsAll = []
for pixelNumber in pixels:
    linear_coefficients = np.polyfit(all_temperatures_sorted, all_solar_counts_sorted[:, pixelNumber], POLYNOMIAL_DEGREE)
    linearCoeffientsAll.append(linear_coefficients)
    linear_fit = np.polyval(linear_coefficients, all_temperatures_sorted)
    if pixelNumber in range(40, 300, 20):
        linear_coefficients_normalised = np.polyfit(all_temperatures_sorted, all_solar_counts_sorted[:, pixelNumber]/np.max(all_solar_counts[:, pixelNumber]), POLYNOMIAL_DEGREE)
        linear_fit_normalised = np.polyval(linear_coefficients_normalised, all_temperatures_sorted)
        plt.scatter(all_temperatures_sorted, all_solar_counts_sorted[:, pixelNumber]/np.max(all_solar_counts[:, pixelNumber]), label="Pixel %i" %pixelNumber, color=colours[pixelNumber])
        plt.plot(all_temperatures_sorted, linear_fit_normalised, "--", color=colours[pixelNumber])
plt.legend()
plt.xlabel("Instrument temperature (Celsius)")
plt.ylabel("Counts for given pixel (normalised to peak)")
plt.title("Interpolating spectra w.r.t. temperature")


"""make interpolated spectrum and calibrate observation as I/F"""
#make solar reference using measurement temperature and linear coefficients
corrected_solar_spectrum = []
for pixelNumber, linear_coefficients in zip(pixels, linearCoeffientsAll):
    correct_solar_pixel = np.polyval(linear_coefficients, observationTemperature)
    corrected_solar_spectrum.append(correct_solar_pixel)

corrected_solar_spectrum = np.asfarray(corrected_solar_spectrum)
ax6b.plot(observation_wavenumbers[CHOSEN_PIXELS], corrected_solar_spectrum[CHOSEN_PIXELS])

#add conversion factor to account for solar incidence angle
conversion_factor = 1.0 / RADIANCE_TO_IRRADIANCE

#do I/F using shifted observation wavenumber scale
observation_i_f = observation_spectrum / corrected_solar_spectrum * conversion_factor

plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(observation_wavenumbers[CHOSEN_PIXELS], observation_i_f[CHOSEN_PIXELS])
plt.title("Nadir calibrated spectra order %i" %diffractionOrder)
plt.xlabel("Wavenumbers (cm-1)")
plt.ylabel("Radiance factor ratio")


x = observation_wavenumbers[CHOSEN_PIXELS]
y = observation_i_f[CHOSEN_PIXELS]




y_smooth = savgol_filter(y, 5, 3)
y_baseline = baseline_als(y_smooth, 250.0, 0.95)

#plt.figure()
#plt.plot(x, y)
#plt.plot(x, y_smooth)
#plt.plot(x, y_baseline)

plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(x, y_smooth/ y_baseline * np.mean(y_smooth))
plt.title("Nadir calibrated spectra order %i" %diffractionOrder)
plt.xlabel("Wavenumbers (cm-1)")
plt.ylabel("Baseline corrected radiance ratio")

#plt.plot(x_radfac[20:]+0.5, (y_radfac[20:]-0.3)*3e-04+0.00074)


#calibrate each spectrum in a file and store new x and y
yOut = np.asfarray([spectrum[CHOSEN_PIXELS] / corrected_solar_spectrum[CHOSEN_PIXELS] * conversion_factor for spectrum in obsDict["y"]])
xOut = np.asfarray([observation_wavenumbers[CHOSEN_PIXELS] for index in range(len(obsDict["y"]))])

#convert utc strings to hdf5 style
SPICE_DATETIME_FORMAT = "%Y %b %d %H:%M:%S"
utc_strings = np.array([[np.string_(utcTime.strftime(SPICE_DATETIME_FORMAT))]*2 for utcTime in obsDict["utc_start_time"]])

outputTitle = "20180611_131514_1p0a_LNO_1_D_189_test"

#write test file
import h5py
with h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "w") as hdf5File:
    hdf5File["Science/Y"] = yOut
    hdf5File["Science/X"] = xOut
    hdf5File["Channel/DiffractionOrder"] = np.tile(np.asarray(obsDict["diffraction_order"]), [2, 1]).T
    hdf5File["Geometry/Point0/IncidenceAngle"] = np.tile(np.asarray(obsDict["incidence_angle"]), [2, 1]).T
    hdf5File["Geometry/Point0/Lon"] = np.tile(np.asarray(obsDict["longitude"]), [2, 1]).T
    hdf5File["Geometry/Point0/Lat"] = np.tile(np.asarray(obsDict["latitude"]), [2, 1]).T
    hdf5File["Geometry/Point0/LST"] = np.tile(np.asarray(obsDict["local_time"]), [2, 1]).T
    
    
    hdf5File.create_dataset("Geometry/ObservationDateTime", data=utc_strings, dtype='S100')

#read test file

with h5py.File(os.path.join(BASE_DIRECTORY, outputTitle+".h5"), "r") as hdf5File:
    yOut = hdf5File["Science/Y"][...]
    xOut = hdf5File["Science/X"][...]
    lonOut = hdf5File["Geometry/Point0/Lon"][:, 0]
    latOut = hdf5File["Geometry/Point0/Lat"][:, 0]
    angleOut = hdf5File["Geometry/Point0/IncidenceAngle"][:, 0]
    dtOut = hdf5File["Geometry/ObservationDateTime"][:, 0]

plt.figure(figsize=(FIG_X+4, FIG_Y))
for x, y, lon, lat, angle, dt in zip(xOut, yOut, lonOut, latOut, angleOut, dtOut):
    plt.plot(10000.0/x,y, label="(%0.1f, %0.1f), inc angle=%0.1f, %s" %(lon, lat, angle, dt))
plt.legend()
plt.ylabel("Radiance factor")
plt.xlabel("Wavelength um")
plt.title(outputTitle)
plt.plot(10000.0/xOut[0, :], np.mean(yOut, axis=0), "k")
plt.savefig(os.path.join(BASE_DIRECTORY, outputTitle+".png"))
