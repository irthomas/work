# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:38:42 2019

@author: iant
"""

import os
import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#TODO: update with 2019 LNO coefficients when ready. check temperature shift, resolving power, aotf sidelobe


#SAVE_FILES = True
SAVE_FILES = False

#1. VERTICALLY BIN DETECTOR COUNTS AND CONVERT TO PER PIXEL PER SECOND FROM A FULLSCAN OBSERVING A KNOWN SOURCE (BB + CHAMBER WINDOW)
#2. USE AOTF AND BLAZE FUNCTIONS TO CALCULATE RADIANCE HITTING EACH PIXEL. AOTF AND BLAZE ARE NORMALISED TO 1, SO IT IS A RELATIVE RADIANCE (OTHER OPTICS NOT INCLUDED)
#3. CALCULATE COUNTS PER RADIANCE FOR EACH PIXEL

LNO_FLAGS_DICT = {
"AOTF_FUNCTION_X_RANGE_START":-200.0,
"AOTF_FUNCTION_X_RANGE_STOP":200.0,
"AOTF_FUNCTION_X_RANGE_STEP":0.1,
}

VERSION = "03"

FIG_X = 18
FIG_Y = 9

FIRST_PIXEL = 0
N_ADJACENT_ORDERS = 0
#N_ADJACENT_ORDERS = 2 #use this for real calibration


if os.path.exists(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
elif os.path.exists(os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")
elif os.path.exists(os.path.normcase(r"/home/iant/linux")):
    BASE_DIRECTORY = os.path.normcase(r"/home/iant/linux")



title = "LNO_Radiometric_Calibration_Table"
outputFilename = "%s" %(title.replace(" ","_"))
outputFilepath = os.path.join(BASE_DIRECTORY, outputFilename + "_v%s.h5" %VERSION)


#read existing calibration file into dictionary
#calibrationTime = "2017 JAN 01 00:00:00.000"
#calibrationFile = {}
#with h5py.File(outputFilepath, "r") as hdf5File:
#    hdf5Group = hdf5File[calibrationTime]
#    for key, value in hdf5Group.items():
#        calibrationFile[key] = value[...]
#plt.figure(figsize=(FIG_X, FIG_Y))
#chosenPixel = 200
#centralWavenumbers = calibrationFile["CentralWavenumbers"][...]
#centralWavelengths = 10000.0/centralWavenumbers
#plt.scatter(centralWavelengths, calibrationFile["CountsPerRadianceFit"][:, chosenPixel] / 1.0e4, label="Counts per radiance full AOTF")
#plt.scatter(centralWavelengths, calibrationFile["CountsPerRadianceAtWavenumberFit"][:, chosenPixel] / 1.0e4, label="Counts per radiance at wavenumber")
#plt.title("Instrument sensitivity curve for pixel %i" %chosenPixel)
#plt.xlabel("Wavelength (microns)")
#plt.ylabel("DN/pixel/second per unit radiance W/m2/sr/cm-1")
#plt.legend()
#
#
#stop


lnoCoefficientDict = {
"AOTFWnCoefficients":[9.409476e-8, 0.1422382, 300.67657],
"AOTFCentreTemperatureShiftCoefficients":[0.0, -6.5278e-5, 0.0],

"AOTFOrderCoefficients":[3.9186850E-09, 6.3020400E-03, 1.3321030E+01],
"ResolvingPowerCoefficients":[-1.898669696e-05, 0.2015505624, 16509.58391], #calculated below from figure in Liuzzi et al

"BlazeFunction":[0.0, 0.0, 22.478113, 0.0001245622383, 22.56190161, 0.00678411387], #i.e. FSR and centre of grating. Replace with new coefficients in wavenumbers

"PixelSpectralCoefficients":[3.774791e-8, 5.508335e-4, 22.478113],
"FirstPixelCoefficients":[0.0, -6.556383e-1, -8.024164],

"AOTFCoefficientsLiuzzi":[0.6290016297432226, 18.188122, 0.37099837025677734, 12.181137], #i0, w, ig, sigmag
}








"""AOTF shape"""
def func_aotf(x, x0, i0, w, iG, sigmaG): #Goddard model 2018
    x0 = x0 + 0.0001 #fudge to stop infinity at peak
    
    fsinc = (i0 * w**2.0 * (np.sin(np.pi * (x - x0) / w))**2.0) / (np.pi**2.0 * (x - x0)**2.0)
    fgauss = iG * np.exp(-1.0 * (x - x0)**2.0 / sigmaG**2.0)
    f = fsinc + fgauss #slant not included
    return f/np.max(f) #slant not included. normalised
    

def getAOTFFunction(flagsDict, coefficientDict, aotfFrequency):

    xStart = flagsDict["AOTF_FUNCTION_X_RANGE_START"]
    xStop = flagsDict["AOTF_FUNCTION_X_RANGE_STOP"]
    xStep = flagsDict["AOTF_FUNCTION_X_RANGE_STEP"]
    xRange = np.arange(xStart, xStop + xStep, xStep)
    
    c1 = coefficientDict["AOTFWnCoefficients"]
    aotfCentre = np.polyval(c1, aotfFrequency)
    
    
    aotfFunctionI0 = coefficientDict["AOTFCoefficientsLiuzzi"][0]
    aotfFunctionW = coefficientDict["AOTFCoefficientsLiuzzi"][1]
    aotfFunctionIg = coefficientDict["AOTFCoefficientsLiuzzi"][2]
    aotfFunctionSigmaG = coefficientDict["AOTFCoefficientsLiuzzi"][3]

    
    aotfFunction = func_aotf(xRange+aotfCentre, aotfCentre, aotfFunctionI0, aotfFunctionW, aotfFunctionIg, aotfFunctionSigmaG)
    aotfCentralWavenb = aotfCentre
    return xRange + aotfCentralWavenb, aotfFunction, aotfCentralWavenb


def getSpectralResolution(flagsDict, coefficientDict, aotfCentralWavenb):
    """Spectral resolution"""
    
    aotfCentre = aotfCentralWavenb

    c3 = coefficientDict["ResolvingPowerCoefficients"]
    resolvingPower = np.polyval(c3, aotfCentre)
    spectralResolution = aotfCentre / resolvingPower
    
    return spectralResolution



def getX(flagsDict, coefficientDict, diffractionOrder, calibrationTemperature):

        
    #calculate pixel shift based on Goddard analysis and temperature sensor 1.
#    c0 = coefficientDict["AOTFOrderCoefficients"]
    c1 = coefficientDict["FirstPixelCoefficients"]
    c2 = coefficientDict["PixelSpectralCoefficients"]
    t = calibrationTemperature

#    diffractionOrder = np.round(np.polyval(c0, aotfFrequency))
    firstPixelValue = np.polyval(c1, t)
    pixelValues = np.arange(FIRST_PIXEL, 320 + FIRST_PIXEL, 1) + firstPixelValue #apply temperature shift
    x = np.polyval(c2, pixelValues) * diffractionOrder
        
    return x, firstPixelValue

def getBlazeFunction(flagsDict, coefficientDict, diffractionOrder, calibrationTemperature):
    
    #calculate pixel shift based on Goddard analysis and temperature sensor 1.
#    c0 = coefficientDict["AOTFOrderCoefficients"]
    c1 = coefficientDict["FirstPixelCoefficients"]
    c2 = coefficientDict["PixelSpectralCoefficients"]
    t = calibrationTemperature

#    diffractionOrder = np.round(np.polyval(c0, aotfFrequency))
    firstPixelValue = np.polyval(c1, t)
    pixelValues = np.arange(FIRST_PIXEL, 320 + FIRST_PIXEL, 1) + firstPixelValue #apply temperature shift
    x = np.polyval(c2, pixelValues) * diffractionOrder

    c3 = coefficientDict["BlazeFunction"][0:3]
    c4 = coefficientDict["BlazeFunction"][3:6]
    
    blazeWidth = np.polyval(c3, diffractionOrder)
    blazeCentre = np.polyval(c4, diffractionOrder) #in cm-1
    
    blazeFunction = np.sinc((x - blazeCentre)/blazeWidth)**2
    
    return blazeFunction
    
    

def getDiffractionOrders(coefficientDict, aotfFrequencies):
    """get orders from aotf"""
    diffractionOrdersCalculated = [np.int(np.round(np.polyval(coefficientDict["AOTFOrderCoefficients"], aotfFrequency))) for aotfFrequency in aotfFrequencies]
    #set darks to zero
    diffractionOrders = np.asfarray([diffractionOrder if diffractionOrder > 50 else 0 for diffractionOrder in diffractionOrdersCalculated])
    return diffractionOrders




def planck(xscale, temp, units): #planck function W/cm2/sr/spectral unit

    SOLAR_SPECTRUM = np.loadtxt("reference_files"+os.sep+"nomad_solar_spectrum_solspec.txt")
    RADIANCE_TO_IRRADIANCE = 8.77e-5 / 100.0**2 #fudge to make curves match. should be 2.92e-5 on mars, 6.87e-5 on earth
    if units=="microns" or units=="um" or units=="wavel":
        c1=1.191042e8
        c2=1.4387752e4
        return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0) / 1.0e4 # m2 to cm2
    elif units=="wavenumbers" or units=="cm-1" or units=="waven":
        c1=1.191042e-5
        c2=1.4387752
        return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2
    elif units=="solar radiance cm-1" or units=="solar cm-1":
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
            solarRad[pixelIndex] = SOLAR_SPECTRUM[index,1] / RADIANCE_TO_IRRADIANCE #/ 1.387 #1AU to MCO-1 AU
        return solarRad / 1.0e4 #m2 to cm2
            
        
    else:
        print("Error: Unknown units given")



def cslWindow(x_scale):
    data_in = np.loadtxt("reference_files"+os.sep+"sapphire_window.csv", skiprows=1, delimiter=",")
    wavenumbers_in =  10000. / data_in[:,0]
    transmission_in = data_in[:,1] / 100.0
    
    transmission = np.interp(x_scale, wavenumbers_in[::-1], transmission_in[::-1])
    
    return transmission






def getBBRadiancePerPixel(flagsDict, coefficientDict, aotfFrequency, calibrationTemperature, bbTemperature, plot=False):
    """get radiance per pixel of a blackbody, accounting for AOTF, blaze, CSL window, orders+-2"""

    aotfX, aotfFunction, aotfCentralWavenb = getAOTFFunction(flagsDict, coefficientDict, aotfFrequency)
    windowTransmission = cslWindow(aotfX)
    aotfWindow = aotfFunction * windowTransmission
    
    if plot: 
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(aotfX, aotfFunction)
        ax1.plot(aotfX, windowTransmission)
        ax1.plot(aotfX, aotfWindow)
    
    centralOrder = int(getDiffractionOrders(coefficientDict, [aotfFrequency])[0])
    total_radiance_per_pixel = np.zeros(320)
    
    blazeFunction = coefficientDict["BlazeFunction"]
    print("aotfCentralWavenb = %0.5g" %aotfCentralWavenb)
    print("diffractionOrder, bb_radiance, aotfWindow, blazeFunction, resp, bb_radiance_resp")
#    for order_index, diffractionOrder in enumerate(range(centralOrder-2, centralOrder+3, 1)):
    for order_index, diffractionOrder in enumerate(range(centralOrder - N_ADJACENT_ORDERS, centralOrder + N_ADJACENT_ORDERS + 1)):
    
    
        pixelX, firstPixel = getX(flagsDict, coefficientDict, diffractionOrder, calibrationTemperature)
        if diffractionOrder == centralOrder:
            centreX = pixelX
        blazeFunction = getBlazeFunction(flagsDict, coefficientDict, diffractionOrder, calibrationTemperature)
        
        
        
        bb_radiance = planck(pixelX, bbTemperature, "cm-1")
        
        if plot:
            ax1.plot(pixelX, blazeFunction)
            ax2.plot(pixelX, bb_radiance)
        
        bb_radiance_resp = np.zeros(320)
    
        for pixel in range(320): #loop through pixels
            index = np.abs(aotfX - pixelX[pixel]).argmin() #find closest value
            resp = aotfWindow[index] * blazeFunction[pixel]
            bb_radiance_resp[pixel] = bb_radiance[pixel] * resp
            total_radiance_per_pixel[pixel] += bb_radiance_resp[pixel] #find radiance value closest to pixel wavenumber
            
            if pixel == 200:
                if plot: 
                    ax2.plot(pixelX[pixel], bb_radiance_resp[pixel], "x")
                print("%i, %0.5g, %0.5g, %0.5g, %0.5g, %0.5g" %(diffractionOrder, bb_radiance[pixel], aotfWindow[index], blazeFunction[pixel], resp, bb_radiance_resp[pixel]))
            
        if plot: 
            ax2.plot(pixelX, bb_radiance_resp)
    
    if plot: 
        ax2.plot(centreX, total_radiance_per_pixel)
        ax2.plot(centreX[200], total_radiance_per_pixel[200], "x")

    return total_radiance_per_pixel, aotfCentralWavenb


hdf5file_path = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p1a\2015\04\26\20150426_054602_0p1a_LNO_1.h5") #150C BB (cold only)
#hdf5file_path = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_0p1a\2015\04\26\20150426_030851_0p1a_LNO_1.h5") #150C BB (cold only)


hdf5Filename = os.path.basename(hdf5file_path)
if "20150426_054602" in hdf5Filename:
    bbTemperature = 425.0
    print("bbTemperature = %0.1f" %bbTemperature)
#    frameIndices = range(5, 92)
    frameIndices = range(5, 15)

with h5py.File(hdf5file_path, "r") as hdf5FileIn:
    Y = hdf5FileIn["Science/Y"][...]
    AOTFFrequency = hdf5FileIn["Channel/AOTFFrequency"][...]
    sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
    MeasurementTemperature = np.mean(sensor1Temperature[2:10])
    
    #cut unused / repeated orders from file
    Y = Y[frameIndices, :, :]
    AOTFFrequency = AOTFFrequency[frameIndices]
    
    
    IntegrationTime = hdf5FileIn["Channel/IntegrationTime"][0]
    Binning = hdf5FileIn["Channel/Binning"][0]
    NumberOfAccumulations = hdf5FileIn["Channel/NumberOfAccumulations"][0]
nSpectra = Y.shape[0]
nBins = Y.shape[1]
nPixels = Y.shape[2]

integrationTime = np.float(IntegrationTime) / 1.0e3 #microseconds to seconds
nAccumulation = np.float(NumberOfAccumulations)/2.0 #assume LNO nadir background subtraction is on
binning = np.float(Binning) + 1.0 #binning starts at zero

diffractionOrders = getDiffractionOrders(lnoCoefficientDict, AOTFFrequency)

print("integrationTimeFile = %i" %IntegrationTime)
print("integrationTime = %0.2f" %integrationTime)
print("nAccumulationFile = %i" %NumberOfAccumulations)
print("nAccumulation = %i" %nAccumulation)
print("binning = %i" %binning)

YBinned = np.sum(Y[:, :, :], axis=1)
measurementSeconds = integrationTime * nAccumulation
measurementPixels = binning * nBins
print("measurementSeconds = %0.1f" %measurementSeconds)
print("measurementPixels = %i" %measurementPixels)

#normalise to 1s integration time per pixel
YBinnedNorm = YBinned / measurementSeconds / measurementPixels

"""correct obvious detector offset"""
#if hdf5Filename == "20150426_054602_0p1a_LNO_1":
#    detectorData[86,:] = detectorData[86,:] - 1.05 #fudge to correct bad point

radiancesAll = []
radianceAtWavenumberAll = []
centralWavenumbers = []

countsPerRadianceAll = []
countsPerRadianceAtWavenumberAll = []
polyCountsPerRadianceAll = []
polyCountsPerRadianceAtWavenumberAll = []

countsPerRadianceSpectralResolutionAll = []
countsPerRadianceSpectralResolutionAtWavenumberAll = []
polyCountsPerRadianceSpectralResolutionAll = []
polyCountsPerRadianceSpectralResolutionAtWavenumberAll = []


CHOSEN_PIXEL = 200
CHOSEN_DIFFRACTION_ORDERS = [134, 169]

for spectrum_index in range(nSpectra):

    YSpectrum = YBinnedNorm[spectrum_index]
    aotfFrequency = AOTFFrequency[spectrum_index]
    totalRadianceOnPixel, centralWavenumber = getBBRadiancePerPixel(LNO_FLAGS_DICT, lnoCoefficientDict, aotfFrequency, MeasurementTemperature, bbTemperature)
    radianceAtWavenumber = planck(getX(LNO_FLAGS_DICT, lnoCoefficientDict, diffractionOrders[spectrum_index], MeasurementTemperature)[0], bbTemperature, "cm-1")
    spectralResolution = getSpectralResolution(LNO_FLAGS_DICT, lnoCoefficientDict, centralWavenumber)
    centralWavenumbers.append(centralWavenumber)
    countsPerRadiance = YSpectrum / totalRadianceOnPixel
    countsPerRadianceAtWavenumber = YSpectrum / radianceAtWavenumber
    countsPerRadianceSpectralResolution = YSpectrum / totalRadianceOnPixel / spectralResolution
    countsPerRadianceSpectralResolutionAtWavenumber = YSpectrum / radianceAtWavenumber / spectralResolution
    radiancesAll.append(totalRadianceOnPixel)
    radianceAtWavenumberAll.append(radianceAtWavenumber)
    countsPerRadianceAll.append(countsPerRadiance)
    countsPerRadianceAtWavenumberAll.append(countsPerRadianceAtWavenumber)
    countsPerRadianceSpectralResolutionAll.append(countsPerRadianceSpectralResolution)
    countsPerRadianceSpectralResolutionAtWavenumberAll.append(countsPerRadianceSpectralResolutionAtWavenumber)


    POLYFIT_PIXEL_RANGE = [50, 320]
    POLYFIT_PIXEL_RANGE_AT_WAVENUMBER = [0, 320]
    POLYFIT_DEGREE = 5
    POLYFIT_DEGREE_AT_WAVENUMBER = 5
    polyfitCoefficients = np.polyfit(range(POLYFIT_PIXEL_RANGE[0], POLYFIT_PIXEL_RANGE[1]), countsPerRadiance[POLYFIT_PIXEL_RANGE[0]:POLYFIT_PIXEL_RANGE[1]], POLYFIT_DEGREE)
    polyCountsPerRadiance = np.polyval(polyfitCoefficients, range(320))
    if np.any(polyCountsPerRadiance != np.abs(polyCountsPerRadiance)):
        print("Warning: negatives found in polyCountsPerRadiance")
        stop()
    polyCountsPerRadianceAll.append(polyCountsPerRadiance)

    polyfitCoefficientsAtWavenumber = np.polyfit(range(POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[0], POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[1]), countsPerRadianceAtWavenumber[POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[0]:POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[1]], POLYFIT_DEGREE_AT_WAVENUMBER)
    polyCountsPerRadianceAtWavenumber = np.polyval(polyfitCoefficientsAtWavenumber, range(320))
    if np.any(polyCountsPerRadianceAtWavenumber != np.abs(polyCountsPerRadianceAtWavenumber)):
        print("Warning: negatives found in polyCountsPerRadianceAtWavenumber")
        stop()
    polyCountsPerRadianceAtWavenumberAll.append(polyCountsPerRadianceAtWavenumber)

    polyfitCoefficientsSpectralResolution = np.polyfit(range(POLYFIT_PIXEL_RANGE[0], POLYFIT_PIXEL_RANGE[1]), countsPerRadianceSpectralResolution[POLYFIT_PIXEL_RANGE[0]:POLYFIT_PIXEL_RANGE[1]], POLYFIT_DEGREE)
    polyCountsPerRadianceSpectralResolution = np.polyval(polyfitCoefficientsSpectralResolution, range(320))
    polyCountsPerRadianceSpectralResolutionAll.append(polyCountsPerRadianceSpectralResolution)

    polyfitCoefficientsSpectralResolutionAtWavenumber = np.polyfit(range(POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[0], POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[1]), countsPerRadianceSpectralResolutionAtWavenumber[POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[0]:POLYFIT_PIXEL_RANGE_AT_WAVENUMBER[1]], POLYFIT_DEGREE_AT_WAVENUMBER)
    polyCountsPerRadianceSpectralResolutionAtWavenumber = np.polyval(polyfitCoefficientsSpectralResolutionAtWavenumber, range(320))
    polyCountsPerRadianceSpectralResolutionAtWavenumberAll.append(polyCountsPerRadianceSpectralResolutionAtWavenumber)

    if diffractionOrders[spectrum_index] in CHOSEN_DIFFRACTION_ORDERS:
        print("diffractionOrder = %i" %diffractionOrders[spectrum_index])
        print("summed raw counts = %0.0f" %YBinned[spectrum_index, CHOSEN_PIXEL])
        print("counts per px per second = %0.5g" %YSpectrum[CHOSEN_PIXEL])
        print("totalRadianceOnPixel = %0.5g" %totalRadianceOnPixel[CHOSEN_PIXEL])
        print("radianceAtWavenumber = %0.5g" %radianceAtWavenumber[CHOSEN_PIXEL])
        print("countsPerRadiance = %0.5g" %countsPerRadiance[CHOSEN_PIXEL])
        print("countsPerRadianceAtWavenumber = %0.5g" %countsPerRadianceAtWavenumber[CHOSEN_PIXEL])
        
        print("polyCountsPerRadiance = %0.5g" %polyCountsPerRadiance[CHOSEN_PIXEL])
        print("polyCountsPerRadianceAtWavenumber = %0.5g" %polyCountsPerRadianceAtWavenumber[CHOSEN_PIXEL])


centralWavenumbers = np.asfarray(centralWavenumbers)
radiancesAll = np.asfarray(radiancesAll)
radianceAtWavenumberAll = np.asfarray(radianceAtWavenumberAll)

countsPerRadianceAll = np.asfarray(countsPerRadianceAll)
countsPerRadianceAtWavenumberAll = np.asfarray(countsPerRadianceAtWavenumberAll)
polyCountsPerRadianceAll = np.asfarray(polyCountsPerRadianceAll)
polyCountsPerRadianceAtWavenumberAll = np.asfarray(polyCountsPerRadianceAtWavenumberAll)

countsPerRadianceSpectralResolutionAll = np.asfarray(countsPerRadianceSpectralResolutionAll)
countsPerRadianceSpectralResolutionAtWavenumberAll = np.asfarray(countsPerRadianceSpectralResolutionAtWavenumberAll)
polyCountsPerRadianceSpectralResolutionAll = np.asfarray(polyCountsPerRadianceSpectralResolutionAll)
polyCountsPerRadianceSpectralResolutionAtWavenumberAll = np.asfarray(polyCountsPerRadianceSpectralResolutionAtWavenumberAll)



chosenPixel = 200

fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
ax1.scatter(10000.0/centralWavenumbers, countsPerRadianceAll[:, chosenPixel] / 1.0e4, label="Counts per radiance full AOTF")
ax1.scatter(10000.0/centralWavenumbers, countsPerRadianceSpectralResolutionAll[:, chosenPixel] / 1.0e4, label="Counts per radiance full AOTF (inc. spectral resolution)")
ax1.scatter(10000.0/centralWavenumbers, countsPerRadianceAtWavenumberAll[:, chosenPixel] / 1.0e4, label="Counts per radiance at wavenumber")
ax1.scatter(10000.0/centralWavenumbers, countsPerRadianceSpectralResolutionAtWavenumberAll[:, chosenPixel] / 1.0e4, label="Counts per radiance at wavenumber (inc. spectral resolution)")
ax1.set_title("Instrument sensitivity curve for pixel %i" %chosenPixel)
ax1.set_xlabel("Wavelength (microns)")
ax1.set_ylabel("DN/pixel/second per unit radiance W/m2/sr/cm-1")
ax1.legend()

#define colours for eachframe to plot
cmap = plt.get_cmap('jet')
frameColours = [cmap(i) for i in np.arange(len(centralWavenumbers))/len(centralWavenumbers)]

fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
ax2.plot(radiancesAll.T)
ax2.plot(radianceAtWavenumberAll.T, linestyle="--")
ax2.set_title("Calculated radiances for diffraction orders")
ax2.set_xlabel("Pixel Number")

fig3, ax3 = plt.subplots(figsize=(FIG_X, FIG_Y))

for frameIndex in range(len(centralWavenumbers)):
    
    frameColour = frameColours[frameIndex]

    ax3.plot(countsPerRadianceAll[frameIndex, :], color=frameColour, linestyle="-", label="Counts per radiance full AOTF %0.1f" %centralWavenumbers[frameIndex])
    ax3.plot(polyCountsPerRadianceAll[frameIndex, :], color=frameColour, linestyle=":", label="Counts per radiance full AOTF Fit %0.1f" %centralWavenumbers[frameIndex])
    ax3.plot(countsPerRadianceAtWavenumberAll[frameIndex, :], color=frameColour, linestyle="--", label="Counts per radiance at wavenumber %0.1f" %centralWavenumbers[frameIndex])
    ax3.plot(polyCountsPerRadianceAtWavenumberAll[frameIndex, :], color=frameColour, linestyle=":", label="Counts per radiance at wavenumber fit %0.1f" %centralWavenumbers[frameIndex])
    ax3.set_xlabel("Pixel Number")
    ax3.set_ylabel("Counts per unit radiance (per ms per pixel per accumulation")
    ax3.set_title("%s T=%iK - IT=%ims NAcc=%i Binning=%i" %(hdf5Filename, bbTemperature, integrationTime, nAccumulation, binning))
    ax3.set_ylim([0.0,1.5e8])
ax3.legend()


"""write to file"""

datasetDict = {
        "DiffractionOrder":diffractionOrders,
        "CentralWavenumbers":centralWavenumbers,
        "VerticallyBinnedCounts":YBinnedNorm,
        "BlackbodyRadiances":radiancesAll,
        "CountsPerRadiance":countsPerRadianceAll,
        "CountsPerRadianceFit":polyCountsPerRadianceAll,

        "BlackbodyRadiancesAtWavenumber":radianceAtWavenumberAll,
        "CountsPerRadianceAtWavenumber":countsPerRadianceAtWavenumberAll,
        "CountsPerRadianceAtWavenumberFit":polyCountsPerRadianceAtWavenumberAll,

        "CountsPerRadianceSpectralResolution":countsPerRadianceSpectralResolutionAll,
        "CountsPerRadianceSpectralResolutionFit":polyCountsPerRadianceSpectralResolutionAll,
        "CountsPerRadianceSpectralResolutionAtWavenumber":countsPerRadianceSpectralResolutionAtWavenumberAll,
        "CountsPerRadianceSpectralResolutionAtWavenumberFit":polyCountsPerRadianceSpectralResolutionAtWavenumberAll,
        }


#make arrays of coefficients for given calibration date
#at present, values don't change over time. Therefore copy values for dates 2 and 3
calibrationTimes = ["2015 JAN 01 00:00:00.000", "2017 JAN 01 00:00:00.000"]


if SAVE_FILES:
    with h5py.File(outputFilepath, "w") as hdf5File:
    
        for calibrationTime in calibrationTimes:
            hdf5Group = hdf5File.create_group(calibrationTime)
                
            for key, value in datasetDict.items():
                hdf5Dataset = hdf5Group.create_dataset(key, data=value, dtype=np.float64)
        
            
        comments = "Analysis by I. Thomas, data from file %s" %hdf5Filename
        hdf5File.attrs["NumberOfAdjacentOrders"] = N_ADJACENT_ORDERS
        hdf5File.attrs["Comments"] = comments
        hdf5File.attrs["DateCreated"] = str(datetime.now())

