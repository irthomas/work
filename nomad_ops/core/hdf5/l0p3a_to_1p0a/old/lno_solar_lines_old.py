# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:54:51 2020

@author: iant

CHECK SOLAR LINE SPECTRAL POSITIONS FOR ALL ORDERS
INTERPOLATE TEMPERATURE SHIFTS TO ALL ORDERS
"""


import os
import numpy as np
import h5py
import re
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate
from fit_absorption_band_v01 import fit_gaussian_absorption

from hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, FIG_X, FIG_Y, makeFileList
from tools.spectra.smooth_hr import smooth_hr as smoothHighRes#, nu_mp, t_p0


SMOOTHING_LEVEL = 600 #must be even number
#PLOT_ONLY = True
PLOT_ONLY = False



regex = re.compile("(20161121_233000_0p1a_LNO_1|20180702_112352_0p1a_LNO_1|20181101_213226_0p1a_LNO_1|20190314_021825_0p1a_LNO_1|20190609_011514_0p1a_LNO_1)")
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
    
if "EXTERNAL_TEMPERATURE_DATETIMES" not in globals():
    print("Reading in TGO temperatures")
    EXTERNAL_TEMPERATURE_DATETIMES, EXTERNAL_TEMPERATURES = prepExternalTemperatureReadings(2) #2=LNO nominal



def getExternalTemperatureReadings(utc_string, column_number): #input format 2015 Mar 18 22:41:03.916651
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
bestOrders = [118, 120, 142, 151, 156, 162, 166, 167, 178, 189, 194]


"""plot solar lines in solar fullscan data for orders contains strong solar lines"""
temperatureRange = np.arange(-20., 15., 0.1)
cmap = plt.get_cmap('plasma')
colours = [cmap(i) for i in np.arange(len(temperatureRange))/len(temperatureRange)]


selectedDiffractionOrders = bestOrders
#selectedDiffractionOrders = range(168, 169)


for diffractionOrder in selectedDiffractionOrders:

    fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2), sharex=True)
    fig1.suptitle("Diffraction Order %i" %diffractionOrder)
    if not PLOT_ONLY:
        fig2, (ax2a, ax2b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
        fig2.suptitle("Diffraction Order %i" %diffractionOrder)


    """get high res solar spectra"""
    #define spectral range
    pixels = np.arange(320)
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
    
    
    #convolve high res solar spectrum to lower resolution. Scale to avoid swamping figure
    solar_spectrum = smoothHighRes(I0_solar_hr, window_len=(SMOOTHING_LEVEL-1))
#    normalised_solar_spectrum = ((solar_spectrum-np.min(solar_spectrum)) / (np.max(solar_spectrum)-np.min(solar_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]# / 5.0 + 0.8
    normalised_solar_spectrum = (solar_spectrum / (np.max(solar_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]
    ax1b.plot(nu_hr, normalised_solar_spectrum, "k")

    if not PLOT_ONLY:
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


    """get fullscan data"""
    pixelShift = []
    measurementTemperatures = []
    for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        
        yRaw = hdf5_file["Science/Y"][...]
        aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
        sensor1Temperature = hdf5_file["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
        
        #get temperature from TGO readout instead of channel
        utcStartTime = hdf5_file["Geometry/ObservationDateTime"][0, 0]
        measurementTemperature = getExternalTemperatureReadings(utcStartTime, 2)
        temperatureIndex = (np.abs(temperatureRange - measurementTemperature)).argmin()
#        utcEndTime = hdf5_file["Geometry/ObservationDateTime"][-1, 0]

        diffractionOrders = getDiffractionOrders(aotfFrequency)
        frameIndices = list(np.where(diffractionOrders == diffractionOrder)[0])

        if len(frameIndices) > 2:
            frameIndices = frameIndices[:2] #just take first 2 frames
        ySelected = yRaw[frameIndices, :, :]
        aotfFrequencySelected = aotfFrequency[frameIndices]

        #solar spectrum - take centre line only
        yBinned = ySelected[:, 11, :]

        pixels = np.arange(320)
        nu_pixels = nu_mp(diffractionOrder, pixels, measurementTemperature)

        if not PLOT_ONLY:
            #step1: find nearest pixels that contain the solar line
            solarLinePixelIndex1 = (np.abs(nu_pixels - solarLineWavenumber)).argmin()
            nPoints = 3
            solarLinePixelIndices1 = range(solarLinePixelIndex1-nPoints, solarLinePixelIndex1+nPoints+1)
        
            """process fullscan data for each frame containing selected diffraction order"""    
        
        pixelShiftPerFrame = []
        for yBinnedFrame in yBinned:
    
            #remove baseline
            yBaseline = baseline_als(yBinnedFrame) #find continuum of mean spectrum
            yCorrected = yBinnedFrame / yBaseline

            #step2: find pixel containing minimumn points
            solarLinePixelIndex2 = (np.abs(yCorrected[solarLinePixelIndices1] - np.min(yCorrected[solarLinePixelIndices1]))).argmin() + solarLinePixelIndices1[0]
            #step3: get pixel indices on either side
            nPoints = 7
#            nPoints = 3
            solarLinePixelIndices = range(solarLinePixelIndex2-nPoints, solarLinePixelIndex2+nPoints+1)

            
            ax1a.plot(nu_pixels, yCorrected, color=colours[temperatureIndex], label="%0.1fC" %measurementTemperature)

            if not PLOT_ONLY:
                ax1a.scatter(nu_pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices], color=colours[temperatureIndex])
                
#                #do polyfits to find minimum
#                POLYNOMIAL_DEGREE = 2
#                #first in wavenumber
#                polyCoefficientsWavenumber = np.polyfit(nu_pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices], POLYNOMIAL_DEGREE)
#                minimumWavenumber = -1 * polyCoefficientsWavenumber[1] / (2 * polyCoefficientsWavenumber[0])
#                #then in pixel space
#                polyCoefficientsPixel = np.polyfit(pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices], POLYNOMIAL_DEGREE)
#                minimumPixel = -1 * polyCoefficientsPixel[1] / (2 * polyCoefficientsPixel[0])
#                nu_pixels_solar_line_position_hr = np.arange(nu_pixels[solarLinePixelIndices[0]], nu_pixels[solarLinePixelIndices[-1]], 0.01)
#                solar_line_abs_nu_hr = np.polyval(polyCoefficientsWavenumber, nu_pixels_solar_line_position_hr)
                
                #do gaussian fit to find minimum
                #first in wavenumber
                nu_pixels_solar_line_position_hr, solar_line_abs_nu_hr, minimumWavenumber = fit_gaussian_absorption(nu_pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices])
                #then in pixel space
                pixels_solar_line_position_hr, solar_line_abs_pixel_hr, minimumPixel = fit_gaussian_absorption(pixels[solarLinePixelIndices], yCorrected[solarLinePixelIndices])
                
                #make hr grid for plotting quadratic
                ax1a.plot(nu_pixels_solar_line_position_hr, solar_line_abs_nu_hr, color=colours[temperatureIndex], linestyle="--")
                ax1a.axvline(x=minimumWavenumber, color=colours[temperatureIndex])
                ax2a.axvline(x=minimumPixel, color=colours[temperatureIndex])
                print(measurementTemperature, minimumPixel, minimumWavenumber)
                
                ax2a.plot(pixels, yCorrected, color=colours[temperatureIndex], label="%0.1fC" %measurementTemperature)
                
                pixelShiftPerFrame.append((solarLinePixelNoShift - minimumPixel))
        
        pixelShift.append(np.mean(pixelShiftPerFrame))
        measurementTemperatures.append(measurementTemperature)
        
    pixelShift = np.asfarray(pixelShift)
    measurementTemperatures = np.asfarray(measurementTemperatures)
    
    linearCoefficients = np.polyfit(measurementTemperatures, pixelShift, 1.0)
    
    print("temperature shift coefficients order %i =" %diffractionOrder, linearCoefficients)
    
    with open(os.path.join(BASE_DIRECTORY, "output.txt"), "a") as f:
        f.write("%i, %0.3f, %0.5f, %0.5f\n" %(diffractionOrder, minimumPixel, linearCoefficients[0], linearCoefficients[1]))
            
    ax1a.legend()
    ticks = ax1a.get_xticks()
    ax1a.set_xticks(np.arange(ticks[0], ticks[-1], 1.0))
    ax1b.set_xticks(np.arange(ticks[0], ticks[-1], 1.0))
    ax1a.grid()
    ax1b.grid()
    if PLOT_ONLY:
        plt.savefig(os.path.join(BASE_DIRECTORY, "lno_solar_fullscan_order_%i.png" %diffractionOrder))