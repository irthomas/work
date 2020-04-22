# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:40:24 2019

@author: iant

APPLY FULLSCAN TABLE TO DATASETS
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.spectra.smooth_hr import smooth_hr
from tools.spectra.baseline_als import baseline_als

from analysis.retrievals.pytran import pytran
#from analysis.retrievals.NOMADTOOLS import nomadtools
#from analysis.retrievals.NOMADTOOLS.nomadtools import gem_tools
from analysis.retrievals.NOMADTOOLS.nomadtools.paths import NOMADParams
from analysis.retrievals.NOMAD_instrument import freq_mp#, F_blaze, F_aotf_3sinc


PFM_AUXILIARY_FILES = os.path.normcase(r"X:\projects\NOMAD\data\pfm_auxiliary_files")
RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration")
LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v01"
NOMAD_TMP_DIR = paths["BASE_DIRECTORY"]

SMOOTHING_LEVEL = 600 #must be even number




#diffraction order: [nadir mean signal cutoff, minimum signal for absorption, n stds for absorption, n stds for hitran absorption]
nadirDict = {
        189:[8.0, 5.0, 1.0, 1.0, "CO"]}


hdf5_filename = "20180611_131514_0p3a_LNO_1_D_189"
#hdf5_filename = "20180728_115528_0p3a_LNO_1_D_189"
#hdf5_filename = "20180729_192122_0p3a_LNO_1_D_189"

#hdf5_filename = "20180428_232453_0p3a_LNO_1_D_189"


hdf5_file_level = "hdf5_level_" + hdf5_filename[16:20]
year_in = hdf5_filename[0:4]
month_in = hdf5_filename[4:6]
day_in = hdf5_filename[6:8]

hdf5file_path = os.path.join(paths["DATA_DIRECTORY"], hdf5_file_level, year_in, month_in, day_in, hdf5_filename+".h5")

hdf5FileIn = h5py.File(hdf5file_path, "r")
hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))


#get Y data
yIn = hdf5FileIn["Science/Y"][...]
#get X data
xIn = hdf5FileIn["Science/X"][0, :]

error = False

#get observation start time and diffraction order/ AOTF
observationStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0]
diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...]
aotfFrequencies = hdf5FileIn["Channel/AOTFFrequency"][...]

integrationTimes = hdf5FileIn["Channel/IntegrationTime"][...]
bins = hdf5FileIn["Science/Bins"][...]
nAccumulations = hdf5FileIn["Channel/NumberOfAccumulations"][...]

integrationTime = np.float(integrationTimes[0]) / 1.0e3 #milliseconds to seconds
nAccumulation = np.float(nAccumulations[0])/2.0 #assume LNO nadir background subtraction is on
binning = np.float(bins[0,1] - bins[0,0]) + 1.0 #binning starts at zero
nBins = 1.0 #Science/Bins reflects the real binning

measurementSeconds = integrationTime * nAccumulation
measurementPixels = binning * nBins

print("integrationTime = %0.3f, light nAccumulation = %i, binning = %i, measurementSeconds = %0.1f" %(integrationTime, nAccumulation, binning, measurementSeconds))


#check that all aotf freqs are the same (they should be for this function)
if (aotfFrequencies == aotfFrequencies[0]).all():
    diffractionOrder = diffractionOrders[0]
else:
    print("Error: AOTF frequencies are not the same. Use another function for fullscan or calibrations")
    


yBinnedNorm = yIn / measurementSeconds / measurementPixels #scale to counts per second per pixel
yBinnedNorm[np.isnan(yBinnedNorm)] = 0.0 #replace nans

#calculate standard deviation - remove continuum shape to leave random error on first 50 pixels only
yFitted = np.polyval(np.polyfit(range(50), yBinnedNorm[0, 0:50],2), range(50))
yStd = np.std(yBinnedNorm[0, 0:50] - yFitted)




sensor1Temperature = hdf5FileIn["Housekeeping/SENSOR_1_TEMPERATURE_LNO"][...]
observationTemperature = np.mean(sensor1Temperature[2:10])

sun_mars_distance = hdf5FileIn["Geometry/DistToSun"][0,0] #get sun-mars distance in AU


if diffractionOrder in nadirDict.keys():
    nadir_mean_signal_cutoff, minimum_signal_for_absorption, n_stds_for_absorption, n_stds_for_hitran_absorption, molecule = nadirDict[diffractionOrder]
else:
    error = True


#first find spectra where signal is sufficiently high    
#validIndices = np.where(np.nanmean(yBinnedNorm, axis=1) > nadir_mean_signal_cutoff)[0] 
validIndices = np.where(np.nanmean(yBinnedNorm, axis=1) > nadir_mean_signal_cutoff)[0]
if not validIndices.size:
    minimum_signal_for_absorption *= 0.9
    nadir_mean_signal_cutoff *= 0.9
    print("%s: No valid indices found. Attempt 2: Reducing signal cutoff to %0.1f and absorption signal to %0.1f" %(hdf5_filename, nadir_mean_signal_cutoff, minimum_signal_for_absorption))
    validIndices = np.where(np.nanmean(yBinnedNorm, axis=1) > nadir_mean_signal_cutoff)[0]
    if not validIndices.size:
        minimum_signal_for_absorption *= 0.9
        nadir_mean_signal_cutoff *= 0.9
        print("No valid indices found. Attempt 3: Reducing signal cutoff to %0.1f and absorption signal to %0.1f" %(nadir_mean_signal_cutoff, minimum_signal_for_absorption))
        validIndices = np.where(np.nanmean(yBinnedNorm, axis=1) > nadir_mean_signal_cutoff)[0]


#plot raw values
fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
for validIndex in validIndices:
    ax1a.plot(xIn, yBinnedNorm[validIndex, :], alpha=0.3)

#plot mean spectrum
mean_spectrum = np.nanmean(yBinnedNorm[validIndices, :], axis=0)
ax1a.plot(xIn, mean_spectrum, "k")

#plot baseline corrected spectra
mean_spectrum_baseline = baseline_als(mean_spectrum) #find continuum of mean spectrum
ax1a.plot(xIn, mean_spectrum_baseline, "k--")

mean_corrected_spectrum = mean_spectrum / mean_spectrum_baseline
ax1b.plot(xIn, mean_corrected_spectrum, "r")

#do quadratic  fit to find true absorption minima
std_corrected_spectrum = np.std(mean_corrected_spectrum)
abs_points = np.where((mean_corrected_spectrum < (1.0 - std_corrected_spectrum * n_stds_for_absorption)) & (mean_spectrum > minimum_signal_for_absorption))[0]
ax1b.scatter(xIn[abs_points], mean_corrected_spectrum[abs_points], c="r", s=10)

#find pixel indices containing absorptions in nadir data
#split indices for different absorptions into different lists
previous_point = abs_points[0]-1
indices_all = []
indices = []
for point in abs_points:
    if point == previous_point + 1:
        indices.append(point)
        if point == abs_points[-1] and len(indices)>0:
            indices_all.append(indices)
    else:
        indices_all.append(indices)
        indices = []
    previous_point = point
indices_all_extra = []
#add extra point to left and right of found indices
for indices in indices_all:
    if len(indices)>0:
        indices_all_extra.append([indices[0]-2] + [indices[0]-1] + indices + [indices[-1]+1])

#plot quadratic and find wavenumber at minimum
observation_wavenumber_minima = []
for extra_indices in indices_all_extra:
    coeffs = np.polyfit(xIn[extra_indices], mean_corrected_spectrum[extra_indices], 2)
    ax1b.plot(xIn[extra_indices], np.polyval(coeffs, xIn[extra_indices]), "g")
    spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
    ax1b.axvline(x=spectrum_minimum, c="g")
    observation_wavenumber_minima.append(spectrum_minimum)








"""get high res solar spectra and hitran spectra of chosen molecule"""
#define spectral range
nu_hr_min = freq_mp(diffractionOrder, 0.) - 1.
nu_hr_max = freq_mp(diffractionOrder, 320.) + 1.
dnu = 0.001
Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
dnu = nu_hr[1]-nu_hr[0]


#get solar spectrum (only to check if absorption exists)
solspecFile = os.path.join(paths["BASE_DIRECTORY"], "reference_files", "nomad_solar_spectrum_solspec.txt")
with open(solspecFile, "r") as f:
    nu_solar = []
    I0_solar = []
    for line in f:
        nu, I0 = [float(val) for val in line.split()]
        if nu < nu_hr_min - 1.:
            continue
        if nu > nu_hr_max + 1.:
            break
        nu_solar.append(nu)
        I0_solar.append(I0)
f_solar = interpolate.interp1d(nu_solar, I0_solar)
I0_solar_hr = f_solar(nu_hr)


#convolve high res solar spectrum to lower resolution. Scale to avoid swamping figure
solar_spectrum = smooth_hr(I0_solar_hr, window_len=(SMOOTHING_LEVEL-1))
normalised_solar_spectrum = ((solar_spectrum-np.min(solar_spectrum)) / (np.max(solar_spectrum)-np.min(solar_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)] / 5.0 + 0.8
ax1b.plot(nu_hr, normalised_solar_spectrum, "k--")



#get spectrum
M = pytran.get_molecule_id(molecule)
filename = os.path.join(NOMADParams['HITRAN_DIR'], '%02i_hit16_2000-5000_CO2broadened.par' % M)
if not os.path.exists(filename):
    filename = os.path.join(NOMADParams['HITRAN_DIR'], '%02i_hit16_2000-5000.par' % M)
LineList = pytran.read_hitran2012_parfile(filename, nu_hr_min, nu_hr_max, Smin=1.e-26)
nlines = len(LineList['S'])
print('Found %i lines' % nlines)
sigma_hr =  pytran.calculate_hitran_xsec(LineList, M, nu_hr, T=200.0, P=1.0*1e3)


#convolve high res spectrum to lower resolution
atmos_spectrum = (1.0 - 1.0e20 * smooth_hr(sigma_hr, window_len=(SMOOTHING_LEVEL-1)))
normalised_atmos_spectrum = ((atmos_spectrum-np.min(atmos_spectrum)) / (np.max(atmos_spectrum)-np.min(atmos_spectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]
ax1b.plot(nu_hr, normalised_atmos_spectrum, "b--")


#do high res minima finder (no fft needed)
std_atmos_spectrum = np.std(normalised_atmos_spectrum)
atmos_abs_points = np.where(normalised_atmos_spectrum < (1.0-std_atmos_spectrum * n_stds_for_hitran_absorption))[0]
#ax1b.scatter(nu_hr[atmos_abs_points], normalised_atmos_spectrum[atmos_abs_points], c="b", s=10)


#find pixel indices containing absorptions in hitran data
#split indices for different absorptions into different lists
previous_point = atmos_abs_points[0]-1
atmos_indices_all = []
atmos_indices = []
for point in atmos_abs_points:
    if point == previous_point + 1:
        atmos_indices.append(point)
        if point == atmos_abs_points[-1]:
            atmos_indices_all.append(atmos_indices)
    else:
        atmos_indices_all.append(atmos_indices)
        atmos_indices = []
    previous_point = point
#no need to add extra points to left and right of found indices
atmos_indices_all_extra = atmos_indices_all


#plot quadratic and find wavenumber at minimum
true_wavenumber_minima = []
for atmos_extra_indices in atmos_indices_all_extra:
    coeffs = np.polyfit(nu_hr[atmos_extra_indices], normalised_atmos_spectrum[atmos_extra_indices], 2)
    ax1b.plot(nu_hr[atmos_extra_indices], np.polyval(coeffs, nu_hr[atmos_extra_indices]), "b")
    atmos_spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
    ax1b.axvline(x=atmos_spectrum_minimum, c="b")
    true_wavenumber_minima.append(atmos_spectrum_minimum)

plt.grid()
plt.title("%s" %hdf5_filename)


#find mean wavenumber shift
wavenumber_shifts = []
for observation_wavenumber_minimum in observation_wavenumber_minima: #loop through found nadir absorption minima
    found = False
    for true_wavenumber_minimum in true_wavenumber_minima: #loop through found hitran absorption minima
        if true_wavenumber_minimum - 1.0 < observation_wavenumber_minimum < true_wavenumber_minimum + 1.0: #if absorption is within 1.0cm-1 then consider it found
            found = True
            wavenumber_shift = observation_wavenumber_minimum - true_wavenumber_minimum
            wavenumber_shifts.append(wavenumber_shift)
            print("Line found. Shift = ", wavenumber_shift)
    if not found:
        print("Warning: matching line not found")

mean_shift = np.mean(wavenumber_shifts) #get mean shift
observation_wavenumbers = xIn - mean_shift #realign observation wavenumbers to match hitran

#plot corrected nadir spectra with solar / hitran spectra
plt.figure(figsize=(FIG_X, FIG_Y))
plt.plot(observation_wavenumbers, mean_corrected_spectrum, "k")
plt.plot(nu_hr, normalised_atmos_spectrum, "b--")
plt.plot(nu_hr, normalised_solar_spectrum, "g--")
plt.grid()
plt.title("Spectral correction %s" %hdf5_filename)




radiometric_calibration_table = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME)
with h5py.File("%s.h5" % radiometric_calibration_table, "r") as radianceFactorFile:
    
    if "%i" %diffractionOrder in radianceFactorFile.keys():
    
        #read in coefficients and wavenumber grid
        wavenumber_grid_in = radianceFactorFile["%i" %diffractionOrder+"/wavenumber_grid"][...]
        coefficient_grid_in = radianceFactorFile["%i" %diffractionOrder+"/coefficients"][...].T

        #TODO: add conversion factor to account for solar incidence angle
        #TODO: this needs checking. No nadir or so FOV in calculation!
        rSun = 695510.0 #km
        dSun = sun_mars_distance * 1.496e+8 #1AU to km
        angleSolar = np.pi * (rSun / dSun) **2



#find solar fullscan obs coefficients at wavenumbers matching real observation
corrected_solar_spectrum = []
for observation_wavenumber in observation_wavenumbers:
    index = np.abs(observation_wavenumber - wavenumber_grid_in).argmin()
    
    coefficients = coefficient_grid_in[index, :]
    correct_solar_counts = np.polyval(coefficients, observationTemperature)
    corrected_solar_spectrum.append(correct_solar_counts)
corrected_solar_spectrum = np.asfarray(corrected_solar_spectrum)



#do I/F using shifted observation wavenumber scale
Y = yBinnedNorm
nSpectra = Y.shape[0] #calculate size
YRadFac = Y / np.tile(corrected_solar_spectrum, [nSpectra, 1]) / angleSolar #convert counts to radiance factor

#plot radiance factor, mean radiance factor and hitran spectra
plt.figure(figsize=(FIG_X, FIG_Y))
for validIndex in validIndices:
    plt.plot(observation_wavenumbers, YRadFac[validIndex, :], alpha=0.3)
mean_radfac = np.mean(YRadFac[validIndices, :], axis=0)

plt.plot(observation_wavenumbers, mean_radfac, "k")
plt.plot(nu_hr, normalised_atmos_spectrum * np.mean(mean_radfac), "b--")
#plt.plot(convolved_solar_wavenumbers, normalised_solar_spectrum * np.mean(mean_radfac), "g--")
plt.xlabel("Wavenumbers cm-1")
plt.ylabel("Radiance factor")
plt.title("Radiance factor after spectral correction %s" %hdf5_filename)
plt.grid()
plt.savefig("Radiance_factor_%s.png" %hdf5_filename)



                            




lines = []
for nu, normalised_atmos, normalised_solar in zip(nu_hr, normalised_atmos_spectrum, normalised_solar_spectrum):
    line = "%0.3f, %0.5f, %0.5f\n" %(nu, normalised_atmos, normalised_solar)
    lines.append(line)

with open(os.path.join(paths["BASE_DIRECTORY"], "order_%i.txt" %diffractionOrder), "w") as f:
    f.writelines(lines)



hr_spectra = np.loadtxt(os.path.join(paths["BASE_DIRECTORY"], "order_%i.txt" %diffractionOrder), delimiter=",")




