# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:16:27 2020

@author: iant

test with nadir data
"""





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
