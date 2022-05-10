# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:35:26 2019

@author: iant

SEARCH FOR METHANE

READ IN 1p0A 132-137 ORDER FILES FROM DB
CONVERT YUNMODIFIED TO TRANSMITTANCE USING LOW ALTITUDE TOA
PYTHON WATER RETRIEVAL
GET STD*2 FROM RESIDUAL
AVERAGE SPECTRA FROM DIFFERENT BINS/TIMES
PYTHON CH4 RETRIEVAL


"""

import os
#import h5py
import numpy as np
#import numpy.linalg as la
#from scipy import stats
#import scipy.optimize
import re

#import bisect
#from scipy.optimize import curve_fit,leastsq
#from mpl_toolkits.basemap import Basemap

#from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt


from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames
from get_hdf5_data_v01 import getLevel1Data, getLevel0p2Data
#from plot_simulations_v01 import getSimulationDataNew, getSOAbsorptionPixels
from retrievals.SO_retrieval_v02b import NOMAD_ret, Rodgers_OEM


regex = re.compile("(20190619|20190618)_1.*_SO_A_[IE]_134")
fileLevel = "hdf5_level_1p0a"
diffractionOrder = 134

regex = re.compile("20191215_050908.*_SO_A_[IE]_136")
fileLevel = "hdf5_level_1p0a"
diffractionOrder = 136



bin_index = 3
#pixels = np.arange(80, 130)
pixels = np.arange(0, 320)



def fft_zerofilling(row, filling_amount):
    """apply fft, zero fill by a multiplier then reverse fft to give very high resolution spectrum"""
    n_pixels = len(row)
    
    rowrfft = np.fft.rfft(row, len(row))
    rowzeros = np.zeros(n_pixels * int(filling_amount), dtype=np.complex)
    rowfft = np.concatenate((rowrfft, rowzeros))
    row_hr = np.fft.irfft(rowfft).real #get real component for reversed fft
    row_hr *= len(row_hr)/len(row) #need to scale by number of extra points

    pixels_hr = np.linspace(0, n_pixels, num=len(row_hr))    
    return pixels_hr, row_hr


def baseline_als(y, lam=250.0, p=0.95, niter=10):
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


def findTransmittanceMinimumPixel(px, spectrum, zerofilling=10):
    
    px_hr, abs_hr = fft_zerofilling(spectrum, zerofilling)
    px_hr += px[0]
    min_x = px_hr[np.where(abs_hr == min(abs_hr))[0][0]]
    return min_x, px_hr, abs_hr













hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)
#for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]

#obsDict = getLevel0p2Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=100.0) #use mean method, returns dictionary
obsDict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=50.0) #use mean method, returns dictionary

obsDict["molecule"] = "H2O"

#limit spectral range
obsDict["pixels"] = pixels
for dataset in ["x", "y", "y_mean", "y_error", "y_raw"]:
    obsDict[dataset] = obsDict[dataset][:, pixels]

#select spectra
tmin = 0.1
tmax = 0.8

#select based on number of pixels, as it can be variable now
xLength = len(obsDict["x"][0, :])
yCentre = np.mean(obsDict["y_mean"][:, (int(xLength/2)-int(xLength/4)):(int(xLength/2)+int(xLength/4))], axis=1)
nSpectra = len(yCentre)
vals = obsDict["alt"] > -999
vals = vals & (yCentre >= tmin)
vals = vals & (yCentre <= tmax)
zind = np.nonzero(vals)[0]

#plt.figure()
#plt.plot(obsDict["alt"], yCentre)
plt.figure()
plt.plot(obsDict["y_mean"].T)
#stop()
print("Max altitude at tmax=", np.max(obsDict["alt"][zind]))

if np.max(obsDict["alt"][zind]) < 80.0: #add higher altitude indices
    zEnd = np.min(np.where(obsDict["alt"] > 80.0)[0])
    zStart = max(zind)
    zStep = np.linspace(zStart, zEnd, num=30)
    zNewSteps = np.round(zStep)[1:].astype(np.int64)
    zind = np.concatenate((zind, zNewSteps))
    

    
for dataset in ["x", "y", "y_mean", "y_error", "y_raw"]:
    obsDict[dataset] = obsDict[dataset][zind, :]
for dataset in ["alt", "latitude", "longitude", "ls", "lst"]:
    obsDict[dataset] = obsDict[dataset][zind]
    
#correct spectral cal
#correct baseline first
    
spectrum_index = 1
y_mean_baseline = baseline_als(obsDict["y_mean"][spectrum_index, :])
y_mean_corrected = obsDict["y_mean"][spectrum_index, :] / y_mean_baseline

plt.figure()
plt.plot(y_mean_corrected)


plt.figure()
plt.plot(obsDict["x"][0, :], y_mean_corrected)

"""v1: find single minimum only"""
#minimum_pixel_input, pixels_hr_input, absorption_hr_input = findTransmittanceMinimumPixel(obsDict["pixels"], y_mean_corrected)
#plt.plot(obsDict["pixels"], obsDict["y_mean"][spectrum_index, :])
#plt.plot(pixels_hr_input, absorption_hr_input)
#print("minimum_pixel=", minimum_pixel_input)
#
#true_line_wn = 3064.408408400825#3064.4555788925004
#file_line_wn = np.interp(minimum_pixel_input, obsDict["pixels"], obsDict["x"][0,:])


"""v2: find all minima"""
#do high res minima finder (no fft needed)
std_occ_spectrum = np.std(y_mean_corrected)
occ_abs_points = np.where(y_mean_corrected < (1.0-std_occ_spectrum * 3.0))[0]
#ax1b.scatter(nu_hr[atmos_abs_points], normalised_atmos_spectrum[atmos_abs_points], c="b", s=10)

#remove matches in first 50 pixels
if len(pixels) == 320:
    occ_abs_points = occ_abs_points[occ_abs_points > 80]
    occ_abs_points = occ_abs_points[occ_abs_points <280]

#find pixel indices containing absorptions in hitran data
#split indices for different absorptions into different lists
previous_point = occ_abs_points[0]-1
occ_indices_all = []
occ_indices = []
for point in occ_abs_points:
    if point == previous_point + 1:
        occ_indices.append(point)
        if point == occ_abs_points[-1]:
            occ_indices_all.append(occ_indices)
    else:
        occ_indices_all.append(occ_indices)
        occ_indices = []
    previous_point = point
#add extra point to left and right of found indices
occ_indices_all_extra = []
for occ_indices in occ_indices_all:
    if len(occ_indices)>0:
        occ_indices_all_extra.append([occ_indices[0]-2] + [occ_indices[0]-1] + occ_indices + [occ_indices[-1]+1])


#plot quadratic and find wavenumber at minimum
occ_wavenumber_minima = []
for occ_extra_indices in occ_indices_all_extra:
    coeffs = np.polyfit(obsDict["x"][0, occ_extra_indices], y_mean_corrected[occ_extra_indices], 2)
    
    nu_selected = obsDict["x"][0, occ_extra_indices]
    nu_selected_hr = np.arange(np.min(nu_selected), np.max(nu_selected), 0.01)
    
    plt.plot(nu_selected_hr, np.polyval(coeffs, nu_selected_hr), "b")
    occ_spectrum_minimum = -1 * coeffs[1] / (2.0 * coeffs[0])
    plt.axvline(x=occ_spectrum_minimum, c="b")
    occ_wavenumber_minima.append(occ_spectrum_minimum)

plt.grid()
plt.title("%s" %hdf5_filename)


hr_spectra = np.loadtxt(os.path.join(BASE_DIRECTORY, "order_%i.txt" %diffractionOrder), delimiter=",")
true_wavenumber_minima = np.loadtxt(os.path.join(BASE_DIRECTORY, "order_%i_minima.txt" %diffractionOrder), delimiter=",")[:, 0]

for true_wavenumber_minimum in true_wavenumber_minima:
    plt.axvline(x=true_wavenumber_minimum, c="g")
    

#find mean wavenumber shift
wavenumber_shifts = []
for observation_wavenumber_minimum in occ_wavenumber_minima: #loop through found nadir absorption minima
    found = False
    for true_wavenumber_minimum in true_wavenumber_minima: #loop through found hitran absorption minima
        if true_wavenumber_minimum - 0.5 < observation_wavenumber_minimum < true_wavenumber_minimum + 0.5: #if absorption is within 1.0cm-1 then consider it found
            found = True
            wavenumber_shift = observation_wavenumber_minimum - true_wavenumber_minimum
            wavenumber_shifts.append(wavenumber_shift)
            print("Line found. Shift = ", wavenumber_shift)
    if not found:
        print("Warning: matching line not found")

delta_wn = np.mean(wavenumber_shifts) #get mean shift
observation_wavenumbers = obsDict["x"][0, :] - delta_wn #realign observation wavenumbers to match hitran

#plot shifted spectra
plt.figure()
plt.plot(observation_wavenumbers, y_mean_corrected)
plt.grid()
plt.title("%s" %hdf5_filename)
for true_wavenumber_minimum in true_wavenumber_minima:
    plt.axvline(x=true_wavenumber_minimum, c="g")




dnu = obsDict["x"][0, 1] - obsDict["x"][0, 0]
obsDict["x"] = obsDict["x"] - delta_wn
obsDict["first_pixel"] = obsDict["first_pixel"] - delta_wn / dnu

#obsDict["resolving_power"] = 12000.0
obsDict["resolving_power"] = 14000.0
obsDict["gaussian_scalar"] = 0.1
obsDict["gaussian_xoffset"] = -0.15
obsDict["gaussian_width"] = 0.3


#the retrieval bit stolen from Justin + modified for IDE and using dictionaries rather than h5 file and classes
retDict = NOMAD_ret(obsDict)
retDict = Rodgers_OEM(retDict, alpha=1.0)

"""v02 code using class
#plot results
fig, ax = plt.subplots()
for iz in range(ret.NbZ):
    l = ax.plot(ret.nu_p, ret.Y[iz,:])
    c = l[0].get_color()
    ax.plot(ret.XObs[iz,:], ret.YObs[iz,:], color=c, linestyle='--')
#    fig.savefig(obsDict["label"]+".png")

ax.axvline(x=true_line_wn)
cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(ret.NbZ)/ret.NbZ]

 

fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
for iz in range(ret.NbZ):
    ax1a.plot(ret.nu_p, ret.Trans_p[iz,:], color=colours[iz], label="%0.1fkm" %obsDict["alt"][iz])
#        c = l[0].get_color()
    ax1a.plot(ret.nu_p, ret.YObs[iz,:]/ret.Trans_background[iz,:], color=colours[iz], linestyle='--')
    ax1b.plot(ret.nu_p, ret.Trans_p[iz,:]-(ret.YObs[iz,:]/ret.Trans_background[iz,:]), color=colours[iz])
ax1a.axvline(x=true_line_wn)
ax1b.axvline(x=true_line_wn)
ax1a.set_ylim(0.85, 1.05)    
ax1a.legend()
#    fig.savefig(obsDict["label"]+"_residual.png")

#find real wavenumber min of convolved abs line
spectrum = ret.YObs[0,:]/ret.Trans_background[0,:]
minimum_pixel, pixels_hr, absorption_hr = findTransmittanceMinimumPixel(obsDict["pixels"], spectrum)
x_hr = np.interp(pixels_hr, obsDict["pixels"], ret.nu_p)
print(np.interp(minimum_pixel, pixels_hr, x_hr))
##    plt.figure()
#    ax1a.plot(x_hr, absorption_hr)
"""


#plt.figure()
#plt.plot(retDict["nu_hr"], retDict["Trans_hr"][10, :])

"""v02b code using dict"""
#plot results
cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(retDict["NbZ"])/retDict["NbZ"]]
#
#    fig, ax = plt.subplots()
#    for iz in range(retDict["NbZ"]):
#        ax.plot(retDict["nu_p"], retDict["Y"][iz,:], color=colours[iz], linestyle='--')
#        ax.plot(retDict["XObs"][iz,:], retDict["YObs"][iz,:], color=colours[iz])
#    #    fig.savefig(obsDict["label"]+".png")
#    
#    ax.axvline(x=true_line_wn)



fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X, FIG_Y), sharex=True)
for iz in range(retDict["NbZ"]):
    ax1a.plot(retDict["nu_p"], retDict["YObs"][iz,:]/retDict["Trans_background"][iz,:], color=colours[iz], label="Obs %0.1fkm" %obsDict["alt"][iz])
    ax1a.plot(retDict["nu_p"], retDict["Trans_p"][iz,:], color=colours[iz], linestyle='--')
    ax1b.plot(retDict["nu_p"], retDict["Trans_p"][iz,:]-(retDict["YObs"][iz,:]/retDict["Trans_background"][iz,:]), color=colours[iz])
#ax1a.axvline(x=true_line_wn)
#ax1b.axvline(x=true_line_wn)
ax1a.set_ylim(0.85, 1.05)    
ax1a.legend()
for true_wavenumber_minimum in true_wavenumber_minima:
    ax1a.axvline(x=true_wavenumber_minimum, c="g")
    ax1b.axvline(x=true_wavenumber_minimum, c="g")

#    fig.savefig(obsDict["label"]+"_residual.png")

#find real wavenumber min of convolved abs line
spectrum = retDict["YObs"][0,:]/retDict["Trans_background"][0,:] #normalised observation
spectrum = retDict["Trans_p"][0,:] #retrieved observation
minimum_pixel, pixels_hr, absorption_hr = findTransmittanceMinimumPixel(obsDict["pixels"], spectrum)
x_hr = np.interp(pixels_hr, obsDict["pixels"], retDict["nu_p"])
print(np.interp(minimum_pixel, pixels_hr, x_hr))


##    plt.figure()
#    ax1a.plot(x_hr, absorption_hr)


#    plt.figure()
#    #for px in range(retDict["NbP"]):
#    #    plt.plot(retDict["W_conv"][px,:])
#    #    plt.plot(retDict["W_conv_old"][px,:])
#    plt.plot(retDict["W_conv"][48,:], label="W_conv")
#    plt.plot(retDict["W_conv_old"][48,:], label="W_conv_old")
#    plt.legend()


#"""get occultation data"""
##test new calibrations on a mean nadir observation
#from database_functions_v01 import obsDB, makeObsDict
#dbName = "so_1p0a"
#db_obj = obsDB(dbName)
#if diffractionOrder in [136]:
#    searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE altitude < 80 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
#    SIGNAL_CUTOFF = 2000
#elif diffractionOrder in [189]:
##    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND n_orders < 4 AND incidence_angle < 20 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
#    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < 25 AND latitude > -25 AND longitude < 90 AND longitude > 0 AND n_orders < 4 AND incidence_angle < 10 AND temperature > -5 AND temperature < -2 AND diffraction_order == %i" %diffractionOrder)
#    SIGNAL_CUTOFF = 3000
#elif diffractionOrder in [188]:
#    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE n_orders < 4 AND diffraction_order == %i" %diffractionOrder)
#obsDict = makeObsDict(searchQueryOutput)
#db_obj.close()
#plt.figure()
#plt.scatter(obsDict["longitude"], obsDict["latitude"])
