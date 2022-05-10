# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:52:38 2019

@author: iant
"""


#import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import os
#from datetime import datetime
#from scipy.signal import savgol_filter

from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList
#from plot_solar_line_simulations_lno import getPixelSolarSpectrum, getConvolvedSolarSpectrum, nu_mp, t_p0

from database_functions_v01 import obsDB, makeObsDict, findHdf5File

from nomad_obs.obs_inputs import occultationRegionsOfInterest
from get_hdf5_data_v01 import getLevel1Data



dbName = "so_1p0a"
db_obj = obsDB(dbName)


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



#"""get occultation coverage maps from observations of interesting regions"""
#diffractionOrder = 134
#LATITUDE_RANGE = 5 #degrees
#LONGITUDE_RANGE = 5 #degress
#
#
#for regionName, _, cycleName, latStart, latEnd, lonStart, lonEnd in occultationRegionsOfInterest:
#    latMean = np.mean((latStart, latEnd))
#    lonMean = np.mean((lonStart, lonEnd))
#
#    #searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND diffraction_order == %i" %diffractionOrder)
#    searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE latitude < %i AND latitude > %i AND longitude < %i AND longitude > %i AND diffraction_order == %i" %(latMean+LATITUDE_RANGE, latMean-LATITUDE_RANGE, lonMean+LONGITUDE_RANGE, lonMean-LONGITUDE_RANGE, diffractionOrder))
#    
#    longitudes = [row[10] for row in searchQueryOutput]
#    latitudes = [row[11] for row in searchQueryOutput]
#    altitudes = [row[12] for row in searchQueryOutput]
#    
#    #obsDict = makeObsDict("so", searchQueryOutput)
#    plt.figure()
#    plt.title("%s: Order %i" %(regionName, diffractionOrder))
#    sc = plt.scatter(longitudes, latitudes, c=altitudes, s=1)
#    plt.colorbar(sc)
#    
#db_obj.close()


"""just get MSL order 136"""
#CURIOSITY = -4.5895, 137.4417
LATITUDE_RANGE = 15 #degrees
LONGITUDE_RANGE = 15 #degress
regionName = "Curiosity"
latMean = -4.5895
lonMean = 137.4417

bin_index = 2

for diffractionOrder in [134, 136]:
    searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE latitude < %i AND latitude > %i AND longitude < %i AND longitude > %i AND diffraction_order == %i" %(latMean+LATITUDE_RANGE, latMean-LATITUDE_RANGE, lonMean+LONGITUDE_RANGE, lonMean-LONGITUDE_RANGE, diffractionOrder))
    
#    obsDict = makeObsDict("so", searchQueryOutput) #use to download files to local
    
    filenames = [x[3] for x in searchQueryOutput]
    uniqueFilenames = sorted(list(set(filenames)))
    print("Files order %i:" %diffractionOrder)
    for i in uniqueFilenames:
        print(i)
        
    cmap = plt.get_cmap('Spectral')
    colours = [cmap(i) for i in np.arange(len(uniqueFilenames))/len(uniqueFilenames)]
    
    
    longitudes = [row[10] for row in searchQueryOutput]
    latitudes = [row[11] for row in searchQueryOutput]
    altitudes = [row[12] for row in searchQueryOutput]
    
    #obsDict = makeObsDict("so", searchQueryOutput)
    plt.figure()
    plt.title("%s: Order %i" %(regionName, diffractionOrder))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    sc = plt.scatter(longitudes, latitudes, c=altitudes, s=1)
    cbar = plt.colorbar(sc)
    cbar.set_label("Tangent altitude above areoid (km)")
    
    
    plt.figure(figsize=(FIG_X, FIG_Y))
    plt.title("Transmittance vs altitude over Gale Crater")
    plt.xlabel("Transmittance")
    plt.ylabel("Altitude (km)")
    plt.ylim([0, 70])

    for fileIndex, uniqueFilename in enumerate(uniqueFilenames):
        with h5py.File(findHdf5File(uniqueFilename), "r") as hdf5_file:
            obsDict = getLevel1Data(hdf5_file, uniqueFilename, bin_index, silent=True, top_of_atmosphere=100.0)
            
            y_mean_baseline = np.asfarray([baseline_als(spectrum) for spectrum in obsDict["y_mean"]])
            
            y_centre = y_mean_baseline[:, 160]
            plt.plot(y_centre, obsDict["alt"], color=colours[fileIndex], label=obsDict["label_full"])

    plt.legend()
    
db_obj.close()


