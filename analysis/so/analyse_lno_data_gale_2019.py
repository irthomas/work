# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:31:13 2019

@author: iant

LNO CH4 ANALYSIS GALE 2019
"""



import numpy as np
#import os
#from datetime import datetime
import matplotlib.pyplot as plt

from hdf5_functions_v04 import BASE_DIRECTORY, DATA_DIRECTORY, makeFileList #FIG_X, FIG_Y
from database_functions_v01 import obsDB, makeObsDict
from plot_simulations_v01 import plotSimulation, getSimulation


"""to add data to sql"""
#import re
#fileLevel = "hdf5_level_1p0a"
##fileLevel = "hdf5_level_0p3a"
#obspaths = re.compile("201905.*LNO.*_D_.*|201906.*LNO.*_D_.*(134¦136")
#db_obj = obsDB()
#db_obj.processLNOData(fileLevel, obspaths, overwrite=True)
#db_obj.close()
#
#stop()


"""get dictionary from query and plot"""
db_obj = obsDB()
#CURIOSITY = -4.5895, 137.4417
queryInput = "SELECT * FROM lno_nadir where latitude > -15 AND latitude < 5 AND longitude > 127 AND longitude < 147 AND incidence_angle < 40 and diffraction_order == 136 and n_orders < 3"
queryOutput = db_obj.query(queryInput)
obsDict = makeObsDict(queryOutput)
db_obj.close()

plt.figure()
plt.title(queryInput)
plt.scatter(obsDict["longitude"], obsDict["latitude"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.scatter(137.4, -4.59, marker="x", color="r")
uniqueFilenames = []
nFrames = len(obsDict["longitude"])
index = 0
for frameIndex, filename in enumerate(obsDict["filename"]):
    if filename not in uniqueFilenames:
        plt.text(obsDict["longitude"][frameIndex], obsDict["latitude"][frameIndex] + index, filename)
        uniqueFilenames.append(filename)
        index -= 1

fig1, ax1 = plt.subplots()
ax1.set_title(queryInput)

#for frameIndex, (x, y) in enumerate(zip(obsDict["x"], obsDict["y"])):
#    ax1.plot(x, y, alpha=0.3)

yMean = np.mean(np.asfarray(obsDict["y"])[:, :], axis=0)
xMean = np.mean(np.asfarray(obsDict["x"])[:, :], axis=0)
ax1.plot(xMean, yMean, "k")

cmap = plt.get_cmap('jet')
n_files = len(uniqueFilenames)
#colour denotes file index
colours = [cmap(i) for i in np.arange(n_files)/n_files]

for filenameIndex, uniqueFilename in enumerate(uniqueFilenames):
    indices = [index for index,filename in enumerate(obsDict["filename"]) if uniqueFilename == filename]
    if len(indices) > 5:
        yMean = np.mean(np.asfarray(obsDict["y"])[indices, :], axis=0) #average spectra of that observation
        xMean = np.mean(np.asfarray(obsDict["x"])[indices, :], axis=0)
        ax1.plot(xMean, yMean, color=colours[filenameIndex], label=uniqueFilename)
ax1.set_xlabel("Wavenumber")
#ax1.set_ylabel("Counts")
ax1.set_ylabel("Radiance")
ax1.legend()

channel = "lno"
molecule = "H2O 136"
wavenumbers = xMean #order 134
#plotSimulation(ax1, wavenumbers, channel, molecules, normalisation=1000.0)
radianceSimulated = getSimulation(wavenumbers, channel, molecule, new=True)
ax1.plot(wavenumbers, radianceSimulated * 500, "k")

#channel = "so"
#molecules = ["CH4 10ppm", "H2O"]
#wavenumbers = xMean #order 134
#plotSimulation(ax1, wavenumbers, channel, molecules, normalisation=700)
#plotSimulation(ax1, wavenumbers, channel, molecules, normalisation=1.0e-5)




