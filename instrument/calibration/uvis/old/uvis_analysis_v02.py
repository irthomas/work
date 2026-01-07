# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 17:54:54 2017

@author: ithom
"""


import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime
import re

#import bisect
#from mpl_toolkits.basemap import Basemap
#from scipy import interpolate

from scipy.signal import savgol_filter
from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v04 import BASE_DIRECTORY, FIG_X, FIG_Y, makeFileList#, printFileNames


#rcParams["axes.formatter.useoffset"] = False


SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


regex = re.compile("20180608_004700.*_UVIS_.*") # good ozone
fileLevel = "hdf5_level_1p0a"


hdf5Files, hdf5Filenames, titles = makeFileList(regex, fileLevel)


for fileIndex, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

    detector_data = hdf5_file["Science/Y"][...]
    x = hdf5_file["Science/X"][0, :]
    lat = hdf5_file["Geometry/Point0/Lat"][:,0]
    lon = hdf5_file["Geometry/Point0/Lon"][:,0]
    
    for spectrumIndex, spectrum in enumerate(detector_data):
        plt.plot(x, savgol_filter(spectrum/ detector_data[49, :], 39, 2), label="%0.1fN, %0.1fE" %(lat[spectrumIndex], lon[spectrumIndex]))
        
    plt.legend(loc="upper right")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative radiance convoluted grid")
    plt.title(hdf5_filename)

#    plt.plot()