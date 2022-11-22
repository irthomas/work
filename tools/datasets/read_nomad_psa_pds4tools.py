# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:16:18 2022

@author: iant

"""

import os
import pds4_tools
import matplotlib.pyplot as plt
import numpy as np


import glob


NOMAD_DATA_PATH_ROOT = r"C:\Users\iant\Documents\DATA\psa"

NOMAD_DATA_PATH_CALIBRATED = os.path.join(NOMAD_DATA_PATH_ROOT, "data_calibrated", "Science_Phase")


channel = "so"
observation_type = "occultation"

# channel = "uvis"
# observation_type = "nadir"

xml_paths = sorted(glob.glob(NOMAD_DATA_PATH_CALIBRATED + os.sep + "**" + os.sep + "*%s*.xml" %channel.replace("_","*"), recursive=True))






xml_path = xml_paths[0] #read first file

if channel == "so" and observation_type == "occultation":

xml_filename = os.path.basename(xml_path)
orbit_dir = os.path.basename(os.path.dirname(xml_path))
orbit_range_dir = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))


#read PDS4 product (.tab and .xml) and extract the data
structures = pds4_tools.read(xml_path) #path to the .xml file
data = structures["CAL_NOMAD_SO"].data
    
#get a list of available data fields
available_fields = list(data.meta_data.keys())

#extract altitudes, wavenumbers and transmittances from product
tangent_altitudes = data["TangentAltAreoidStart0"]
x = data["Wavenumber, Pixel wavenumber"]
transmittances = data["Transmittance, Pixel transmittance"]
   

#plot x (spectral calibration of each pixel) vs transmittance for all the spectra in the file
#add a legend with the tangent altitude of each spectrum
plt.figure()
plt.title("NOMAD data: %s" %os.path.basename(xml_path))
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Transmittance")

for tangent_altitude, pixel_x, transmittance in zip(tangent_altitudes, x, transmittances):
    plt.plot(pixel_x, transmittance, label="%0.1fkm" %tangent_altitude)

plt.grid()
plt.legend()


if channel == "uvis" and observation_type == "nadir":

xml_filename = os.path.basename(xml_path)
orbit_dir = os.path.basename(os.path.dirname(xml_path))
orbit_range_dir = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))


#read PDS4 product (.tab and .xml) and extract the data
structures = pds4_tools.read(xml_path) #path to the .xml file
data = structures["CAL_NOMAD_UVIS"].data
    
#get a list of available data fields
available_fields = list(data.meta_data.keys())

#extract altitudes, wavenumbers and transmittances from product
latitudes = data["LatStart0"]
x = data["Wavelength, Pixel wavelength"]
radiances = data["Radiance, Pixel radiance"]

radiances[radiances == -999] = np.nan
   

#plot x (spectral calibration of each pixel) vs transmittance for all the spectra in the file
#add a legend with the tangent altitude of each spectrum
plt.figure()
plt.title("NOMAD data: %s" %os.path.basename(xml_path))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance")


for latitude, pixel_x, radiance in zip(latitudes, x, radiances):
    plt.plot(pixel_x, radiance, label="%0.1f latitude" %latitude)

plt.grid()
plt.legend()
