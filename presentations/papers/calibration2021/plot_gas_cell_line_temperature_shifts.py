# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:30:30 2021

@author: iant

LNO CALIBRATION PAPER FIG 10 (GAS CELL SPECTRA, COLOUR CODED BY TEMPERATURE)
"""

import numpy as np

import matplotlib.pyplot as plt

from tools.spectra.baseline_als import baseline_als
from tools.file.hdf5_functions import make_filelist


SAVE_FIGS = False
#SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True


model = "PFM"
title = ""
fileLevel = "hdf5_level_0p1a"
#obspaths = ["*2015042*LNO"]
obspaths = [
    "20150329_183323_0p1a_LNO_1", #CH4 -15.1C
    "20150320_091400_0p1a_LNO_1", #CH4 -14.3C
    "20150320_204147_0p1a_LNO_1", #CH4 -13.9C
    "20150330_084826_0p1a_LNO_1", #CH4 -10.9C
    "20150330_022349_0p1a_LNO_1", #CH4 -8.8C

    "20150315_085754_0p1a_LNO_1", #CH4 1.4C
    "20150316_052446_0p1a_LNO_1", #CH4 2.9C
    "20150315_152945_0p1a_LNO_1", #CH4 3.7C

    "20150318_050153_0p1a_LNO_1", #CH4 12.4C
    "20150402_221740_0p1a_LNO_1", #CH4 13.7C 
]



def findAbsorptionMinimum(spectrum, continuum_range, plot=False):
    
    continuum_centre = int((continuum_range[3] - continuum_range[0])/2)
    ABSORPTION_WIDTH_INDICES = list(range(continuum_centre - 3, continuum_centre + 3, 1))
        
    pixels = np.arange(320)
    
    continuum_pixels = pixels[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]    
    continuum_spectra = spectrum[list(range(continuum_range[0], continuum_range[1])) + list(range(continuum_range[2], continuum_range[3]))]
    
    #fit polynomial to continuum on either side of absorption band
    coefficients = np.polyfit(continuum_pixels, continuum_spectra, 2)
    continuum = np.polyval(coefficients, pixels[range(continuum_range[0], continuum_range[3])])
    #divide by continuum to get absorption
    absorption = spectrum[range(continuum_range[0], continuum_range[3])] / continuum
    
    #fit polynomial to centre of absorption
    abs_coefficients = np.polyfit(pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES], absorption[ABSORPTION_WIDTH_INDICES], 2)
#    detector_row = illuminated_window_tops[frame_index]+bin_index*binning
    
    if plot:
        plt.plot(pixels[list(range(continuum_range[0], continuum_range[3]))], absorption)
        fitted_absorption = np.polyval(abs_coefficients, pixels[list(range(continuum_range[0], continuum_range[3]))][ABSORPTION_WIDTH_INDICES])
        plt.plot(pixels[list(range(continuum_range[0],continuum_range[3]))][ABSORPTION_WIDTH_INDICES], fitted_absorption)
    
    absorption_minima = (-1.0 * abs_coefficients[1]) / (2.0 * abs_coefficients[0])
    
    return absorption_minima




    
orders = []
gradients = []

molecule = "ch4"
    
    

    
# ABSORPTION_COEFFS1 = {"ch4":[[0.85, 251.0]], "c2h2":[[0.804, 202.8]], "co2":[[0.874, 201.9]], "co":[[0.878, 196.0], [0.924, 105.0]]}[molecule] #before detector swap

AOTF_FREQUENCIES = {"ch4":[19055.938]}[molecule]
ORDERS = {"ch4":[135]}[molecule]
ABSORPTION_COEFFS1 = {"ch4":[[0.85, 84.0], [0.85, 212.0], [0.85, 251.0]]}[molecule] #before detector swap

hdf5Files, hdf5Filenames, _ = make_filelist(obspaths, fileLevel, model=model)

diffraction_order = ORDERS[0]
orders.append(diffraction_order)
lineNumbers = [0,1,2]

def checkSignal(normalised_spectrum):
    return True



pixels = np.arange(320)

CHOSEN_BIN = 11


nColours = 50 #from -20C to +30C
cmap = plt.get_cmap('plasma')
colours = [cmap(i) for i in np.arange(nColours)/nColours]


fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
ax1.grid(True)

temperatures1 = []
temperatures2 = []
absorption_minima1 = []
absorption_minima2 = []

for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
    spectrum_found = False
    
    plt.title("%s gas cell: spectra of order %s for different instrument temperatures" %(molecule.upper(), diffraction_order))
    plt.xlabel("Pixel number")
    plt.ylabel("Normalised gas cell transmittance")
    channel = hdf5_filename.split("_")[3].lower()


    detector_data_all = hdf5_file["Science/Y"][...]
    window_top_all = hdf5_file["Channel/WindowTop"][...]
    binning = hdf5_file["Channel/Binning"][0] + 1
    integration_time = hdf5_file["Channel/IntegrationTime"][0]
    sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
    measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
    
    aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
    
    if sbsf == 1:
        spectra = detector_data_all[:, CHOSEN_BIN, :]
        
        for spectrum, aotf_frequency in zip(spectra, aotf_frequencies):
            if not spectrum_found:
                if np.any([(frequency-1.1) < aotf_frequency < (frequency+1.1) for frequency in AOTF_FREQUENCIES]):
                    # normalised_spectrum = spectrum / np.max(spectrum)
                    spectrum_als = baseline_als(spectrum)
                    normalised_spectrum = spectrum / spectrum_als
                    
                    #bad pixel removal
                    normalised_spectrum[138] = np.mean(normalised_spectrum[100:150])
                    normalised_spectrum[139] = np.mean(normalised_spectrum[101:151])
                    normalised_spectrum[284] = np.mean(normalised_spectrum)
                    
                    if checkSignal(normalised_spectrum):
                        
                        spare_detector = False
                        linestyle = "-"
                        #find pixel with absorption minimum
                        #absorption minimum varies with temperature - use approx calibration
                        for lineNumber in lineNumbers:
                            approx_centre = int(ABSORPTION_COEFFS1[lineNumber][0] * measurement_temperature + ABSORPTION_COEFFS1[lineNumber][1])
                            
                            if lineNumber == lineNumbers[0]:
                                label = "%s; %0.1fC" %(hdf5_filename[0:15], measurement_temperature)
                            else:
                                label = ""
                            
                            ax1.plot(normalised_spectrum, color=colours[int(measurement_temperature)+20], linestyle=linestyle, label=label)
                            print("\"%s\", #%s %0.1fC" %(hdf5_filename, molecule.upper(), measurement_temperature))
                            spectrum_found = True
                            
                            minimum_range = [approx_centre-5, approx_centre+5]
                            
                            
                            minimum_position = np.argmin(spectrum[range(minimum_range[0], minimum_range[1])])
                            minimum_pixel = pixels[[range(minimum_range[0], minimum_range[1])]][minimum_position]
                            #define fitting range from this
                            continuum_range = [minimum_pixel-12, minimum_pixel-7, minimum_pixel+7, minimum_pixel+12]
                            
                            absorption_minimum = findAbsorptionMinimum(spectrum, continuum_range, plot=False)
                            plt.axvline(x=absorption_minimum, linestyle="--", color=colours[int(measurement_temperature)+20])
                            
                            if spare_detector:
                                temperatures2.append(measurement_temperature)
                                absorption_minima2.append(absorption_minimum)
                            else:
                                temperatures1.append(measurement_temperature)
                                absorption_minima1.append(absorption_minimum)


ax1.legend()

fig1.savefig("%s_gas_cell_spectra_vs_temperature.png" %molecule.upper(), dpi=300)

