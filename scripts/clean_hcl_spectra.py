# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:38:03 2020

@author: iant

PLOT HCL PROFILES

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import h5py

from tools.file.hdf5_functions import get_file, make_filelist
from tools.file.save_dict_to_hdf5 import save_dict_to_hdf5

# from tools.file.hdf5_functions_v04 import getFile, makeFileList
from tools.file.get_hdf5_data_v01 import getLevel1Data
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_polynomial import fit_polynomial
from tools.general.get_nearest_index import get_nearest_index
from tools.plotting.colours import get_colours
from instrument.nomad_so_instrument import nu_mp



#for plotting
FIG_X = 18
FIG_Y = 8

#plot_type = [1,2,3,4,5,6,7,8]
plot_type = [1,2,3,4,7,8]
#plot_type = [2]

#select obs for deriving correction
#be careful of which detector rows are used. Nom pointing after 2018 Aug 11


#regex = re.compile("20180(615|804|813|816|818|820|823|827)_.*_SO_A_I_129") #bad
#regex = re.compile("20180(813|816|818|820|823|827)_.*_SO_A_I_129") #row120 used 

#regex = re.compile("20.*_SO_A_[IE]_130")
regex = re.compile("20(180828|180830|180901|181125|181201|181207|190203|190311|190504|191211)_.*_SO_A_[IE]_130") #row120 used 


file_level = "hdf5_level_1p0a"
hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True)



fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))


pixels = np.arange(320.0)
bin_indices = list(range(4))

#calibrate transmittances for all 4 bins
correction_dict = {}
for bin_index in bin_indices:
    spectra_in_bin = [] #get all spectra 
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
    
        toa_alt = 100.0
        
        #use mean method, returns dictionary
        obs_dict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=toa_alt)
        
        if 0 in plot_type:
            if bin_index == 3:
                plt.figure()
    #            colours = ["b", "g", "r", "m"]
                for px in [160, 200, 240]:
                    plt.plot(obs_dict["alt"], obs_dict["y_mean"][:, px])
                plt.title(hdf5_filename)
                plt.savefig(os.path.join("output", hdf5_filename+"_bin3_vs_alt.png"))
                plt.close()
    
        #cat all data and sort by altitude
#        yAll = np.concatenate([obs_dict["y_mean"] for obs_dict in obs_dicts])
#        altAll = np.concatenate([obs_dict["alt"] for obs_dict in obs_dicts])
#        sortIndices = altAll.argsort()
#        altSorted = altAll[sortIndices]
#        ySorted = yAll[sortIndices, :]
        
        good_indices = np.where((obs_dict["y_mean"][:, 200] > 0.1) & (obs_dict["y_mean"][:, 200] < 0.9))[0]
        
#        nu_obs = obs_dicts[0]["x"][0,:]
    #    diffraction_order = int(hdf5_filename.split("_")[-1])
    #    nu_obs = nu_mp(diffraction_order, pixels, instrument_temperature)
        
        for spectrum_index in good_indices:
            spectra_in_bin.append(obs_dict["y_mean"][spectrum_index, :])
        
    ### derive correction
    spectra_in_bin = np.asfarray(spectra_in_bin)
    correction_dict[bin_index] = {}
    correction_dict[bin_index]["spectra"] = spectra_in_bin

    continuum = np.zeros_like(spectra_in_bin)
    deviation = np.zeros_like(spectra_in_bin)
    
    for spectrum_index, spectrum in enumerate(spectra_in_bin):
        polyfit = np.polyfit(pixels, spectrum, 5)
        continuum[spectrum_index, :] = np.polyval(polyfit, pixels)
        deviation[spectrum_index, :] = spectrum - continuum[spectrum_index, :]

    correction_dict[bin_index]["continuum"] = continuum
    correction_dict[bin_index]["deviation"] = deviation

    fit_coefficients = []
    for pixel in pixels:
        pixel = int(pixel)
        linear_fit, coefficients = fit_polynomial(continuum[:, pixel], deviation[:, pixel], coeffs=True)
        fit_coefficients.append(coefficients)

    fit_coefficients = np.asfarray(fit_coefficients).T
    correction_dict[bin_index]["coefficients"] = fit_coefficients
    
    
save_dict_to_hdf5(correction_dict, "test2")





if 0 in plot_type:
    print(0)
    sys.exit()


colours = get_colours(len(pixels), cmap="viridis")
if 5 in plot_type:

    pixel = 69
    bin_index = 3
    
    plt.figure()
    plt.scatter(correction_dict[bin_index]["continuum"][:, pixel], correction_dict[bin_index]["deviation"][:, pixel], color=colours[pixel])
    plt.plot(np.arange(0.0, 1.0, 0.1), np.polyval(correction_dict[bin_index]["coefficients"][:, pixel], np.arange(0.0, 1.0, 0.1)), color=colours[pixel])
    plt.xlabel("Continuum transmittance of pixel")
    plt.ylabel("Deviation from continuum")
    print(5)
    sys.exit()


#plot all deviations bin3
if 6 in plot_type:
    
    bin_index = 3
    colours2 = get_colours(correction_dict[bin_index]["continuum"].shape[0], cmap="jet")
    plt.figure()
    for spectrum_index, spectrum_deviation in enumerate(correction_dict[bin_index]["deviation"]):
        plt.plot(pixels, spectrum_deviation, alpha=0.3, color=colours2[int(correction_dict[bin_index]["continuum"][spectrum_index, 200]*200)])
    plt.xlabel("Pixels")
    plt.ylabel("Deviation from continuum")
    print(6)
    sys.exit()

    
    

    
### get data from best occultation

#chosen_hdf5_filename = "20190102_190343_1p0a_SO_A_E_129" #for presentation
#instrument_temperature = -3.5
#toa_alt = 27.0
#plot_altitude = 9.9
#transmittance_range = [0.25, 0.388]
#apply_correction = True
#indices_no_strong_abs = list(range(100))+list(range(120, 320))
###resize_len = 374
####resize_index = 25


#chosen_hdf5_filename = "20180720_225026_1p0a_SO_A_E_129"
#instrument_temperature = -3.5
#toa_alt = 40.0
#plot_altitude = 16.5
#transmittance_range = [0.2, 0.6]
#apply_correction = True
#indices_no_strong_abs = list(range(100))+list(range(120, 320))

#chosen_hdf5_filename = "20191122_210929_1p0a_SO_A_E_126"
#instrument_temperature = -3.5
#toa_alt = 30.0
#plot_altitude = 2.9
#transmittance_range = [0.3, 0.35]
#apply_correction = True
#indices_no_strong_abs = list(range(100))+list(range(120, 320))

#chosen_hdf5_filename = "20180522_054221_1p0a_SO_A_E_130" #detection at unusual time
#instrument_temperature = 3.5
#toa_alt = 65.0
#plot_altitude = 24.7
#transmittance_range = [0.2, 0.4]
#apply_correction = True
#indices_no_strong_abs = list(range(64))+list(range(74, 121))+list(range(131,164))+\
#    list(range(181,191))+list(range(200,215))+list(range(226, 320))


#chosen_hdf5_filename = "20181108_154033_1p0a_SO_A_I_130"
#instrument_temperature = -1.8
#toa_alt = 50.0
#plot_altitude = 20.0
#transmittance_range = [0.2, 0.4]
#apply_correction = True
##64-74, 121-131, 164-181, 190-202, etc.
#indices_no_strong_abs = list(range(64))+list(range(74, 121))+list(range(131,164))+\
#    list(range(181,191))+list(range(200,215))+list(range(226, 320))


#chosen_hdf5_filename = "20180719_184005_1p0a_SO_A_I_130" #noisy difficult to evaluate
#instrument_temperature = 3.5
#toa_alt = 33.0
#plot_altitude = 16.5
#transmittance_range = [0.1, 0.3]
#apply_correction = True
#indices_no_strong_abs = list(range(64))+list(range(74, 121))+list(range(131,164))+\
#    list(range(181,191))+list(range(200,215))+list(range(226, 320))


chosen_hdf5_filename = "20180724_143318_1p0a_SO_A_I_130" #noisy difficult to evaluate
instrument_temperature = 3.5
toa_alt = 63.0
plot_altitude = 18.0
transmittance_range = [0.1, 0.3]
apply_correction = True
indices_no_strong_abs = list(range(64))+list(range(74, 121))+list(range(131,164))+\
    list(range(181,191))+list(range(200,215))+list(range(226, 320))





diffraction_order = int(chosen_hdf5_filename.split("_")[-1])

_, chosen_hdf5_file = get_file(chosen_hdf5_filename, file_level, 0, silent=True)


alts_in = np.mean(chosen_hdf5_file["Geometry/Point0/TangentAltAreoid"][...], axis=1)


#calibrate transmittances for all 4 bins
#use mean method, returns dictionary
obs_dicts = [
    getLevel1Data(chosen_hdf5_file, chosen_hdf5_filename, 0, silent=True, top_of_atmosphere=toa_alt), 
    getLevel1Data(chosen_hdf5_file, chosen_hdf5_filename, 1, silent=True, top_of_atmosphere=toa_alt),
    getLevel1Data(chosen_hdf5_file, chosen_hdf5_filename, 2, silent=True, top_of_atmosphere=toa_alt),
    getLevel1Data(chosen_hdf5_file, chosen_hdf5_filename, 3, silent=True, top_of_atmosphere=toa_alt),
]

#x = obs_dicts[0]["x"][0,:]
#x = pixels
x = nu_mp(diffraction_order, pixels, instrument_temperature)

if 2 in plot_type:
    #plot bin 3 chosen altitude uncorrected
    bin_index = 3
    alts = obs_dicts[bin_index]["alt"]
    spectrum_index = get_nearest_index(plot_altitude, alts)

    alt = obs_dicts[bin_index]["alt"][spectrum_index]
    
    
    y = obs_dicts[bin_index]["y_mean"][spectrum_index, :]
    y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
    y_corrected = y / y_continuum
    ax1.plot(x, y_corrected, "g", alpha=0.7, label="%0.1fkm before correction" %alt)
    y_corrected_std = np.std(y_corrected[indices_no_strong_abs])
    ax1.axhline(y=1.0 - 1.0 * y_corrected_std, color="g", linestyle=":", label="stdev before correction")
    ax1.axhline(y=1.0 + y_corrected_std, color="g", linestyle=":")

#plot transmittance vs alt all bins
if 1 in plot_type:

    plt.figure()
    colours = ["b", "g", "r", "m"]
    for px in [160, 200, 240]:
        for bin_index, obs_dict in enumerate(obs_dicts):
            plt.scatter(obs_dict["alt"], obs_dict["y_mean"][:, px], color=colours[bin_index], label="bin %i px %i" %(bin_index, px))
    plt.title(chosen_hdf5_filename)


y_list = [obs_dict["y_mean"] for obs_dict in obs_dicts]
alt_list = [obs_dict["alt"] for obs_dict in obs_dicts]
goodIndices = [np.where((y[:, 200] > transmittance_range[0]) & (y[:, 200] < transmittance_range[1]))[0] for y in y_list]

for i in range(4):
    print("y200 bin %i:" %i, y_list[i][goodIndices[i], 200])
for i in range(4):
    print("alt bin %i:" %i, alt_list[i][goodIndices[i]])

#plot raw spectra from all bins
if 1 in plot_type:
    plt.figure()
    for bin_index, (ys, alts, indices) in enumerate(zip(y_list, alt_list, goodIndices)):
        for spectrum_index in range(len(alts)):
            y = ys[spectrum_index, :]
            if spectrum_index in indices:
                plt.plot(x, y, label="bin %i %0.1fkm" %(bin_index, alts[spectrum_index]))
            else:
                plt.plot(x, y, alpha=0.1)
    plt.xlabel("Wavenumbers (cm-1)")
    plt.ylabel("Transmittance")
    plt.legend()
    plt.title(chosen_hdf5_filename)
#    sys.exit()


if 3 in plot_type:
    #plot normalised spectra from bin 3
    plt.figure()
    i = 3
    for spectrum_index in goodIndices[i]:
        
        y = y_list[i][spectrum_index, :]
        y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
        y_corrected = y / y_continuum
        
    #    plt.plot(pixels, y, label="%0.1fkm" %alt_list[3][spectrum_index])
        plt.plot(x, y_corrected, label="%0.1fkm" %alt_list[i][spectrum_index])
    plt.legend()
    plt.title(chosen_hdf5_filename + ": unbinned before correction")
    plt.xlabel("Pixel number")
    plt.ylabel("Normalised transmittance")



### apply correction


#plot corrected normalised spectra from bin 3
if 4 in plot_type:
    plt.figure()

i = 3
y_offset_corrected_all = []
alts_all = []
for spectrum_index in goodIndices[i]:
    
    y = y_list[i][spectrum_index, :]
    y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
    
    if apply_correction:
    
        for pixel in pixels:
            coefficients = correction_dict[i]["coefficients"][:, int(pixel)]
            y[int(pixel)] -= np.polyval(coefficients, y_continuum[int(pixel)])
        y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)

    y_corrected = y / y_continuum
    y_offset_corrected_all.append(y_corrected)
    alts_all.append(alt_list[i][spectrum_index])

    if 4 in plot_type:
#       plt.plot(pixels, y, label="%0.1fkm" %alt_list[3][spectrum_index])
        plt.plot(x, y_corrected, label="%0.1fkm" %alt_list[i][spectrum_index])
if 4 in plot_type:
    plt.legend()
    plt.title(chosen_hdf5_filename + ": unbinned after correction")
    plt.xlabel("Pixels")
    plt.ylabel("Normalised transmittance")


#plot mean of bin 3 data
y_offset_corrected_all = np.asfarray(y_offset_corrected_all)
y_offset_corrected_mean_1bin = np.mean(y_offset_corrected_all, axis=0)
y_offset_corrected_std = np.std(y_offset_corrected_mean_1bin[indices_no_strong_abs])

if 2 in plot_type:
    ax1.plot(x, y_offset_corrected_mean_1bin, "b", label="bin %i after correction" %i)
    ax1.axhline(y=1.0 - 1.0 * y_offset_corrected_std, color="b", linestyle=":", label="stdev after correction")
    ax1.axhline(y=1.0 + y_offset_corrected_std, color="b", linestyle=":")

    y_continuum = baseline_als(y_offset_corrected_mean_1bin)
    y_baseline_corrected_mean_1bin = y_offset_corrected_mean_1bin / y_continuum


    ax2.plot(x, y_baseline_corrected_mean_1bin, "b", label="bin %i after baseline correction" %i)
    ax2.axhline(y=1.0 - 1.0 * y_offset_corrected_std, color="b", linestyle=":", label="stdev after correction")
    ax2.axhline(y=1.0 + y_offset_corrected_std, color="b", linestyle=":")

#    sys.exit()











#plot transmittance vs alt all bins
#plt.figure()
#colours = ["b", "g", "r", "m"]
#for px in [160, 200, 240]:
#    for bin_index, obs_dict in enumerate(obs_dicts):
#        plt.scatter(obs_dict["alt"], obs_dict["y_mean"][:, px], color=colours[bin_index])
#plt.title(chosen_hdf5_filename)

#y_list = [obs_dict["y_mean"] for obs_dict in obs_dicts]
#alt_list = [obs_dict["alt"] for obs_dict in obs_dicts]
#goodIndices = [np.where((y[:, 200] > 0.2) & (y[:, 200] < 0.4))[0] for y in y_list]


#plot raw spectra from all bins
#plt.figure()
#for bin_index, (ys, alts, indices) in enumerate(zip(y_list, alt_list, goodIndices)):
#    for spectrum_index in indices:
#        y = ys[spectrum_index, :]
#        plt.plot(x, y, label="bin %i %0.1fkm" %(bin_index, alts[spectrum_index]))
#plt.xlabel("Wavenumbers (cm-1)")
#plt.ylabel("Transmittance")
#plt.legend()
#plt.title(chosen_hdf5_filename)

if 7 in plot_type:
    #plot normalised spectra from all bins
    for bin_index in bin_indices:
        plt.figure()
        for spectrum_index in goodIndices[bin_index]:
            
            y = y_list[bin_index][spectrum_index, :]
            y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
            y_corrected = y / y_continuum
            
        #    plt.plot(pixels, y, label="%0.1fkm" %alt_list[3][spectrum_index])
            plt.plot(x, y_corrected, label="%0.1fkm" %alt_list[bin_index][spectrum_index])
        plt.legend()
        plt.title(chosen_hdf5_filename + ": bin %i unbinned before correction" %bin_index)
        plt.xlabel("Pixel number")
        plt.ylabel("Normalised transmittance")


### apply correction


#plot corrected normalised spectra from all bins
y_offset_corrected_all_bins = []
alts_all = []
y_uncorrected_all = []

for bin_index in bin_indices:
    if 8 in plot_type:
        plt.figure()

    for spectrum_index in goodIndices[bin_index]:
    
        y = y_list[bin_index][spectrum_index, :]
        y_uncorrected_all.append(y)
        y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
        
        if apply_correction:
            for pixel in pixels:
                coefficients = correction_dict[bin_index]["coefficients"][:, int(pixel)]
                y[int(pixel)] -= np.polyval(coefficients, y_continuum[int(pixel)])

        y_continuum = fit_polynomial(pixels, y, 5, indices=indices_no_strong_abs)
    
        y_corrected = y / y_continuum
        y_offset_corrected_all_bins.append(y_corrected)
        alts_all.append(alt_list[bin_index][spectrum_index])
    
        if 8 in plot_type:
    #       plt.plot(pixels, y, label="%0.1fkm" %alt_list[3][spectrum_index])
            plt.plot(x, y_corrected, label="%0.1fkm" %alt_list[i][spectrum_index])

    if 8 in plot_type:
        plt.legend()
        plt.title(chosen_hdf5_filename + ": bin %i unbinned after correction" %bin_index)
        plt.xlabel("Pixels")
        plt.ylabel("Normalised transmittance")

mean_alt = np.mean(alts_all)
mean_alt_plus_minus = (np.max(alts_all) - np.min(alts_all))/2.0
print("altitudes=", mean_alt, "+-", mean_alt_plus_minus)

mean_trans = np.mean(y_uncorrected_all)
print("mean_transmittance=", mean_trans)


#plot mean of all bins data
y_offset_corrected_all_bins = np.asfarray(y_offset_corrected_all_bins)
y_offset_corrected_mean_all_bins = np.mean(y_offset_corrected_all_bins, axis=0)
y_offset_corrected_std = np.std(y_offset_corrected_mean_all_bins[indices_no_strong_abs])

if 2 in plot_type:
    ax1.plot(x, y_offset_corrected_mean_all_bins, "r", label="all bins after correction")
    ax1.axhline(y=1.0 - 1.0 * y_offset_corrected_std, color="r", linestyle=":", label="stdev after correction")
    ax1.axhline(y=1.0 + y_offset_corrected_std, color="r", linestyle=":")
    
    ax1.set_title(chosen_hdf5_filename + " before and after correction 2")
    ax1.set_xlabel("Wavenumber from file (cm-1)")
    ax1.set_ylabel("Normalised transmittance")
ax1.legend()


y_continuum = baseline_als(y_offset_corrected_mean_all_bins, lam=10.0, p=0.95)
y_baseline_corrected_mean_all_bins = y_offset_corrected_mean_all_bins / y_continuum

ax2.plot(x, y_baseline_corrected_mean_all_bins, "k", label="all bins after baseline correction")
ax2.axhline(y=1.0 - 1.0 * y_offset_corrected_std, color="k", linestyle=":", label="stdev after correction")
ax2.axhline(y=1.0 + y_offset_corrected_std, color="k", linestyle=":")

ax2.set_title(chosen_hdf5_filename + " before and after correction bin3 / all bins")
ax2.set_xlabel("Wavenumber from file (cm-1)")
ax2.set_ylabel("Normalised transmittance")
ax2.legend()

from analysis.py_retrievals.so_simple_retrieval import simple_retrieval, forward_model

alt = mean_alt
resize_index = get_nearest_index(alt, alts_in)
resize_len = len(alts_in)


print("Index = %i" %resize_index)

sim_spectra = {"HCl":{"scaler":4.0, "label":"HCl 4ppbv", "colour":"r--"},
               "H2O":{"scaler":4.8, "label":"H2O", "colour":"m--"},
               "CO2":{"scaler":2.0, "label":"CO2", "colour":"c--"},
               }

for molecule in ["HCl", "H2O", "CO2"]:
#for molecule in ["HCl"]:
    retDict = simple_retrieval(y_offset_corrected_mean_all_bins, alt, molecule, diffraction_order, instrument_temperature)
    retDict = forward_model(retDict, xa_fact=[sim_spectra[molecule]["scaler"]])
    sim_spectra[molecule]["spectrum"] = retDict["Y"][0, :]
    sim_spectra[molecule]["xa"] = retDict["xa"][0]
    sim_spectra[molecule]["xa_fact"] = retDict["xa_fact"][0]
    ax2.plot(x, sim_spectra[molecule]["spectrum"], sim_spectra[molecule]["colour"], label=sim_spectra[molecule]["label"])
    print(molecule, sim_spectra[molecule]["xa"] * sim_spectra[molecule]["xa_fact"] * 1.0e6, "ppmv")
ax2.legend()
plt.savefig(chosen_hdf5_filename+"_corrected_spectrum.png")



from tools.file.read_write_hdf5 import read_hdf5_to_dict, write_hdf5_from_dict



hdf5_filename_new = os.path.join("output", chosen_hdf5_filename+"_corrected_spectrum")
y_out = y_baseline_corrected_mean_all_bins * mean_trans
#with open(hdf5_filename_new+".txt", "w") as f:
#    for nu, y_px in zip(x, y_out):
#        f.write("%0.3f, %0.5f\n" %(nu, y_px))
replace_datasets = {"Science/X":x, "Science/Y":y_out}
replace_attributes = {"NSpec":np.int64(1)}
hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(chosen_hdf5_filename)
write_hdf5_from_dict(hdf5_filename_new, hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes, resize_len=resize_len, resize_index=resize_index)




hdf5_filename_new = os.path.join("output", chosen_hdf5_filename+"_corrected_spectrum_nobaseline")
y_out = y_offset_corrected_mean_all_bins * mean_trans
#with open(hdf5_filename_new+".txt", "w") as f:
#    for nu, y_px in zip(x, y_out):
#        f.write("%0.3f, %0.5f\n" %(nu, y_px))
replace_datasets = {"Science/X":x, "Science/Y":y_out}
replace_attributes = {"NSpec":np.int64(1)}
hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(chosen_hdf5_filename)
write_hdf5_from_dict(hdf5_filename_new, hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes, resize_len=resize_len, resize_index=resize_index)





hdf5_filename_new = os.path.join("output", chosen_hdf5_filename+"_corrected_spectrum_bin3")
y_out = y_baseline_corrected_mean_1bin * mean_trans
#with open(hdf5_filename_new+".txt", "w") as f:
#    for nu, y_px in zip(x, y_out):
#        f.write("%0.3f, %0.5f\n" %(nu, y_px))
replace_datasets = {"Science/X":x, "Science/Y":y_out}
replace_attributes = {"NSpec":np.int64(1)}
hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(chosen_hdf5_filename)
write_hdf5_from_dict(hdf5_filename_new, hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes, resize_len=resize_len, resize_index=resize_index)





hdf5_filename_new = os.path.join("output", chosen_hdf5_filename+"_corrected_spectrum_bin3_nobaseline")
y_out = y_offset_corrected_mean_1bin * mean_trans
#with open(hdf5_filename_new+".txt", "w") as f:
#    for nu, y_px in zip(x, y_out):
#        f.write("%0.3f, %0.5f\n" %(nu, y_px))
replace_datasets = {"Science/X":x, "Science/Y":y_out}
replace_attributes = {"NSpec":np.int64(1)}
hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(chosen_hdf5_filename)
write_hdf5_from_dict(hdf5_filename_new, hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes, resize_len=resize_len, resize_index=resize_index)

