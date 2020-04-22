# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:35:02 2020

@author: iant

CHECK F.SCHMIDT CO2 LINE OBSERVATIONS

"""

import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np

from tools.file.hdf5_functions_v04 import getFile
from tools.file.get_hdf5_data_v01 import getLevel1Data
from tools.spectra.baseline_als import baseline_als
from instrument.nomad_so_instrument import nu_mp
from tools.spectra.solar_spectrum_so import so_solar_line_temperature_shift
from tools.file.observation_orders import list_other_measured_orders

matFilepath = os.path.normcase(r"C:\Users\iant\Documents\DATA\order134_15sources.mat")
lineListFilepath = os.path.normcase(r"C:\Users\iant\Documents\DATA\12CO-16O2__E2__T296Kb_01111_00010_reduced.txt")

#for plotting
FIG_X = 18
FIG_Y = 8

SOURCE_NUMBER = 3 #4th source in Frederic's data
CHOSEN_BIN = 3 #SO bin 0-3 if analysis using only 1 bin


#choose a filename to plot and spectrum indices
#hdf5Filename = "20191014_175211_1p0a_SO_A_I_134" #best but only 1 index in best 1000
#spectraIdAllBins = [31, 35, 39, 43, 47]
#instrumentTemperature = -0.75

#hdf5Filename = "20190919_221937_1p0a_SO_A_I_133" #best occ containing both orders 133 and 134
#hdf5Filename = "20190919_221937_1p0a_SO_A_I_134" #best occ containing both orders 133 and 134
#spectraIdAllBins = [23, 28, 24, 27, 32, 31, 25, 29, 20, 30, 34, 26, 36]
#instrumentTemperature = -5.


#hdf5Filename = "20191105_043151_1p0a_SO_A_E_132"
#hdf5Filename = "20191105_043151_1p0a_SO_A_E_133"
hdf5Filename = "20191105_043151_1p0a_SO_A_E_134"
spectraIdAllBins = [56, 60, 64, 55, 52, 58, 59, 53, 68, 67, 63, 62, 48, 57, 66, 54, 50,
       61, 49, 51, 72, 71, 65, 75, 76, 70, 74, 69, 73, 80, 45, 47, 79]
instrumentTemperature = -2.5


def best_observation_list(dot_mat_filepath, n_best_spectra):
    """make a list of 1.0a hdf5 files and spectrum indices from Frederic Schmidt's
    source file. Consider only the first n_best_spectra and ignore files where
    only 1 spectrum is found.
    
    chosen_source = the chosen source taken directly from the .mat file
    source_list = a list of hdf5 files and their spectrum indices"""

    #cheat to avoid wasting time loading the mat file each run!
#    if 'mat' not in globals():
    mat = scipy.io.loadmat(dot_mat_filepath)
    source_diffraction_order = 134
    
    #A : abundance (dimension : 365985 x 15)
    #S : source spectra (dimension : 320 x 15)
    #observationid_all : name of the file (dimension : 365985 x 1)
    #spectranumber_all : spectra number within a file (dimension : 365985 x 1)
    
    A = mat["A"][:, SOURCE_NUMBER]
    all_sources = mat["S"]
    
    chosen_source = all_sources[:, SOURCE_NUMBER]
    observation_id_all = np.ndarray.flatten(mat["observationid_all"])
    spectra_number_all = np.ndarray.flatten(mat["spectranumber_all"])
       
    hdf5_filenames = [str(i[0]).replace(".h5","") for i in observation_id_all]
    spectra_id = [int(i) for i in spectra_number_all]
    abundance = np.asfarray([float(i) for i in A])
    
    sort_indices = np.argsort(abundance)[::-1]
    spectra_filelist = []
    for sort_index in sort_indices[:n_best_spectra]:
        spectra_filelist.append([hdf5_filenames[sort_index], spectra_id[sort_index]])
    
    
    unique_filenames = list(set([i[0] for i in spectra_filelist]))
    chosen_filenames = [i[0] for i in spectra_filelist]
    chosen_ids = [i[1] for i in spectra_filelist]
    
    source_list = []
    for unique_filename in unique_filenames:
        file_indices = np.where(unique_filename == np.array(chosen_filenames))[0]
        if len(file_indices)>1:
            source_list.append([unique_filename, np.array(chosen_ids)[file_indices]])

    return chosen_source, all_sources, source_diffraction_order, source_list




def get_co2_line_list(line_list_filepath, scale_each_branch=True):
    """read in co2 electric quadrupole. Output dictionary containing wavenumbers
    relative intensity and branch letter.
    If scale_each_branch then the intensities for each branch will be scaled separately"""
    
    co2_line_list_dict = {"co2_nu":[], "co2_e_intensity":[], "co2_branch":[]}
    
    with open(line_list_filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != "#":
                lineSplit = line.split()
                nuLine = float(lineSplit[0])
    #            if np.min(nuSource) < nuLine < np.max(nuSource):
                co2_line_list_dict["co2_nu"].append(nuLine)
                co2_line_list_dict["co2_e_intensity"].append(float(lineSplit[1]))
                co2_line_list_dict["co2_branch"].append(lineSplit[2])
                
    co2_line_list_dict["co2_nu"] = np.array(co2_line_list_dict["co2_nu"])
    co2_line_list_dict["co2_e_intensity"] = np.array(co2_line_list_dict["co2_e_intensity"])
    co2_line_list_dict["co2_branch"] = np.array(co2_line_list_dict["co2_branch"])
                  
    if scale_each_branch:
        unique_branches = list(set(co2_line_list_dict["co2_branch"]))
        for current_branch in unique_branches: #loop through each branch letter
            branches_indices = np.array([i for i, value in enumerate(co2_line_list_dict["co2_branch"]) if value == current_branch])
            branch_intensities = co2_line_list_dict["co2_e_intensity"][branches_indices]
            co2_line_list_dict["co2_e_intensity"][branches_indices] = branch_intensities / np.max(branch_intensities)
    else: #scale relative to all lines in all branches
        co2_line_list_dict["co2_e_intensity"] = co2_line_list_dict["co2_e_intensity"] / np.max(co2_line_list_dict["co2_e_intensity"])

    co2_line_list_dict["co2_nu"] = list(co2_line_list_dict["co2_nu"])
    co2_line_list_dict["co2_e_intensity"] = list(co2_line_list_dict["co2_e_intensity"])
    co2_line_list_dict["co2_branch"] = list(co2_line_list_dict["co2_branch"])

    return co2_line_list_dict


def plot_co2_line_list(ax, co2_line_list_dict, text_y_position, alpha_intensity=True):
    """plot the line list onto the given axis.
    If alpha_intensity then shade lines according to intensity"""

    for nu, intensity, branch in zip(co2_line_list_dict["co2_nu"], \
         co2_line_list_dict["co2_e_intensity"], co2_line_list_dict["co2_branch"]):
        if branch == "O":
            colour = "c"
        elif branch == "P":
            colour = "m"
        elif branch == "Q":
            colour = "r"
        elif branch == "R":
            colour = "g"
        elif branch == "S":
            colour = "b"
        else:
            colour = "grey"
            
        if alpha_intensity:
            alpha = np.max((0.0, np.log(intensity*1.5+(2.71-1.5))))
#            alpha = np.max((0.0, np.log(intensity*2.71)))
            if alpha>0.1:
                ax.axvline(x=nu, alpha=alpha, color=colour, linestyle="--")
                ax.annotate(branch, xy=(nu, text_y_position), fontsize=16)
        else:
            ax.axvline(x=nu, alpha=0.7, color=colour, linestyle="--")
            ax.annotate(branch, xy=(nu, text_y_position), fontsize=16)
            
        ax.set_xlabel("Wavenumber cm-1")
        ax.set_ylabel("Source relative absorbance")




def plot_sources(ax, all_sources, nu, co2_line_list_dict):
    """plot all sources from Frederic Schmidt .mat file"""

    for spectrum_index, spectrum in enumerate(all_sources.T):
        if spectrum_index == 3:
            ax.plot(nu, spectrum, "k", label=spectrum_index, linewidth=2)
        else:
            ax.plot(nu, spectrum, label=spectrum_index, linestyle=":")
    ax.legend()
    ax.set_xlim((np.min(nu)-1.0,np.max(nu)+1.0))



def plot_acs(ax, offset=-0.01):
    
    data = np.loadtxt("ACS_CO2_M1_lines.csv", skiprows=1, delimiter=",")
    ax.plot(data[:, 0], data[:, 1] + 1.0 + offset, label="ACS combined spectrum offset=%0.3f" %offset)


def plot_source(ax, chosen_source, nu, offset=0.005):
    """plot chosen source in transmittance and rescale
    from Frederic Schmidt .mat file on existing axes"""

    ax.plot(nu, 1.0 - chosen_source / np.max(chosen_source) / 100.0 + offset, "k", label="Frederic Source %i (0=1st source)" %SOURCE_NUMBER, linewidth=2)




def plot_solar_line_shift(ax, instrument_temperature, delta_temperature, diffraction_order, offset=0.005):
    """plot solar line shift due to grating temperature change on axis"""

    instrument_temperatures = [instrument_temperature, instrument_temperature + delta_temperature]
    solspec_filepath = os.path.join("reference_files", "nomad_solar_spectrum_solspec.txt")
    
    lineShift = so_solar_line_temperature_shift(diffraction_order, instrument_temperatures, solspec_filepath, adj_orders=2)

    pixels = np.arange(320)
    nu = nu_mp(diffraction_order, pixels, instrument_temperature)
    ax.plot(nu, lineShift+offset, color="brown", linestyle="--", label="%0.2fC solar line shift offset+%0.3f" %(delta_temperature, offset))



def plot_single_bin_occultation(ax, hdf5_filename, spectra_id_all_bins, use_file_nu=False, instrument_temperature=-999.):
    """plot the selected spectra from a single bin of one hdf5 file.
    If use_file_nu then take the wavenumbers from the file,
    else if an instrument temperature to recalculate nu grid.
    Output = waenumber grid"""
   
    
    hdf5_filepath, hdf5_file = getFile(hdf5_filename, "hdf5_level_1p0a", 0)
    
    #find which spectrum indices in the source file correspond to those in the chosen bin
    bins = hdf5_file["Science/Bins"][:, 0]
    uniqueBins = sorted(list(set(bins)))
    binIndices = np.where(bins == uniqueBins[CHOSEN_BIN])[0]
    binIds = []
    for binId, binIndex in enumerate(binIndices):
        if binIndex in spectra_id_all_bins:
            binIds.append(binId)

    #get data for the chosen bin
    obsDict = getLevel1Data(hdf5_file, hdf5_filename, CHOSEN_BIN, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    
    #use wavenumber X from file, or remake from instrument temperature?
    #replace with line detection
    if use_file_nu:
        nu_obs = obsDict["x"][0, :]
    else:
        diffraction_order = int(hdf5_filename.split("_")[-1])
        pixels = np.arange(320)
        nu_obs = nu_mp(diffraction_order, pixels, instrument_temperature)
    
    #remove baseline continuum and plot
    for spectrumIndex in binIds:
        y_mean_baseline = baseline_als(obsDict["y_mean"][spectrumIndex, :])
        y_mean_corrected = obsDict["y_mean"][spectrumIndex, :] / y_mean_baseline
    
        ax.plot(nu_obs, y_mean_corrected, label="SO bin %i @ %0.1fkm " %(CHOSEN_BIN, obsDict["alt"][spectrumIndex]))
        
    return nu_obs


###start script###

nBestSpectra = 1000
print("Getting data for best spectra from .mat file")
chosenSource, allSources, sourceDiffractionOrder, sourceList = best_observation_list(matFilepath, nBestSpectra)
co2LineListDict = get_co2_line_list(lineListFilepath)

#get wavenumber scale for Frederic analysis
sourceInstTemperature = -7.5 #this temperature is required to match wavenumber scale to Frederic's plot
pixels = np.arange(320)
nuSource = nu_mp(sourceDiffractionOrder, pixels, sourceInstTemperature)


"""plot all sources on a figure"""
#fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))
#plot_sources(ax, allSources, nuSource, co2LineListDict)
#plot_co2_line_list(ax, co2LineListDict, 0.07)



"""plot a single observation selected above"""
obsDiffractionOrder = int(hdf5Filename.split("_")[-1])

fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))
ax.set_title(hdf5Filename)
nuObservation = plot_single_bin_occultation(ax, hdf5Filename, spectraIdAllBins, instrument_temperature=instrumentTemperature)
plot_co2_line_list(ax, co2LineListDict, 1.01, alpha_intensity=True)
#plot_co2_line_list(ax, co2LineListDict, 1.01, alpha_intensity=False)

plot_acs(ax)

#print("Getting solar line shift")
#plot_solar_line_shift(ax, sourceInstTemperature, 0.25, obsDiffractionOrder)

#Frederic's data only available for 134. Don't plot if not
if obsDiffractionOrder == sourceDiffractionOrder:
    print("Plotting source on figure")
    plot_source(ax, chosenSource, nuSource)


ax.set_ylim((0.980,1.015))
ax.set_xlim((np.min(nuObservation)-1.0,np.max(nuObservation)+1.0))
ax.legend(loc="lower right")
ax.set_xlabel("Wavenumber cm-1")
ax.set_ylabel("Baseline corrected transmittance")
#fig.savefig("CO2_M1_E2_solar_line_shift_%s.png" %hdf5Filename)
fig.savefig("CO2_M1_E2_with_ACS_%s.png" %hdf5Filename)

stop()

"""find other diffraction orders measured at same time as best 134 obs"""
#hdf5_filenames = [i[0] for i in sourceList]
#otherOrdersMeasured = list_other_measured_orders(hdf5_filenames)
#for fileIndex, orders in enumerate(otherOrdersMeasured):
#    if 132 in orders and 133 in orders:
#        print(hdf5_filenames[fileIndex][0:15], ": orders 132, 133 and 134 measured")
#    elif 132 in orders:
#        print(hdf5_filenames[fileIndex][0:15], ": orders 132 and 134 measured")
#    elif 133 in orders:
#        print(hdf5_filenames[fileIndex][0:15], ": orders 133 and 134 measured")
#stop()



def plot_all_bins_occultation(ax, hdf5_filename, spectra_id_all_bins, use_file_nu=False, instrument_temperature=-999.):
    """plot the selected spectra from all the bins of one hdf5 file.
    If use_file_nu then take the wavenumbers from the file,
    else if an instrument temperature to recalculate nu grid.
    Output = waenumber grid"""

    hdf5_filepath, hdf5_file = getFile(hdf5_filename, "hdf5_level_1p0a", 0)
    
    #calibrate transmittances for all 4 bins
    obsDict0 = getLevel1Data(hdf5_file, hdf5_filename, 0, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    obsDict1 = getLevel1Data(hdf5_file, hdf5_filename, 1, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    obsDict2 = getLevel1Data(hdf5_file, hdf5_filename, 2, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    obsDict3 = getLevel1Data(hdf5_file, hdf5_filename, 3, silent=True, top_of_atmosphere=60.0) #use mean method, returns dictionary
    
    #cat all data and sort by altitude
    yAll = np.concatenate((obsDict0["y_mean"], obsDict1["y_mean"], obsDict2["y_mean"], obsDict3["y_mean"]))
    altAll = np.concatenate((obsDict0["alt"], obsDict1["alt"], obsDict2["alt"], obsDict3["alt"]))
    sortIndices = altAll.argsort()
    altSorted = altAll[sortIndices]
    ySorted = yAll[sortIndices, :]
    
    

    #use wavenumber X from file, or remake from instrument temperature?
    #replace with line detection
    if use_file_nu:
#       nu = obsDict0["x"][0, :] - 0.1
        nu_obs = obsDict0["x"][0, :]
    else:
        diffraction_order = int(hdf5_filename.split("_")[-1])
        pixels = np.arange(320)
        nu_obs = nu_mp(diffraction_order, pixels, instrument_temperature)
    
    
    for spectrum_index in spectra_id_all_bins:
        yBaseline = baseline_als(ySorted[spectrum_index, :])
        yCorrected = ySorted[spectrum_index, :] / yBaseline
    
        ax.plot(nu_obs, yCorrected, label="%0.1fkm" %altSorted[spectrum_index])

    return nu_obs



for file_index, (hdf5Filename, spectraIdAllBins) in enumerate(sourceList):

    otherOrdersMeasured = list_other_measured_orders([hdf5Filename])[0]

    hdf5FilenamesObservation = [hdf5Filename]
    if 132 in otherOrdersMeasured:
        hdf5FilenamesObservation.append("_".join(hdf5Filename.split("_")[:-1]) + "_132")
    elif 133 in otherOrdersMeasured:
        hdf5FilenamesObservation.append("_".join(hdf5Filename.split("_")[:-1]) + "_133")
    
    if len(hdf5FilenamesObservation)==1:
        hdf5FilenamesObservation = []
    
    for hdf5FilenameEach in hdf5FilenamesObservation:
        obsDiffractionOrder = int(hdf5FilenameEach.split("_")[-1])
    
        fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))
        ax.set_title(hdf5FilenameEach)
        
#        plot_acs(ax)
    
        nuObservation = plot_all_bins_occultation(ax, hdf5FilenameEach, spectraIdAllBins, instrument_temperature=instrumentTemperature)
        #plot_co2_line_list(ax, co2LineListDict, 1.01, alpha_intensity=True)
        plot_co2_line_list(ax, co2LineListDict, 1.01, alpha_intensity=False)
        
    #    print("Getting solar line shift")
    #    plot_solar_line_shift(ax, sourceInstTemperature, 0.25, obsDiffractionOrder)
        
        #Frederic's data only available for 134. Don't plot if not
        if obsDiffractionOrder == sourceDiffractionOrder:
            print("Plotting source on figure")
            plot_source(ax, chosenSource, nuSource)
        
        
        ax.set_ylim((0.980,1.015))
        ax.set_xlim((np.min(nuObservation)-1.0,np.max(nuObservation)+1.0))
        ax.legend(loc="lower center")
        ax.set_xlabel("Wavenumber cm-1")
        ax.set_ylabel("Baseline corrected transmittance")
        fig.savefig("CO2_dipole_%s.png" %hdf5FilenameEach)
        plt.close()


