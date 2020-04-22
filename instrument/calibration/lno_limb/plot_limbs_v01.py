# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:06:29 2018

@author: iant

PLOT LNO LIMB DATA


"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

#from hdf5_functions_v02b import get_dataset_contents,get_hdf5_attribute
from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, getFile, makeFileList, printFileNames
from filename_lists_v01 import getFilenameList

"""Save figures to current working directory?"""
#SAVE_FIGS = True
SAVE_FIGS = False
#

"""Select file names to be read in"""
"""blank"""
fileLevel = "hdf5_level_0p1a"
obspaths = []
title = ""

"""limbs"""
fileLevel = "hdf5_level_0p3a"
obspaths = ["*_LNO_1_L_164"]

#title = "plot geometry each file"
#title = "plot spectra each file"
#title = "plot mean counts each file"
#title = "all file spectra for altitude bins"
title = "check altitudes"



def plotBins(hdf5Files, hdf5Filenames, whatToPlot):
    """Choose what to plot. Note that total number of figures produced = number of bins chosen x number of files chosen!
    Detector bins for limb observations are [80, 92, 104, 116, 128, 140, 152, 164, 176, 188, 200, 212]
    Row 80 is the lowest altitude bin"""
    #chosenBinsToPlot = [80, 92, 104, 116, 128, 140, 152, 164, 176, 188, 200, 212] #all bins
    chosenBinsToPlot = [80] #lowest bin
    #chosenBinsToPlot = [212] #highest bin
    


    #loop through files, plotting each one.
    for fileIndex,hdf5File in enumerate(hdf5Files):
    
        hdf5File = hdf5Files[fileIndex]
            
        
        """get FOV geometry from chosen file
        LNO is given by 5 points, 0=centre, 1-4=FOV corners
        Get tangent point longitude/latitude/local time/altitude above surface
        And sub-observer and sub-solar points"""
#        nPoints = get_hdf5_attribute(hdf5File, "GeometryPoints")
#        geometryNames = ["Geometry/Point%i" %point for point in list(range(1, nPoints))+[1]]
#        geometryPointsLons = np.asfarray([get_dataset_contents(hdf5File, "Lon", chosen_group=geometryName)[0] for geometryName in geometryNames])
#        geometryPointsLats = np.asfarray([get_dataset_contents(hdf5File, "Lat", chosen_group=geometryName)[0] for geometryName in geometryNames])
#        geometryPointsLst = np.asfarray([get_dataset_contents(hdf5File, "LST", chosen_group=geometryName)[0] for geometryName in geometryNames])
#        geometryPointsAlts = np.asfarray([get_dataset_contents(hdf5File, "TangentAlt", chosen_group=geometryName)[0] for geometryName in geometryNames])
#        geometrySubObsLon = get_dataset_contents(hdf5File, "SubObsLon")[0]
#        geometrySubObsLat = get_dataset_contents(hdf5File, "SubObsLat")[0]
#        geometrySubSolLon = get_dataset_contents(hdf5File, "SubSolLon")[0]
#        geometrySubSolLat = get_dataset_contents(hdf5File, "SubSolLat")[0]


        #get centre of FOV only
        geometryPointsLons = hdf5File["Geometry/Point0/Lon"][...]
        geometryPointsLats = hdf5File["Geometry/Point0/Lat"][...]
        geometryPointsAlts = hdf5File["Geometry/Point0/TangentAlt"][...]


        geometrySubObsLon = hdf5File["Geometry/SubObsLon"][...]
        geometrySubObsLat = hdf5File["Geometry/SubObsLat"][...]
        geometrySubSolLon = hdf5File["Geometry/SubSolLon"][...]
        geometrySubSolLat = hdf5File["Geometry/SubSolLat"][...]
        
        
        #Limb observations are unbinned - find the detector rows associated with each observation.
#        bins = get_dataset_contents(hdf5File, "Bins")[0]
        bins = hdf5File["Science/Bins"][...]
    
        #get detector data
#        detectorDataAll = get_dataset_contents(hdf5File, "Y")[0] #get data
        detectorDataAll = hdf5File["Science/Y"][...]
    
        #get wavenumber scale
#        wavenumbersAll = get_dataset_contents(hdf5File, "X")[0] #get spectral axis
        wavenumbersAll = hdf5File["Science/X"][...]
        wavenumbers = wavenumbersAll[0, :]
        
        if "Mean Counts" in whatToPlot: #preload figure
            fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
        
        for chosenBinNumber in chosenBinsToPlot:
    
            #find indices for spectra of chosen bin
            chosenIndices = np.asarray([spectrumIndex for spectrumIndex, eachBin in enumerate(bins) if eachBin[0] == chosenBinNumber])
                
    
            #select data in that chosen detector bin
            detectorData = detectorDataAll[chosenIndices, :]
            
            """1. plot LNO geometry: tangent altitude of lowest detector rows"""
            if "Geometry" in whatToPlot:
           
                fig = plt.figure(figsize=(FIG_X, FIG_Y))
                ax = fig.add_subplot(111, projection="mollweide")
                ax.grid(True)
                colours = geometryPointsAlts[chosenIndices, 0]
                plot1 = ax.scatter(geometryPointsLons[chosenIndices, 0] * np.pi / 180.0, geometryPointsLats[chosenIndices, 0] * np.pi / 180.0, c=colours, cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.7)
                ax.scatter(geometrySubObsLon[chosenIndices,0] * np.pi / 180.0, geometrySubObsLat[chosenIndices,0] * np.pi / 180.0, c=colours, cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.7)
                ax.scatter(geometrySubSolLon[chosenIndices,0] * np.pi / 180.0, geometrySubSolLat[chosenIndices,0] * np.pi / 180.0, c=colours, cmap=plt.cm.jet, marker='o', linewidth=0, alpha=0.7)
                cbar = fig.colorbar(plot1, fraction=0.046, pad=0.04)
                cbar.set_label("Tangent altitude of bin: Min=%0.1f, Max=%0.1f" %(np.min(geometryPointsAlts[chosenIndices, 0]),np.max(geometryPointsAlts[chosenIndices, 0])), rotation=270, labelpad=20)
                plt.annotate("Sub Observer Start", [geometrySubObsLon[0,0] * np.pi / 180.0, (geometrySubObsLat[0,0] - 5.0) * np.pi / 180.0])
                plt.annotate("Sub Solar Point Start", [geometrySubSolLon[0,0] * np.pi / 180.0, (geometrySubSolLat[0,0] - 5.0) * np.pi / 180.0])
                plt.annotate("Tangent Point Start", [geometryPointsLons[0, 0] * np.pi / 180.0, (geometryPointsLats[0, 0] - 15.0) * np.pi / 180.0])
                fig.tight_layout()
                if SAVE_FIGS:
                    plt.savefig(BASE_DIRECTORY+os.sep+hdf5Filenames[fileIndex]+"_groundtrack_altitude_bin_%i.png" %chosenBinNumber)
            
            
            
            """2. plot LNO detector data, corrected for offset"""
    
            if "Spectra" in whatToPlot:
                plt.figure(figsize=(FIG_X, FIG_Y))
                for detectorRow in detectorData:
                    plt.plot(wavenumbers,detectorRow)
    #                plt.plot(detectorRow)
                plt.title(hdf5Filenames[fileIndex]+": counts vs. time")
                plt.xlabel("Wavenumbers (cm-1)")
                plt.ylabel("Detector counts (background subtraction on)")
                plt.annotate("Min observing altitude = %0.1fkm\nMax observing altitude = %0.1fkm"\
                             %(np.min(geometryPointsAlts[:, chosenIndices, :]), np.max(geometryPointsAlts[:, chosenIndices, :])), [0.02, 0.80], xycoords="axes fraction")
                if SAVE_FIGS:
                    plt.savefig(BASE_DIRECTORY+os.sep+hdf5Filenames[fileIndex]+"_spectra_bin_%i.png" %chosenBinNumber)
        
        
            """3. plot LNO detector data - mean of detector centre (80 pixels) vs observing altitude"""
            
            if "Mean Counts" in whatToPlot:
                ax1.scatter(geometryPointsAlts[0, chosenIndices, 0], np.mean(detectorData[:,160:240], axis=1), label="Detector bin=%i"%chosenBinNumber, alpha=0.3)
                ax1.set_title(hdf5Filenames[fileIndex]+": mean order counts vs. time")
                ax1.set_xlabel("Tangent Altitude at Centre of FOV (km)")
                ax1.set_ylabel("Mean of detector columns 160-240")
                if SAVE_FIGS:
                    fig1.savefig(BASE_DIRECTORY+os.sep+hdf5Filenames[fileIndex]+"_mean_detector_centre_vs_altitude_bin_%i.png" %chosenBinNumber)
        
        
        hdf5File.close()


def getAltitudeRange(hdf5Files, hdf5Filenames, altitudeStart, altitudeEnd):

    matchingSpectra = []
    #loop through files
    for fileIndex,hdf5File in enumerate(hdf5Files):
    
        hdf5File = hdf5Files[fileIndex]
            
        geometryPointsAlts = np.mean(hdf5File["Geometry/Point0/TangentAltAreoid"][...], axis=1)
#        bins = get_dataset_contents(hdf5File, "Bins")[0]
#        return geometryPointsAlts
        #get detector data
        detectorDataAll = hdf5File["Science/Y"][...] #get data
    
        #get wavenumber scale
        wavenumbersAll = hdf5File["Science/X"][...] #get spectral axis
        wavenumbers = wavenumbersAll[0, :]

        matchingIndices = (altitudeStart < geometryPointsAlts) & (altitudeEnd > geometryPointsAlts)
        for matchingIndex,value in enumerate(matchingIndices):
            if value:
                matchingSpectra.append(detectorDataAll[matchingIndex, :])
        nSpectra = len(matchingSpectra)
#        hdf5File.close()
            
    return wavenumbers, np.mean(np.asfarray(matchingSpectra), axis=0), nSpectra
#    return wavenumbers, matchingSpectra


#make lists of filenames, file levels and filepaths
hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)



if title == "plot geometry each file":
    whatToPlot = ["Geometry"] #plot everything - groundtracks, spectra and mean counts vs altitude
    plotBins(hdf5Files, hdf5Filenames, whatToPlot)


if title == "plot spectra each file":
    whatToPlot = ["Spectra","Mean Counts"] #plot all except geometry
    plotBins(hdf5Files, hdf5Filenames, whatToPlot)

if title == "plot mean counts each file":
    whatToPlot = ["Mean Counts"] #plot all except geometry
    plotBins(hdf5Files, hdf5Filenames, whatToPlot)

if title == "all file spectra for altitude bins":
    plt.figure(figsize=(FIG_X, FIG_Y))
    for altitudeIndex, altitudeStart in enumerate(range(45, 105, 5)):
        altitudeEnd = altitudeStart + 5.0
        wavenumbers, spectrum, nSpectra = getAltitudeRange(hdf5Files, hdf5Filenames, altitudeStart, altitudeEnd)
        offset = altitudeIndex * 5.0
        plt.plot(wavenumbers, np.zeros(320) + offset, "k")
        plt.plot(wavenumbers, spectrum + offset, label="%i-%ikm (%i spectra)" %(altitudeStart, altitudeEnd, nSpectra))
    plt.legend()
        
    
if title == "check altitudes":
    
    plt.figure(figsize=(FIG_X+4, FIG_Y))
    plt.title("Limb tangent heights for all limb measurements of order 164")
    plt.xlabel("Observation month")
    plt.ylabel("Limb point tangent height (km)")
    plt.ylim((0, 150))
    plt.xticks([])
    nColours = len(hdf5Filenames)
#    cmap = plt.get_cmap('Spectral')
#    colours = [cmap(i) for i in np.arange(nColours)/nColours]
    colours = ["gray"] * 1000
    
    old_offset = -999
    
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        
        year = np.float32(hdf5_filename[0:4])
        month = np.float32(hdf5_filename[4:6])
        
        offset = ((year - 2018.0) + (month - 4.0) / 10.0) * 3600
        
        if offset != old_offset:
#            plt.axvline(offset, color="k", linestyle="--")
            plt.text(offset+10, 10, "%0d-%02d" %(year, month))
            old_offset = offset
        
        bin_start = hdf5_file["Science/Bins"][:, 0]
        
        first_bin = sorted(set(bin_start))[0]
        last_bin = sorted(set(bin_start))[-1]
        centre_bin = sorted(set(bin_start))[6] #note: just above centre
        
        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
        
        alts_first = alts[bin_start == first_bin]
        plt.plot(np.arange(len(alts_first))+offset, alts_first, color=colours[file_index])
        
        alts_last = alts[bin_start == last_bin]
        plt.plot(np.arange(len(alts_first))+offset, alts_last, color=colours[file_index])
        
        plt.fill_between(np.arange(len(alts_first))+offset, alts_last, alts_first, color=colours[file_index], alpha=0.3)
        
        if SAVE_FIGS:
            plt.savefig(os.path.join(BASE_DIRECTORY,"all_limb_tangent_heights.png"))
        
        
        