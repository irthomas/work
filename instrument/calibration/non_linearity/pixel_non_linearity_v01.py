# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:46:22 2019

@author: iant
"""

import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from scipy import stats
import scipy.optimize

#import bisect
from scipy.optimize import curve_fit
#from mpl_toolkits.basemap import Basemap

from datetime import datetime
#from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
#import matplotlib.cm as cm
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import struct
from matplotlib.backends.backend_pdf import PdfPages


from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, getFile, makeFileList, printFileNames
from hdf5_functions_v03 import getFilesFromDatastore
from analysis_functions_v01b import write_log
from filename_lists_v01 import getFilenameList

#from plot_occultations_v02 import joinTransmittanceFiles, polynomialFit

if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
    print("Running on windows")
    import spiceypy as sp
#    from plot_simulations_v01 import findSimulations, getSimulationData, getOrderSimulation



#SAVE_FIGS = False
SAVE_FIGS = True

SAVE_FILES = False
#SAVE_FILES = True

####CHOOSE FILENAMES######
title = ""
obspaths = []
fileLevel = ""


"""plot solar lines in order"""
title = "convolve solar lines"


"""make CH4 corrected SNR spectra for paper"""
#fileLevel = "hdf5_level_0p3a"
#obspaths = [
#        "20180507_050656_0p3a_SO_1_I_136",
#        "20180620_103219_0p3a_SO_1_I_134"
#        ]
#obspaths = ["*201809*_0p3a_SO_*I_134"]

#title = "detector correction"
#obspaths = getFilenameList(title)

"""correct micro non linearity"""
#get data from 0p1a files before other detector corrections are applied
#title = "pixel non linearity"
#fileLevel = "hdf5_level_0p1a"
#obspaths = [
#        "*201812*_0p1a_SO_1",
#        "20181205_030659_0p1a_SO_1"

#    "20181202_093756_0p1a_SO_1", #133
#    "20181205_030659_0p1a_SO_1", #135
#    "20181206_103504_0p1a_SO_1", #134
#    "20181207_014349_0p1a_SO_1", #132
#    "20181207_212339_0p1a_SO_1", #133
#    "20181209_092342_0p1a_SO_1", #134
#    "20181210_165147_0p1a_SO_1", #135
#    "20181210_204753_0p1a_SO_1", #134
#    "20181211_222149_0p1a_SO_1", #134
#    "20181212_112839_0p1a_SO_1", #132
#    "20181213_070753_0p1a_SO_1", #133
#    "20181214_111907_0p1a_SO_1", #134
#    "20181215_164827_0p1a_SO_1", #135
#    "20181216_201945_0p1a_SO_1", #134
#    "20181217_092315_0p1a_SO_1", #132
#    "20181218_070002_0p1a_SO_1", #133
#    "20181220_164221_0p1a_SO_1", #134
#    "20181223_045314_0p1a_SO_1", #135
#    "20181225_104034_0p1a_SO_1", #134
#    "20181226_073237_0p1a_SO_1", #132
#    "20181227_185428_0p1a_SO_1", #133
#            ]








#hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
#binIndices = [0,1,2,3]
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#
#obsDicts = []
#for binIndex in binIndices:
#    obsDicts.append(joinTransmittanceFiles(hdf5Files[0], hdf5Filenames[0], binIndex, silent=False, top_of_atmosphere=60.0))
#
#fig, ax1 = plt.subplots()
#for binIndex in binIndices:
##    ax1.plot(obsDicts[binIndex]["alt"], obsDicts[binIndex]["y"][:,160], label="Bin %i" %binIndex, color=colors[binIndex]) #plot centre pixel
#    for spectrumIndex,spectrum in enumerate(obsDicts[binIndex]["y"]):
#        if obsDicts[binIndex]["alt"][spectrumIndex] < 8.0:
#            ax1.plot(spectrum, label=obsDicts[binIndex]["alt"][spectrumIndex], color=colors[binIndex])
#plt.legend()





def applyFilter(data, plot=False):
    from scipy.signal import butter, lfilter
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    order = 3
    fs = 15.0       # sample rate, Hz
    cutoff = 0.5#3.667  # desired cutoff frequency of the filter, Hz
    
    
    
    pixel_in = np.arange(len(data))

    dataFit = butter_lowpass_filter(data, cutoff, fs, order)
    
    pixelInterp = np.arange(pixel_in[0], pixel_in[-1]+1.0, 0.1)
    dataInterp = np.interp(pixelInterp, pixel_in, data)
    dataFitInterp = np.interp(pixelInterp, pixel_in, dataFit)
    
    firstPoint = 200
    pixelInterp = pixelInterp[firstPoint:]
    dataInterp = dataInterp[firstPoint:]
    dataFitInterp = dataFitInterp[firstPoint:]
    
    nPoints = len(dataInterp)
    
    
    chi = np.asfarray([np.sum((dataInterp[0:(nPoints-index)] - dataFitInterp[index:(nPoints)])**2) / (nPoints - index) \
                       for index in np.arange(0, 1000, 1)])
    minIndex = np.argmin(chi)-1

    if plot:
        plt.subplots(figsize=(14,10), sharex=True)
        plt.subplot(2, 1, 1)
        plt.plot(pixelInterp, dataInterp, 'b-', label='data')
        plt.plot(pixelInterp[0:(nPoints-minIndex)], dataFitInterp[minIndex:(nPoints)], 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    
    x = pixelInterp[0:(nPoints-minIndex)]
    yfit = dataFitInterp[minIndex:(nPoints)]
    y = dataInterp[0:(nPoints-minIndex)] / yfit
    
    
    
    if plot:
        plt.subplot(2, 1, 2)
        plt.plot(x, y, label="residual")
        plt.ylim([0.95, 1.02])
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return x, yfit, y


def spectralCalibration(order, temperature):
    
    temperatureCoeffs = np.asfarray([0.0, -0.7299039, -6.267734])
    spectralCoeffs = np.asfarray([1.751279e-8, 0.00055593, 22.473422])
    pixel1 = np.polyval(temperatureCoeffs, temperature)
    wavenumbers = np.polyval(spectralCoeffs, np.arange(320) + pixel1) * order
    return wavenumbers, pixel1


def check_solar_lines(order, pixel_shift):
    
    from test_justin_simulation_v02 import NOMAD_sim
    
    sim = NOMAD_sim(order=order, pixel_shift=pixel_shift)
    solarConvolved = sim.I0_p
    
    solarFitted = applyFilter(solarConvolved)
    solarResidual = np.interp(range(20, 300),  solarFitted[0], solarFitted[2])
    
    return solarResidual


def expFuncDown(x, a, b, c):
    return 2.0 - np.exp(-(x+a)/b)+c

def expFuncUp(x, a, b, c):
    return np.exp(-(x+a)/b)+c


if title == "pixel non linearity":            
    hdf5Files, hdf5Filenames, _ = makeFileList(obspaths, fileLevel, silent=True)
    
    goodFilenames = []
    goodFilenamesOrders = []
    
    
    
    
    plot_type = []#["fits"]
    
    colours = {132:'#1f77b4', 133:'#ff7f0e', 134:'#2ca02c', 135:'#d62728', 136:'#9467bd'}
    cmap = plt.get_cmap('jet')
    coloursSpectral = [cmap(i) for i in np.arange(10)/10.0] * 10
    
    bins = np.logspace(6,13,base=np.e,num=30)
    binsMean = np.mean([bins, np.roll(bins, 1)], axis=0)
    binnedResidualsAll = [ [] for _ in range(30)]
    pdfOut = PdfPages('test.pdf')           
    
    
    #pixels = [100,101,102]
    pixels = [100]
    for pixel in pixels:
        fig, ax = plt.subplots(figsize=(11.69,8.27), dpi=100)
        colourLoop = -1
    
    
        for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
            detectorData = hdf5_file["Science/Y"][...]
            aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
            sensor1Temperature = hdf5_file["Housekeeping/SENSOR_1_TEMPERATURE_SO"][...]
            measurementTemperature = np.mean(sensor1Temperature[2:10])
            
            singleAotfFrequencies = [17566.0, 17712.0, 17859.0, 18005.0, 18152.0]
            singleDiffractionOrders = [132, 133, 134, 135, 136]
            if aotfFrequency[0] in singleAotfFrequencies and aotfFrequency[0] == aotfFrequency[1] == aotfFrequency[2] == aotfFrequency[3]:
                diffractionOrder = singleDiffractionOrders[singleAotfFrequencies.index(aotfFrequency[0])]
                print("File %s is good (%i)" %(hdf5_filename, diffractionOrder))
                
                wavenumbers, pixelShift = spectralCalibration(diffractionOrder, measurementTemperature)
                reducedWavenumbers = wavenumbers[20:300]
                
                goodFilenames.append(hdf5_filename)
                goodFilenamesOrders.append(diffractionOrder)
            
                #TODO: check this is correct
                detectorDataDark = np.asfarray([np.tile(detectorData[index,:,:], [5, 1, 1]) for index in range(5, len(detectorData[:,0,0]), 6)])
                detectorDataLight = np.asfarray([detectorData[index:(index+5),:,:] for index in range(0, len(detectorData[:,0,0]), 6)])
                
                detectorSubtracted = detectorDataLight - detectorDataDark
                detectorFlattened = detectorSubtracted.reshape([-1, 4, 320])
                
                detectorNormalised = [(detectorFlattened[:, index, :] - np.min(detectorFlattened[:, index, :], axis=0)) / (np.max(detectorFlattened[:, index, :], axis=0) - np.min(detectorFlattened[:, index, :], axis=0)) for index in range(4)]
    #            detectorCounts = [detectorFlattened[:, index, :] for index in range(4)]
        
                #take data only during occultation
                if detectorNormalised[0][0, 200] > detectorNormalised[0][-1, 200]: #if ingress
                    atmosphericIndices = list(range((detectorNormalised[0][:, 200] < 0.95).argmax(), (detectorNormalised[0][:, 200] > 0.02).argmin()))
                else: #if egress
                    atmosphericIndices = list(range((detectorNormalised[0][:, 200] > 0.02).argmax(), (detectorNormalised[0][:, 200] < 0.95).argmin()))
        
        
                #apply filter 
                flatBin = detectorFlattened[atmosphericIndices,0,:]
                fittedBin = [applyFilter(spectrum) for spectrum in flatBin]
        
                #next part - interpolate to find fitted pixel value, compare to real value throughout occultation
                fittedBinCounts = np.asfarray([np.interp(range(20, 300),  fittedBin[index][0], fittedBin[index][1]) for index in range(len(fittedBin))])
                fittedBinResiduals = np.asfarray([np.interp(range(20, 300),  fittedBin[index][0], fittedBin[index][2]) for index in range(len(fittedBin))])
        
               
                if "fits" in plot_type:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    plotIndices = range(0, len(flatBin[:,0]), 10)
                    for plotIndex in plotIndices:
                        ax1.plot(wavenumbers, flatBin[plotIndex])
                        ax1.plot(reducedWavenumbers, fittedBinCounts[plotIndex, :])
                        ax2.plot(reducedWavenumbers, fittedBinResiduals[plotIndex, :])
        
    #            solarResidual = check_solar_lines(diffractionOrder, pixelShift)
    #            ax2.plot(reducedWavenumbers, solarResidual, "k--")
                
                if True:#diffractionOrder == 134:
                    colourLoop += 1
                    #fit to exponential curve from Frank
                    try:
                        popt, pcov = curve_fit(expFuncDown, fittedBinCounts[:, pixel], fittedBinResiduals[:, pixel], p0=[-1,1e3,1])
                    except RuntimeError:
                        continue
                    #if fit works, plot residual vs counts for data and fitted line, both normalised to 1
                    fittedFunction = expFuncDown(fittedBinCounts[:, pixel], popt[0], popt[1], popt[2])
        #            ax.plot(fittedBinCounts[:, pixel], fittedBinResiduals[:, pixel]/np.max(fittedFunction), label="%s (%i)" %(hdf5_filename, diffractionOrder), color=colours[diffractionOrder], alpha=0.5)
                    ax.scatter(fittedBinCounts[:, pixel], fittedBinResiduals[:, pixel]/np.max(fittedFunction), label="%s (%i)" %(hdf5_filename, diffractionOrder), color=coloursSpectral[colourLoop], marker="*")
        #            ax.plot(fittedBinCounts[:, pixel], fittedFunction/np.max(fittedFunction), linestyle="--", color=colours[diffractionOrder], alpha=0.5)
                    
                    digitised = np.digitize(fittedBinCounts[:, pixel], bins)
                    binnedResiduals = [fittedBinResiduals[digitised == i, pixel] for i in range(len(bins))]
                    
                    for residualIndex in range(len(binnedResiduals)):
                        if len(binnedResiduals[residualIndex])>0:
                            for value in binnedResiduals[residualIndex]:
                                binnedResidualsAll[residualIndex].append(value/np.max(fittedFunction))
                        
                    
                    binnedResidualsCounts = np.asfarray([binsMean[index] for index, residuals in enumerate(binnedResiduals) if len(residuals)>0])
                    binnedResidualsMean = np.asfarray([np.mean(residuals) for residuals in binnedResiduals if len(residuals)>0])
                    binnedResidualsStd = np.asfarray([np.std(residuals) for residuals in binnedResiduals if len(residuals)>0])
                    
                    ax.errorbar(binnedResidualsCounts, binnedResidualsMean/np.max(fittedFunction), yerr=binnedResidualsStd, fmt="o", capsize=5, color=coloursSpectral[colourLoop])
                    ax.plot(binnedResidualsCounts, binnedResidualsMean/np.max(fittedFunction), color=coloursSpectral[colourLoop], linestyle="--")
                    vlines = [ax.axvline(x=eachbin) for eachbin in bins]
                    ax.set_xscale("log")
        
        plt.legend()
        
        binnedResidualsAllCounts = np.asfarray([binsMean[index] for index, residuals in enumerate(binnedResidualsAll) if len(residuals)>0])
        binnedResidualsAllMean = np.asfarray([np.mean(residuals) for residuals in binnedResidualsAll if len(residuals)>0])
        binnedResidualsAllStd = np.asfarray([np.std(residuals) for residuals in binnedResidualsAll if len(residuals)>0])
        
        ax.errorbar(binnedResidualsAllCounts, binnedResidualsAllMean, yerr=binnedResidualsAllStd, fmt="o", capsize=5, color="k")
        ax.plot(binnedResidualsAllCounts, binnedResidualsAllMean, color="k", linestyle="--")
        
        ax.errorbar(binnedResidualsAllCounts, binnedResidualsAllMean-0.05, yerr=binnedResidualsAllStd, fmt="o", capsize=5, color="k")
        
        pdfOut.savefig()
    
    pdfOut.close()
    #fit curve to all points in all files
    #try:
    #    poptAll, pcovAll = curve_fit(expFuncDown, binnedResidualsAllCounts, binnedResidualsAllMean, p0=[-1,1e3,1])
    #except RuntimeError:
    #    print("error")
    #fittedFunction = expFuncDown(binnedResidualsAllCounts, popt[0], popt[1], popt[2])
    #ax.plot(binnedResidualsAllCounts, fittedFunction/np.max(fittedFunction), linestyle="--", color="k")



if title == "convolve solar lines":
    #plot all convolved solar lines
#    plt.figure()
#    singleAotfFrequencies = [17566.0, 17712.0, 17859.0, 18005.0, 18152.0]
#    singleDiffractionOrders = [132, 133, 134, 135, 136]
    singleDiffractionOrders = [134]
    for order in singleDiffractionOrders:
        solarLines = check_solar_lines(order, 0)
        plt.plot(range(20, 300), solarLines, label="Order %i" %order)
    plt.legend()


#list relevant files
#for hdf5_file, hdf5_filename in zip(hdf5Files, hdf5Filenames):
#    aotfFrequency = hdf5_file["Channel/AOTFFrequency"][...]
#    singleAotfFrequencies = [17566.0, 17712.0, 17859.0, 18005.0, 18152.0]
#    singleDiffractionOrders = [132, 133, 134, 135, 136]
#    if aotfFrequency[0] in singleAotfFrequencies and aotfFrequency[0] == aotfFrequency[1] == aotfFrequency[2] == aotfFrequency[3]:
#        diffractionOrder = singleDiffractionOrders[singleAotfFrequencies.index(aotfFrequency[0])]
#        print(""""%s", #%i""" %(hdf5_filename, diffractionOrder))

    

    
    
    