# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:24:37 2018

@author: iant
"""

from plot_occultations_v02 import joinTransmittanceFiles
from hdf5_functions_v03 import makeFileList


import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

#
#hdf5Files, hdf5Filenames, _ = makeFileList(["20180421_202111_0p3a_SO_1_E_134"], "hdf5_level_0p3a", silent=True)
#hdf5Files, hdf5Filenames, _ = makeFileList(["20180618_001805_0p3a_SO_1_E_134"], "hdf5_level_0p3a", silent=True)
hdf5Files, hdf5Filenames, _ = makeFileList(["20180613_073640_0p3a_SO_2_I_134"], "hdf5_level_0p3a", silent=True)
#
#
obsDict = joinTransmittanceFiles(hdf5Files[0], hdf5Filenames[0], 3, silent=False, top_of_atmosphere=60.0)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def applyFilter(data, plot=False):
    # Filter requirements.
    order = 3
    fs = 15.0       # sample rate, Hz
    cutoff = 0.5#3.667  # desired cutoff frequency of the filter, Hz
    
    if plot:
        plt.subplots(figsize=(14,10), sharex=True)
        plt.subplot(2, 1, 1)
    
    
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    pixel_in = np.arange(len(obsDict["y_raw"][150,:]))
    wavenumber_in = obsDict["x"]
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    
    # Filter the data, and plot both the original and filtered signals.
    dataFit = butter_lowpass_filter(data, cutoff, fs, order)
    
    pixelInterp = np.arange(pixel_in[0], pixel_in[-1]+1.0, 0.1)
    dataInterp = np.interp(pixelInterp, pixel_in, data)
    dataFitInterp = np.interp(pixelInterp, pixel_in, dataFit)
    wavenumberInterp = np.interp(pixelInterp, pixel_in, wavenumber_in)
    
    
    
    firstPoint = 200
    pixelInterp = pixelInterp[firstPoint:]
    dataInterp = dataInterp[firstPoint:]
    dataFitInterp = dataFitInterp[firstPoint:]
    wavenumberInterp = wavenumberInterp[firstPoint:]
    
    nPoints = len(dataInterp)
    
    
    #chi = [chisquare(data[0:(319-index)] - y[index:319])[0]**2 for index in np.arange(0, 20, 1)]
    #minIndex = np.argmin(chi)-1
    
    chi = np.asfarray([np.sum((dataInterp[0:(nPoints-index)] - dataFitInterp[index:(nPoints)])**2) / (nPoints - index) \
                       for index in np.arange(0, 1000, 1)])
    minIndex = np.argmin(chi)-1
    
    
    
    if plot:
        plt.plot(pixelInterp, dataInterp, 'b-', label='data')
        plt.plot(pixelInterp[0:(nPoints-minIndex)], dataFitInterp[minIndex:(nPoints)], 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    
    x = pixelInterp[0:(nPoints-minIndex)]
    v = wavenumberInterp[0:(nPoints-minIndex)]
    y = dataInterp[0:(nPoints-minIndex)]/dataFitInterp[minIndex:(nPoints)]
    
    
    
    if plot:
        plt.subplot(2, 1, 2)
        plt.plot(x, y, label="residual")
        plt.ylim([0.95, 1.02])
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return x, v, y

def writeCsvFile(file_name, lines_to_write):
    """function to write to csv file"""
    logFile = open(file_name, "w")
    for line_to_write in lines_to_write:
        logFile.write(line_to_write+'\n')
    logFile.close()


#write to csv file
linesToWrite = ["Altitude (km), "+"".join(["Pixel%i, " %i for i in range(320)])]
for altitude, spectrum in zip(obsDict["alt"], obsDict["y"]):
    linesToWrite.append("%0.2f, " %altitude + "".join(["%0.5f, " %i for i in spectrum]))
writeCsvFile("20180613_073640_0p3a_SO_2_I_134.csv", linesToWrite)


#high_altitude_indices_to_plot = [150, 200, 232]
high_altitude_indices_to_plot = [150, 200, 230]

fig, ax1 = plt.subplots()
for i in list(range(25,60,5))+high_altitude_indices_to_plot: 
    ax1.plot(obsDict["y"][i,:], label=obsDict["alt"][i])
plt.legend()

#ax2 = ax1.twinx()
#ax2.plot(obsDict["y_raw"][50,:])
#for index in high_altitude_indices_to_plot:
#    ax2.plot(obsDict["y_raw"][index,:])

#stop()
fig, ax1 = plt.subplots()
ax1.plot(obsDict["y_raw"][:,200])
ax2 = ax1.twinx()
ax2.plot(obsDict["y_dark"][:,200])




applyFilter(obsDict["y_raw"][50,:], plot=True)


indexStart=40
indexEnd=200

filtered1 = [applyFilter(data) for data in obsDict["y_raw"][indexStart:indexEnd,:]]
nPixels = np.min([len(spectrum) for _,_,spectrum in filtered1])
filtered = np.asfarray([spectrum[0:nPixels] for _,_,spectrum in filtered1])
wavenumber = np.asfarray([wavenumbers[0:nPixels] for _,wavenumbers,_ in filtered1])[0,:]

meanFiltered = np.mean(filtered, axis=0)
cmap = plt.get_cmap('jet')
colours = [cmap(i) for i in np.arange(len(filtered))/len(filtered)]


labels = ["%0.1fkm, signal=%0.0f" %(alt, signal) for alt, signal in zip(obsDict["alt"][indexStart:indexEnd],obsDict["y_raw"][indexStart:indexEnd,200])]

plt.figure()
for index, spectrum in enumerate(filtered):
    plt.plot(wavenumber, spectrum, color=colours[index], label=labels[index])
plt.legend()

plt.figure()
for index, spectrum in enumerate(filtered):
    plt.plot(wavenumber, spectrum/meanFiltered, color=colours[index], label=labels[index])
#plt.legend()


