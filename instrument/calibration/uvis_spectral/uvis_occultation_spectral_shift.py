# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:53:09 2022

@author: iant

UVIS SPECTRAL CALIBRATION SHIFT
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.get_nearest_index import get_nearest_index
from tools.spectra.baseline_als import baseline_als
from tools.plotting.colours import get_colours
from tools.general.get_minima_maxima import get_local_minima

from tools.spectra.fft_zerofilling import fft_hr_nu_spectrum


from uvis_meftah_solar_spectrum import uvis_solar_superhr



regex = re.compile("20..(01|02|03|04|05|08|12).._......_.*_UVIS_I")
file_level = "hdf5_level_1p0a"
min_altitude = 100. #mean all spectra above this


def make_solar_spectra_obs_dict(regex, file_level, min_altitude):

    _, h5s, _ = make_filelist(regex, file_level, open_files=False)
    spectra_dict = {}
    
    print("Collecting spectra from files")
    for i, h5 in enumerate(h5s):
        
        if np.mod(i, 100) == 0:
            print("%i/%i" %(i, len(h5s)))
        
        h5_f = open_hdf5_file(h5)
    
        mode = h5_f["Channel/AcquisitionMode"][0]
        #if not full frame, skip file
        if mode != 0:
            continue
        
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        x = h5_f["Science/X"][0, :]
        
        if len(x) != 1024:
            continue
        ix = get_nearest_index(min_altitude, alts)
        
        
        sun_spectra = h5_f["Science/YUnmodified"][ix:, :]
        spectrum = np.nanmean(sun_spectra, axis=0)
        
        
        if np.sum(np.isnan(spectrum)) > 0:
            continue
        
        # if np.median(spectrum) < 0.5:
        #     continue
        
        cont = baseline_als(spectrum)
        # dt= datetime.strptime(h5[0:15], "%Y%m%d_%H%M%S")
        
        spectra_dict[h5] = {"spectrum":spectrum/cont}
    
        # plt.plot(x, spectrum/cont, label=h5)
        
    return spectra_dict, x
    


def plot_division(spectra_dict, x):
    """plot raw spectra and division of each spectrum by the mean spectrum to see sub-pixel differences"""
    
    plt.figure(figsize=(12, 6))
    plt.title("UVIS occultation spectra above the atmosphere")
    plt.xlabel("Wavelength nm")
    plt.ylabel("Continuum removed counts")
    plt.grid()
    spectra = np.zeros([len(spectra_dict.keys()), 1024])
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        spectra[i, :] = spectrum
        
    mean_spectrum = np.mean(spectra, axis=0)
    plt.plot(x, mean_spectrum, color="k")

    colours = get_colours(len(spectra))
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        
        plt.plot(x, spectrum, color=colours[i], label=h5)

    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.title("UVIS occultation spectra above the atmosphere")
    plt.xlabel("Wavelength nm")
    plt.ylabel("Spectrum / mean spectrum")
    plt.grid()
    colours = get_colours(len(spectra))
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        
        plt.plot(x, spectrum/mean_spectrum, color=colours[i], label=h5)
    plt.legend()



def write_solar_line_file(spectra_dict, x):
    """write solar line positions to text file"""
    
    print("Finding solar line positions and writing to file")
    with open("uvis_solar_lines.tsv", "w") as f:
        f.write("Filename\tAbsorption wavelength\n")
    
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        
        uvis_x_fft, uvis_y_fft = fft_hr_nu_spectrum(x, spectrum, 100)
        
        
        local_minima = get_local_minima(uvis_y_fft)
        good_local_minima = local_minima[np.where(uvis_y_fft[local_minima] < 0.8)[0]]
        
        uvis_x_mins = uvis_x_fft[good_local_minima]
        
        for uvis_x_min in uvis_x_mins:
        
            with open("uvis_solar_lines.tsv", "a") as f:
                f.write(f"{h5}\t{uvis_x_min:#0.4f}\n")
        




def write_mean_solar_line_file(spectra_dict, x):

    #mean spectrum by month/year
    sorted_spectra = {"201804":[], "201805":[], "201808":[], "201812":[], \
                    "201901":[], "201902":[], "201903":[], "201904":[], "201905":[], "201908":[], "201912":[], \
                    "202201":[], "202202":[], "202203":[], "202204":[], "202205":[]}
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        sorted_spectra[h5[0:6]].append(spectrum)
    
    mean_spectra = {}
    
    for year in sorted_spectra.keys():
        if len(sorted_spectra[year]) > 1:
            # plt.figure(figsize=(18, 6))
            # plt.title(year)
            # plt.plot(np.array(sorted_spectra[year]).T)
            mean_spectra[year] = np.mean(np.array(sorted_spectra[year]), axis=0)

    with open("uvis_mean_spectrum_solar_lines.tsv", "w") as f:
        f.write("Filename\tAbsorption wavelength\n")

    
    plt.figure(figsize=(12, 6))
    for year in mean_spectra.keys():
        spectrum = mean_spectra[year]
        
        uvis_x_fft, uvis_y_fft = fft_hr_nu_spectrum(x, spectrum, 100)
        
        plt.plot(x, spectrum, label=year)
        plt.plot(uvis_x_fft, uvis_y_fft, linestyle="--")
    
        local_minima = get_local_minima(uvis_y_fft)
        good_local_minima = local_minima[np.where(uvis_y_fft[local_minima] < 0.8)[0]]
        
        uvis_x_mins = uvis_x_fft[good_local_minima]
        
        for uvis_x_min in uvis_x_mins:
        
            with open("uvis_mean_spectrum_solar_lines.tsv", "a") as f:
                f.write(f"{year}\t{uvis_x_min:#0.4f}\n")
                
    plt.legend()
    
    return mean_spectra
    # plt.figure(figsize=(12, 6))
    # plt.plot(x, mean_spectrum["2018"]/mean_spectrum["2019"])
    # plt.plot(x, mean_spectrum["2022"]/mean_spectrum["2019"])

# spectra_dict, x = make_solar_spectra_obs_dict(regex, file_level, min_altitude)
# plot_division(spectra_dict, x)
# mean_spectra = write_mean_solar_line_file(spectra_dict, x)


# write_solar_line_file(spectra_dict, x)
# stop()


solar_lines = []
dts = []
with open("uvis_mean_spectrum_solar_lines.tsv", "r") as f:
    _ = f.readline()
    for line in f.readlines():
        # dts.append(datetime.strptime(line.split("\t")[0][0:15], "%Y%m%d_%H%M%S"))
        dts.append(datetime.strptime(line.split("\t")[0][0:6], "%Y%m"))
        solar_lines.append(np.float32(line.split("\t")[1]))
        
solar_lines = np.array(solar_lines)

# # line_centre = 


nm_fft, solar_hr_fft = uvis_solar_superhr()
local_minima = get_local_minima(solar_hr_fft)
good_local_minima = local_minima[np.where(solar_hr_fft[local_minima] < 0.8)[0]]

#get wavelengths of solar minima
solar_x_mins = nm_fft[good_local_minima]

#remove points that are too close to one another
solar_x_deltas = np.diff(solar_x_mins)
# a = [i for i in solar_x_mins if ]
ix = []
for index, solar_x_delta in enumerate(solar_x_deltas):
    if solar_x_delta < 1:
        ix.append(index)
        ix.append(index+1)
    
solar_x_mins2 = []
for index, solar_x_min in enumerate(solar_x_mins):
    if index not in ix:
        solar_x_mins2.append(solar_x_min)



# plt.figure()
# plt.plot(nm_fft, solar_hr_fft)
# for solar_x_min in solar_x_mins2:
#     plt.axvline(x=solar_x_min, color="k")





colours = get_colours(len(solar_x_mins2))

plt.figure()
for index, solar_x_min in enumerate(solar_x_mins2):
    idx = np.where((solar_lines > solar_x_min - 0.2) & (solar_lines < solar_x_min + 0.2))[0]
    datetimes = [dts[i] for i in idx]
    
    if len(idx) > 1:
        x = [(datetime - datetimes[0]).total_seconds() for datetime in datetimes]
        y = solar_lines[idx] - np.mean(solar_lines[idx])
        # plt.scatter(datetimes, solar_lines[idx] - solar_x_min, label="%0.3fnm" %solar_x_min)
        plt.scatter(datetimes, y, label="%0.3fnm" %solar_x_min)
        polyfit = np.polyfit(x, y, 1)
        y_fit = np.polyval(polyfit, [x[0], x[-1]])
        plt.plot([datetimes[0], datetimes[-1]], y_fit)

plt.legend()
