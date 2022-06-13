# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:53:09 2022

@author: iant

UVIS SPECTRAL CALIBRATION SHIFT
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tools.file.paths import paths
from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.get_nearest_index import get_nearest_index
from tools.spectra.baseline_als import baseline_als
from tools.plotting.colours import get_colours
from tools.general.get_minima_maxima import get_local_minima, get_local_maxima

# from tools.spectra.fft_zerofilling import fft_hr_nu_spectrum
from tools.spectra.savitzky_golay import savitzky_golay
from tools.spectra.fit_gaussian_absorption import fit_gauss, make_gauss

from instrument.calibration.uvis_spectral.uvis_meftah_solar_spectrum import uvis_solar_sg



regex = re.compile("20......_......_.*_UVIS_D")
file_level = "hdf5_level_1p0a"
max_incidence_angle = 30.
temperature_field = "Housekeeping/TEMP_2_CCD"

#list of meftah sg convolved local minima and maxima wavelengths
# good_minima_nm = {227.050:[], 274.5:[], 280.0:[], 285.3:[], 309.7:[], 323.3:[], 344.4:[], 383.3:[], 393.5:[], 396.95:[], 430.7:[], 445.95:[], 453.2:[], 470.6:[], 527.05:[], 586.3:[], 589.55:[], 616.8:[], 632.05:[]}
# good_maxima_nm = {242.6:[], 277.05:[], 290.7:[], 311.2:[], 359.85:[], 395.4:[], 436.3:[], 451.:[], 477.95:[], 490.25:[], 498.6:[]}

good_minima_nm = {
    227.05: [56, 57, 58, 59, 60, 61, 62], 
    274.5: [159, 160, 161, 162, 163], 
    280.0: [169, 170, 171, 172, 173, 174, 175, 176], 
    # 285.3: [182, 183, 184, 185, 186], #?
    # 309.7: [235, 236, 237, 238], #?
    323.3: [264, 265, 266, 267, 268], 
    344.4: [310, 311, 312, 313, 314], 
    383.3: [395, 396, 397, 398, 399, 400], 
    393.5: [418, 419, 420, 421, 422], 
    396.95: [425, 426, 427, 428, 429, 430], 
    430.7: [499, 500, 501, 502, 503, 504], 
    445.95: [534, 535, 536, 537, 538, 539], 
    453.2: [550, 551, 552, 553, 554, 555], 
    470.6: [590, 591, 592, 593, 594], 
    527.05: [719, 720, 721, 722, 723], 
    589.55: [865, 866, 867, 868, 869], 
    616.8: [930, 931, 932, 933, 934], 
    }


good_maxima_nm = {
    242.6: [91, 92, 93, 94, 95, 96], 
    277.05: [163, 164, 165, 166, 167, 168, 169], 
    290.7: [193, 194, 195, 196, 197, 198], 
    # 311.2: [238, 239, 240, 241, 242], #?
    # 359.85: [344, 345, 346, 347, 348], #?
    # 436.3: [513, 514, 515, 516, 517], #?
    # 451.0: [545, 546, 547, 548, 549, 550], #?
    # 477.95: [607, 608, 609, 610, 611], #?
    # 490.25: [635, 636, 637], 
    }

good_minmax_nm_d = {**good_minima_nm, **good_maxima_nm}
good_minmax_nm = sorted(list(good_minmax_nm_d.keys()))


def make_solar_spectra_obs_dict(regex, file_level, max_incidence_angle):

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
        
        incidence = h5_f["Geometry/Point0/IncidenceAngle"][:, 0]
        x = h5_f["Science/X"][0, :]
        
        #if not full spectrum mode
        if len(x) != 1024:
            continue
        ix = np.argmin(incidence)
        
        #if solar angle too large everywhere
        if incidence[ix] > max_incidence_angle:
            continue
        
        #if too close to start/end of file
        if ix < 10 or ix > len(incidence) - 10:
            continue
        
        best_spectra = h5_f["Science/Y"][(ix-10):(ix+10), :]

        #if all nans
        if np.all(np.isnan(best_spectra)):
            continue
        
        #make mean spectrum from best solar angles in a file
        spectrum = np.nanmean(best_spectra, axis=0)
        
        #check no nans
        if np.sum(np.isnan(spectrum)) > 0:
            continue
        
        temps = h5_f[temperature_field][(ix-10):(ix+10)]
        t = np.mean(temps)
        
        # if np.median(spectrum) < 0.5:
        #     continue
        
        cont = baseline_als(spectrum)
        # dt= datetime.strptime(h5[0:15], "%Y%m%d_%H%M%S")
        
        spectra_dict[h5] = {"spectrum":spectrum/cont, "temperature":t}
    
    
    
    #make mean spectrum from all files
    spectra = np.zeros([len(spectra_dict.keys()), 1024])
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        spectra[i, :] = spectrum
        
    mean_spectrum = np.mean(spectra, axis=0)

    #remove spectra with bad stdevs - check only px 500 onwards
    bad_filenames = []
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        stdev = np.std(spectrum[500:] - mean_spectrum[500:])
        
        if stdev > 0.01:
            print("Removing %s" %h5)
            bad_filenames.append(h5)
            continue

    for bad_filename in bad_filenames:
        spectra_dict.pop(bad_filename)
        
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
    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "w") as f:
        f.write("Filename\tMin or max\tHDF5 absorption wavelength\tPixel\tSolar line nm\tTemperature\n")
    
    # colours = get_colours(len(spectra_dict.keys()))
    for i, h5 in enumerate(spectra_dict.keys()):
        
        if np.mod(i, 100) == 0:
            print("%i/%i" %(i, len(spectra_dict.keys())))

        spectrum = spectra_dict[h5]["spectrum"]
        temperature = spectra_dict[h5]["temperature"]
        
        #FFT doesn't work
        # uvis_x_fft, uvis_y_fft = fft_hr_nu_spectrum(x, spectrum, 100)
        # local_minima = get_local_minima(uvis_y_fft)
        # good_local_minima = local_minima[np.where(uvis_y_fft[local_minima] < 0.8)[0]]
        # uvis_x_mins = uvis_x_fft[good_local_minima]
        # for uvis_x_min in uvis_x_mins:
        #     with open("uvis_nadir_solar_lines.tsv", "a") as f:
        #         f.write(f"{h5}\t{uvis_x_min:#0.4f}\n")

        """fit individual polynomials: minima"""
        local_minmax = get_local_minima(spectrum)
        
        for ix in local_minmax:
            #check if no minima nearby
            
            if ix < 435: #if less than 400 nm
                min_absorption = 0.9
            else:
                min_absorption = 0.95
            
            
            if ix > 30 and ix < 1020 and spectrum[ix] < min_absorption:
                
                #check if close to a pre-selected solar line min/max
                ix_good_minmax = get_nearest_index(x[ix], list(good_minima_nm.keys()))
                closest_good_line = list(good_minima_nm.keys())[ix_good_minmax]
                
                if np.abs(x[ix] - closest_good_line) < 0.3:
                
                    # ix_range = np.arange(ix-2, ix+3, 1)
                    ix_range = good_minima_nm[closest_good_line]
                    
                    #check if values in correct order
                    diff = np.diff(spectrum[ix_range])
                    if diff[0] < 0 and diff[1] < 0 and diff[2] > 0 and diff[3] > 0:
                    
                        polyfit = np.polyfit(np.arange(-2, 3, 1), spectrum[ix_range], 2)
                        
                        uvis_x_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + x[ix]
                        uvis_px_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + ix
                        
                        for i in ix_range:
                            if i not in good_minima_nm[closest_good_line]:
                                good_minima_nm[closest_good_line].append(i)
                        
                        # if closest_good_line == 445.95:
                        #     polyval = np.polyval(polyfit, np.arange(-2, 3, 1))
                        #     plt.plot(x[ix_range], spectrum[ix_range], color=colours[i], alpha=0.3)
                        #     plt.scatter(x[ix_range], polyval, color=colours[i])
                        #     plt.axvline(uvis_x_minmax, color="k")
                        
                        with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "a") as f:
                            f.write(f"{h5}\tMin\t{uvis_x_minmax:#0.4f}\t{uvis_px_minmax:#0.4f}\t{closest_good_line:#0.3f}\t{temperature:#0.2f}\n")

        """fit individual polynomials: maxima"""
        local_minmax = get_local_maxima(spectrum)
        
        for ix in local_minmax:
            #check if no minima nearby
            
            if ix < 435: #if less than 400 nm
                min_peak = 0.98
            else:
                min_peak = 0.98
            
            
            if ix > 30 and ix < 1020 and spectrum[ix] > min_peak:
                
                #check if close to a pre-selected solar line min/max
                ix_good_minmax = get_nearest_index(x[ix], list(good_maxima_nm.keys()))
                closest_good_line = list(good_maxima_nm.keys())[ix_good_minmax]
                
                if np.abs(x[ix] - closest_good_line) < 0.3:
                
                    # ix_range = np.arange(ix-2, ix+3, 1)
                    ix_range = good_maxima_nm[closest_good_line]
                    
                    #check if values in correct order
                    diff = np.diff(spectrum[ix_range])
                    if diff[0] > 0 and diff[1] > 0 and diff[2] < 0 and diff[3] < 0:
                    
                        polyfit = np.polyfit(np.arange(-2, 3, 1), spectrum[ix_range], 2)
                        
                        uvis_x_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + x[ix]
                        uvis_px_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + ix
                        
                        for i in ix_range:
                            if i not in good_maxima_nm[closest_good_line]:
                                good_maxima_nm[closest_good_line].append(i)

                        # polyval = np.polyval(polyfit, np.arange(-2, 3, 1))
                        # plt.scatter(x[ix_range], polyval)
                        # plt.axvline(uvis_x_min, color="k")
                        
                        with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "a") as f:
                            f.write(f"{h5}\tMax\t{uvis_x_minmax:#0.4f}\t{uvis_px_minmax:#0.4f}\t{closest_good_line:#0.3f}\t{temperature:#0.2f}\n")




def write_solar_line_file2(spectra_dict):
    """write solar line positions to text file. This version uses the pre-defined solar line min and max points and indices from dictionary"""
    
    print("Finding solar line positions and writing to file")
    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "w") as f:
        f.write("Filename\tMin or max\tHDF5 absorption wavelength\tPixel\tSolar line nm\tTemperature\n")
    
    # colours = get_colours(len(spectra_dict.keys()))
    #loop through observations
    for i, h5 in enumerate(spectra_dict.keys()):
        
        if np.mod(i, 100) == 0:
            print("%i/%i" %(i, len(spectra_dict.keys())))

        spectrum = spectra_dict[h5]["spectrum"]
        temperature = spectra_dict[h5]["temperature"]
        
        for index, solar_x in enumerate(good_minmax_nm):
            ix_range = good_minmax_nm_d[solar_x]
            ix_range_relative = ix_range - np.mean(ix_range)
            
            #check if values in correct order - no need if data checked by hand
            # diff = np.diff(spectrum[ix_range])
            # if diff[0] < 0 and diff[1] < 0 and diff[2] > 0 and diff[3] > 0:
                
            
            """quadratic fit"""
            # polyfit = np.polyfit(ix_range_relative, spectrum[ix_range], 2)
            
            # uvis_px_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + np.mean(ix_range)
            # uvis_px_minmaxs.append(uvis_px_minmax)
            # uvis_temperatures.append(temperatures[i])
           
            # polyval = np.polyval(polyfit, ix_range_relative)
            # # plt.plot(x[ix_range], spectrum[ix_range], alpha=0.3)
            # plt.scatter(ix_range, polyval, color=colours[i])
            # plt.axvline(uvis_px_minmax, color=colours[i], alpha=0.2)
            # plt.xlim((ix_range[0]-5, ix_range[-1]+5))

            """gaussian fit"""
            error, gauss_coeffs = fit_gauss(ix_range_relative, spectrum[ix_range])
            if not error:
            
                uvis_px_minmax = gauss_coeffs[1] + np.mean(ix_range)
               
                if solar_x in good_minima_nm:
                    minmax = "Min"
                else:
                    minmax = "Max"

                with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "a") as f:
                    f.write(f"{h5}\t{minmax}\t{uvis_px_minmax:#0.4f}\t{solar_x:#0.3f}\t{temperature:#0.2f}\n")



def write_mean_solar_line_file(spectra_dict, x):

    #mean spectrum by month/year
    sorted_spectra = {}
    for i, h5 in enumerate(spectra_dict.keys()):
        
        spectrum = spectra_dict[h5]["spectrum"]
        temperature = spectra_dict[h5]["temperature"]
        year = h5[0:6]
        
        #add month/year to keys
        if year not in sorted_spectra.keys():
            sorted_spectra[year] = {"spectrum":[], "temperature":[]}
        
        sorted_spectra[year]["spectrum"].append(spectrum)
        sorted_spectra[year]["temperature"].append(temperature)
    
    mean_spectra = {}
    
    for year in sorted_spectra.keys():
        if len(sorted_spectra[year]) > 1:
            # plt.figure(figsize=(18, 6))
            # plt.title(year)
            # plt.plot(np.array(sorted_spectra[year]).T)

            #add month/year to keys
            if year not in mean_spectra.keys():
                mean_spectra[year] = {}

            mean_spectra[year]["spectrum"] = np.mean(np.array(sorted_spectra[year]["spectrum"]), axis=0)
            mean_spectra[year]["temperature"] = np.mean(np.array(sorted_spectra[year]["temperature"]))


    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_mean_spectrum_solar_lines.tsv"), "w") as f:
        f.write("Filename\tAbsorption wavelength\tTemperature\n")

    
    plt.figure(figsize=(12, 6))
    for year in mean_spectra.keys():
        spectrum = mean_spectra[year]["spectrum"]
        temperature = mean_spectra[year]["temperature"]
        
        """fft gives spurious results"""
        # uvis_x_fft, uvis_y_fft = fft_hr_nu_spectrum(x, spectrum, 100)
        # plt.plot(uvis_x_fft, uvis_y_fft, linestyle="--")
        # local_minima = get_local_minima(uvis_y_fft)
        # good_local_minima = local_minima[np.where(uvis_y_fft[local_minima] < 0.8)[0]]
        # uvis_x_mins = uvis_x_fft[good_local_minima]
        
        """fit individual polynomials"""
        local_minima = get_local_minima(spectrum)
        
        plt.plot(x, spectrum, label=year)

        for ix in local_minima:
            #check if no minima nearby
            if np.sort(np.abs(local_minima - ix))[1] > 3 and ix > 30 and ix < 1020 and spectrum[ix] < 0.9:
                
                ix_range = np.arange(ix-2, ix+3, 1)
                
                #check if values in correct order
                diff = np.diff(spectrum[ix_range])
                if diff[0] < 0 and diff[1] < 0 and diff[2] > 0 and diff[3] > 0:
                
                    polyfit = np.polyfit(np.arange(-2, 3, 1), spectrum[ix_range], 2)
                    
                    uvis_x_min = -1 * polyfit[1] / (2 * polyfit[0]) + x[ix]
                    
                    polyval = np.polyval(polyfit, np.arange(-2, 3, 1))
                    plt.scatter(x[ix_range], polyval)
                    plt.axvline(uvis_x_min, color="k")
                    
                    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_mean_spectrum_solar_lines.tsv"), "a") as f:
                        f.write(f"{year}\t{uvis_x_min:#0.4f}\t{temperature:#0.2f}\n")
                
        
        
    
        
        
                
    plt.legend()
    
    return mean_spectra
    # plt.figure(figsize=(12, 6))
    # plt.plot(x, mean_spectrum["2018"]/mean_spectrum["2019"])
    # plt.plot(x, mean_spectrum["2022"]/mean_spectrum["2019"])

# spectra_dict, x = make_solar_spectra_obs_dict(regex, file_level, max_incidence_angle)
# plot_division(spectra_dict, x)
# mean_spectra = write_mean_solar_line_file(spectra_dict, x)
# write_solar_line_file(spectra_dict, x)
# stop()

def plot_random_spectra_solar_ref(spectra_dict, x):
    """plotting stuff"""
    # nm_fft, solar_hr_fft = uvis_solar_superhr()
    # nm_lr, solar_lr = uvis_solar_lr()
    # nm_hr, solar_hr = uvis_solar_hr()
    nm_hr, solar_sg = uvis_solar_sg()
    local_minima = get_local_minima(solar_sg)
    local_maxima = get_local_maxima(solar_sg)
    good_local_minima = local_minima[np.where(solar_sg[local_minima] < 0.98)[0]]
    good_local_maxima = local_maxima[np.where(solar_sg[local_maxima] > 0.98)[0]]
    
    #get wavelengths of solar minima
    solar_x_mins = nm_hr[good_local_minima]
    solar_x_maxs = nm_hr[good_local_maxima]
    
    
    fig1, (ax1a, ax1b) = plt.subplots(figsize=(12, 8), nrows=2, sharex=True)
    ax1a.plot(nm_hr, solar_sg, label="Solar SG")
    for solar_x_min in solar_x_mins:
        ax1a.axvline(x=solar_x_min, color="k", alpha=0.2)
        ax1a.text(solar_x_min, 0.95, "%0.3f" %solar_x_min)
    for solar_x_max in solar_x_maxs:
        ax1a.axvline(x=solar_x_max, color="r", alpha=0.2)
        ax1a.text(solar_x_max, 1.05, "%0.3f" %solar_x_max)
    
    #plot 10 random spectra
    for i in np.linspace(0, len(spectra_dict.keys())-1, num=10):
        
        key = list(spectra_dict.keys())[int(i)]
        
        
        ax1b.plot(x[30:], spectra_dict[key]["spectrum"][30:])
    
    ax1a.legend()
    ax1a.grid()
    ax1b.grid()
    # ax1a.set_xlim((40, 300))



def plot_shift_binned_monthly():
    dts = []
    solar_lines = []
    temperatures = []
    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_mean_spectrum_solar_lines.tsv"), "r") as f:
        _ = f.readline()
        for line in f.readlines():
            # dts.append(datetime.strptime(line.split("\t")[0][0:15], "%Y%m%d_%H%M%S"))
            dts.append(datetime.strptime(line.split("\t")[0][0:6], "%Y%m"))
            solar_lines.append(np.float32(line.split("\t")[1]))
            temperatures.append(np.float32(line.split("\t")[2]))
            
    solar_lines = np.array(solar_lines)
    temperatures = np.array(temperatures)
    
    
    
    
    # colours = get_colours(len(good_minima_nm))
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("Observation date")
    plt.ylabel("Displacement of solar line from mean value (nm)")
    plt.title("Displacement of strongest solar lines vs time (binned monthly)")
    for index, solar_x_min in enumerate(good_minima_nm):
        
        idx = np.where((solar_lines > solar_x_min - 0.2) & (solar_lines < solar_x_min + 0.2))[0]
        datetimes = [dts[i] for i in idx]
        ts = [temperatures[i] for i in idx]
        
        if len(idx) > 2:
            plot_x = [(datetime - datetimes[0]).total_seconds() for datetime in datetimes]
            # plot_y = solar_lines[idx] - solar_x_min
            plot_y = solar_lines[idx] - np.mean(solar_lines[idx])
            # plt.scatter(datetimes, solar_lines[idx] - solar_x_min, label="%0.3fnm" %solar_x_min)
            plt.scatter(datetimes, plot_y, label="%0.3fnm" %solar_x_min)
            polyfit = np.polyfit(plot_x, plot_y, 1)
            y_fit = np.polyval(polyfit, [plot_x[0], plot_x[-1]])
            # plt.plot([datetimes[0], datetimes[-1]], y_fit)
    plt.grid()
    plt.legend()
    plt.twinx()
    plt.plot(dts, temperatures, "k--", label="CCD Temperature")
    plt.legend(loc="lower left")
    plt.ylabel("CCD temperature")
    plt.savefig("uvis_nadir_mean_spectrum_solar_lines.png")
    
    
    best_solar_line_ranges = [[285., 286.], [393., 487.]]
    best_solar_lines = []
    best_solar_lines_x = []
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("CCD temperature")
    plt.ylabel("Displacement of solar line from Meftah 2018 (nm)")
    plt.title("Displacement of strongest solar lines vs temperature (binned monthly)")
    for index, solar_x_min in enumerate(good_minima_nm):
        idx = np.where((solar_lines > solar_x_min - 0.2) & (solar_lines < solar_x_min + 0.2))[0]
        datetimes = [dts[i] for i in idx]
        ts = [temperatures[i] for i in idx]
        
        if len(idx) > 2:
            plot_x = ts
            plot_y = solar_lines[idx] - solar_x_min
            # plot_y = solar_lines[idx] - np.mean(solar_lines[idx])
    
            plt.scatter(plot_x, plot_y, label="%0.3fnm" %solar_x_min)
            polyfit = np.polyfit(plot_x, plot_y, 1)
            y_fit = np.polyval(polyfit, [min(plot_x), max(plot_x)])
            plt.plot([min(plot_x), max(plot_x)], y_fit)
            
            for best_solar_line_range in best_solar_line_ranges:
                if best_solar_line_range[0] < solar_x_min < best_solar_line_range[1]:
                    best_solar_lines.extend(list(plot_y))
                    best_solar_lines_x.extend(ts)
    
    best_solar_lines = np.array(best_solar_lines)
    best_solar_lines_x = np.array(best_solar_lines_x)
    
    polyfit = np.polyfit(best_solar_lines_x, best_solar_lines, 1)
    y_fit2 = np.polyval(polyfit, [min(best_solar_lines_x), max(best_solar_lines_x)])
    plt.plot([min(best_solar_lines_x), max(best_solar_lines_x)], y_fit2, "k--", label="Mean of best lines")
    
       
    plt.grid()
    plt.legend()
    
    plt.savefig("uvis_nadir_mean_spectrum_solar_line_temperature.png")





def read_solar_line_tsv_to_dict():
    
    d = {"filename":[], "dt":[], "min":[], "uvis_px":[], "solar_x":[], "t":[]}
    with open(os.path.join(paths["REFERENCE_DIRECTORY"], "uvis_nadir_solar_lines.tsv"), "r") as f:
        _ = f.readline()
        for line in f.readlines():
            split = line.split("\t")
            # dts.append(datetime.strptime(line.split("\t")[0][0:15], "%Y%m%d_%H%M%S"))
            d["filename"].append(split[0])
            d["dt"].append(datetime.strptime(split[0][0:8], "%Y%m%d"))
            d["min"].append({"Min":True, "Max":False}[split[1]])
            d["uvis_px"].append(np.float32(split[2]))
            d["solar_x"].append(np.float32(split[3]))
            d["t"].append(np.float32(split[4]))
    
    for key in d.keys():
        d[key] = np.array(d[key])
        
    return d



def plot_shifts_vs_temperature():
    """plot shifts by solar line"""
    d = read_solar_line_tsv_to_dict()
    
    d_solar = {"solar_x":[], "fit0":[], "fit1":[]}
    
    colours = get_colours(len(good_minmax_nm))
    fig1, ax1 = plt.subplots()
    for index, solar_x in enumerate(good_minmax_nm):
        
        idx = np.where(d["solar_x"] == solar_x)[0]
        
        
        
        if len(idx) > 10:
        
            delta_x = d["uvis_x"][idx] - solar_x #plot relative to expected solar line wavelength
            # delta_x = d["uvis_x"][idx] - np.mean(d["uvis_x"][idx])
    
    
            temperatures = d["t"][idx]
            
            ax1.scatter(temperatures, delta_x, color=colours[index], label="%0.3fnm" %solar_x, alpha=0.8)
            
            polyfit2 = np.polyfit(temperatures, delta_x, 1)
            y_fit = np.polyval(polyfit2, [min(temperatures), max(temperatures)])
            ax1.plot([min(temperatures), max(temperatures)], y_fit, color=colours[index])
            
            print("%0.3fnm" %solar_x, polyfit2)
            
            d_solar["solar_x"].append(solar_x)
            d_solar["fit0"].append(polyfit2[0])
            d_solar["fit1"].append(polyfit2[1])


"""investigate fitting of polynomial to each solar line"""
# d = read_solar_line_tsv_to_dict()

# for index, solar_x in enumerate(good_minmax_nm):
    
#     idx = np.where(d["solar_x"] == solar_x)[0]
    
    
    
#     if len(idx) > 10:
    
#         uvis_px_minmaxs = []
#         uvis_temperatures = []
        
#         filenames = d["filename"][idx]
#         temperatures = d["t"][idx]
#         plt.figure(figsize=(16, 12))
#         plt.title("%0.3fnm" %(solar_x))
#         plt.ylim((0.4, 1.3))
#         colours = get_colours(len(filenames))
#         for i, filename in enumerate(filenames):
#             spectrum = spectra_dict[filename]["spectrum"]
#             plt.plot(spectrum, color=colours[i], alpha=0.2)


#             ix_range = good_minmax_nm_d[solar_x]
#             ix_range_relative = ix_range - np.mean(ix_range)
            
            
#             """quadratic fit"""
#             # polyfit = np.polyfit(ix_range_relative, spectrum[ix_range], 2)
            
#             # uvis_px_minmax = -1 * polyfit[1] / (2 * polyfit[0]) + np.mean(ix_range)
#             # uvis_px_minmaxs.append(uvis_px_minmax)
#             # uvis_temperatures.append(temperatures[i])
           
#             # polyval = np.polyval(polyfit, ix_range_relative)
#             # # plt.plot(x[ix_range], spectrum[ix_range], alpha=0.3)
#             # plt.scatter(ix_range, polyval, color=colours[i])
#             # plt.axvline(uvis_px_minmax, color=colours[i], alpha=0.2)
#             # plt.xlim((ix_range[0]-5, ix_range[-1]+5))

#             """gaussian fit"""
#             error, gauss_coeffs = fit_gauss(ix_range_relative, spectrum[ix_range])
#             if not error:
            
#                 uvis_px_minmax = gauss_coeffs[1] + np.mean(ix_range)
#                 uvis_px_minmaxs.append(uvis_px_minmax)
#                 uvis_temperatures.append(temperatures[i])
               
#                 gauss = make_gauss(ix_range_relative, gauss_coeffs)
#                 plt.scatter(ix_range, gauss, color=colours[i])
#                 plt.axvline(uvis_px_minmax, color=colours[i], alpha=0.2)
#                 plt.xlim((ix_range[0]-5, ix_range[-1]+5))


#         polyfit2 = np.polyfit(uvis_temperatures, uvis_px_minmaxs, 1)

#         print("%0.3fnm" %solar_x, polyfit2)



def calculate_nm_coeffs(poly_degree, plot=False):
    """group by temperature, then fit best polynomial to each group. Plot coefficients vs temperature"""
    d = read_solar_line_tsv_to_dict()
    bin_delta = 1.0
    
    t_bins = []
    coeffs = []
    
    # for t_bin in np.arange(-15, 15, bin_delta):
    for t_bin in np.arange(-15, 15, bin_delta):
        idx_t = np.where((d["t"] > t_bin) & (d["t"] < (t_bin + bin_delta)))[0]
        
        if len(idx_t) > 10:
            pxs = d["uvis_px"][idx_t]
            ys = d["solar_x"][idx_t]
            
            polyfit = np.polyfit(pxs, ys, poly_degree)
            polyval = np.polyval(polyfit, np.arange(1024))
            
            diff = ys-np.polyval(polyfit, pxs)
    
            if plot:
                fig, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True)
                fig.suptitle(t_bin)
                ax1a.scatter(pxs, ys)
                ax1a.plot(np.arange(1024), polyval)
                ax1b.scatter(pxs, diff)
                
            good_points = np.where(diff > -0.5)[0]
    
            #repeat fitting with bad points removed
            pxs = pxs[good_points]
            ys = ys[good_points]
    
            polyfit = np.polyfit(pxs, ys, poly_degree)
            polyval = np.polyval(polyfit, np.arange(1024))
            
            diff = ys-np.polyval(polyfit, pxs)
    
            if plot:
                ax1a.scatter(pxs, ys)
                ax1a.plot(np.arange(1024), polyval)
                ax1b.scatter(pxs, diff)
            
            
            
            t_bins.append(t_bin)
            coeffs.append(polyfit)
        
    t_bins = np.array(t_bins)
    coeffs = np.array(coeffs)
    
    return t_bins, coeffs, poly_degree



"""linear fit to make each px->nm coefficient from the temperature"""
poly_degree = 4

t_bins, coeffs, poly_degree = calculate_nm_coeffs(poly_degree)
# t_bins, coeffs, poly_degree = calculate_nm_coeffs(poly_degree, plot=True)


coefficients_vs_temperature = np.zeros((poly_degree+1, 4))
for i in range(poly_degree+1):
    polyfit = np.polyfit(t_bins, coeffs[:, i], 3)
    polyval = np.polyval(polyfit, t_bins)


    # plt.figure()
    # plt.scatter(t_bins, coeffs[:, i])
    # plt.plot(t_bins, polyval)
    
    print("["+",".join([str(f) for f in polyfit])+"],")
    # print(i, str(polyfit))
    coefficients_vs_temperature[i, :] = polyfit

            
            

def get_x_from_temperature(temperature, coefficients_vs_temperature, full_spectrum=True):
    """make a function to calculate spectral calibration, given the temperature"""
    """using 4th order polynomial fit to px-nm data, and 3rd orbit fit to coefficients

    [1.2668651393731336e-15,-6.762390987967002e-15,-2.9130158902353364e-13,-5.113388527679992e-12],
    [-2.33271620171716e-12,1.0562355230625095e-11,5.572599545842369e-10,2.9465610513351482e-09],
    [1.2487227999347614e-09,-3.543384495278793e-09,-3.25905132888265e-07,-2.6764778054044725e-05],
    [-1.4712608864333634e-07,-7.908713661540165e-07,6.072742850076307e-05,0.474985871609256],
    [1.9105326041204307e-05,-1.327509536090508e-05,-0.008797269974871506,198.81879191516828],
    
    """
    
    px_coeffs = np.polyval(coefficients_vs_temperature.T, temperature)
    
    px = np.arange(1024)
    nm_1024 = np.polyval(px_coeffs, px)
    
    nm_1048 = np.zeros(1048)
    nm_1048[8:1032] = nm_1024
    
    if full_spectrum:
        return nm_1048
    else:
        return nm_1024
    

x_m12 = get_x_from_temperature(-12, coefficients_vs_temperature)
x_12 = get_x_from_temperature(12, coefficients_vs_temperature)

plt.figure()
plt.scatter(x_m12, x_m12 - x_12)