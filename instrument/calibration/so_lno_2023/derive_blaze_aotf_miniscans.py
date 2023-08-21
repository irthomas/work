# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:13:26 2022

@author: iant

INVESTIGATE BLAZE FUNCTION VS TEMPERATURE FROM MINISCANS
USE THIS TO CHECK THE CALIBRATION AND FITS, NOT FOR CONVERTING FILES
USE correct_miniscan_diagonals.py TO MAKE DIAGONALLY CORRECTED H5 FILES
"""

# import sys
import os
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py

import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, BSpline

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.cprint import cprint
from tools.spectra.running_mean import running_mean_1d

from instrument.nomad_so_instrument_v03 import aotf_peak_nu, lt22_waven
from instrument.nomad_lno_instrument_v02 import nu0_aotf, nu_mp




# inflight

# channel = "SO"
channel = "LNO"


file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_%s_.*_CM" %channel)

regex = re.compile("20200201.*_%s_.*_CM" %channel)
# regex = re.compile("20191002.*_%s_.*_CM" %channel)

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


HR_SCALER = 10.

MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")


if channel == "SO":
    from instrument.nomad_so_instrument_v03 import m_aotf
elif channel == "LNO":
    from instrument.nomad_lno_instrument_v02 import m_aotf


if __name__ == "__main__":
    if channel == "SO":
        # aotf_steppings = [8.0]
        aotf_steppings = [4.0]
        binnings = [0]
        # starting_orders = [188]
        starting_orders = [191]
        # dictionary of fft_cutoff for each aotf_stepping
        fft_cutoff_dict = {
            1:4,
            2:15,
            4:15,
            8:40,
            }
    
    elif channel == "LNO":
        # aotf_steppings = [8.0]
        aotf_steppings = [4.0, 8.0]
        binnings = [0]
        starting_orders = [194]
        # dictionary of fft_cutoff for each aotf_stepping
        fft_cutoff_dict = {} #don't apply FFT - no ringing
    



illuminated_row_dict = {24:slice(6, 19)}



#solar line dict
# solar_line_dict = {
    # "20181206_171850-191-4":{"left":np.arange(215, 222), "centre":229, "right":np.arange(234, 245)}, #6.4C
    # "20211105_155547-191-4":{"left":np.arange(208, 215), "centre":222, "right":np.arange(227, 238)}, #-2.1C
    # "20230112_084925-191-4":{"left":np.arange(201, 208), "centre":215, "right":np.arange(220, 231)}, #-8.3C

    # "20181206_171850":{"left":np.arange(144, 148), "centre":149, "right":np.arange(153, 157)},
    # "20211105_155547":{"left":np.arange(137, 141), "centre":142, "right":np.arange(146, 150)},

    # "20181206_171850":{"left":np.arange(201, 205), "centre":206, "right":np.arange(208, 212)},
    # "20211105_155547":{"left":np.arange(194, 198), "centre":200, "right":np.arange(201, 205)},
    
    # "20190416_020948-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
    # "20210717_072315-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
    # "20220120_125011-194-1":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},

    # "20201010_113533-188-8":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},
    # "20210201_111011-188-8":{"left":np.arange(209, 213), "centre":218, "right":np.arange(220, 224)},
    # "20210523_001053-188-8":{"left":np.arange(205, 209), "centre":212, "right":np.arange(216, 220)},
    # "20221011_132104-188-8":{"left":np.arange(209, 213), "centre":215, "right":np.arange(218, 222)},

    #LNO    
    # "20191002_000902-176-8":{"left":np.arange(196, 200), "centre":210, "right":np.arange(220, 224), "centres":[510, 850, 950, 2100, 2760, 2390]}, #-5.4C
    # "20200812_135659-176-8":{"left":np.arange(196, 200), "centre":210, "right":np.arange(220, 224), "row":39}, #-3.6C
    # "20210606_021551-176-8":{"left":np.arange(196, 200), "centre":206, "right":np.arange(220, 224), "row":39}, #-9.5C

    # "20220619_140101-176-4":{"centres":[206, 235, 272]},
    
    # "20190408_032951-194-1":{"centres":[]},
    # "20210724_201241-194-1":{"centres":[]},
    
    # "20181021_071529-194-4":{"centres":[129, 139, 178, 195, 236]},
    # # "20181121_133754-194-2":{"centres":[]},
    # "20190212_145904-194-8":{"centres":[128, 139, 157, 194]},
    # "20190408_040458-194-4":{"centres":[125, 134, 184, 191]},
    # "20200201_001633-194-4":{"centres":[124, 134, 183, 190]},
    # "20200827_133646-194-8":{"centres":[242, 257]},
    

    # "20191002_000902-192-8":{"left":np.arange(203, 207), "centre":214, "right":np.arange(221, 224)},
    # }


# get data from h5 files?
# list_files = True
list_files = False

if "d" not in globals():
    list_files = True

#find new solar lines
plot_miniscans = True
# plot_miniscans = False


plot_fft = True
# plot_fft = False

plot_blaze = True
# plot_blaze = False

plot_absorptions = True
# plot_absorptions = False

# plot_aotf = True
plot_aotf = False





# def get_miniscan_data_0p1a(regex):


#     h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True)

#     d = {}
#     for h5_f, h5 in zip(h5_files, h5_filenames):
    
#         h5_split = h5.split("_")
#         h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}"
        
#         d[h5_prefix] = {}
        
        
        
#         y = h5_f["Science/Y"][...] #y is 3d in 0.1A
#         aotf_freqs = h5_f["Channel/AOTFFrequency"][...]
#         unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        
#         orders = np.array([m_aotf(i) for i in unique_aotf_freqs])
#         unique_orders = sorted(list(set(orders)))
        
        
#         aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]
        
#         print("Miniscan stepping = %0.1fkHz" %(aotf_freqs_step))
#         print(h5, "orders", unique_orders[0])
    
        
#         bin_ = 12
        
#         y_bin = y[:, bin_, :]
    
#         for unique_aotf_freq in unique_aotf_freqs:
            
#             aotf_ixs = np.where(aotf_freqs == unique_aotf_freq)[0]
            
#             for aotf_ix in aotf_ixs[0:1]:
#                 y_spectrum = y_bin[aotf_ix, :]
        
#                 d[h5_prefix][unique_aotf_freq] = {0.0:y_spectrum} #set temperature to 0
            
#     return d





def list_miniscan_data_1p0a(regex, starting_orders, aotf_steppings, binnings):
    h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True)
    
            
    matching_h5 = []
    matching_h5_prefix = []
    for h5_f, h5 in zip(h5_files, h5_filenames): #loop through orders of that observation
    
    
        binning_ = h5_f["Channel/Binning"][0]
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]
        
        
        
        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        # unique_bins = sorted(list(set(bins)))
        
        orders = np.array([m_aotf(i) for i in unique_aotf_freqs])
        unique_orders = sorted(list(set(orders)))
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]

        h5_split = h5.split("_")
        h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}-%i-%i" %(np.min(unique_orders), np.round(aotf_freqs_step))
        
        
        
        if unique_orders[0] not in starting_orders:
            print("%s order %i stepping %0.1fkHz" %(h5, unique_orders[0], aotf_freqs_step))
            continue


        if aotf_freqs_step not in aotf_steppings or binning_ not in binnings:
            cprint("%s order %i stepping %0.1fkHz" %(h5, unique_orders[0], aotf_freqs_step), "y")
            continue

        cprint("%s order %i stepping %0.1fkHz" %(h5, unique_orders[0], aotf_freqs_step), "c")
        matching_h5.append(h5)
        matching_h5_prefix.append(h5_prefix)
    
    return matching_h5, matching_h5_prefix





def get_miniscan_data_1p0a(h5_filenames):
        
    print("Getting data for %i files" %len(h5_filenames))

    d = {}
    for h5 in h5_filenames:
        # if "SO" in h5:
        #     good_bins = np.arange(126, 131)
        # elif "LNO" in h5:
        #     good_bins = np.arange(150, 155)


        print("Getting data for %s" %h5)
        h5_f = open_hdf5_file(h5)

        # observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
        # bins = h5_f["Science/Bins"][:, 0]
        y = h5_f["Science/Y"][...]
        t = h5_f["Channel/InterpolatedTemperature"][...]
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]

        #number of bins per aotf_freq
        n_bins = np.where(np.diff(aotf_freqs) > 0)[0][0] + 1 #first index where diff is nonzero
        
        #reshape by bin
        y = np.reshape(y, (-1, n_bins, 320))
        t = np.reshape(t, (-1, n_bins))[:, 0]
        aotf_freqs = np.reshape(aotf_freqs, (-1, n_bins))[:, 0]
        aotf_freqs = np.array([int(np.round(i)) for i in aotf_freqs])

        illuminated_rows = illuminated_row_dict[n_bins]
        y_mean = np.mean(y[:, illuminated_rows, :], axis=1) #mean of illuminated rows

        #number of aotf freqs 
        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        n_aotf_freqs = len(unique_aotf_freqs)
        
        #remove incomplete aotf repetitions
        n_repetitions = int(np.floor(np.divide(y.shape[0], n_aotf_freqs)))
        
        y_rep = np.reshape(y_mean[0:(n_repetitions*n_aotf_freqs), :], (-1, n_aotf_freqs, 320))
        t_rep = np.reshape(t[0:(n_repetitions*n_aotf_freqs)], (-1, n_aotf_freqs)).T

        # unique_bins = sorted(list(set(bins)))
        starting_order = m_aotf(np.min(unique_aotf_freqs))
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]
        


        h5_split = h5.split("_")
        h5_prefix = f"{h5_split[3]}-{h5_split[0]}-{h5_split[1]}-%i-%i" %(starting_order, np.round(aotf_freqs_step))
        
        #output mean y for illuminated bins (2D), temperatures (1D) and aotf freqs (1D)
        #also output truncated arrays containing n repeated aotf freqs: y_rep (3D), t_rep (2D), a_rep(1D)
        d[h5_prefix] = {"y":y_mean, "t":t, "a":aotf_freqs, "y_rep":y_rep, "t_rep":t_rep, "a_rep":unique_aotf_freqs}
        
        
        
        
    
    return d






def remove_oscillations(d, fft_cutoff_dict, cut_off=["inner"], plot=False):
    """make dictionary of miniscan arrays before and after oscillation removal
    input: dictionary of raw miniscan data,
    fft_cutoff: index to start setting fft to zero (symmetrical from centre)
    cut_off_inner: whether to set the inner indices to zero (removes high res oscillations) 
    or outer indices to zero (removes large features)"""
    
    d2 = {}
    for h5_prefix in d.keys():
        
        stepping = int(h5_prefix.split("-")[-1])
        fft_cutoff = fft_cutoff_dict[stepping]

        
        # miniscan_array = np.zeros((len(d[h5_prefix].keys()), 320))
        # for i, aotf_freq in enumerate(d[h5_prefix].keys()):
            
        #     for temperature in list(d[h5_prefix][aotf_freq].keys())[0:1]:
        #         miniscan_array[i, :] = d[h5_prefix][aotf_freq][temperature] #get 2d array for 1st temperature in file
        
        # miniscan_array = d[h5_prefix]["y_rep"][0, :, :]  #get 2d array for 1st repetition in file

        n_reps = d[h5_prefix]["y_rep"].shape[0]
        d2[h5_prefix] = {}
        d2[h5_prefix]["nreps"] = n_reps
        d2[h5_prefix]["aotf"] = d[h5_prefix]["a_rep"]
        d2[h5_prefix]["t"] = np.mean(d[h5_prefix]["t_rep"])
        
        #bad pixel correction
        for rep_ix in range(n_reps):
            miniscan_array = d[h5_prefix]["y_rep"][rep_ix, :, :]  #get 2d array for 1st repetition in file
            miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)
            # d2[h5_prefix]["array_raw%i" %(rep_ix)] = miniscan_array
        

            fft = np.fft.fft2(miniscan_array)
            
            if plot:
                fig, ax = plt.subplots()
                ax.set_title(h5_prefix)
                ax.plot(fft.real[:, 200])
                # ax.plot(fft.imag[:, 200])
            
            if "inner" in cut_off:
                fft.real[fft_cutoff:(256-fft_cutoff), :] = 0.0
                fft.imag[fft_cutoff:(256-fft_cutoff), :] = 0.0
                
            if "outer" in cut_off:
                fft.real[0:fft_cutoff, :] = 0.0
                fft.real[(256-fft_cutoff):, :] = 0.0
                fft.imag[0:fft_cutoff, :] = 0.0
                fft.imag[(256-fft_cutoff):, :] = 0.0
    
            if plot:
                ax.plot(fft.real[:, 200], linestyle=":")
                # ax.plot(fft.imag[:, 200])
    
    
            ifft = np.fft.ifft2(fft).real
    
            d2[h5_prefix]["array%i" %(rep_ix)] = ifft
        

    return d2





"""vertical slices"""
# plt.figure(figsize=(8, 5), constrained_layout=True)
# for file_ix, h5_prefix in enumerate(d2.keys()):
#     aotf_freqs =  [f for f in d[h5_prefix].keys()]
#     miniscan_array = d2[h5_prefix]["array"]
#     for line_ix, line in enumerate([180, 200, 220]):
#         if file_ix == 0:
#             label = "Pixel number %i" %line
#         else:
#             label = ""
#         plt.plot(aotf_freqs, miniscan_array[:, line]+line_ix*100000, label=label, color="C%i" %line_ix)

# for k, v in A_aotf.items():
#     if v in aotf_freqs:
#         plt.axvline(x=v, color="k", linestyle="dashed")

# plt.legend()
# plt.xlabel("AOTF frequency (kHz)")
# plt.ylabel("Signal on detector")
# plt.grid()
# plt.savefig("miniscan_vertical_slice.png")


"""horizontal slices"""
# plt.figure(figsize=(8, 5), constrained_layout=True)
# h5_prefix = list(d2.keys())[-1]
# aotf_freqs =  [f for f in d[h5_prefix].keys()]
# miniscan_array = d2[h5_prefix]["array"]
# for line_ix, line in enumerate([135, 137, 139, 141, 143]):
#     label = "AOTF frequency %0.1f" %aotf_freqs[line]
#     plt.plot(np.arange(320), miniscan_array[line, :], label=label, color="C%i" %line_ix)
# plt.legend()
# plt.xlabel("Pixel number")
# plt.ylabel("Signal on detector")
# plt.grid()
# plt.savefig("miniscan_horizontal_slice.png")




def find_peak_aotf_pixel(t, aotf_freqs, px_ixs):
    """get pixel number corresponding to the peak of the AOTF in each spectrum"""

    
    #spectral calibration to find peak aotf and pixel wavenumbers
    orders = [m_aotf(i) for i in aotf_freqs]
    if channel == "SO":
        aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
        px_nus = [lt22_waven(i, t) for i in orders]
    elif channel == "LNO":
        aotf_nus = [nu0_aotf(i) for i in aotf_freqs]
        px_nus = [nu_mp(i, px_ixs, t) for i in orders]

    #pixel position of peak AOTF in each frame
    px_peaks = []
    for aotf_nu, px_nu in zip(aotf_nus, px_nus):
        px_peak = (np.abs(px_nu - aotf_nu)).argmin()
        px_peaks.append(px_peak)
        
    return px_peaks, aotf_nus



def get_diagonal_blaze_indices(px_peaks, px_ixs):
    
    #make array of blaze functions, one blaze for each row offset value
    blazes = []
    px_ix = 0
    blaze = []
    for array_row_ix, px_peak in enumerate(px_peaks):
        
        if px_peak < px_ix: #start new list when order changes
            blazes.append(blaze)
            blaze = []
        
        px_ix = px_peak
        
        blaze.append([px_ix, array_row_ix]) #save pixel indices, column and row



    blaze_diagonal_ixs = []
    for blaze in blazes:
        blaze = np.array(blaze)
        
        #extrapolate to get pixel indices for whole detector and interpolate between pixels
        polyfit = np.polyfit(blaze[:, 0], blaze[:, 1], 1)
        px_extrap = np.array([int(np.round(i)) for i in np.polyval(polyfit, px_ixs)])
        
        if np.any(px_extrap < 0.0):
            continue
        
        blaze_diagonal_ixs.append(px_extrap)
        
    return np.asarray(blaze_diagonal_ixs)




# def make_blaze_functions(d2, row_offsets, array_name="array", plot=False):
#     """inputs:
#     oscillation corrected/uncorrected miniscan array
#     row_offsets: find blaze functions for rows above and below the peak blaze
#     These have the same shape but a lower intensity which is corrected for by the normalisation
#     array_name: the name of the dictionary key in the miniscan array"""
    
#     extrapolated_blazes = {}
    
#     for h5_ix, h5_prefix in enumerate(d2.keys()):
#         extrapolated_blazes[h5_prefix] = []
        
#         px_peaks, _ = find_peak_aotf_pixel(h5_prefix) #get pixel number where AOTF peaks for every spectrum 
    
#         array = d2[h5_prefix][array_name] #get data
            
#         for offset in row_offsets:
#             #make array of blaze functions, one blaze for each row offset value
#             blazes = []
#             px_ix = 0
#             blaze = []
#             for array_row_ix, px_peak in enumerate(px_peaks):
                
#                 if px_peak + offset < px_ix: #start new list when order changes
#                     blazes.append(blaze)
#                     blaze = []
                
#                 px_ix = px_peak + offset
                
#                 blaze.append([px_ix, array_row_ix]) #save pixel indices, column and row
            
            
#             #for plotting only
#             if offset == 0 and plot:
#                 array_peaks = array.copy()
    
#             for blaze in blazes:
#                 blaze = np.array(blaze)
                
#                 #extrapolate to get pixel indices for whole detector and interpolate between pixels
#                 polyfit = np.polyfit(blaze[:, 0], blaze[:, 1], 1)
#                 px_range = np.arange(320)
#                 px_extrap = np.array([int(np.round(i)) for i in np.polyval(polyfit, px_range)])
                
#                 if np.any(px_extrap < 0.0):
#                     continue
                
#                 #not interpolated or extrapolated
#                 # px_row = array[blaze[:, 1], blaze[:, 0]]
                
#                 #interpolated/extrapolated
#                 px_row = array[px_extrap, px_range]
    
#                 extrapolated_blazes[h5_prefix].append(px_row/np.max(px_row))
    
#                 if offset == 0 and plot:
            
#                     """plot on miniscan array where pixel nu = aotf nu i.e. the diagonals"""
#                     for px_row_ix, px_column_ix in zip(px_range, px_extrap):
#                         array_peaks[px_column_ix, px_row_ix] = -999
                
#                 # for row, px_peak in enumerate(px_peaks):
#                 #     array_peaks[row, px_peak] = -999
    
    
#             if offset == 0 and plot:
#                 plt.figure(figsize=(8, 5), constrained_layout=True)
#                 plt.title("Miniscan corrected array")
#                 plt.imshow(array_peaks)

#     return extrapolated_blazes




# def plot_blazes(d2, extrapolated_blazes):
#     fig, ax = plt.subplots()
#     for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
#         blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
#         ax.plot(blazes.T, color="C%i" %h5_ix, alpha=0.1) #linestyle=linestyles[h5_ix], alpha=0.2)
    
    
#     blaze_all = []
#     for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
#         blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
#         blazes_median = np.median(blazes, axis=0)
        
#         t = d2[h5_prefix]["t"]
        
#         #smooth the median and plot
#         blazes_median_rm = running_mean_1d(blazes_median, 9)
#         ax.plot(blazes_median_rm, color="C%i" %h5_ix, label="%s, %0.2fC" %(h5_prefix, t))
#         blaze_all.append(blazes_median_rm)
        
#     ax.legend()
#     ax.grid()
#     ax.set_title("Derived blaze functions")
#     ax.set_xlabel("Pixel number")
#     ax.set_ylabel("Normalised blaze function")
    
#     mean_blaze = np.mean(np.array(blaze_all), axis=0)
    
#     return mean_blaze / np.max(mean_blaze)



 

# def band_depth(line, centre_px, ixs_left, ixs_right, ax=None):
#     """solar line band depth"""

#     abs_pxs = np.concatenate((ixs_left, ixs_right))
#     abs_vals = line[abs_pxs]
    
#     cont_pxs = np.arange(ixs_left[0], ixs_right[-1] + 1)
#     cont_vals = np.polyval(np.polyfit(abs_pxs, abs_vals, 2), cont_pxs)
    
#     abs_vals = line[cont_pxs] / cont_vals
#     abs_depth = 1.0 - abs_vals[centre_px - ixs_left[0]]
    
#     if ax:
#         ax.plot(cont_pxs, abs_vals)
    
#     return abs_depth


# def make_aotf_functions(d2, array_name="array", plot=plot_absorptions):
#     """get aotf function from depth of a solar line"""
    
#     aotfs = {}
#     for h5_prefix in d2.keys():
        
        
#         t = d2[h5_prefix]["t"]
    
#         aotf_freqs = d2[h5_prefix]["aotf"]

#         if channel == "SO":
#             aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
#         elif channel == "LNO":
#             aotf_nus = [nu0_aotf(i) for i in aotf_freqs]

#         array = d2[h5_prefix][array_name]
    
#         if plot:
#             fig1, (ax1a, ax1b) = plt.subplots(nrows=2, sharex=True)
#             ax1a.set_title(h5_prefix)
#             ax1a.grid()
#             ax1b.grid()
        
#         ixs_left = solar_line_dict[h5_prefix]["left"]
#         ixs_right = solar_line_dict[h5_prefix]["right"]
#         centre_px = solar_line_dict[h5_prefix]["centre"]
        
#         #absorption depth variations
#         abs_depths = []
#         for row_ix in np.arange(array.shape[0]):
#             line = array[row_ix, :]
            
#             if plot:
#                 abs_depth = band_depth(line, centre_px, ixs_left, ixs_right, ax=ax1b)
#                 if abs_depth > 0.1:
#                     cont_pxs = np.arange(ixs_left[0] - 3, ixs_right[-1] + 4)
#                     ax1a.plot(cont_pxs, line[cont_pxs])

#             else:
#                 abs_depth = band_depth(line, centre_px, ixs_left, ixs_right)
#             abs_depths.append(abs_depth)
            
#             # plt.plot(line)
#             # plt.plot(cont_pxs, cont_vals)
            
#         aotfs[h5_prefix] = {"aotf_freqs":aotf_freqs, "aotf_nus":aotf_nus, "abs_depths":np.array(abs_depths)}

#     return aotfs


# def plot_aotf(d2, array_name="array"):
#     aotfs = make_aotf_functions(d2, array_name=array_name)
    
#     plt.figure()
#     for h5_prefix in aotfs.keys():
    
#         t = d2[h5_prefix]["t"]
        
#         aotf_nus = aotfs[h5_prefix]["aotf_nus"]
#         aotf_func = aotfs[h5_prefix]["abs_depths"]
        
#         plt.plot(aotf_nus, aotf_func/np.max(aotf_func), label="%s, %0.2fC" %(h5_prefix, t))
#         plt.xlabel("Wavenumber cm-1")
#         plt.ylabel("Normalised AOTF Function")
#         plt.title("AOTF Functions")

#         with open("aotf_%s.tsv" %h5_prefix, "w") as f:
#             f.write("Wavenumber\tAOTF function\n")
#             for aotf_nu, aotf_f in zip(aotf_nus, aotf_func):
#                 f.write("%0.4f\t%0.4f\n" %(aotf_nu, aotf_f))


#     plt.grid()
#     plt.legend()



def make_hr_array(array, aotfs):
    
    
    array_shape = array.shape
    x = np.arange(array_shape[0])
    y = np.arange(array_shape[1])
    
    interp = RegularGridInterpolator((x, y), array,
                                 bounds_error=False, fill_value=None, method="linear")
    x_hr = np.arange(0, array_shape[0], 1.0/HR_SCALER)
    y_hr = np.arange(0, array_shape[1], 1.0/HR_SCALER)
    X, Y = np.meshgrid(x_hr, y_hr, indexing='ij')
    
    array_hr = interp((X, Y))
    
    #interpolate aotf freqs onto same grid
    aotf_hr = np.interp(x_hr, x, aotfs)

    return array_hr, aotf_hr


#find absorptions in 2d grid
#find local minima



"""get data"""
if __name__ == "__main__":
    if file_level == "hdf5_level_1p0a":
        if list_files:
            h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, starting_orders, aotf_steppings, binnings)
            if "d" not in globals():
                d = get_miniscan_data_1p0a(h5_filenames)
            if h5_prefixes != list(d.keys()):
                d = get_miniscan_data_1p0a(h5_filenames)
    
    # h5_prefixes = h5_prefixes[0:1] #TODO: remove
    
        
    if channel == "SO":
        d2 = remove_oscillations(d, fft_cutoff_dict, cut_off=["inner"], plot=plot_fft)


    elif channel == "LNO":
        #no FFT ringing correction
        d2 = {}
        for h5_prefix in h5_prefixes:
            
            n_reps = d[h5_prefix]["y_rep"].shape[0]

            
            #plot LR array spectra for different repetitions
            plt.figure()
            plt.title("Miniscan spectra for different repetitions")
            for rep_ix in range(n_reps):
                plt.plot(d[h5_prefix]["y_rep"][rep_ix, 128, :], label=rep_ix)
            for rep_ix in range(n_reps):
                plt.plot(d[h5_prefix]["y_rep"][rep_ix, :, 160], label=rep_ix)
            plt.legend()

            d2[h5_prefix] = {"aotf":d[h5_prefix]["a_rep"], "t":np.mean(d[h5_prefix]["t_rep"])} #don't do anything

            #bad pixel correction
            for rep_ix in range(n_reps):
                miniscan_array = d[h5_prefix]["y_rep"][rep_ix, :, :]  #get 2d array for 1st repetition in file
                # miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)
                d2[h5_prefix]["array%i" %(rep_ix)] = miniscan_array
            d2[h5_prefix]["nreps"] = n_reps

    # """plot miniscan arrays"""
    if plot_miniscans:
        for h5_prefix in h5_prefixes:
            temperature = d2[h5_prefix]["t"]
            # array = d2[h5_prefix]["array_raw"]
            array_corrected = d2[h5_prefix]["array1"]
        
            # plt.figure(figsize=(8, 5), constrained_layout=True)
            # plt.title("Miniscan: %s, %0.2fC" %(h5_prefix, temperature))
            # plt.imshow(array)
            # plt.xlabel("Pixel number")
            # plt.ylabel("Frame index (AOTF frequency)")
        
        
            plt.figure(figsize=(8, 5), constrained_layout=True)
            plt.title("Miniscan corrected array: %s, %0.2fC" %(h5_prefix, temperature))
            plt.imshow(array_corrected)
            plt.xlabel("Pixel number")
            plt.ylabel("Frame index (AOTF frequency)")
        
            # plt.figure(figsize=(8, 5), constrained_layout=True)
            # plt.title("Miniscan difference: %s, %0.2fC" %(h5_prefix, temperature))
            # plt.imshow(array - array_corrected)
            # plt.xlabel("Pixel number")
            # plt.ylabel("Frame index (AOTF frequency)")

    
    """make HR arrays"""
    for h5_prefix in h5_prefixes:

        n_reps = d2[h5_prefix]["nreps"]
        #HR array spectra for all repetitions
        aotfs = d2[h5_prefix]["aotf"]
        for rep in range(n_reps):
            
            #interpolate onto high res grid
            array = d2[h5_prefix]["array%i" %rep]
            array_hr, aotf_hr = make_hr_array(array, aotfs)
            d2[h5_prefix]["array%i_hr" %rep] = array_hr
            
            
        d2[h5_prefix]["aotf_hr"] = aotf_hr


        t_rep = [np.mean(d[h5_prefix]["t_rep"][:, rep]) for rep in range(n_reps)]
        d2[h5_prefix]["t"] = t_rep


        
    """Correct diagonals"""
    for h5_prefix in h5_prefixes:
        for rep in range(d2[h5_prefix]["nreps"]):
        #calc blaze diagonals
        
            t = d2[h5_prefix]["t"][rep]
            px_ixs = np.arange(d2[h5_prefix]["array%i_hr" %rep].shape[1])

            px_peaks, aotf_nus = find_peak_aotf_pixel(t, d2[h5_prefix]["aotf_hr"], px_ixs)
            px_peaks = np.asarray(px_peaks)  * int(HR_SCALER)
            aotf_nus = np.asarray(aotf_nus)
            blaze_diagonal_ixs_all = get_diagonal_blaze_indices(px_peaks, px_ixs)
        
        
            #make diagonally corrected array
            diagonals = []
            diagonals_aotf = []
             
            for row in range(d2[h5_prefix]["array%i_hr" %rep].shape[0]-5):
                #find closest diagonal pixel number (in first column)
                closest_ix = np.argmin(np.abs(blaze_diagonal_ixs_all[:, 0] - row))
                row_offset = blaze_diagonal_ixs_all[closest_ix, 0] - row
                
                
                #apply offset to diagonal indices
                blaze_diagonal_ixs = (blaze_diagonal_ixs_all[closest_ix, :] - row_offset)
                
                if np.all(blaze_diagonal_ixs < d2[h5_prefix]["array%i_hr" %rep].shape[0]):
                    diagonals.append(d2[h5_prefix]["array%i_hr" %rep][blaze_diagonal_ixs, px_ixs])
                    diagonals_aotf.append(d2[h5_prefix]["aotf_hr"][blaze_diagonal_ixs])

            diagonals = np.asarray(diagonals)
            diagonals_aotf = np.asarray(diagonals_aotf)
            d2[h5_prefix]["array_diag%i_hr" %rep] = diagonals
            d2[h5_prefix]["aotf_diag%i_hr" %rep] = diagonals_aotf


    """Save figures and files"""
    for h5_prefix in h5_prefixes:
        #save diagonally-correct array and aot freqs to hdf5
        with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" %h5_prefix), "w") as f:
            for rep in range(d2[h5_prefix]["nreps"]):
                f.create_dataset("array%02i" %rep, dtype=np.float32, data=d2[h5_prefix]["array_diag%i_hr" %rep], \
                                 compression="gzip", shuffle=True)
            f.create_dataset("aotf", dtype=np.float32, data=d2[h5_prefix]["aotf_diag%i_hr" %rep], \
                             compression="gzip", shuffle=True)
                
        #save miniscan png
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.title(h5_prefix)
        plt.imshow(d2[h5_prefix]["array_diag%i_hr" %rep], aspect="auto")
        # plt.savefig(os.path.join(MINISCAN_PATH, channel, "%s.png" %h5_prefix))
        plt.close()


    # for h5_prefix in h5_prefixes:
    #     fig1, (ax1a, ax1b) = plt.subplots(nrows=2)
    #     fig1.suptitle("Diagonals")
    #     for rep in range(d2[h5_prefix]["nreps"])[0:1]:
            
    #         diagonals = d2[h5_prefix]["array_diag%i_hr" %rep]
            
    #         stripe = diagonals[int(78*HR_SCALER), int(170*HR_SCALER):int(210*HR_SCALER)]
    #         plt.figure(); plt.plot(stripe)
        
    #         left = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(180*HR_SCALER)]
    #         right = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(200*HR_SCALER)]
            
    #         left_corr = left# * np.mean(right) / np.mean(left)
    #         left_right = left#np.mean([right, left_corr], axis=0)
    
    #         centre1 = diagonals[int(35*HR_SCALER):int(120*HR_SCALER), int(189*HR_SCALER)]
    #         centre_corr1 = centre1# * np.mean(left_right[0:10]) / np.mean(centre1[0:10])
        
    #         # plt.plot(left_corr)
    #         # plt.plot(right)
    #         # ax1a.plot(centre_corr1, color="C%i" %rep)
    #         # plt.plot(centre_corr2)
    #         # plt.plot(centre_corr3)
    #         # ax1a.plot(left_right, color="C%i" %rep)
            
    #         centre_corr1_sg = savgol_filter(centre_corr1, 7, 1)
    #         left_right_sg = savgol_filter(left_right, 7, 1)
            
    #         ax1a.plot(centre_corr1_sg, label=rep, color="C%i" %rep)
    #         ax1a.plot(left_right_sg, label=rep, color="C%i" %rep)
        
    #     # plt.figure()
    #         # ax1b.plot(centre_corr1 / left_right, label=rep)
    #         ax1b.plot(centre_corr1_sg / left_right_sg, label=rep, color="C%i" %rep)
    #         # plt.plot(centre_corr2 / left_right)
    #         # plt.plot(centre_corr3 / left_right)
    #     ax1a.legend()
    #     ax1b.legend()
        
    # stop()
    for h5_prefix in h5_prefixes:
        if plot_miniscans:
            
            #get blaze diagonal indices and set array values to nan for plotting
            plt.figure()
            for blaze_diagonal_ixs in blaze_diagonal_ixs_all:
                plt.plot(array_hr[blaze_diagonal_ixs, np.arange(array_hr.shape[1])])
                array_hr[blaze_diagonal_ixs, np.arange(array_hr.shape[1])] = np.nan
                
            
            plt.figure(figsize=(8, 5), constrained_layout=True)
            plt.title("Miniscan HR array: %s, %0.2fC" %(h5_prefix, temperature))
            plt.imshow(array_hr)
            plt.xlabel("Pixel number")
            plt.ylabel("Frame index (AOTF frequency)")
            
            
    
        
        
        
    #     # band depths
    #     depths = []
    #     for centre in solar_line_dict[h5_prefix]["centres"]:
    #         centre_px = int(centre * HR_SCALER)
    #         ixs_left = np.arange(centre_px - int(6 * HR_SCALER), centre_px - int(5 * HR_SCALER))
    #         ixs_right = np.arange(centre_px + int(5 * HR_SCALER), centre_px + int(6 * HR_SCALER))
            
    #         depth = []
    #         for line in diagonals:
    #             depth.append(band_depth(line, centre_px, ixs_left, ixs_right, ax=None))
    #         depths.append(depth)
        
    #     depths = np.asarray(depths)
        
    #     plt.figure()
    #     plt.plot(depths.T)
        

    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonals[:, (centre-1):(centre+1)], axis=1), label=centre)
    #     # plt.legend()

        
    #     # a = np.mean(diagonals[:, 1820:1860], axis=1)
        
        
    #     # diagonals = np.array
        
    #     diagonals_norm = diagonals / np.repeat(np.max(diagonals[:, :], axis=1), array_hr.shape[1]).reshape((-1, array_hr.shape[1]))

    #     plt.figure(figsize=(8, 5), constrained_layout=True)
    #     plt.title("Miniscan diagonally-corrected array: %s, %0.2fC" %(h5_prefix, temperature))
    #     plt.imshow(diagonals_norm, aspect="auto")
    #     plt.xlabel("Diagonally corrrected pixel number")
    #     plt.ylabel("Diagonally corrrected frame index (AOTF frequency)")
 
    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonals_norm[:, (centre-1):(centre+1)], axis=1), label=centre)
    #     # plt.legend()
    



    #     diagonal_median = np.median(diagonals_norm, axis=0)
        
    #     diagonal_div = diagonals_norm / diagonal_median

    #     # plt.figure()
    #     # plt.imshow(diagonal_div, aspect="auto")

        
    #     # plt.figure()
    #     # for i in range(diagonals_norm.shape[0]):
    #     #     plt.plot(diagonals_norm[i, :], alpha=0.1)
    #     # plt.plot(diagonal_median, "k")
            
            
    #     # plt.figure()
    #     # for centre in solar_line_dict[h5_prefix]["centres"]:
    #     #     plt.plot(np.sum(diagonal_div[:, (centre-10):(centre+10)], axis=1), label=centre)
    #     # plt.legend()



    
    # # if plot_blaze:
    # #     row_offsets = [-2, -1, 0, 1, 2]
    # #     extrapolated_blazes = make_blaze_functions(d2, row_offsets, array_name="array", plot=plot_miniscans)
    # #     mean_blaze = plot_blazes(d2, extrapolated_blazes)
        
    #     # #save mean blaze function to file
    #     # with open("blaze_%s.tsv" %channel.lower(), "w") as f:
    #     #     f.write("Pixel number\tBlaze function\n")
    #     #     for i, px_blaze in enumerate(mean_blaze):
    #     #         f.write("%i\t%0.4f\n" %(i, px_blaze))
    
    
    
    # #method 1 - normal band depth calc
    # # if plot_aotf:
    # #     plot_aotf(d2)
        
    # #method 2 - vertical stripe band depth calc

    # # for h5_prefix in h5_prefixes:        
    # #     pixel_number = solar_line_dict[h5_prefix]["centre"]
    # #     np.savetxt("raw_aotf_%s.tsv" %h5_prefix, d[h5_prefix]["y_rep"][0, :, pixel_number])
    # # # stop()
    
    # # for h5_prefix in h5_prefixes[0:1]:

    # #     pixel_number = solar_line_dict[h5_prefix]["centre"]
    # #     band_centre = d[h5_prefix]["y_rep"][0, :, pixel_number]
    # #     band_left = d[h5_prefix]["y_rep"][0, :, pixel_number-10]
    # #     band_right = d[h5_prefix]["y_rep"][0, :, pixel_number+10]
        
    # #     # row_centre = 
        
    # #     plt.figure()
    # #     plt.plot(band_centre)
    # #     plt.plot(band_left)
    # #     plt.plot(band_right)

