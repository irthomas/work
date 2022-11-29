# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:13:26 2022

@author: iant

INVESTIGATE BLAZE FUNCTION VS TEMPERATURE FROM MINISCANS
"""

# import sys
import re
import numpy as np

import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.cprint import cprint
from tools.spectra.running_mean import running_mean_1d
from instrument.nomad_so_instrument_v03 import m_aotf, aotf_peak_nu, lt22_waven


# inflight
file_level = "hdf5_level_1p0a"
# regex = re.compile(".*_LNO_.*_CM")
# regex = re.compile(".*_SO_.*_CM")
regex = re.compile("20.*_SO_.*_CM")

# #ground
# file_level = "hdf5_level_0p1a"
# regex = re.compile("20150404_(08|09|10)...._.*")  #all observations with good lines (CH4 only)


# aotf_steppings = [1.0, 2.0, 4.0]
aotf_steppings = [8.0]
binnings = [0]
starting_orders = [188]


# aotf_steppings = [4.0]
# binnings = [0]
# starting_orders = [191]


#TODO: make dictionary of fft_cutoff for each aotf_stepping
fft_cutoff_dict = {
    1:4,
    2:15,
    4:15,
    8:40,
    }


#solar line dict
solar_line_dict = {
    # "20181206_171850-191-4":{"left":np.arange(215, 222), "centre":229, "right":np.arange(234, 245)},
    # "20211105_155547-191-4":{"left":np.arange(208, 215), "centre":222, "right":np.arange(227, 238)},

    # "20181206_171850":{"left":np.arange(144, 148), "centre":149, "right":np.arange(153, 157)},
    # "20211105_155547":{"left":np.arange(137, 141), "centre":142, "right":np.arange(146, 150)},

    # "20181206_171850":{"left":np.arange(201, 205), "centre":206, "right":np.arange(208, 212)},
    # "20211105_155547":{"left":np.arange(194, 198), "centre":200, "right":np.arange(201, 205)},
    
    # "20190416_020948-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
    # "20210717_072315-194-1":{"left":np.arange(205, 209), "centre":217, "right":np.arange(223, 227)},
    # "20220120_125011-194-1":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},

    "20201010_113533-188-8":{"left":np.arange(209, 213), "centre":221, "right":np.arange(227, 231)},
    "20210201_111011-188-8":{"left":np.arange(209, 213), "centre":218, "right":np.arange(220, 224)},
    "20210523_001053-188-8":{"left":np.arange(205, 209), "centre":212, "right":np.arange(216, 220)},
    "20221011_132104-188-8":{"left":np.arange(209, 213), "centre":215, "right":np.arange(218, 222)},
    }



list_files = True
# list_files = False


#find new solar lines
plot_miniscans = True
# plot_miniscans = False


plot_fft = True
# plot_fft = False

plot_blaze = True
# plot_blaze = False

plot_absorptions = True
# plot_absorptions = False

plot_aotf = True
# plot_aotf = False





def get_miniscan_data_0p1a(regex):


    h5_files, h5_filenames, _ = make_filelist(regex, file_level, silent=True)

    d = {}
    for h5_f, h5 in zip(h5_files, h5_filenames):
    
        h5_prefix = h5[0:15]
        
        d[h5_prefix] = {}
        
        
        
        y = h5_f["Science/Y"][...] #y is 3d in 0.1A
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]
        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        
        orders = np.array([m_aotf(i) for i in unique_aotf_freqs])
        unique_orders = sorted(list(set(orders)))
        
        
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]
        
        print("Miniscan stepping = %0.1fkHz" %(aotf_freqs_step))
        print(h5, "orders", unique_orders[0])
    
        
        bin_ = 12
        
        y_bin = y[:, bin_, :]
    
        for unique_aotf_freq in unique_aotf_freqs:
            
            aotf_ixs = np.where(aotf_freqs == unique_aotf_freq)[0]
            
            for aotf_ix in aotf_ixs[0:1]:
                y_spectrum = y_bin[aotf_ix, :]
        
                d[h5_prefix][unique_aotf_freq] = {0.0:y_spectrum} #set temperature to 0
            
    return d





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

        h5_prefix = "%s-%i-%i" %(h5[0:15], np.min(unique_orders), np.round(aotf_freqs_step))
        
        
        
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
        if "SO" in h5:
            good_bins = np.arange(126, 131)
        elif "LNO" in h5:
            good_bins = np.arange(150, 155)


        print("Getting data for %s" %h5)
        h5_f = open_hdf5_file(h5)

        # observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
        bins = h5_f["Science/Bins"][:, 0]
        y = h5_f["Science/Y"][...]
        t = h5_f["Channel/InterpolatedTemperature"][...]
        aotf_freqs = h5_f["Channel/AOTFFrequency"][...]
        
        aotf_freqs = [int(np.round(i)) for i in aotf_freqs]

        unique_aotf_freqs = sorted(list(set(aotf_freqs)))
        # unique_bins = sorted(list(set(bins)))
        starting_order = m_aotf(np.min(unique_aotf_freqs))
        aotf_freqs_step = unique_aotf_freqs[1] - unique_aotf_freqs[0]

        h5_prefix = "%s-%i-%i" %(h5[0:15], starting_order, np.round(aotf_freqs_step))
        
        d[h5_prefix] = {}
        for unique_aotf_freq in unique_aotf_freqs:
            indices = [0]+[i for i, (bin_, aotf_freq) in enumerate(zip(bins, aotf_freqs)) if ((bin_ in good_bins) and (aotf_freq == unique_aotf_freq))]+[9999999]
            
        
        
            joining_indices = list(np.where(np.diff(indices) > 1)[0])
            
            d[h5_prefix][unique_aotf_freq] = {}
            for i in range(len(joining_indices) - 2):
                aotf_bin_consec_indices = np.arange(joining_indices[i]+1, joining_indices[i+1]+1, 1)
                aotf_bin_indices = np.array(indices)[aotf_bin_consec_indices]
                t_mean = np.mean(t[aotf_bin_indices])
                
                y_binned = np.mean(y[aotf_bin_indices, :], axis=0)
            
                d[h5_prefix][unique_aotf_freq][t_mean] = y_binned
    
    return d






def remove_oscillations(d, cut_off_inner=True, plot=False):
    """make dictionary of miniscan arrays before and after oscillation removal
    input: dictionary of raw miniscan data,
    fft_cutoff: index to start setting fft to zero (symmetrical from centre)
    cut_off_inner: whether to set the inner indices to zero (removes high res oscillations) 
    or outer indices to zero (removes large features)"""
    
    d2 = {}
    for h5_prefix in d.keys():
        
        stepping = int(h5_prefix.split("-")[-1])
        fft_cutoff = fft_cutoff_dict[stepping]

        
        miniscan_array = np.zeros((len(d[h5_prefix].keys()), 320))
        for i, aotf_freq in enumerate(d[h5_prefix].keys()):
            
            for temperature in list(d[h5_prefix][aotf_freq].keys())[0:1]:
                miniscan_array[i, :] = d[h5_prefix][aotf_freq][temperature] #get 2d array for 1st temperature in file
        
        #bad pixel correction
        miniscan_array[:, 269] = np.mean(miniscan_array[:, [268, 270]], axis=1)
        

        fft = np.fft.fft2(miniscan_array)
        
        if plot:
            fig, ax = plt.subplots()
            ax.set_title(h5_prefix)
            ax.plot(fft.real[:, 200])
            # ax.plot(fft.imag[:, 200])
        
        if cut_off_inner:
            fft.real[fft_cutoff:(256-fft_cutoff), :] = 0.0
            fft.imag[fft_cutoff:(256-fft_cutoff), :] = 0.0
            
        else:
            fft.real[0:fft_cutoff, :] = 0.0
            fft.real[(256-fft_cutoff):, :] = 0.0
            fft.imag[0:fft_cutoff, :] = 0.0
            fft.imag[(256-fft_cutoff):, :] = 0.0

        if plot:
            ax.plot(fft.real[:, 200], linestyle=":")
            # ax.plot(fft.imag[:, 200])


        ifft = np.fft.ifft2(fft).real
    
        d2[h5_prefix] = {"array_raw":miniscan_array, "array_corrected":ifft, "aotf":np.asfarray(list(d[h5_prefix].keys())), "t":temperature}

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





def make_blaze_functions(d2, row_offsets, array_name="array_corrected", plot=False):
    """inputs:
    oscillation corrected/uncorrected miniscan array
    row_offsets: find blaze functions for rows above and below the peak blaze
    These have the same shape but a lower intensity which is corrected for by the normalisation
    array_name: the name of the dictionary key in the miniscan array"""
    
    extrapolated_blazes = {}
    
    for h5_ix, h5_prefix in enumerate(d2.keys()):
        extrapolated_blazes[h5_prefix] = []
        
        #spectral calibration to find peak aotf and pixel wavenumbers
        t = d2[h5_prefix]["t"]
        aotf_freqs = d2[h5_prefix]["aotf"]
        aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
        orders = [m_aotf(i) for i in aotf_freqs]
        
        px_nus = [lt22_waven(i, t) for i in orders]
        
        #pixel position of peak AOTF in each frame
        px_peaks = []
        for aotf_nu, px_nu in zip(aotf_nus, px_nus):
            px_peak = (np.abs(px_nu - aotf_nu)).argmin()
            px_peaks.append(px_peak)
    
        array_corrected = d2[h5_prefix][array_name]
            
        for offset in row_offsets:
            #make array of blaze functions, one blaze for each row offset value
            blazes = []
            px_ix = 0
            blaze = []
            for array_row_ix, px_peak in enumerate(px_peaks):
                
                if px_peak + offset < px_ix: #start new list when order changes
                    blazes.append(blaze)
                    blaze = []
                
                px_ix = px_peak + offset
                
                blaze.append([px_ix, array_row_ix]) #save pixel indices, column and row
            
            
            if offset == 0 and plot:
                array_peaks = array_corrected.copy()
    
            for blaze in blazes:
                blaze = np.array(blaze)
                
                #extrapolate to get pixel indices for whole detector and interpolate between pixels
                polyfit = np.polyfit(blaze[:, 0], blaze[:, 1], 1)
                px_range = np.arange(320)
                px_extrap = np.array([int(np.round(i)) for i in np.polyval(polyfit, px_range)])
                
                if np.any(px_extrap < 0.0):
                    continue
                
                #not interpolated or extrapolated
                # px_row = array_corrected[blaze[:, 1], blaze[:, 0]]
                
                #interpolated/extrapolated
                px_row = array_corrected[px_extrap, px_range]
    
                extrapolated_blazes[h5_prefix].append(px_row/np.max(px_row))
    
                if offset == 0 and plot:
            
                    """plot on miniscan array where pixel nu = aotf nu i.e. the diagonals"""
                    for px_row_ix, px_column_ix in zip(px_range, px_extrap):
                        array_peaks[px_column_ix, px_row_ix] = -999
                
                # for row, px_peak in enumerate(px_peaks):
                #     array_peaks[row, px_peak] = -999
    
    
            if offset == 0 and plot:
                plt.figure(figsize=(8, 5), constrained_layout=True)
                plt.title("Miniscan corrected array")
                plt.imshow(array_peaks)

    return extrapolated_blazes




def plot_blazes(d2, extrapolated_blazes):
    fig, ax = plt.subplots()
    for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
        blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
        ax.plot(blazes.T, color="C%i" %h5_ix, alpha=0.1) #linestyle=linestyles[h5_ix], alpha=0.2)
    
    
    blaze_all = []
    for h5_ix, h5_prefix in enumerate(extrapolated_blazes.keys()):
        blazes = np.array(extrapolated_blazes[h5_prefix]) #N x 320 pixels
        blazes_median = np.median(blazes, axis=0)
        
        t = d2[h5_prefix]["t"]
        
        #smooth the median and plot
        blazes_median_rm = running_mean_1d(blazes_median, 9)
        ax.plot(blazes_median_rm, color="C%i" %h5_ix, label="%s, %0.2fC" %(h5_prefix, t))
        blaze_all.append(blazes_median_rm)
        
    ax.legend()
    ax.grid()
    ax.set_title("Derived blaze functions")
    ax.set_xlabel("Pixel number")
    ax.set_ylabel("Normalised blaze function")
    
    mean_blaze = np.mean(np.array(blaze_all), axis=0)
    
    return mean_blaze / np.max(mean_blaze)



 

def band_depth(line, centre_px, ixs_left, ixs_right, ax=None):
    """solar line band depth"""

    abs_pxs = np.concatenate((ixs_left, ixs_right))
    abs_vals = line[abs_pxs]
    
    cont_pxs = np.arange(ixs_left[0], ixs_right[-1] + 1)
    cont_vals = np.polyval(np.polyfit(abs_pxs, abs_vals, 2), cont_pxs)
    
    abs_vals = line[cont_pxs] / cont_vals
    abs_depth = 1.0 - abs_vals[centre_px - ixs_left[0]]
    
    if ax:
        ax.plot(cont_pxs, abs_vals)
    
    return abs_depth


def make_aotf_functions(d2, array_name="array_corrected", plot=plot_absorptions):
    """get aotf function from depth of a solar line"""
    
    aotfs = {}
    for h5_prefix in d2.keys():
        
        
        t = d2[h5_prefix]["t"]
    
        aotf_freqs = d2[h5_prefix]["aotf"]
        aotf_nus = [aotf_peak_nu(i, t) for i in aotf_freqs]
        array = d2[h5_prefix][array_name]
    
        if plot:
            fig, ax = plt.subplots()
            ax.set_title(h5_prefix)
        
        ixs_left = solar_line_dict[h5_prefix]["left"]
        ixs_right = solar_line_dict[h5_prefix]["right"]
        centre_px = solar_line_dict[h5_prefix]["centre"]
        
        #absorption depth variations
        abs_depths = []
        for row_ix in np.arange(array.shape[0]):
            line = array[row_ix, :]
            
            if plot:
                abs_depth = band_depth(line, centre_px, ixs_left, ixs_right, ax=ax)
            else:
                abs_depth = band_depth(line, centre_px, ixs_left, ixs_right)
            abs_depths.append(abs_depth)
            
            # plt.plot(line)
            # plt.plot(cont_pxs, cont_vals)
            
        aotfs[h5_prefix] = {"aotf_freqs":aotf_freqs, "aotf_nus":aotf_nus, "abs_depths":np.array(abs_depths)}

    return aotfs


def plot_aotf(d2, array_name="array_corrected"):
    aotfs = make_aotf_functions(d2, array_name=array_name)
    
    plt.figure()
    for h5_prefix in aotfs.keys():
    
        t = d2[h5_prefix]["t"]
        
        aotf_nus = aotfs[h5_prefix]["aotf_nus"]
        aotf_func = aotfs[h5_prefix]["abs_depths"]
        
        plt.plot(aotf_nus, aotf_func/np.max(aotf_func), label="%s, %0.2fC" %(h5_prefix, t))
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Normalised AOTF Function")
        plt.title("AOTF Functions")

        with open("aotf_%s.tsv" %h5_prefix, "w") as f:
            f.write("Wavenumber\tAOTF function\n")
            for aotf_nu, aotf_f in zip(aotf_nus, aotf_func):
                f.write("%0.4f\t%0.4f\n" %(aotf_nu, aotf_f))


    plt.grid()
    plt.legend()



"""get data"""
if __name__ == "__main__":
    if file_level == "hdf5_level_1p0a":
        if list_files:
            h5_filenames, h5_prefixes = list_miniscan_data_1p0a(regex, starting_orders, aotf_steppings, binnings)
            if "d" not in globals():
                d = get_miniscan_data_1p0a(h5_filenames)
            if h5_prefixes != list(d.keys()):
                d = get_miniscan_data_1p0a(h5_filenames)
    
    
    
    # elif file_level == "hdf5_level_0p1a":
    #     d = get_miniscan_data_0p1a(regex)
    
    
    
    d2 = remove_oscillations(d, plot=plot_fft)
    h5_prefix = list(d2.keys())[-1]
    
    
    """plot miniscan arrays"""
    if plot_miniscans:
        for h5_prefix in d2.keys():
            temperature = d2[h5_prefix]["t"]
            array = d2[h5_prefix]["array_raw"]
            array_corrected = d2[h5_prefix]["array_corrected"]
        
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
    
    
    
    
    if plot_blaze:
        row_offsets = [-2, -1, 0, 1, 2]
        extrapolated_blazes = make_blaze_functions(d2, row_offsets, array_name="array_corrected", plot=plot_miniscans)
        mean_blaze = plot_blazes(d2, extrapolated_blazes)
        
        with open("blaze.tsv", "w") as f:
            f.write("Pixel number\tBlaze function\n")
            for i, px_blaze in enumerate(mean_blaze):
                f.write("%i\t%0.4f\n" %(i, px_blaze))
    
    
    
    if plot_aotf:
        plot_aotf(d2)

"""simulate miniscan from solar spectrum"""


"""investigate oscillations on solar spectrum"""