# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:29:14 2022

@author: iant

CALIBRATE LNO PHOBOS OBSERVATIONS
"""

import sys
import re
import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter



from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours
from tools.spectra.non_uniform_savgol import non_uniform_savgol
from tools.datasets.get_phobos_crism_data import get_phobos_crism_data

# solar_or_bb = "bb"
solar_or_bb = "solar"

if solar_or_bb == "bb":
    from instrument.calibration.lno_phobos.bb_ground_cal import rad_cal_order
else:
    from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order



PLOT_FIT_TO_PX_VALUES = False
# PLOT_FIT_TO_PX_VALUES = True

file_level = "hdf5_level_0p3a"



obs_types = {
    # "Hydration band":["20220713.*_LNO_._P"],
    "Hydration band":["20220710.*_LNO_._P", "20220713.*_LNO_._P", "20220725.*_LNO_._P", "20220826_03.*_LNO_._P", "20220826_19.*_LNO_._P"],
    "Carbonates":["20220714.*_LNO_._P", "20220719.*_LNO_._P"],
    "Phyllosilicates":["20220808.*_LNO_._P", "20220823.*_LNO_._P"]
    }




bad_pixel_d = {
    (146, 2):[188, 228, 121, 236, 294, ],
    (149, 2):[292, 310, 317, 301, 258, ],
    (152, 2):[5, 245, 129, 135, 178, ],
    (155, 2):[125, 232, 308, 78, 254, 294, 291, ],
}

crism_d = get_phobos_crism_data()


def get_lno_phobos_data(obs_type):

    data_d = {}
    
    for obs_ix, regex_str in enumerate(obs_types[obs_type]): #loop through observations of that type
        regex = re.compile(regex_str)
    
        h5_files, h5_filenames, _ = make_filelist(regex, file_level, path=r"E:\DATA\hdf5_phobos") #no detector offset!
        
        
                
        for file_ix, (h5_f, h5) in enumerate(zip(h5_files, h5_filenames)): #loop through orders of that observation
        
            h5_prefix = h5[0:15]
        
            # observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
            bins = h5_f["Science/Bins"][...]
            
            binning = h5_f["Channel/Binning"][0]
            
            # exponent = h5_f["Channel/Exponent"][...]
            # print(np.min(exponent), np.max(exponent))
            
            x = h5_f["Science/X"][0, :]
            x_mean = np.mean(x)
            y = h5_f["Science/Y"][...]
            t_mean = np.mean(h5_f["Temperature/NominalLNO"][...])
            
            order = h5_f["Channel/DiffractionOrder"][0]
            
            #on first run find unique bins and colours
            if file_ix == 0:
                unique_bins = sorted(list(set(bins[:, 0])))
        
        
            
            for bin_ix, unique_bin in enumerate(unique_bins):
                
                if unique_bin not in data_d.keys():
                    data_d[unique_bin] = {}
                
                    
                indices = np.where(bins[:, 0] == unique_bin)[0]
                y_bin = y[indices, :]
                
                bad_pixels = bad_pixel_d[(unique_bin, binning)]
                
                y_bin[:, bad_pixels] = np.nan

                if h5_prefix not in data_d[unique_bin].keys():
                    data_d[unique_bin][h5_prefix] = {}

                if order not in data_d[unique_bin][h5_prefix].keys():
                    data_d[unique_bin][h5_prefix][order] = {"x":x, "x_mean":x_mean, "y":y_bin, "t":t_mean}
                    
            h5_f.close()

    return data_d



data_d = get_lno_phobos_data("Hydration band")
# data_d = get_lno_phobos_data("Carbonates")
# data_d = get_lno_phobos_data("Phyllosilicates")

#bin -> h5 prefix -> order -> {x[320], y[:, 320], t}

order = 164
# order = 170
# order = 177
h5_prefixes = ["20220710_200313", "20220713_164911"]
bin_ = 146
binning = 2
order_dict = {
    148:{"colour":"C0"},
    153:{"colour":"C1"},
    158:{"colour":"C2"},
    164:{"colour":"C3"},
    170:{"colour":"C4"},
    177:{"colour":"C5"},
}

if solar_or_bb == "bb":
    cal_h5 = "20150426_054602_0p1a_LNO_1"
else:
    cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
cal_d = {order:rad_cal_order(cal_h5, order) for order in order_dict.keys()}

solar_scalars = {order:cal_d[order]["y_centre_mean"] / 2.0e6 for order in cal_d.keys()}


#fix detector offsets



# for order in data_d[bin_][h5_prefixes[0]].keys():
#     y = data_d[bin_][h5_prefixes[0]][order]["y"]
#     plt.figure()
#     plt.title("%i" %order)
#     plt.plot(y)


# plt.figure()
# plt.title("%i" %order)
# for bin_ in data_d.keys():
#     y = data_d[bin_][h5_prefixes[1]][order]["y"]
#     y_mean = np.nanmean(y, axis=1)
#     plt.plot(y_mean, label=bin_)
# plt.legend()

# for i in range(22, 30):
#     plt.figure()
#     plt.title("%i" %order)
#     for bin_ in data_d.keys():
#         y = data_d[bin_][h5_prefixes[0]][order]["y"]
#         plt.plot(y[i, :], label=bin_)
#     plt.legend()


#compare standard deviation of all bins in a single order
fig1, ax1 = plt.subplots(figsize=(10, 8), constrained_layout=True)
fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
ax1.set_title(order)

d = {}
for h5_prefix in data_d[bin_].keys():
    
    d[h5_prefix] = {}
    
    for order in data_d[bin_][h5_prefix].keys():
    
        y = np.array([data_d[bin_][h5_prefix][order]["y"] for bin_ in data_d.keys()]) #get data for all bins
        y_frames_std = np.nanstd(y, axis=(0, 2))
        ax1.plot(y_frames_std, label=h5_prefix)
        
        good_ixs = np.where(y_frames_std < 12.0)[0]
        
        #remove offsets by comparing to last bin 
        #pixel by pixel doesn't work -> too digitised
        # y_good = y[:, good_ixs, :]
        # y_good_ref = np.zeros_like(y_good)
        # for i in range(len(y_good[:, 0, 0])):
        #     y_good_ref[i, :, :] = y_good[-1, :, :] #divide by one bin
        # y_norm = y_good - y_good_ref
        # ax2.plot(np.nanmean(y_norm, axis=2).T, label=h5_prefix)
    
    
        # #get mean of all spectral pixels then divide
        y_good = np.nanmean(y[:, good_ixs, :], axis=2)
        
        y_good_ref = np.zeros_like(y_good)
        for i in range(len(y_good[:, 0])):
            y_good_ref[i, :] = y_good[-1, :]
        
        y_norm = y_good - y_good_ref
        
        ax2.plot(y_norm.T, label=h5_prefix)
        
        d[h5_prefix][order] = [np.mean(y_norm[i, :]) / solar_scalars[order]  for i in range(4)]
    
ax1.legend()
ax2.legend()




fig2, ax2a = plt.subplots(figsize=(15, 10), constrained_layout=True)
fig2.suptitle("Phobos Radiance Calibration")
ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="r", marker="x")
ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="b", marker="x")


for h5_prefix in data_d[bin_].keys():
    
    orders = data_d[bin_][h5_prefix].keys()
    
    phobos_corrected_counts = [d[h5_prefix][order][1] for order in orders]
    
    phobos_scaled_counts = phobos_corrected_counts / np.max(phobos_corrected_counts) * 0.07
    
    x_means = [10000.0 / cal_d[order]["x_mean"] for order in orders]
    

    ax2a.plot(x_means, phobos_scaled_counts, label=h5_prefix)
    ax2a.scatter(x_means, phobos_scaled_counts, label=h5_prefix)

ax2a.set_ylim((0, np.max(crism_d["phobos_blue"]) * 1.05))
# ax2a.errorbar(x, y_scaled, yerr=y_err_scaled, color=colour, capsize=4, label=label)





stop()



# y_means = []
# for order_ix, order in enumerate(data_d[bin_][h5_prefixes[0]].keys()):
#     y = data_d[bin_][h5_prefixes[0]][order]["y"]
#     y_mean = np.nanmean(y, axis=1)
    
#     y_means.append(y_mean)

# plt.figure()
# plt.title("%i %i" %(bin_, order))
# for y_mean in y_means:
#     plt.plot(y_mean)



# fig1, ax1 = plt.subplots(figsize=(10, 8), constrained_layout=True)

nu_order = np.zeros(len(data_d[bin_][h5_prefixes[0]].keys()))
y_filtered = np.zeros(len(data_d[bin_][h5_prefixes[0]].keys()))

for h5_prefix in h5_prefixes:

    for order_ix, order in enumerate(data_d[bin_][h5_prefix].keys()):
        
        colour = order_dict[order]["colour"]
    
        #get all spectra for a bin from one file at one order
        y = data_d[bin_][h5_prefix][order]["y"]
        
        if PLOT_FIT_TO_PX_VALUES:
            plt.figure()
        
        x_centre = np.arange(y.shape[0])
        y_centre = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            y_spectrum = y[i, :]

            not_nan_indices = np.isfinite(y_spectrum)

            y_sorted = np.array(sorted(y_spectrum[not_nan_indices]))
            
            x_reduced = np.arange(50, 270)
            y_reduced = y_sorted[50:270]
            
            
            # plt.plot(y_spectrum)
            if PLOT_FIT_TO_PX_VALUES:
                plt.plot(y_sorted)
                plt.scatter(x_reduced, y_reduced)
            
            polyfit = np.polyfit(x_reduced, y_reduced, 2)
            y_fit = np.polyval(polyfit, x_reduced)
            y_point = np.polyval(polyfit, 160.)

            if PLOT_FIT_TO_PX_VALUES:
                plt.plot(x_reduced, y_fit)
                plt.scatter(160., y_point, color="k")
            
            y_centre[i] = y_point
            
        if PLOT_FIT_TO_PX_VALUES:
            sys.exit()
        
        #solar reference
        y_ref = cal_d[order]["y_spectrum"] #get solar spectrum measured by LNO
        
        
        bad_pixels = bad_pixel_d[(bin_, binning)] #get bad pixel indices
        y_ref[bad_pixels] = np.nan #set as nan
        not_nan_indices = np.isfinite(y_ref) #apply same bad pixel

        
        y_ref_sorted = np.array(sorted(y_ref[not_nan_indices])) #sort the good pixels
        y_ref_reduced = y_ref_sorted[50:270] #remove detector edges
        x_reduced = np.arange(50, 270)
        polyfit_ref = np.polyfit(x_reduced, y_ref_reduced, 2) #fit quadratic to sorted values
        y_ref_fit = np.polyval(polyfit_ref, x_reduced) #make quadratic from fit
        y_ref_point = np.polyval(polyfit_ref, 160.) #fit to pixel 160
        
        y_centre_scaled = y_centre / y_ref_point #calibrate with solar spectrum: scale phobos fit to pixel 160 to solar cal fit to pixel 160
    
        # plt.plot(y_ref_sorted)
        # plt.scatter(x_reduced, y_ref_reduced, label=order)
    
    
    
        y_filter = non_uniform_savgol(x_centre, y_centre_scaled, 19, 1)
        # y_filter = np.polyval(np.polyfit(x_centre, y_centre_scaled, 5), x_centre)
        # y_filter = savgol_filter(y_centre, 39, 1)
        plt.plot(y_centre_scaled, label=order, color=colour)
        plt.plot(y_filter, label="%s: %i" %(h5_prefix, order), color=colour)
        
        
        nu_order[order_ix] = data_d[bin_][h5_prefix][order]["x_mean"]
        y_filtered[order_ix] = np.mean(y_filter)
    
    stop()
    
    # plt.legend()
        
    y_filtered_scaled = y_filtered / np.max(y_filtered) * 0.065
    
    ax1.plot(10000. / nu_order, y_filtered_scaled, color="C%i" %(bin_ + 2), marker="x", label="LNO %s bin %i first calibration (scaled)" %(h5_prefix, bin_))
ax1.scatter(crism_d["x"], crism_d["phobos_red"], color="tab:red", marker="x", label="CRISM Phobos Red Unit")
ax1.scatter(crism_d["x"], crism_d["phobos_blue"], color="tab:blue", marker="x", label="CRISM Phobos Blue Unit")

ax1.set_title("LNO Phobos Spectra")
ax1.set_xlabel("Wavelength um")
ax1.set_ylabel("CRISM Radiance Factor")
ax1.legend()
ax1.grid()






# plt.figure()
# plt.plot(y.T)
# plt.plot(np.mean(y, axis=0), "k")

# fig2, ax2a = plt.subplots(figsize=(15, 10), constrained_layout=True)
# fig2.suptitle("Phobos Radiance Calibration: %s" %obs_type)



# fig1, (ax1a, ax1b) = plt.subplots(figsize=(15, 10), nrows=2, sharex=True, constrained_layout=True)

# rad_d = {"wavenumber":[], "bin":[], "radiance":[], "radiance_error":[], "h5":[]}


# counts_per_rad = cal_d["counts_per_rad"]



# ydimensions = y.shape
# nSpectra = ydimensions[0]

#on first run find unique bins and colours
# if file_ix == 0:
#     unique_bins = sorted(list(set(bins[:, 0])))
#     colours = get_colours(len(unique_bins))


                
                # y_mean = np.mean(y[indices, :], axis=0)
        
                # if file_ix == 0:
                #     label = unique_bin
                # else:
                #     label = ""
        
                # y_rad = y_mean# / counts_per_rad        
        
                # ax1a.plot(x, y_mean, label=label, color=colours[bin_ix])
                # ax1b.plot(x, y_rad, label=label, color=colours[bin_ix])
                
                # x_mean = np.mean(x)
                # y_rad_mean = np.mean(y_rad[160:240])
                # y_rad_std = np.std(y_rad[160:240])
    
                # y_rad_mean = np.mean(y_rad)
                # y_rad_std = np.std(y_rad)
                
                # rad_d["h5"].append(h5)
                # rad_d["wavenumber"].append(x_mean)
                # rad_d["bin"].append(unique_bin)
                # rad_d["radiance"].append(y_rad_mean)
                # rad_d["radiance_error"].append(y_rad_std)
                
        
        # ax1a.legend()
        # ax1a.grid()
        # ax1b.legend()
        # ax1b.grid()
        
        
        #convert to np arrays
    #     for key in rad_d.keys():
    #         rad_d[key] = np.array(rad_d[key])
        
        
        
        
    #     for bin_ix, unique_bin in enumerate(unique_bins):
            
    #         bin_colours = {146:"tab:purple", 149:"tab:orange", 152:"tab:green", 155:"tab:cyan"}
    #         colour = bin_colours[unique_bin]
    #         if obs_ix == 0:
    #             label = unique_bin
    #         else:
    #             label = ""
            
    #         indices = np.where(rad_d["bin"] == unique_bin)[0]
            
    #         # x = rad_d["wavenumber"][indices]
    #         x = 10000.0 / rad_d["wavenumber"][indices]
    #         y = rad_d["radiance"][indices]
    #         y_err = rad_d["radiance_error"][indices]
    #         y_max = np.max(y)

    #         y_scaled = y
    #         y_err_scaled = y_err
            
    #         # if unique_bin == 155:
    #         #     y_scaled = y
    #         #     y_err_scaled = y_err / y_scaled

    #         if unique_bin != 155:
    #             y_scaled = y / y_max * 0.04
    #             y_err_scaled = y_err / y_scaled
            
    #         # ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="r", marker="x")
    #         # ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="b", marker="x")
            
    #         ax2a.scatter(x, y_scaled, color=colour, label=label)
    #         ax2a.errorbar(x, y_scaled, yerr=y_err_scaled, color=colour, capsize=4, label=label)
        
    # ax2a.legend()
    # ax2a.grid()
    # ax2a.set_xlabel("Diffraction order central wavelength")
    # ax2a.set_ylabel("Mean radiance of diffraction order centre W/cm2/sr/cm-1")