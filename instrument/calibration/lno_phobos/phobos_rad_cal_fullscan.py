# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:09:47 2022

@author: iant

PHOBOS FULLSCAN ANALYSIS
"""

# import os
import sys
import numpy as np
import matplotlib.pyplot as plt



from instrument.nomad_lno_instrument_v01 import m_aotf
from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
# from tools.general.normalise_values_to_range import normalise_values_to_range
from tools.datasets.get_phobos_crism_data import get_phobos_crism_data

"""decide what to plot: 
    on first run, select -2 to show where spectra get noisy, then add good_indices to obs_type dictionary
    then select -1 to check for bad pixels. If spikes are observed, add to bad_pixel_d
    then 0 runs through the full calibration plotting everything
    to just plot the final result, select option 4"""

# plot_level = -2 #find where signal gets too noisy
# plot_level = -1 #check bad pixels

# plot_level = 0
# plot_level = 2
# plot_level = 3
plot_level = 4

bad_pixel_d = {
    (148, 1):[288, 294, ],
    (149, 1):[292, 310, 317, ],
    (150, 1):[301, ],
    (151, 1):[258, ],
    (152, 1):[],
    (153, 1):[5, 40, 245, ],
    (154, 1):[129, 135, 178, ],
    (155, 1):[125, 232, 308, ],
    
    #new 60s fullscan with 16 rows
    (144, 2):[12, 102, 312],
    (146, 2):[121, 236, 314],
    (148, 2):[13, 92],
    (150, 2):[],
    (152, 2):[37, 40, 245],
    (154, 2):[],
    (156, 2):[78],
    (158, 2):[92, 105, 203, 245],
    
    (146, 3):[13, 188, 228, 121, 151, 236, 294, 314, ],
    (149, 3):[1, 3, 46, 92, 292, 296, 310, 317, 301, 258, ],
    (152, 3):[5, 37, 40, 49, 129, 135, 178, 245, 266, ],
    (155, 3):[125, 232, 308, 78, 181, 209, 254, 294, 291, ],

}


subtract_last_bin = True
# subtract_last_bin = False

good_indices = ""


"""observation name:{"h5":hdf5 filename, "good_indices":frames with best signal and least noise, "good_row_indices":bins with highest signal, "orders_crism":orders to fit to CRISM spectra}"""
obs_types = {
    "Gap 2 orders 1 60s":{"h5":"20230713_092806_0p1a_LNO_1", "good_indices":[*range(40, 96)], "good_row_indices":[1], "orders_crism":[160, 168, 170]},
    "Gap 3 orders 1 60s":{"h5":"20230716_061435_0p1a_LNO_1", "good_indices":[*range(1, 100)], "good_row_indices":[1], "orders_crism":[160, 169, 172]}, #frames 1-100 are good, but low signal on 1-50

    "Hydration band 10":{"h5":"20230320_122418_0p1a_LNO_1", "good_indices":[*range(2, 100)], "good_row_indices":[1], "orders_crism":[158, 170, 177]},
    "Hydration band 11":{"h5":"20230323_011917_0p1a_LNO_1", "good_indices":[*range(2, 115)], "good_row_indices":[1], "orders_crism":[158, 170, 177]},
    "Hydration band 12":{"h5":"20230323_091017_0p1a_LNO_1", "good_indices":[*range(2, 116)], "good_row_indices":[1], "orders_crism":[158, 170, 177]},
    "Hydration band 13":{"h5":"20230701_222009_0p1a_LNO_1", "good_indices":[*range(5, 60)], "good_row_indices":[1], "orders_crism":[158, 170, 177]},
    "Hydration band 14":{"h5":"20230703_193210_0p1a_LNO_1", "good_indices":[*range(5, 85)], "good_row_indices":[1], "orders_crism":[158, 170, 177]},

    "Carbonates 6":{"h5":"20230429_181516_0p1a_LNO_1", "good_indices":[*range(0, 94)], "good_row_indices":[1], "orders_crism":[174, 175, 176, 190, 191, 192]},
    "Carbonates 7":{"h5":"20230505_114846_0p1a_LNO_1", "good_indices":[*range(0, 91)], "good_row_indices":[0,1], "orders_crism":[174, 175, 176, 190, 191, 192]},
    "Carbonates 8":{"h5":"20230511_131415_0p1a_LNO_1", "good_indices":[*range(0, 80)], "good_row_indices":[1], "orders_crism":[174, 175, 176, 190, 191, 192]},


    # "Fullscan 1":{"h5":"20221011_024508_0p1a_LNO_1", "good_indices":[*range(20, 106)]},
    # "Fullscan 3 60s":{"h5":"20230223_011421_0p1a_LNO_1", "good_indices":[*range(0, 59)], "good_row_indices":range(2,4), "orders_crism":[166, 172, 178, 184]},
    # "Fullscan 4 60s":{"h5":"20230225_220121_0p1a_LNO_1", "good_indices":[*range(5, 38)], "good_row_indices":range(2,4), "orders_crism":[166, 172, 178, 184]},

    # "Hydration band 1":{"h5":"20220710_200313_0p1a_LNO_1", "good_indices":[*range(20, 149)]},
    # "Hydration band 2":{"h5":"20220713_164911_0p1a_LNO_1", "good_indices":[*range(20, 149)]},
    # "Hydration band 3":{"h5":"20220725_035605_0p1a_LNO_1", "good_indices":[*range(20, 127)]},
    # "Hydration band 4":{"h5":"20220826_031920_0p1a_LNO_1", "good_indices":[*range(21, 97)]},
    # "Hydration band 5":{"h5":"20220826_190220_0p1a_LNO_1", "good_indices":[*range(21, 99)]},

    # "Carbonates 1":{"h5":"20220714_004111_0p1a_LNO_1", "good_indices":[*range(20, 113)]},
    # "Carbonates 2":{"h5":"20220719_102309_0p1a_LNO_1", "good_indices":[*range(20, 180)]},

    # "Phyllosilicates 1":{"h5":"20220808_223604_0p1a_LNO_1", "good_indices":[*range(27, 130)]},
    # "Phyllosilicates 2":{"h5":"20220823_063050_0p1a_LNO_1", "good_indices":[*range(15, 90)]},


}







# colour_bin_dict = {
#     148:{"colour":"C0"},
#     149:{"colour":"C1"},
#     150:{"colour":"C2"},
#     151:{"colour":"C3"},
#     152:{"colour":"C4"},
#     153:{"colour":"C5"},
#     154:{"colour":"C6"},
#     155:{"colour":"C7"},
# }

ls_bin_dict = {
    146:"solid",
    149:"dotted",
    152:"dashed",
    155:"dashdot",
    }

crism_d = get_phobos_crism_data()


for obs_type in obs_types.keys():
    
    h5 = obs_types[obs_type]["h5"]
    good_indices = obs_types[obs_type]["good_indices"]
    good_row_indices = obs_types[obs_type]["good_row_indices"]

    h5_f = open_hdf5_file(h5, path=r"E:\DATA\hdf5_phobos") #no detector offset!)




    y = h5_f["Science/Y"][...]
    unique_bins = sorted(h5_f["Science/Bins"][0, :, 0])
    binning = unique_bins[1] - unique_bins[0]
    
    aotf_f = h5_f["Channel/AOTFFrequency"][...]
    unique_freqs = sorted(list(set(aotf_f)))
    
    orders = np.array([m_aotf(i) for i in aotf_f])
    unique_orders = sorted(list(set(orders)))
    
    print(obs_type, "orders:", ", ".join([str(i) for i in unique_orders]))
    
    colour_order_dict = {order:{"colour":"C%i" %i} for i, order in enumerate(unique_orders)}
    
    #add offset?
    # y += 0
    
    
    #correct bad pixels
    for i, unique_bin in enumerate(unique_bins):
        bad_pixels = bad_pixel_d[(unique_bin, binning)]
        y[:, i, bad_pixels] = np.nan
    
    y_spectral_mean = np.nanmean(y[:, :, 100:300], axis=2) #spectral mean
    
    
    
    #subtract the first bin?
    if subtract_last_bin:
        y_spectral_mean -= np.tile(y_spectral_mean[:, -1], (len(unique_bins), 1)).T #subtract the first bin
    
    if good_row_indices == "": #average all rows
        y_column_mean = np.nanmean(y_spectral_mean[:, :], axis=1) #mean of all rows 
    else: #mean of selected rows
        y_column_mean = np.nanmean(y_spectral_mean[:, good_row_indices], axis=1) #mean of some rows 
    
    
    # im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
    # stop()
    
    
    if plot_level < -1:
        #plot raw data for every bin to check for noisy regions
        plt.figure(figsize=(14, 8))
        plt.plot(y_spectral_mean)
        sys.exit()
    
    
    
    if good_indices == "":
        good_indices = np.arange(y.shape[0])
    
    if plot_level < 0:
        #plot all good spectra for a specific bin
        for bin_ in range(y.shape[1]):
            plt.figure(figsize=(14, 8))
            plt.title("Raw spectra for bin %i (detector row %i; binning %i)" %(bin_, unique_bins[bin_], binning))
            plt.xlabel("Pixel number")
            plt.ylabel("Signal (counts)")
            for i, y_spectrum in enumerate(y[good_indices, bin_, :]):
                colour = colour_order_dict[orders[i]]["colour"]
                plt.plot(y_spectrum, alpha=0.3, color=colour)
                # if np.any(y_spectrum > 100):
                #     plt.plot(y_spectrum, color="k")
                #     print(unique_bins[bin_])
    
        sys.exit() #stop execution
    
    
    
    if plot_level < 1:
        #plot raw y data spectrally binned
        plt.figure(figsize=(12, 3))
        plt.title("%s: raw Y data, spectrally binned" %h5)
        plt.xlabel("Frame index (all orders)")
        plt.ylabel("Bin number")
        im = plt.imshow(y_spectral_mean[good_indices, :].T, aspect="auto")
        cbar = plt.colorbar(im)
        cbar.set_label("Signal (counts)", rotation=270, labelpad=10)
        plt.subplots_adjust(bottom=0.15)
    
    
    # stop()
    
    """get solar calibration info from an LNO solar cal fullscan. This gives the sensitivity of the instrument in each order"""
    cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
    # cal_d = {order:rad_cal_order(cal_h5, order, centre_indices=None) for order in unique_orders}
    cal_d = {order:rad_cal_order(cal_h5, order, centre_indices=range(100, 300)) for order in unique_orders}
    solar_scalars = {order:cal_d[order]["y_centre_mean"] / 2.0e6 for order in cal_d.keys()}
    
    
    
    
    if plot_level < 2:
        #plot solar calibration spectra
        plt.figure()
        plt.title("Solar calibration spectra")
        for order in cal_d.keys():
            plt.plot(cal_d[order]["y_spectrum"], color = colour_order_dict[order]["colour"], label=order)
            plt.plot([0, len(cal_d[order]["y_spectrum"])-1], [cal_d[order]["y_centre_mean"], cal_d[order]["y_centre_mean"]], \
                      color = colour_order_dict[order]["colour"], label=order)
        plt.legend()
    
    
    
    
    
    counts_d = {}
    for order in unique_orders:
        indices = [i for i, each_order in enumerate(orders) if order == each_order and i in good_indices]
    
        solar_scalar = solar_scalars[order]
    
        y_column_mean_good = y_column_mean[indices]
        y_column_solar_scaled = y_column_mean_good / solar_scalar
    
        counts_d[order] = {"indices":indices, "y_column_mean":y_column_mean_good, "y_column_solar_scaled":y_column_solar_scaled, "colour":colour_order_dict[order]["colour"]}
    
    
        counts_d[order]["y_all_column_mean"] = np.mean(y_column_mean_good)
        counts_d[order]["y_all_column_std"] = np.std(y_column_mean_good)
        counts_d[order]["y_all_column_solar_scaled_mean"] = np.mean(y_column_solar_scaled)
        counts_d[order]["y_all_column_solar_scaled_std"] = np.std(y_column_solar_scaled)
        
            
            
        y_spectral_mean_good = y_spectral_mean[indices, :]
        
        y_spectral_solar_scaled = y_spectral_mean_good / solar_scalar

        counts_d[order]["y_spectral_mean"] = y_spectral_mean_good
        counts_d[order]["y_spectral_solar_scaled"] = y_spectral_solar_scaled

        counts_d[order]["y_all_spectral_mean"] = np.mean(y_spectral_mean_good, axis=0)
        counts_d[order]["y_all_spectral_std"] = np.std(y_spectral_mean_good, axis=0)
        counts_d[order]["y_all_spectral_solar_scaled_mean"] = np.mean(y_spectral_solar_scaled, axis=0)
        counts_d[order]["y_all_spectral_solar_scaled_std"] = np.std(y_spectral_solar_scaled, axis=0)
    
    
    
    if plot_level < 2.5:
        n_bins = y_spectral_mean.shape[1]
        #plot spectrally averaged signal for each bin individually
        fig, axes = plt.subplots(nrows=2, ncols=int(n_bins/2), figsize=(n_bins*2, 8))
        axes = axes.flatten()
        fig.suptitle("%s: spectrally binned signal for each order and bin" %h5)
        # plt.subplots_adjust(bottom=0.15)
        for bin_ix in range(y_spectral_mean.shape[1]):
            for order in unique_orders:
            
                
                # linestyle = ls_bin_dict[unique_bins[bin_ix]]
                linestyle = "-"
                colour = colour_order_dict[order]["colour"]
           
                axes[bin_ix].set_title("Bin %i" %bin_ix)
                axes[bin_ix].scatter(counts_d[order]["indices"], counts_d[order]["y_spectral_mean"][:, bin_ix], color=colour, label="Order %i" %order)
                axes[bin_ix].plot([np.min(counts_d[order]["indices"]), np.max(counts_d[order]["indices"])], \
                                  [np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix]), np.mean(counts_d[order]["y_spectral_mean"][:, bin_ix])], \
                                  color=colour, linestyle=linestyle)
        
            axes[bin_ix].set_xlabel("Frame index")
            axes[bin_ix].set_ylabel("Signal (counts)")
            if bin_ix == 0:
                axes[bin_ix].legend(loc="upper left")
            axes[bin_ix].grid()
        fig.subplots_adjust(bottom=0.15)
        
    
    
    
    
    
    if plot_level < 3:
        #plot spectrally averaged signal for all bins averaged
        plt.figure(figsize=(12, 5))
        plt.title("%s: spectrally binned signal for each order" %h5)
        plt.xlabel("Frame index")
        plt.ylabel("Signal (counts)")
        plt.subplots_adjust(bottom=0.15)
    
        for order in unique_orders:
    
            colour = colour_order_dict[order]["colour"]
            
            x_plt = counts_d[order]["indices"]
            y_plt = counts_d[order]["y_column_mean"]
            y_plt_std = counts_d[order]["y_all_column_std"]
            y_plt_mean = counts_d[order]["y_all_column_mean"]

            plt.scatter(x_plt, y_plt, color=colour, label="Order %i" %order)
            plt.plot([x_plt[0], x_plt[-1]], [y_plt_mean, y_plt_mean], color=colour)
            # plt.fill_between(x_plt, y1=y_plt_mean-y_plt_std, y2=y_plt_mean+y_plt_std, color=colour, alpha=0.3)
    
        plt.legend(loc="upper left")
        plt.grid()
    
    
    
    
    
    
    
    
    
    if plot_level < 4:
    
        fig1, axes1 = plt.subplots(nrows=3, figsize=(16, 9))
        fig1.suptitle("%s: Instrument sensitivity correction" %h5)
        axes1[0].plot(cal_d.keys(), [cal_d[order]["y_centre_mean"] for order in cal_d.keys()], label="Solar calibration scalar")
        axes1[0].scatter(cal_d.keys(), [cal_d[order]["y_centre_mean"] for order in cal_d.keys()])
        axes1[0].legend(loc="upper left")
        axes1[0].grid()
        axes1[0].set_ylabel("Counts")
        
        for bin_ix in range(len(unique_bins)):
            # axes1[1].plot(unique_orders, [counts_d[order]["y_all_spectral_mean"][bin_ix] for order in unique_orders], label="Y counts spectral and frame mean bin %i" %bin_ix, linestyle="dashed")
            axes1[1].errorbar(unique_orders, [counts_d[order]["y_all_spectral_mean"][bin_ix] for order in unique_orders], \
                              yerr=[counts_d[order]["y_all_spectral_std"][bin_ix] for order in unique_orders], capsize=2, label="Y counts spectral and frame mean bin %i" %bin_ix)
        # axes1[1].plot(unique_orders, [counts_d[order]["y_all_column_mean"] for order in unique_orders], color="r", label="Y counts spectral and frame mean all bins")
        # axes1[1].errorbar(unique_orders, [counts_d[order]["y_all_column_mean"] for order in unique_orders], \
        #                   yerr=[counts_d[order]["y_all_column_std"] for order in unique_orders], color="k", capsize=2, label="Y counts spectral and frame mean all bins")
        axes1[1].legend(loc="upper left")
        axes1[1].grid()
        axes1[1].set_ylabel("Counts")
    
    
        #plot counts, scaled by instrument sensitivity, for each bin separately with error
        for bin_ix in range(len(unique_bins)):
            axes1[2].errorbar(unique_orders, [counts_d[order]["y_all_spectral_solar_scaled_mean"][bin_ix] for order in unique_orders], \
                              yerr=[counts_d[order]["y_all_spectral_solar_scaled_std"][bin_ix] for order in unique_orders], capsize=2, label="Y counts mean scaled to solar bin %i" %bin_ix)
    
        
        axes1[2].plot(unique_orders, [counts_d[order]["y_all_column_solar_scaled_mean"] for order in unique_orders], color="k", label="Y counts mean scaled to solar")
        # axes1[2].scatter(unique_orders, [counts_d[order]["y_all_column_solar_scaled_mean"] for order in unique_orders], color="g")
        axes1[2].legend(loc="upper left")
        axes1[2].grid()
        axes1[2].set_ylabel("Counts scaled to instrument sensitivity")
        
        axes1[2].set_xlabel("Diffraction order")


    
    
    #scale solar-corrected counts to CRISM
    # plot CRISM red and blue first
    if plot_level < 5:
    
    
        fig2, ax2a = plt.subplots(figsize=(12, 5))
        fig2.suptitle("Phobos Radiance Calibration: %s (%s)" %(obs_type, h5))
        ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="tab:red", marker="x", alpha=0.7, label="CRISM Phobos red")
        ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="tab:blue", marker="x", alpha=0.7, label="CRISM Phobos blue")
    
    
    #for each good row, get the mean and standard deviation
    good_bin_means = []
    good_bin_stds = []
    for bin_ix in good_row_indices:
        good_bin_means.append([counts_d[order]["y_all_spectral_solar_scaled_mean"][bin_ix] for order in unique_orders])
        good_bin_stds.append([counts_d[order]["y_all_spectral_solar_scaled_std"][bin_ix] for order in unique_orders])

    good_bin_means = np.array(good_bin_means)
    good_bin_stds = np.array(good_bin_stds)
    good_bin_snrs = good_bin_means / good_bin_stds
    good_bin_snr = np.sqrt(np.sum(good_bin_snrs**2, axis=0))


    # scale to crism
    #get column-scaled data for each order
    y_column_mean_norm = {order:counts_d[order]["y_all_column_solar_scaled_mean"] for order in unique_orders}
    y_column_std_norm = {order:counts_d[order]["y_all_column_solar_scaled_std"] for order in unique_orders}



    orders_to_scale = np.array(obs_types[obs_type]["orders_crism"])
    order_ums = [10000./cal_d[order]["x_mean"] for order in orders_to_scale]

    
    #calculate scaling factor to calibrate to crism red
    matching_crism_indices = np.where((crism_d["x"] > np.min(order_ums)) & (crism_d["x"] < np.max(order_ums)))[0]
    crism_red_mean = np.mean(crism_d["phobos_red"][matching_crism_indices])
    crism_blue_mean = np.mean(crism_d["phobos_blue"][matching_crism_indices])
    lno_mean = np.mean([y_column_mean_norm[order] for order in orders_to_scale])
    
    red_scalar = crism_red_mean / lno_mean
    blue_scalar = crism_blue_mean / lno_mean
    
    y_column_mean_norm_red = {order:y_column_mean_norm[order]*red_scalar for order in y_column_mean_norm.keys()}
    y_column_mean_norm_blue = {order:y_column_mean_norm[order]*blue_scalar for order in y_column_mean_norm.keys()}
    y_column_std_norm_red = {order:y_column_std_norm[order]*red_scalar for order in y_column_std_norm.keys()}
    y_column_std_norm_blue = {order:y_column_std_norm[order]*blue_scalar for order in y_column_std_norm.keys()}
    
    print("SNRs per order:", good_bin_snr)
    
    if plot_level < 5:
    
    
        x_plt = [10000./cal_d[order]["x_mean"] for order in y_column_mean_norm.keys()]


        y_plt = np.array([y_column_mean_norm_red[order] for order in y_column_mean_norm.keys()])
        # y_err = np.array([y_column_std_norm_red[order] for order in y_column_std_norm.keys()])
        y_err = y_plt / good_bin_snr
        
        # ax2a.plot(x_plt, y_plt, color="darkred", label="LNO scaled to Phobos red")
        ax2a.errorbar(x_plt, y=y_plt, yerr=y_err, color="darkred", capsize=2, label="LNO scaled to Phobos red")
        ax2a.scatter(x_plt, y_plt, color="darkred")

        y_plt = np.array([y_column_mean_norm_blue[order] for order in y_column_mean_norm.keys()])
        # y_err = np.array([y_column_std_norm_blue[order] for order in y_column_std_norm.keys()])
        y_err = y_plt / good_bin_snr


        # ax2a.plot(x_plt, y_plt, color="darkblue", label="LNO scaled to Phobos blue")
        ax2a.errorbar(x_plt, y=y_plt, yerr=y_err, color="darkblue", capsize=2, label="LNO scaled to Phobos blue")
        ax2a.scatter(x_plt, y_plt, color="darkblue")
    
        ax2a.grid()
        ax2a.legend()
        ax2a.set_ylim((0.0, 0.1))
        ax2a.set_xlabel("Wavelength (microns)")
        ax2a.set_ylabel("CRISM Phobos I/F (Fraeman 2014)")
        fig2.subplots_adjust(bottom=0.15)
        fig2.savefig("phobos_radcal_%s.png" %h5)
