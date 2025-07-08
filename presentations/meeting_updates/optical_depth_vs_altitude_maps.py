# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:26:44 2023

@author: iant

SO / UVIS OCCULTATION OPTICAL DEPTHS VS ALTITUDE

RUN THE CODE IN 2 STEPS, FIRST READ IN ALL DATA AND CREATE H5 FILE OF 
ALTITUDES VS GEOMETRY PARAMETERS FOR THE THREE ORDERS 132, 134 AND 136

THEN COMBINE THE DATA FROM ALL 3 ORDERS AND INPUT INTO THE MODEL

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import pandas as pd
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.general.get_mars_year_ls import get_mars_year_ls
from tools.general.progress_bar import progress_bar
from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict
from tools.plotting.make_ls_labels import make_ls_labels


# TO CREATE H5 FILE, SELECT ONE ORDER
# order = 121
# order = 132
# order = 134
order = 136

# TO CREATE H5 FILE, USE EMPTY ORDER LIST
# combined_orders = {"orders":[]}


# combine orders - needs h5 dictionaries to be already made
combined_orders = {"orders": [132, 134, 136], "reference_order": 134}


# plot_level = 2
# plot_level = 3
plot_level = 4
# plot_level = 5
# plot_level = 9 #plot nothing


# save as pngs?
SAVE_PLOTS = True
# SAVE_PLOTS = False


# TO CREATE H5 FILE, READ IN ALL DATA OF THE CHOSEN ORDER
file_level = "hdf5_level_1p0a"

"""for processing"""
regex = re.compile("20......_......_.*_SO_A_[IE]_%s" % order)
PLOT_BASELINE_FITTING = False
PLOT_ALTITUDE_TRANSMITTANCE_FITTING = False


"""only for testing/defining baseline points and checking transmittances"""
# regex = re.compile("20230701_01...._.*_SO_A_[IE]_136")
# PLOT_BASELINE_FITTING = True
# PLOT_ALTITUDE_TRANSMITTANCE_FITTING = True


# find the lowest altitude with this transmittance
CHOSEN_TRANS = np.exp(-1)


# not used - make the good_ixs.txt containing indices of good observations only
# FIND_GOOD_CORRELATED_OBS = True
FIND_GOOD_CORRELATED_OBS = False


# ML_VARIABLES = ["lat", "lon", "ls", "lst", "my"] #all params
ML_VARIABLES = ["lat", "lon", "ls", "lst"]  # all except MY
# ML_VARIABLES = ["lat", "ls"] #best fit to 2 parameters


"""total number of occultations"""
# _, h5s, _ = make_filelist(re.compile("20......_......_.*_SO_A_[IE]_..."), file_level, open_files=False, silent=True)
# h5_prefixes = [h5[0:15] for h5 in h5s]
# unique_prefixes = sorted(list(set(h5_prefixes)))
# print("N files=", len(unique_prefixes))


MIN_ALT = 0.0
MAX_ALT = 60.0


# define the wavenumbers in each order where no lines are present
# much faster baseline fitting then baseline ALS
baseline_points = {
    121: np.array([2718, 2726.8, 2731.4, 2734.1, 2736.8, 2739.6]),
    132: np.array([2974, 2976.5, 2978.7, 2983.2, 2986, 2988.1]),
    134: np.array([3014, 3017, 3020, 3024, 3028, 3033]),
    136: np.array([3058, 3062, 3065, 3068, 3073, 3077, 3078.5]),
}


# convert altitude from order tuple[1] to tuple[0]
order_conversion_coeffs = {
    (134, 132): [0.9975402, 0.68779766],
    (134, 136): [0.9778404,  -0.56015717],
}
# to find coefficients, set below to True

# find coefficients between two orders, if not present in order_conversion_coeffs
# CORRELATE_ORDERS = True
CORRELATE_ORDERS = False


def get_altitude(x, y, alts, order, trans, plot_alts=False, plot_baselines=False):
    """get altitude where transmittance = chosen value (trans)"""

    baseline_nus = baseline_points[order]
    baseline_ixs = np.searchsorted(x, baseline_nus)

    polyfits = np.polyfit(baseline_ixs, y[:, baseline_ixs].T, 4)

    # interpolate to spectral continuum at centre of detector
    polyvals_det_centre = np.polyval(polyfits, 160)

    if plot_baselines:
        plt.figure()
        plt.xlabel("Wavenumber (cm-1)")
        plt.ylabel("Transmittance at centre of order")
        plt.grid()
        for i in np.arange(y.shape[0]):
            plt.plot(x, y[i, :], "k-", alpha=0.3)
            plt.plot(x, np.polyval(polyfits[:, i], np.arange(y.shape[1])), "k--", alpha=0.3)
            plt.scatter(x[160], polyvals_det_centre[i], color="k", alpha=0.7)

    if plot_alts:
        plt.figure()
        plt.xlabel("Altitude above aeroid (km)")
        plt.ylabel("Transmittance")
        plt.grid()
        plt.axhline(y=CHOSEN_TRANS, linestyle="--", color="k")
        plt.plot(alts, polyvals_det_centre, "k--")

    smoothed = savgol_filter(polyvals_det_centre, 19, 2)
    if plot_alts:
        plt.plot(alts, smoothed, "k-")

    # index of first alt above chosen transmittance
    above_ix = np.where(smoothed > CHOSEN_TRANS)[0][0]
    below_ix = above_ix - 1

    if below_ix > 0:
        # interpolate between two bounding points to get true altitude at the transmittance value
        alt = np.interp(CHOSEN_TRANS, smoothed[below_ix:above_ix+1], alts[below_ix:above_ix+1])

        if plot_alts:
            plt.scatter(alts[above_ix], smoothed[above_ix], color="k")
            plt.scatter(alts[below_ix], smoothed[below_ix], color="k")
            plt.scatter(alt, CHOSEN_TRANS, color="k")
            plt.axvline(x=alt, linestyle="--", color="k")

    else:
        alt = -999.0

    ix = above_ix

    return alt, ix


def make_trans_dict(regex, file_level):

    out_d = {"h5": [], "h5_no_order": [], "dt": [], "my": [], "ls": [], "lat": [], "lon": [], "lst": [], "alt": []}

    h5_fs, h5s, _ = make_filelist(regex, file_level, open_files=False, silent=True)
    print("N files for order=", len(h5s))
    if len(h5s) < 10:
        # print file list
        for h5 in h5s:
            print(h5)

    for h5 in progress_bar(h5s):

        h5_f = open_hdf5_file(h5)

        order = h5_f["Channel/DiffractionOrder"][0]

        lats = h5_f["Geometry/Point0/Lat"][:, 0]
        lons = h5_f["Geometry/Point0/Lon"][:, 0]
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        lsts = h5_f["Geometry/Point0/LST"][:, 0]
        y = h5_f["Science/Y"][...]
        x = h5_f["Science/X"][0, :]
        # bins = h5_f["Science/Bins"][:, 0]

        dt = datetime(int(h5[0:4]), int(h5[4:6]), int(h5[6:8]), int(h5[9:11]), int(h5[11:13]))
        my, ls = get_mars_year_ls(dt)

        if order not in baseline_points.keys():
            # just plt raw spectra

            ix = np.where(y[:, 200] > 0.3)[0][0]
            plt.plot(x, y[ix, :])
            plt.grid()

        else:

            alt, ix = get_altitude(x, y, alts, order, CHOSEN_TRANS,
                                   plot_alts=PLOT_ALTITUDE_TRANSMITTANCE_FITTING,
                                   plot_baselines=PLOT_BASELINE_FITTING)

            # if not error
            if alt > -998.:

                out_d["h5"].append(b'%s' % h5.encode('utf8'))
                out_d["h5_no_order"].append(b'%s' % h5[:-4].encode('utf8'))
                out_d["dt"].append(b"%s" % str(dt).encode('utf8'))
                out_d["my"].append(my)
                out_d["ls"].append(ls)
                out_d["lat"].append(lats[ix])
                out_d["lon"].append(lons[ix])
                out_d["lst"].append(lsts[ix])
                out_d["alt"].append(alt)

    for key in out_d.keys():
        out_d[key] = np.asarray(out_d[key])

    return out_d


"""for correlating different orders"""
if CORRELATE_ORDERS:
    # get matching dts for two orders
    # calculate coeffs to convert from order_1 to order_2
    # order_1 = 136
    # order_2 = 134

    order_1 = 132
    order_2 = 134

    out_ds = {}
    for order in [order_1, order_2]:
        filename = "so_order_%s" % order
        out_d = read_hdf5_to_dict(filename)[0]
        print("N files for order %i=" % order, len(out_d["alt"]))

        out_ds[order] = out_d

    if (order_2, order_1) in order_conversion_coeffs.keys():
        coeffs = order_conversion_coeffs[(order_2, order_1)]
    else:
        coeffs = [1.0, 0.0]

    h5s_1 = list([f.decode() for f in out_ds[order_1]["h5_no_order"]])
    h5s_2 = list([f.decode() for f in out_ds[order_2]["h5_no_order"]])

    matching_h5s = set(h5s_1).intersection(h5s_2)
    print("N matching files for orders %i and %i=" % (order_1, order_2), len(matching_h5s))
    print("Total files for both orders=", len(h5s_1)+len(h5s_2)-len(matching_h5s))
    match_h51_ix = [i for i, v in enumerate(h5s_1) if v in matching_h5s]
    match_h52_ix = [i for i, v in enumerate(h5s_2) if v in matching_h5s]

    match_alts_1 = [out_ds[order_1]["alt"][i] for i in match_h51_ix]
    match_alts_2 = [out_ds[order_2]["alt"][i] for i in match_h52_ix]

    match_alts_1 = np.polyval(coeffs, match_alts_1)

    plt.figure()
    plt.scatter(match_alts_1, match_alts_2, alpha=0.1)

    polyfit = np.polyfit(match_alts_1, match_alts_2, 1)
    print("Polyfit=", polyfit)
    polyvals = np.polyval(polyfit, sorted(match_alts_1))
    plt.plot(sorted(match_alts_1), polyvals)
    plt.plot([0, 60], [0, 60])

    km_error = match_alts_2 - np.polyval(polyfit, match_alts_1)
    h5_error_ix = [match_h52_ix[i] for i in np.where(np.abs(km_error) > 5)[0]]

    bad_h5s = [out_ds[order_2]["h5"][i].decode() for i in h5_error_ix]
    print("N bad matches found=", len(bad_h5s))

    # for h52 in bad_h5s[0:20]:

    #     h5_f2 = open_hdf5_file(h52)
    #     h51 = h52[:-3] + "%s" %order_1
    #     h5_f1 = open_hdf5_file(h51)

    #     alts = h5_f1["Geometry/Point0/TangentAltAreoid"][:, 0]
    #     y = h5_f1["Science/Y"][...]
    #     x = h5_f1["Science/X"][0, :]

    #     plt.figure()
    #     alt, ix = get_altitude(x, y, alts, order_1, CHOSEN_TRANS, plot=True)

    #     alts = h5_f2["Geometry/Point0/TangentAltAreoid"][:, 0]
    #     y = h5_f2["Science/Y"][...]
    #     x = h5_f2["Science/X"][0, :]

    #     alt, ix = get_altitude(x, y, alts, order_2, CHOSEN_TRANS, plot=True)

    sys.exit()


if len(combined_orders["orders"]) == 0:
    """simple version, just use 1 order"""

    filename = "so_order_%s" % order

    # load if exists, if not make it from the h5 files
    if os.path.exists(filename+".h5"):
        out_d = read_hdf5_to_dict(filename)[0]
    else:
        out_d = make_trans_dict(regex, file_level)

        write_hdf5_from_dict(filename, out_d, {}, {}, {})
        # stop()

else:
    """combine multiple orders into 1 dictionary. Must already exist in h5 format"""

    reference_order = combined_orders["reference_order"]
    orders = combined_orders["orders"]

    out_ds = {}
    for order in orders:
        filename = "so_order_%s" % order

        # load if exists, if not make it from the h5 files
        if os.path.exists(filename+".h5"):
            out_d = read_hdf5_to_dict(filename)[0]
        else:
            regex = re.compile("20......_......_.*_SO_A_[IE]_%s" % order)
            out_d = make_trans_dict(regex, file_level)
            write_hdf5_from_dict(filename, out_d, {}, {}, {})

        # out_d = read_hdf5_to_dict(filename)[0]
        # print("N files for order %i=" %order, len(out_d["alt"]))

        out_ds[order] = out_d

    # combine into 1 dictionary taking elements not in the reference order dict
    keys = list(out_ds[reference_order].keys())
    comb_d = {k: [] for k in keys}
    for order in orders:

        if order == reference_order:
            # copy all keys
            for key in keys:
                comb_d[key].extend(list(out_ds[order][key]))

        else:
            # find indices where not in reference order
            h5s = list(out_ds[order]["h5_no_order"])
            h5s_ref = list(out_ds[reference_order]["h5_no_order"])

            non_matching_ixs = [i for i, v in enumerate(h5s) if v not in h5s_ref]

            for key in keys:

                if key == "alt":

                    coeffs = order_conversion_coeffs[(reference_order, order)]
                    converted_alts = np.polyval(coeffs, out_ds[order]["alt"][non_matching_ixs])
                    comb_d["alt"].extend(list(converted_alts))
                else:
                    comb_d[key].extend(list(out_ds[order][key][non_matching_ixs]))

    out_d = comb_d
    for key in out_d.keys():
        out_d[key] = np.asarray(out_d[key])

    order_str = "_".join(["%s" % i for i in orders])
# stop()


# sort by ls
lss = out_d["ls"] + out_d["my"] * 360.0
sort_ixs = np.argsort(lss)
for key in out_d.keys():
    out_d[key] = out_d[key][sort_ixs]


# list of all Martian years
mys = list(set(out_d["my"]))

max_alt = np.max(out_d["alt"])

# plot each MY separately
# for my in mys:

#     ixs = np.where(out_d["my"] == my)[0]

#     #make x axis smaller if only partial year (note: change for MY37 if new data added)
#     if my == 34:
#         plt.figure(figsize=(6.3,5), constrained_layout=True)
#     elif my == 37:
#         plt.figure(figsize=(5,5), constrained_layout=True)
#     else:
#         plt.figure(figsize=(10,5), constrained_layout=True)
#     plt.title("MY%0.0f: %s" %(my, regex.pattern))
#     plt.xlabel("Ls")
#     plt.ylabel("Latitude")
#     plt.grid()
#     sc = plt.scatter(out_d["ls"][ixs], out_d["lat"][ixs], c=out_d["alt"][ixs], vmin=0, vmax=np.max(out_d["alt"])+3)
#     cbar = plt.colorbar(sc)
#     cbar.set_label("Tangent altitude above areoid (km)")
#     if SAVE_PLOTS:
#         plt.savefig("so_order_%i_my%0.0f.png" %(order_str, my))


ls_range, ls_labels, ls_label_strs = make_ls_labels(out_d["ls"], out_d["my"], 60.0)

if plot_level < 2:
    if len(mys) > 0:
        # plot all MYs on same figure
        plt.figure(figsize=(10, 5), constrained_layout=True)
        plt.title("Minimum measured altitude where T>%0.3f" % CHOSEN_TRANS)
        plt.xlabel("Ls")
        plt.ylabel("Latitude")
        plt.grid()
        sc = plt.scatter(ls_range, out_d["lat"], c=out_d["alt"], vmin=MIN_ALT, vmax=MAX_ALT)
        cbar = plt.colorbar(sc)
        cbar.set_label("Tangent altitude above areoid (km)")

        plt.xticks(ls_labels, ls_label_strs, rotation='vertical')

        if SAVE_PLOTS:
            plt.savefig("so_order_%s_all_mys.png" % (order_str))

        plt.figure(figsize=(10, 5), constrained_layout=True)
        plt.title("Minimum measured altitude where T>%0.3f" % CHOSEN_TRANS)
        plt.xlabel("Ls")
        plt.ylabel("Latitude")
        plt.grid()
        sc = plt.scatter(out_d["ls"], out_d["lat"], c=out_d["alt"], vmin=MIN_ALT, vmax=MAX_ALT)
        cbar = plt.colorbar(sc)
        cbar.set_label("Tangent altitude above areoid (km)")
        if SAVE_PLOTS:
            plt.savefig("so_order_%s_all_mys_overlap.png" % (order_str))


# plt.figure(figsize=(10,5), constrained_layout=True)
# plt.title("%s" %regex.pattern)
# plt.xlabel("Latitude")
# plt.ylabel("Minimum altitude")
# plt.grid()
# sc = plt.scatter(out_d["lat"], out_d["alt"], c=out_d["ls"])
# cbar = plt.colorbar(sc)
# cbar.set_label("Ls")


# for my in mys:

#     ixs = np.where(out_d["my"] == my)[0]

#     plt.figure(figsize=(10,5), constrained_layout=True)
#     plt.title("MY%0.0f: %s" %(my, regex.pattern))
#     plt.xlabel("Latitude")
#     plt.ylabel("Minimum altitude")
#     plt.grid()
#     plt.scatter(out_d["lat"][ixs], out_d["alt"][ixs])


if not FIND_GOOD_CORRELATED_OBS:
    # read in good ixs and select these points only
    good_ixs = np.loadtxt("good_ixs_order_132_134_136_lat_lon_ls_lst_my.txt", dtype=int)

    for key in out_d.keys():
        out_d[key] = out_d[key][good_ixs]


df = pd.DataFrame.from_dict(out_d)

scale = StandardScaler()

X = df[ML_VARIABLES]
y = df["alt"]


# print("Fitting transform")
scaledX = scale.fit_transform(X.values)

# print("Linear regression")
regr = linear_model.LinearRegression()
# print("Fitting")
regr.fit(X.values, y)

# print("Transforming output and predicting")
# scaled = scale.transform([[-60.0, 80.0]])

# predicted = regr.predict([scaled[0]])
# print(predicted)


# my35 only
# train_ixs = np.where((out_d["my"] == 35) | (out_d["my"] == 36))[0]
# train_ixs = np.where((out_d["my"] == 34))[0]
train_ixs = np.where((out_d["my"] == 35))[0]  # most like other years
# train_ixs = np.where((out_d["my"] == 36))[0]
# train_ixs = np.where((out_d["my"] > 33))[0] #all data


model = RandomForestRegressor()
model.fit(X.values[train_ixs, :], y[train_ixs])
score = model.score(X.values[train_ixs, :], y[train_ixs])
# print("R-squared:", score)
print(", ".join(ML_VARIABLES), ":", "%0.3f" % score)

ypred = model.predict(X.values)


diff = out_d["alt"]-ypred
diff_std = np.std(diff)
diff_mean = np.mean(diff)


ls_range = out_d["ls"] + 360.0 * (out_d["my"] - 34.0)


# lat_ixs = np.where((out_d["lat"] > -40) & (out_d["lat"] < -40))[0]


# if FIND_GOOD_CORRELATED_OBS:
#     corr_ixs = np.where((np.abs(diff) < diff_std*2.0))[0]
#     np.savetxt("good_ixs_order_%s_%s.txt" %(order_str, "_".join(ML_VARIABLES)), corr_ixs, fmt="%i")


if plot_level < 3:
    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Predicted altitudes for inputs %s for order %s" % (", ".join(ML_VARIABLES), order_str))
    plt.xlabel("Ls")
    plt.ylabel("Latitude")
    plt.grid()
    sc = plt.scatter(ls_range, out_d["lat"], c=ypred, vmin=MIN_ALT, vmax=MAX_ALT)
    cbar = plt.colorbar(sc)
    cbar.set_label("Tangent altitude above areoid (km)")

    plt.xticks(ls_labels, ls_label_strs, rotation='vertical')

    if SAVE_PLOTS:
        plt.savefig("so_order_%s_all_mys_predicted.png" % (order_str))

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Predicted altitudes for inputs %s for order %s" % (", ".join(ML_VARIABLES), order_str))
    plt.xlabel("Ls")
    plt.ylabel("Latitude")
    plt.grid()
    sc = plt.scatter(out_d["ls"], out_d["lat"], c=ypred, vmin=MIN_ALT, vmax=MAX_ALT)
    cbar = plt.colorbar(sc)
    cbar.set_label("Tangent altitude above areoid (km)")
    if SAVE_PLOTS:
        plt.savefig("so_order_%s_%s_all_mys_predicted_overlap.png" % (order_str, "_".join(ML_VARIABLES)))


if plot_level < 4:

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Measured minus predicted altitudes for inputs %s for order %s" % (", ".join(ML_VARIABLES), order_str))
    plt.xlabel("Ls")
    plt.ylabel("Latitude")
    plt.grid()
    sc = plt.scatter(ls_range, out_d["lat"], c=diff, cmap="coolwarm")
    cbar = plt.colorbar(sc)
    cbar.set_label("Measured minus predicted altitude (km)")

    plt.xticks(ls_labels, ls_label_strs, rotation='vertical')

    if SAVE_PLOTS:
        plt.savefig("so_order_%s_%s_all_mys_diff.png" % (order_str, "_".join(ML_VARIABLES)))

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Measured minus predicted altitudes for inputs %s for order %s" % (", ".join(ML_VARIABLES), order_str))
    plt.xlabel("Ls")
    plt.ylabel("Latitude")
    plt.grid()
    sc = plt.scatter(out_d["ls"], out_d["lat"], c=diff, cmap="coolwarm")
    cbar = plt.colorbar(sc)
    cbar.set_label("Measured minus predicted altitude (km)")

    if SAVE_PLOTS:
        plt.savefig("so_order_%s_%s_all_mys_diff_overlap.png" % (order_str, "_".join(ML_VARIABLES)))

    lat_ixs = np.where((out_d["lat"] < -60) | (out_d["lat"] > 60))[0]

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Measured minus predicted altitudes for inputs %s for order %s" % (", ".join(ML_VARIABLES), order_str))
    plt.xlabel("Ls")
    plt.ylabel("Altitude difference between measured and predicted (km)")
    plt.grid()
    plt.scatter(ls_range, diff, s=2, alpha=0.5)
    plt.scatter(ls_range[lat_ixs], diff[lat_ixs], s=2, alpha=0.7, color="red")

    plt.xticks(ls_labels, ls_label_strs, rotation='vertical')

    # plt.fill_between([0, 360], y1=diff_mean-diff_std, y2=diff_mean+diff_std, color="C0", alpha=0.3)
    plt.text(min(ls_range)+10, np.min(diff)+3, "1-sigma standard deviation = %0.2fkm" % (diff_std))
    # for x in range(360, 1081, 360):
    #     plt.axvline(x=x, c="k", linestyle="--")

    # plt.scatter(np.arange(len(out_d["alt"])), diff, s=2, alpha=0.7)
    # plt.fill_between(np.arange(len(out_d["alt"])), y1=diff_mean-diff_std, y2=diff_mean+diff_std, color="C0", alpha=0.3)
    # plt.savefig("so_order_%s_all_mys_diff.png" %(order_str))

    # error in ls ranges
    d_ls = 2.0
    for ls_start in np.arange(np.min(ls_range), np.max(ls_range), d_ls):
        # get ixs in bin
        ixs = np.where((ls_range > ls_start) & (ls_range < ls_start+d_ls))
        # get std

        if len(ixs) > 0:
            ls_bin_mean = ls_start + d_ls/2.0

            alt_bin_mean = np.mean(diff[ixs])
            alt_bin_std = np.std(diff[ixs])

            plt.scatter(ls_bin_mean, alt_bin_mean, color="green")

    if SAVE_PLOTS:
        plt.savefig("so_order_%s_%s_all_mys_diff_lat_bin.png" % (order_str, "_".join(ML_VARIABLES)))


if plot_level < 5:
    for my in mys:

        plt.figure(figsize=(12, 4), constrained_layout=True)
        plt.title("Predicted altitude with measured-predicted error bar for inputs %s for MY%i" % (", ".join(ML_VARIABLES), my))
        plt.xlabel("Ls")
        plt.ylabel("Predicted visibility cutoff altitude (km)")
        plt.grid()
        # plt.scatter(ls_range, out_d["alt"], s=2, alpha=0.5)
        # plt.errorbar(ls_range, y=out_d["alt"], yerr=np.abs(diff), alpha=0.5, capsize=2, ls="none")

        lat_ixs = np.where((out_d["lat"] < -60) & (out_d["my"] == my))[0]
        color = "red"
        label = "<60 degrees S"

        plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
        plt.errorbar(out_d["ls"][lat_ixs], y=ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

        lat_ixs = np.where((out_d["lat"] > 60) & (out_d["my"] == my))[0]
        color = "green"
        label = ">60 degrees N"

        plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
        plt.errorbar(out_d["ls"][lat_ixs], y=ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

        lat_ixs = np.where((out_d["lat"] > -30) & (out_d["lat"] < 30) & (out_d["my"] == my))[0]
        color = "blue"
        label = "-30 to +30 degrees"

        plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
        plt.errorbar(out_d["ls"][lat_ixs], ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

        # plt.xticks(ls_labels, ls_label_strs, rotation='vertical')
        plt.legend()
        plt.xlim((-5, 365))

        if SAVE_PLOTS:
            plt.savefig("so_order_%s_%s_my_%i_predicted_alts_error_bars.png" % (order_str, "_".join(ML_VARIABLES), my))

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.title("Predicted altitude with measured-predicted error bar for inputs %s for orders %s" % (", ".join(ML_VARIABLES), order_str.replace("_", ", ")))
    plt.xlabel("Ls")
    plt.ylabel("Predicted visibility cutoff altitude (km)")
    plt.grid()
    # plt.scatter(ls_range, out_d["alt"], s=2, alpha=0.5)
    # plt.errorbar(ls_range, y=out_d["alt"], yerr=np.abs(diff), alpha=0.5, capsize=2, ls="none")

    lat_ixs = np.where((out_d["lat"] < -60) & (out_d["my"] != 35.0))[0]
    color = "red"
    label = "<60 degrees S"

    plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
    plt.errorbar(out_d["ls"][lat_ixs], y=ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

    lat_ixs = np.where((out_d["lat"] > 60) & (out_d["my"] != 35.0))[0]
    color = "green"
    label = ">60 degrees N"

    plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
    plt.errorbar(out_d["ls"][lat_ixs], y=ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

    # lat_ixs = np.where((out_d["lat"] > -30) & (out_d["lat"] < 30) & (out_d["my"] != 35))[0]
    # color = "blue"
    # label = "-30 to +30 degrees"

    # plt.scatter(out_d["ls"][lat_ixs], ypred[lat_ixs], s=4, alpha=0.7, color=color, label=label)
    # plt.errorbar(out_d["ls"][lat_ixs], ypred[lat_ixs], yerr=np.abs(diff[lat_ixs]), alpha=0.2, capsize=2, ls="none", color=color)

    # plt.xticks(ls_labels, ls_label_strs, rotation='vertical')
    plt.legend()

    if SAVE_PLOTS:
        plt.savefig("so_order_%s_%s_all_mys_predicted_alts_error_bars.png" % (order_str, "_".join(ML_VARIABLES)))


"""make maps for given ls"""
# print("Making maps")
# grids = []
# lats = np.arange(-70., 70., 1.)
# lons = np.arange(-179., 179, 1.)
# lss = np.arange(0., 359., 1.)

# grid = np.zeros((len(lons), len(lats), len(ls)))
# X, Y, Z = np.meshgrid(lats, lons, lss)
# grid = np.zeros_like(meshgrid)

# grid = model.predict(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)

# grid2 = grid.reshape((len(lons), len(lats), len(lss)))


# plt.figure(figsize=(10,5), constrained_layout=True)
# im = plt.imshow(np.mean(grid2[:, :, 0:10], axis=2).T, extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect="auto")
# cbar = plt.colorbar(im)

# ls_ixs = np.where(out_d["ls"] < 10.)[0]
# plt.scatter(out_d["lon"][ls_ixs], out_d["lat"][ls_ixs], c=out_d["alt"][ls_ixs])


# #average together longitudes
# plt.figure(figsize=(10,5), constrained_layout=True)
# im = plt.imshow(np.flipud(np.mean(grid2[:, :, :], axis=0)), aspect="auto")
# cbar = plt.colorbar(im)

# grids.append(grid)


# model MY35 ls vs latitude
# set training data to MY35


lats = np.arange(-90., 90., 5.)
lss = np.arange(0., 359., 5.)

X, Y = np.meshgrid(lats, lss)
grid = np.zeros((len(lats), len(lss)))

grid = model.predict(np.array([X.ravel(), Y.ravel()]).T)

grid2 = grid.reshape((len(lss), len(lats))).T


plt.figure(figsize=(10, 5), constrained_layout=True)
plt.title("Model MY36")
im = plt.imshow(grid2[:, :], extent=[lss[0], lss[-1], lats[0], lats[-1]], aspect="auto", origin="lower")
cbar = plt.colorbar(im)


# plt.figure(figsize=(10,5), constrained_layout=True)
# im = plt.imshow(np.mean(grid2[:, :, 0:10], axis=2).T, extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect="auto")
# cbar = plt.colorbar(im)

# ls_ixs = np.where(out_d["ls"] < 10.)[0]
# plt.scatter(out_d["lon"][ls_ixs], out_d["lat"][ls_ixs], c=out_d["alt"][ls_ixs])


# #average together longitudes
# plt.figure(figsize=(10,5), constrained_layout=True)
# im = plt.imshow(np.flipud(np.mean(grid2[:, :, :], axis=0)), aspect="auto")
# cbar = plt.colorbar(im)
