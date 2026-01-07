# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:27:05 2025

@author: iant

CHECK UVIS OCC ILLUMINATION ROWS VS TEMPERATURE

"""


from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import pandas as pd

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import scipy

from tools.file.hdf5_functions import make_filelist2
from tools.plotting.colours import get_colours

from tools.general.progress_bar import progress

# regex = re.compile("2023012._.*_UVIS_[IE]")
regex = re.compile("20....10_......_.*_UVIS_[IE]")


# check for h5 files matching the regex? If not it is much faster but won't load new data
FORCE_RELOAD = True
FORCE_RELOAD = False

# to avoid reloading from the json, delete or move the json

PLOT = ["linear_fit"]
PLOT = []

DATASETS = [
    "Housekeeping/TEMP_1_PROXIMITY_BOARD",
    "Housekeeping/TEMP_2_CCD",
    "Housekeeping/TEMP_3_DETECTOR_BOARD"
]

# which UVIS detector columns to save info from?
nms = [400, 500, 600]
# nms = [600]

# on which rows approx do the illumination patterns rise and fall?
line_ranges = {400: [[29, 34], [42, 47]], 500: [[25, 30], [44, 49]], 600: [[19, 24], [46, 51]]}


def make_prefix(h5):
    if "UVIS" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress"}[h5.split("_")[4]]
    elif "SO" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress", "S": "Fullscan"}[h5.split("_")[5]]
    return h5_prefix


def get_uvis_data(regex, done=[]):
    file_level = "hdf5_level_0p3b"

    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for file_ix, (h5, h5_f) in enumerate(zip(progress(h5s), h5_fs)):

        h5_prefix = make_prefix(h5)

        if h5_prefix in done:
            print("Skipping", h5_prefix)
            continue

        if h5_prefix not in d.keys():
            # print("Loading %i/%i" % (file_ix, len(h5s)))

            y = h5_f["Science/Y"][:, 0]
            if y.shape[0] == 71:

                x = h5_f["Science/X"][0, :]
                if len(x) == 128:

                    print("Getting data from", h5_prefix)

                    # lats = h5_f["Geometry/Point0/Lat"][:, 0]
                    # lons = h5_f["Geometry/Point0/Lon"][:, 0]
                    rfadfr = h5_f["Channel/ReverseFlagAndDataTypeFlagRegister"][...]
                    alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
                    nspectra = len(rfadfr)
                    ts = {dset: h5_f[dset][...] for dset in DATASETS}
                    vstart = h5_f["Channel/VStart"][0]

                    y = h5_f["Science/Y"][...]

                    d[h5_prefix] = {"alts": alts, "rfadfr": rfadfr, "ts": ts, "y": y, "x": x, "vstart": vstart, "nspectra": nspectra}

    return d


# load existing from json file if it exists
print("Loading from json")
if os.path.exists('uvis_row_illumination.json'):
    with open('uvis_row_illumination.json', 'r') as json_file:
        d_shift = json.load(json_file)

        # convert keys back to integers
        d_shift = {int(k): v for k, v in d_shift.items()}

    h5s_loaded = d_shift[nms[0]]["h5"]
else:
    h5s_loaded = []
    d_shift = {nm: {"h5": [], "t1": [], "t2": [], "t3": [], "x_rise": [], "x_fall": [], "vstart": [], "nspectra": []} for nm in nms}


if FORCE_RELOAD:
    d = get_uvis_data(regex, done=h5s_loaded)
else:
    d = {}


temps = np.arange(-20, 10, 0.1)
colours = get_colours(len(temps))


# for dset in DATASETS:
if "linear_fit" in PLOT:
    fig0, ax0 = plt.subplots(figsize=(16, 10))

# if no new files, this has zero length
for h5 in list(d.keys()):
    x = d[h5]["x"]

    if len(x) == 128:
        y = d[h5]["y"]

        if y.shape[1] == 71:

            alts = d[h5]["alts"]
            rfadfr = d[h5]["rfadfr"]
            rfadfr = d[h5]["rfadfr"]
            t1s = d[h5]["ts"][DATASETS[0]]
            t2s = d[h5]["ts"][DATASETS[1]]
            t3s = d[h5]["ts"][DATASETS[2]]

            if "linear_fit" in PLOT:
                ts = t1s

            v = d[h5]["vstart"]

            toa_ixs = np.where((alts > 100.0) & (rfadfr == 4))[0]  # top of atmosphere

            # make indices based on number of spectra
            if len(toa_ixs) < 200:
                pos_ixs = [0, -1]
            if len(toa_ixs) > 200:
                pos_ixs = [0, int(len(toa_ixs)/2), -1]
            if len(toa_ixs) > 400:
                pos_ixs = [0, int(len(toa_ixs)/4), int(len(toa_ixs)/2), int(len(toa_ixs)*3/4), -1]
            # make apply position indices to make spectral indices
            ixs = [toa_ixs[i] for i in pos_ixs]

            for nm in nms:
                nm_ix = np.argmin(np.abs(x - nm))
                line_range_rise, line_range_fall = line_ranges[nm]

                for ix in ixs:

                    col = y[ix, :, nm_ix]/np.max(y[ix, :, nm_ix])

                    if nm == 500 and "linear_fit" in PLOT:
                        line = np.arange(col.shape[0]) + v
                        colour = colours[np.argmin(np.abs(temps - ts[ix]))]
                        ax0.plot(line, col, color=colour, label="%0.1f C" % ts[ix])

                    x_rise = np.interp(0.5, col[line_range_rise[0]:line_range_rise[1]], np.arange(line_range_rise[0], line_range_rise[1]) + v)
                    x_fall = np.interp(0.5, col[line_range_fall[0]:line_range_fall[1]][::-1], np.arange(line_range_fall[0], line_range_fall[1])[::-1] + v)

                    if nm == 500 and "linear_fit" in PLOT:
                        ax0.scatter([x_rise, x_fall], [0.5, 0.5], color=colour)

                    d_shift[nm]["h5"].append(h5)
                    d_shift[nm]["t1"].append(float(t1s[ix]))
                    d_shift[nm]["t2"].append(float(t2s[ix]))
                    d_shift[nm]["t3"].append(float(t3s[ix]))
                    d_shift[nm]["x_rise"].append(float(x_rise))
                    d_shift[nm]["x_fall"].append(float(x_fall))
                    d_shift[nm]["vstart"].append(float(d[h5]["vstart"]))
                    d_shift[nm]["nspectra"].append(float(d[h5]["nspectra"]))

# save to json file
print("Saving to json")
with open('uvis_row_illumination.json', 'w') as json_file:
    json.dump(d_shift, json_file, indent=4)


# sort by h5 filename
x = d_shift[400]["h5"]
sort_ix = np.asarray(sorted(range(len(x)), key=lambda index: x[index]))

for nm in nms:
    for key in d_shift[nm].keys():
        d_shift[nm][key] = np.asarray(d_shift[nm][key])[sort_ix]

# add time
for nm in nms:
    d_shift[nm]["dt"] = [int(h5[0:8]) for h5 in d_shift[nm]["h5"]]

if "linear_fit" in PLOT:

    ax0.legend()
    ax0.set_xlabel("Detector Row")
    ax0.set_ylabel("Normalised Illumination Pattern")

    fig1, (axes) = plt.subplots(figsize=(22, 6), ncols=6)
    axes[0].set_ylabel("Illumination pattern shift (detector row)")

    for i, nm in enumerate(nms):

        axes[i*2].scatter(d_shift[nm]["t"], d_shift[nm]["x_rise"])
        axes[i*2+1].scatter(d_shift[nm]["t"], d_shift[nm]["x_fall"])

        lr_rise = scipy.stats.linregress(d_shift[nm]["t"], d_shift[nm]["x_rise"])
        polyfit_rise = [lr_rise.slope, lr_rise.intercept]
        polyval_rise = np.polyval(polyfit_rise, d_shift[nm]["t"])

        lr_fall = scipy.stats.linregress(d_shift[nm]["t"], d_shift[nm]["x_fall"])
        polyfit_fall = [lr_fall.slope, lr_fall.intercept]
        polyval_fall = np.polyval(polyfit_fall, d_shift[nm]["t"])

        axes[i*2].plot(d_shift[nm]["t"], polyval_rise)
        axes[i*2+1].plot(d_shift[nm]["t"], polyval_fall)

        fig1.suptitle(DATASETS[0])

        axes[i*2].set_title("%i nm rise, R squared = %0.2f" % (nm, lr_rise.rvalue))
        axes[i*2+1].set_title("%i nm fall, R squared = %0.2f" % (nm, lr_fall.rvalue))

        axes[i*2].set_xlabel("UVIS Temperature")

        axes[i*2+1].set_xlabel("UVIS Temperature")


ML_VARIABLES = [
    # ["t1"],
    # ["t2"],
    # ["t3"],
    # ["t1", "t2"],
    # ["t2", "t3"],
    # ["t1", "t3"],
    # ["t1", "dt"],
    # ["t3", "dt"],
    # ["t1", "t2", "t3"],
    # ["t1", "t3", "nspectra"],
    # ["t1", "t2", "t3", "nspectra"],
    ["t1", "t3", "dt"],
    # ["t1", "t3", "nspectra", "dt"],
]

for nm in [500]:  # nms:
    for type_ in ["x_rise", "x_fall"]:
        df = pd.DataFrame.from_dict(d_shift[nm])

        # scale = StandardScaler()

        for ml_vars in ML_VARIABLES:

            X = df[ml_vars]
            y = df[type_]
            # y = df["x_rise"]

            train_ixs = np.arange(len(X.values))  # all points

            model = RandomForestRegressor()
            model.fit(X.values[train_ixs, :], y[train_ixs])
            score = model.score(X.values[train_ixs, :], y[train_ixs])
            print(nm, ":", ", ".join(ml_vars), ":", "%0.3f" % score)

        ypred = model.predict(X.values)

        np.std(y-ypred)

        plt.scatter(train_ixs, y-ypred, label={"x_rise": "Top (rising) edge", "x_fall": "Bottom (falling) edge"}[type_], alpha=0.7)
plt.title("Observed vs predicted shift at 500 nm")
plt.xlabel("HDF5 file index")
plt.ylabel("Observed-predicted")
plt.legend()
# plt.errorbar(train_ixs, y, yerr=np.abs(y-ypred), fmt="none")

# find worst files
bad_ixs = np.where(np.abs(y-ypred) > 0.3)[0]
print([d_shift[500]["h5"][ix] for ix in bad_ixs])
print([d_shift[500]["t1"][ix] for ix in bad_ixs])
print([d_shift[500]["t2"][ix] for ix in bad_ixs])
print([d_shift[500]["t3"][ix] for ix in bad_ixs])
print([d_shift[500]["nspectra"][ix] for ix in bad_ixs])


# make grid
t1_grid = np.arange(-18, 15, 0.1)
t3_grid = np.arange(-22, 0, 0.1)


# list(product(a, b))

grid_points = np.asarray(list(product(t1_grid, t3_grid)))

# gpred = model.predict(grid_points)


# gpred = gpred.reshape(len(t1_grid), len(t3_grid))


# im = plt.imshow(gpred.T, extent=(t1_grid[0], t1_grid[-1], t3_grid[0], t3_grid[-1]))
# plt.colorbar(im)
# plt.xlabel(DATASETS[0])
# plt.ylabel(DATASETS[2])

# plt.scatter(X["t1"], X["t3"], c=y)


"""check how dt evolution affects the results"""


# """check if the shift changes with wavelength"""
# # %%
# plt.figure()

# nm_ref = nms[0]
# nm_others = nms[1:]

# rise_ref = d_shift[nm_ref]["x_rise"]
# fall_ref = d_shift[nm_ref]["x_fall"]
# for nm_other in nm_others:
#     rise = d_shift[nm_other]["x_rise"]
#     fall = d_shift[nm_other]["x_fall"]

#     d_rise = rise_ref - rise
#     d_fall = fall_ref - fall

#     plt.plot(d_rise - np.mean(d_rise), label="%i nm rise" % nm_other, alpha=0.4)
#     plt.plot(d_fall - np.mean(d_fall), label="%i nm fall" % nm_other, alpha=0.4)

# plt.legend()
