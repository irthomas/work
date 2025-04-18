# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:27:05 2025

@author: iant

CHECK UVIS OCC ILLUMINATION ROWS VS TEMPERATURE

"""


import re
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import scipy

from tools.file.hdf5_functions import make_filelist2
from tools.plotting.colours import get_colours


regex = re.compile("2023012._.*_UVIS_[IE]")


FORCE_RELOAD = True
FORCE_RELOAD = False

DATASETS = [
    "Housekeeping/TEMP_1_PROXIMITY_BOARD",
    "Housekeeping/TEMP_2_CCD",
    "Housekeeping/TEMP_3_DETECTOR_BOARD"
]


def make_prefix(h5):
    if "UVIS" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress"}[h5.split("_")[4]]
    elif "SO" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress", "S": "Fullscan"}[h5.split("_")[5]]
    return h5_prefix


def get_uvis_data(regex):
    file_level = "hdf5_level_0p3b"

    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        print("Loading %i/%i" % (file_ix, len(h5s)))

        # lats = h5_f["Geometry/Point0/Lat"][:, 0]
        # lons = h5_f["Geometry/Point0/Lon"][:, 0]
        rfadfr = h5_f["Channel/ReverseFlagAndDataTypeFlagRegister"][...]
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]

        ts = {dset: h5_f[dset][...] for dset in DATASETS}

        y = h5_f["Science/Y"][...]
        x = h5_f["Science/X"][0, :]
        vstart = h5_f["Channel/VStart"][0]

        h5_prefix = make_prefix(h5)
        if h5_prefix not in d.keys():
            d[h5_prefix] = {"alts": alts, "rfadfr": rfadfr, "ts": ts, "y": y, "x": x, "vstart": vstart}

    return d


if "d" not in globals() or FORCE_RELOAD:
    d = get_uvis_data(regex)


temps = np.arange(-20, 10, 0.1)
colours = get_colours(len(temps))


nms = [400, 500, 600]

# nms = [600]

line_ranges = {400: [[29, 34], [42, 47]], 500: [[25, 30], [44, 49]], 600: [[19, 24], [46, 51]]}


for dset in DATASETS:
    fig0, ax0 = plt.subplots(figsize=(16, 10))

    d_shift = {nm: {"t": [], "x_rise": [], "x_fall": []} for nm in nms}

    for h5 in list(d.keys()):
        x = d[h5]["x"]

        if len(x) == 128:
            y = d[h5]["y"]

            if y.shape[1] == 71:

                alts = d[h5]["alts"]
                rfadfr = d[h5]["rfadfr"]
                rfadfr = d[h5]["rfadfr"]
                ts = d[h5]["ts"][dset]
                v = d[h5]["vstart"]

                toa_ixs = np.where((alts > 100.0) & (rfadfr == 4))[0]  # top of atmosphere
                ixs = [toa_ixs[0], toa_ixs[-1]]

                for nm in nms:
                    nm_ix = np.argmin(np.abs(x - nm))
                    line_range_rise, line_range_fall = line_ranges[nm]

                    for ix in ixs:

                        col = y[ix, :, nm_ix]/np.max(y[ix, :, nm_ix])

                        if nm == 500:
                            line = np.arange(col.shape[0]) + v
                            colour = colours[np.argmin(np.abs(temps - ts[ix]))]
                            ax0.plot(line, col, color=colour, label="%0.1f C" % ts[ix])

                        x_rise = np.interp(0.5, col[line_range_rise[0]:line_range_rise[1]], np.arange(line_range_rise[0], line_range_rise[1]) + v)
                        x_fall = np.interp(0.5, col[line_range_fall[0]:line_range_fall[1]][::-1], np.arange(line_range_fall[0], line_range_fall[1])[::-1] + v)

                        if nm == 500:
                            ax0.scatter([x_rise, x_fall], [0.5, 0.5], color=colour)

                        d_shift[nm]["t"].append(ts[ix])
                        d_shift[nm]["x_rise"].append(x_rise)
                        d_shift[nm]["x_fall"].append(x_fall)

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

        fig1.suptitle(dset)

        axes[i*2].set_title("%i nm rise, R squared = %0.2f" % (nm, lr_rise.rvalue))
        axes[i*2+1].set_title("%i nm fall, R squared = %0.2f" % (nm, lr_fall.rvalue))

        axes[i*2].set_xlabel("UVIS Temperature")

        axes[i*2+1].set_xlabel("UVIS Temperature")
