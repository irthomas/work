# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:41:37 2024

@author: iant

SELECT GOOD SPECTRA, PLOT PEAK OF SPECTRUM VS TEMPERATURE TO SHOW HOW THE PEAK (DUE TO AOTF) VARIES WITH TEMPERATURE

    
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
# from matplotlib.backends.backend_pdf import PdfPages

from tools.general.progress_bar import progress
from tools.file.hdf5_functions import make_filelist2
from instrument.calibration.so_lno_2023.fit_blaze_shape import fit_blaze

file_level = "hdf5_level_1p0a"

# orders measured regularly and not with 5 other orders: 121, 134, 136, 167, 168, 189, 190
order = 168

regex = re.compile("20......_......_.*_LNO_1_D._%i" % order)

# where to find data
root_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
root_path = r"D:\DATA\hdf5"

max_sza = 20.0  # only use data where solar zenith angle is low i.e where SNR is the best

# use blaze fit for orders with good signal and clearly visible absorption lines, otherwise use polyfit
orders_with_absorptions = [167, 168, 189, 190]

# reload data every time? Always select True if using a different order
FORCE_RELOAD = True
# FORCE_RELOAD = False


def get_lno_data(regex, file_level, max_sza):
    """get data from files matching the regex and save to a simple dictionary"""
    h5fs, h5s, h5_paths = make_filelist2(regex, file_level, path=root_path)

    d = {}
    for h5f, h5 in zip(h5fs, h5s):

        szas = h5f["Geometry/Point0/SunSZA"][:, 0]
        good_ixs = np.where(szas < max_sza)[0]

        # skip if more than 3 subdomains
        nsubs = int(h5f.attrs["NSubdomains"])

        # print(len(good_ixs), nsubs)
        if len(good_ixs) > 3 and nsubs < 4:
            y_raw = h5f["Science/YUnmodified"][good_ixs, :]
            ts = h5f["Channel/InterpolatedTemperature"][good_ixs]
            d[h5] = {"y_raw": y_raw, "szas": szas[good_ixs], "ts": ts}

    return d


# get data from hdf5 files, or skip if already loaded
if FORCE_RELOAD or "d" not in globals():
    d = get_lno_data(regex, file_level, max_sza)

ts = []
peaks = []
chi_sqs = []

y_bad = {}
# loop through data dictionary, fitting a blaze-type function to each spectrum to find the peak
print("Finding peaks of spectra")
for h5 in progress(d.keys()):
    y_raw = d[h5]["y_raw"]

    for spec_ix, y in enumerate(y_raw):

        # if nans, skip
        if np.any(np.isnan(y)):
            continue

        if order in orders_with_absorptions:
            # fit the rough blaze function
            # ignore polyfit errors, quality checks will deal with bad fitting
            blaze = fit_blaze(y, max_rms=0.1, degree=5)  # increase rms to deal with bad pixels
        else:
            # if order without absorption lines, just polyfit through the centre of the spectrum
            blaze_fit = np.polyfit(np.arange(y.shape[0]), y, 5)
            blaze = np.polyval(blaze_fit, np.arange(y.shape[0]))

        ix = np.where(blaze == np.max(blaze))[0][0]

        # get temperature
        t = d[h5]["ts"][spec_ix]

        # quality check: if peak is reasonably located
        if ix > 150 and ix < 250:

            # ix is the peak to the nearest pixel (integer), so next fit a quadratic to the 3 points around the peak to find the subpixel max
            peak_fit = np.polyfit(np.arange(-1, 2), blaze[ix-1:ix+2], 2)
            peak_ix = -peak_fit[1]/(2 * peak_fit[0]) + ix

            chi = np.sum(((y[150:250] - blaze[150:250])**2) / y[150:250])

            # quality check: if chi squared fit is sensible
            if chi > 0.0 and chi < 1500:

                # if quality checks passed, add temperature, peak position and chi squared to lists for plotting later
                ts.append(t)
                peaks.append(peak_ix)
                chi_sqs.append(chi)

                # if something is not right, save to check it out later
                if peak_ix > 200:
                    y_bad[h5] = {"spec": y, "blaze": blaze, "spec_ix": spec_ix, "peak_ix": ix}
                #     # stop()

# plot
plt.figure(figsize=(12, 6))
plt.title("Raw spectrum peak vs temperature for LNO diffraction order %i" % order)
plt.scatter(ts, peaks, c=chi_sqs, alpha=0.3)
plt.xlabel("Instrument temperature")
plt.ylabel("Peak pixel column")

best_fit = np.polyfit(ts, peaks, 1)
best_vals = np.polyval(best_fit, ts)

plt.plot(ts, best_vals, "k")
plt.text(ts[100], best_vals[100], "y = %0.5f x + %0.5f" % (best_fit[0], best_fit[1]))

plt.grid()
plt.savefig("AOTF_peak_vs_temperature_LNO_order_%i.png" % order)


# check bad fits, only plot first 19 if many found
for h5 in list(y_bad.keys())[0:18]:
    plt.figure()
    plt.plot(y_bad[h5]["spec"])
    plt.plot(y_bad[h5]["blaze"])
