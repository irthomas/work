# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:05:20 2020

@author: iant
"""
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp

import numpy.linalg as la
from scipy.optimize import minimize

from tools.file.paths import FIG_X, FIG_Y
from tools.file.hdf5_functions import open_hdf5_file

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.datetime_functions import utc2et

from tools.spice.read_webgeocalc import read_webgeocalc

from tools.spectra.non_uniform_savgol import non_uniform_savgol
from tools.plotting.custom_colourmaps import red_widegrey_blue

# load_spice_kernels()
load_spice_kernels()


"""user modifiable"""
# SAVE_FIG = True
SAVE_FIG = False


# print out ephemeris times so that they can be put into ESA WebGeoCalc?
# PRINT_WGC_ETS = True
PRINT_WGC_ETS = False

# read in WebGeoCalc state vectors from a file. File must be made first from the ephemeris times
# needed for cruise phase calibrations where full kernels are not available
# WGC = True
WGC = False


# path to h5 root directory
# hdf5_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
hdf5_path = r"C:\Users\iant\Documents\DATA\hdf5"


# select a channel. For SO, plot all the scans in subplots on one figure. LNO/UVIS: plot one at a time
# channel = "so"
channel = "uvis"
# channel = "lno"

# interpolate the results onto a contour map?

# plot_types = ["raw_xy"]
# plot_types = ["meshgrid"]
# plot_types = ["raw_sep"]
plot_types = ["fit_zero"]


# plot_types = ["raw_xy", "fit_zero"]

"""end user modifiable options"""


SPICE_ABERRATION_CORRECTION = "None"
SPICE_OBSERVER = "-143"

DETECTOR_CENTRE_LINES = {"so": 128, "lno": 152}

grid_size = 0.005

# for lno and uvis, just select one at a time!
linescan_dict = {
    "so": {
        # "Initial boresight (Nov 2016)": ["20161120_231420_0p1a_SO_1", "20161121_012420_0p1a_SO_1"],
        # "Mission start (Apr 2018)": ["20180428_023343_0p1a_SO_1", "20180511_084630_0p1a_SO_1"],
        # "UVIS-prime (Aug 2018)": ["20180821_193241_0p1a_SO_1", "20180828_223824_0p1a_SO_1"],
        # "UVIS-prime (Dec 2018)": ["20181219_091740_0p1a_SO_1", "20181225_025140_0p1a_SO_1"],  # not a nomad linescan
        # "UVIS-prime (Jan 2019)": ["20190118_183336_0p1a_SO_1", "20190125_061434_0p1a_SO_1"],
        # "SO-prime (Oct 2019)": ["20191022_013944_0p1a_SO_1", "20191028_003815_0p1a_SO_1"],
        # "SO-prime (Feb 2020)": ["20200226_024225_0p1a_SO_1", "20200227_041530_0p1a_SO_1"],
        # "SO-prime (Dec 2020)": ["20201224_011635_0p1a_SO_1", "20210102_092937_0p1a_SO_1"],
    },
    "lno": {
        # "Initial boresight (June 2016)":["20160613_001950_0p1a_LNO_1", "20160613_022203_0p1a_LNO_1"], \
        # "Initial boresight (June 2016)": ["20160615_233950_0p1a_LNO_1", "20160616_015450_0p1a_LNO_1"], \
        # "Refined boresight (Nov 2016)":["20161121_000420_0p1a_LNO_1", "20161121_021920_0p1a_LNO_1"], \
        # "MTP001":["201905", "20190704"],
        # "MTP015":["", ""],
        # "SO-prime (Jul 2020)":["20200724_125331_0p1a_LNO_1", "20200728_144718_0p1a_LNO_1"],
    },
    "uvis": {
        # "Before gyroless update (May 2024)": ["20240519_025013_1p0a_UVIS_CL", "20240521_134320_1p0a_UVIS_CL"], \
        # "Before gyroless update (Sep 2024)": ["20240915_141221_1p0a_UVIS_CL", "20240916_135327_1p0a_UVIS_CL"], \
        # "After gyroless update (Jan 2025)": ["20250113_132313_1p0a_UVIS_CL", "20250115_143500_1p0a_UVIS_CL"], \
        # "After gyroless update (Feb 2025)": ["20250219_180516_1p0a_UVIS_CL", "20250221_191311_1p0a_UVIS_CL"], \

        # "Before gyroless update (Sep 2024) 1": {"h5s": ["20240915_141221_1p0a_UVIS_CL"], "npoints": 149, "smoothing": 159, "max_angle": 0.00315}, \
        # "Before gyroless update (Sep 2024) 1": {"h5s": ["20240915_141221_1p0a_UVIS_CL"], "centre_xy": [-0.00013281-0.0001, 0.00002734-0.0001], "npoints": 149, "smoothing": 159, "max_angle": 0.0035}, \
        # "After gyroless update (Jan 2025) 1": {"h5s": ["20250113_132313_1p0a_UVIS_CL"], "npoints": 249, "smoothing": 229, "max_angle": 0.0035}, \
        # "After gyroless update (Jan 2025) 1": {"h5s": ["20250113_132313_1p0a_UVIS_CL"], "centre_xy": [-0.00023281000000000002, -7.266e-05], "npoints": 149, "smoothing": 229, "max_angle": 0.00325}, \
        # "After gyroless update (Jan 2025) 2": {"h5s": ["20250115_143500_1p0a_UVIS_CL"], "centre_xy": [-2.6171875e-04, 8.5937500e-05], "npoints": 149, "smoothing": 229, "max_angle": 0.00315}, \
        # "After gyroless update (Feb 2025) 1": {"h5s": ["20250219_180516_1p0a_UVIS_CL"], "centre_xy": [0.00007813, -0.00025781], "npoints": 149, "smoothing": 229, "max_angle": 0.00325}, \
        "After gyroless update (Feb 2025) 2": {"h5s": ["20250221_191311_1p0a_UVIS_CL"], "npoints": 149, "smoothing": 229, "max_angle": 0.00325}, \
        # "After gyroless update (July 2025) 1": {"h5s": ["20250704_005109_1p0a_UVIS_CL"], "centre_xy": [-0.00023281, -7.266e-05], "npoints": 149, "smoothing": 229, "max_angle": 0.00325}, \
        # "After gyroless update (July 2025) 2": {"h5s": ["20250704_104036_1p0a_UVIS_CL"], "centre_xy": [-0.00023281, -7.266e-05], "npoints": 149, "smoothing": 229, "max_angle": 0.00325}, \
    },
}

if channel == "uvis":
    referenceFrame = "TGO_NOMAD_UVIS_OCC"
elif channel == "so":
    referenceFrame = "TGO_NOMAD_SO"
elif channel == "lno":
    referenceFrame = "TGO_NOMAD_LNO_OPS_OCC"
# referenceFrame = "TGO_SPACECRAFT"


def get_vector2(hdf5_filename):
    # read state vectors from file generated by WebGeoCalc
    dt, obs2SunVector = read_webgeocalc(hdf5_filename, "spkpos")
    print(dt[0])
    obs2SunUnitVector = obs2SunVector / np.tile(la.norm(obs2SunVector, axis=1), (3, 1)).T
    return -1 * obs2SunUnitVector  # -1 is there to switch the directions to be like in cosmographia


def get_vector(date_time, reference_frame):
    # calculate state vectors
    # print("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)
    obs2SunVector = sp.spkpos("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
    obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
    return -1 * obs2SunUnitVector  # -1 is there to switch the directions to be like in cosmographia


if "raw_xy" in plot_types or "meshgrid" in plot_types or "fit_zero" in plot_types:
    if channel == "so":
        # plot all in subplots
        fig1, axes = plt.subplots(nrows=2, ncols=4, figsize=(FIG_X+5, FIG_Y+2))
        axes = axes.flatten()

        labelpad = 15
        fig1.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    else:
        # plot just one at a time
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(FIG_X, FIG_Y),)
        labelpad = 0

    if channel == "uvis":
        # no spatial/spectral dimensions for uvis
        ax1.set_xlabel("%s SPICE FRAME X" % referenceFrame)
        ax1.set_ylabel("%s SPICE FRAME Y" % referenceFrame, labelpad=labelpad)
    else:
        ax1.set_xlabel("%s SPICE FRAME X (Spatial direction)" % referenceFrame)
        ax1.set_ylabel("%s SPICE FRAME Y (Spectral direction)" % referenceFrame, labelpad=labelpad)

linescan_d = {}
for linescan_ix, linescan_name in enumerate(linescan_dict[channel].keys()):

    if "raw_xy" in plot_types or "meshgrid" in plot_types:
        if channel == "so":
            ax1 = axes[linescan_ix]

    h5s = linescan_dict[channel][linescan_name]["h5s"]

    for scan_ix, h5 in enumerate(h5s):

        linescan_d[h5] = {"xs": [], "ys": [], "counts": []}

        hdf5_file = open_hdf5_file(h5, path=hdf5_path)

        detector_data_all = hdf5_file["Science/Y"][...]
        datetime_all = hdf5_file["Geometry/ObservationDateTime"][...]

        # convert data to times and boresights using spice
        et_all = np.asfarray([np.mean([utc2et(i[0]), utc2et(i[1])]) for i in datetime_all])

        if channel != "uvis":
            # get SO/LNO data
            detector_centre_line = DETECTOR_CENTRE_LINES[channel]
            window_top_all = hdf5_file["Channel/WindowTop"][...]
            window_height = hdf5_file["Channel/WindowHeight"][0]+1
            binning = hdf5_file["Channel/Binning"][0]+1
            sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]

            print(window_top_all[0], window_height, binning, sbsf)

            if binning == 2:  # stretch array if binned
                detector_data_all = np.repeat(detector_data_all, 2, axis=1)
                detector_data_all /= 2
            if binning == 4:  # stretch array
                detector_data_all = np.repeat(detector_data_all, 4, axis=1)
                detector_data_all /= 4

            if sbsf == 0:
                # if background subtration not used, use rough reduction
                detector_data_all -= 50000.0

            if binning == 1 or binning == 2:
                # find which window top contains the line - this is not correct for binning
                unique_window_tops = list(set(window_top_all))
                for unique_window_top in unique_window_tops:
                    if unique_window_top <= detector_centre_line <= (unique_window_top + window_height):
                        centre_window_top = unique_window_top
                        centre_row_index = detector_centre_line - unique_window_top

                window_top_indices = np.where(window_top_all == centre_window_top)[0]
                detector_data_line = detector_data_all[window_top_indices, centre_row_index, :]
                et_line = et_all[window_top_indices]

            if binning == 4:
                # if all binned like an occultation
                detector_data_line = np.mean(detector_data_all[:, 7:9, :], axis=1)
                et_line = et_all[:]

            detector_line_mean = np.mean(detector_data_line[:, 160:240], axis=1)
            detector_line_min = (np.max(detector_line_mean) + np.min(detector_line_mean)) * 0.5
            # detector_line_max = np.max(detector_line_mean)
            detector_line_mean[detector_line_mean < detector_line_min] = detector_line_min
            # detector_line_mean[detector_line_mean > detector_line_max] = detector_line_max

        else:
            # if uvis, take mean of whole spectrum
            detector_data_line = np.mean(detector_data_all[:, 780:820], axis=1)
            #  simple bg subtraction
            detector_data_line -= np.min(detector_data_line)
            et_line = et_all[:]
            detector_line_mean = detector_data_line

        if PRINT_WGC_ETS:
            with open("%s_ets.txt" % h5, "w") as f:
                for et in et_line:
                    f.write("%0.3f\n" % et)
            continue

        # print("%s: max value = %0.0f, min value = %0.0f" % (h5, np.max(detector_line_mean), np.min(detector_line_mean)))

        if not WGC:
            # calculate pointing with spiceypy
            unitVectors = np.asfarray([get_vector(datetime, referenceFrame) for datetime in et_line])

        else:
            # get pointing from file after calculating with WebGeoCalc
            unitVectors = get_vector2(h5)
            print(et_line[0])


#        marker_colour = np.log(detector_line_mean)
        marker_colour = detector_line_mean

        linescan_d[h5]["xs"].extend(unitVectors[:, 0])
        linescan_d[h5]["ys"].extend(unitVectors[:, 1])
        linescan_d[h5]["counts"].extend(marker_colour)

        xys = np.asarray([linescan_d[h5]["xs"], linescan_d[h5]["ys"]]).T

        linescan_d[h5]["angles"] = [sp.vnorm([xy[0], xy[1]]) for xy in xys]

        for dset in ["xs", "ys", "counts", "angles"]:
            linescan_d[h5][dset] = np.asarray(linescan_d[h5][dset])

        if "fit_zero" in plot_types:
            # find offsets that best fit data
            # first plot the original boresight vector

            def apply_offset(params, xys):
                [x_offset, y_offset] = params
                xys[:, 0] -= x_offset
                xys[:, 1] -= y_offset
                return xys

            if "centre_xy" not in linescan_dict[channel][linescan_name].keys():
                plt.figure()
                plt.plot(linescan_d[h5]["angles"], linescan_d[h5]["counts"])

                # get subset of data where sep angle should be the same
                # fit sun angles to minimise difference in counts
                ix_midvals1 = np.where((linescan_d[h5]["counts"] > 2.5e6) & (linescan_d[h5]["counts"] < 2.55e6))[0]
                # ix_midvals1 = np.where((linescan_d[h5]["counts"] > 1.0e6) & (linescan_d[h5]["counts"] < 1.25e6))[0]
                # ix_midvals1 = np.where((linescan_d[h5]["counts"] > 1.0e6) & (linescan_d[h5]["counts"] < 1.25e6))[0]
                # ix_midvals2 = np.where((linescan_d[h5]["angles"] > 0.001) & (linescan_d[h5]["angles"] < 0.0012))[0]
                ix_midvals2 = ix_midvals1

                print("%i points for fitting" % len(ix_midvals1))

                xys_new1 = xys[ix_midvals1, :]
                xys_new2 = xys[ix_midvals2, :]

                def min_std_angle(params, args):

                    [xys1, xys2] = args

                    xys1 = apply_offset(params, xys1.copy())
                    xys2 = apply_offset(params, xys2.copy())

                    angles1 = np.asarray([sp.vnorm([xy[0], xy[1]]) for xy in xys1])
                    angles2 = np.asarray([sp.vnorm([xy[0], xy[1]]) for xy in xys2])
                    angles_std = np.std(angles1) + np.std(angles2)
                    # print(params, np.std(angles1), np.std(angles2))
                    return float(angles_std)

                first_guess = [0.0, 0.0]
                res = minimize(min_std_angle, first_guess, args=[xys_new1, xys_new2], method="Nelder-Mead")

                x_offset = res.x[0]
                y_offset = res.x[1]

                print("Calculated values are %0.8f %0.8f" % (x_offset, y_offset))

            if "centre_xy" in linescan_dict[channel][linescan_name].keys():
                x_offset = linescan_dict[channel][linescan_name]["centre_xy"][0]
                y_offset = linescan_dict[channel][linescan_name]["centre_xy"][1]

                print("Used values are %0.5f %0.5f" % (x_offset, y_offset))

            xys_fit = apply_offset([x_offset, y_offset], xys.copy())
            angles = np.asarray([sp.vnorm([xy[0], xy[1]]) for xy in xys_fit])
            print(np.std(angles))

            # darkening calc
            on_disk_ixs = np.where(angles < linescan_dict[channel][linescan_name]["max_angle"])[0]

            angles_norm = angles[on_disk_ixs] / np.max(angles[on_disk_ixs])
            counts_norm = linescan_d[h5]["counts"][on_disk_ixs] / np.max(linescan_d[h5]["counts"][on_disk_ixs])
            # i0 = np.max(detector_line_mean)

            angles_sorted_ixs = np.argsort(angles_norm)
            angles_sorted = angles_norm[angles_sorted_ixs]
            counts_sorted = counts_norm[angles_sorted_ixs]

            n_points = len(angles_sorted_ixs)

            vals = np.linspace(0, n_points, num=linescan_dict[channel][linescan_name]["npoints"], dtype=int)
            angles_steps = []
            counts_steps = []
            for i in range(len(vals)-1):
                start = vals[i]
                stop = vals[i+1]

                counts_step = np.copy(counts_sorted[start:stop])

                # # remove further points in a loop
                # for n in range(int(np.floor((stop-start)/2))):
                #     counts_step_mean = np.mean(counts_step)
                #     ix_rem = np.argmax(np.abs(counts_step - counts_step_mean))
                #     counts_step = np.asarray([counts_step[i] for i in range(len(counts_step)) if i != ix_rem])

                # print(counts_step, angles_sorted[start:stop])

                angles_step_mean = np.mean(angles_sorted[start:stop])
                counts_step_mean = np.mean(counts_step)
                angles_steps.append(angles_step_mean)
                counts_steps.append(counts_step_mean)

            # plt.scatter(angles_steps, counts_steps, c="k")

            # smooth discrete means
            # sg_filter = non_uniform_savgol(angles_steps, counts_steps, linescan_dict[channel][linescan_name]["smoothing"], 2)

            # smooth raw data
            sg_filter = non_uniform_savgol(angles_sorted, counts_sorted, linescan_dict[channel][linescan_name]["smoothing"], 2)

            # interpolate smoothed onto angles grid
            # counts_interp = np.interp(angles_norm, angles_steps, sg_filter)
            counts_interp = np.interp(angles_norm, angles_sorted, sg_filter)

            distance = counts_norm - counts_interp

            marker_colour = distance * 100

            plt.figure()
            plt.scatter(angles_norm, counts_norm, alpha=1, c=marker_colour, cmap=red_widegrey_blue(), linewidths=0, vmin=-30, vmax=30,
                        label="Raw data points")
            plt.title("%s" % (linescan_name))
            plt.scatter(angles_norm, counts_interp, color="k", label="Running mean")
            plt.xlabel("Distance from Sun centre (normalised)")
            plt.ylabel("Signal on detector (normalised)")
            plt.grid()
            plt.legend()

            ax1.scatter(unitVectors[:, 0], unitVectors[:, 1], color="green", alpha=0.2, linewidths=0)
            scat = ax1.scatter(unitVectors[:, 0][on_disk_ixs], unitVectors[:, 1][on_disk_ixs], c=marker_colour,
                               alpha=1, cmap=red_widegrey_blue(), linewidths=0, vmin=-30, vmax=30)
            cbar = fig1.colorbar(scat)
            cbar.set_label("Relative difference between measured and expected signal (%)", rotation=270, labelpad=20)
            ax1.set_aspect("equal")
            # ax.set_title("%s: %s & %s" % (title, hdf5_filenames[0][:8], hdf5_filenames[1][:8]))
            ax1.set_title("%s" % (linescan_name))
            ax1.grid()

            # fit (1-cos theta)^N function to shape (doesn't work)
            # def calc_limb_counts(params, theta):
            #     # [a1, a2, a3] = params
            #     i0 = 4345713.6

            #     # 1- cos theta shape
            #     vals = np.sum(np.asarray([a * (1 - np.cos(theta))**(i+1) for i, a in enumerate(params)]), axis=0)
            #     i = i0 * (1 + vals)

            #     # inverse polynomial
            # #     i = i0 - np.sum(np.asarray([a * theta**i for i, a in enumerate(params)]), axis=0)
            #     return i

            # def fit_limb_counts(params, args):

            #     [theta, counts] = args

            #     chisq = np.sum(calc_limb_counts(params, theta) - counts)**2

            #     print(params, chisq)
            #     return float(chisq)

            # a1 = -0.46
            # a2 = -0.23
            # a3 = -0.05

            # for scalar in [0.9, 0.95, 1.0, 1.05]:
            #     theta = angles[on_disk_ixs] * 333 * 90 * (np.pi / 180.0) * scalar
            #     counts = linescan_d[h5]["counts"][on_disk_ixs]

            #     first_guess = [a1, a2, a3, -0.05, -0.15, 0.0, 0.0, 0.0, 0.0]
            #     res = minimize(fit_limb_counts, first_guess, args=[theta, counts], method="Nelder-Mead", options={"maxfev": 1000000})

            #     fitted_counts = calc_limb_counts(list(res.x), theta)
            #     plt.plot(angles[on_disk_ixs], fitted_counts)

        if "raw_xy" in plot_types and "meshgrid" not in plot_types:
            # plot raw points
            ax1.scatter(unitVectors[:, 0], unitVectors[:, 1], c=marker_colour, alpha=1, cmap="gnuplot", linewidths=0)

    if "raw_xy" in plot_types or "meshgrid" in plot_types:

        if "meshgrid" not in plot_types:
            # plot a circle where the Sun should be
            circle1 = plt.Circle((0, 0), 0.0016, color='yellow', alpha=0.1)
            ax1.add_artist(circle1)

        ax1.set_xlim([-grid_size, grid_size])
        ax1.set_ylim([-grid_size, grid_size])
        ax1.set_aspect("equal")
        # ax1.set_title("%s: %s & %s" % (title, hdf5_filenames[0][:8], hdf5_filenames[1][:8]))
        ax1.set_title("%s" % (linescan_name))
        ax1.grid()

        if "meshgrid" in plot_types:

            # plot meshgrid interpolation of the points

            # Create grid values first.
            ngridx = 100
            ngridy = 100
            xi = np.linspace(-grid_size, grid_size, ngridx)
            yi = np.linspace(-grid_size, grid_size, ngridy)

            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            import matplotlib.tri as tri
            triang = tri.Triangulation(linescan_d[h5]["xs"], linescan_d[h5]["ys"])
            interpolator = tri.LinearTriInterpolator(triang, linescan_d[h5]["counts"])
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            ax1.contourf(xi, yi, zi, levels=50, cmap="gnuplot")

    if "raw_sep" in plot_types:
        plt.figure()
        plt.plot(linescan_d[h5]["angles"], linescan_d[h5]["counts"])


if "raw_xy" in plot_types or "meshgrid" in plot_types:
    fig1.tight_layout()

if SAVE_FIG:
    fig1.savefig("%s_linescan_boresight.png" % channel, dpi=300)
# fig1.suptitle("SO Linescans")
