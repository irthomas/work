# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:14:38 2023

@author: iant

CO ORDERS OCCULTATION
"""
from analysis.so_lno_2023.functions.geometry import make_path_lengths
from analysis.so_lno_2023.forward_model import forward
from analysis.so_lno_2023.molecules import get_molecules
from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from tools.file.h5_obj import h5_obj
# from tools.plotting.anim import make_line_anim
from tools.spectra.baseline_als import baseline_als

from tools.general.get_minima_maxima import get_local_minima_or_equals


good_px_ixs = np.arange(50, 320)


# fullscans with orders 185 to 195
h5_no_order = "20230826_015002_1p0a_SO_A_I_"
mol_scaler = 0.25
# h5_no_order = "20230826_212855_1p0a_SO_A_I_"
# mol_scaler = 1.0
# h5_no_order = "20230828_172637_1p0a_SO_A_E_"
# h5_no_order = "20230831_231912_1p0a_SO_A_I_"
# h5_no_order = "20230920_225731_1p0a_SO_A_I_"
# h5_no_order = "20230921_223324_1p0a_SO_A_I_"

# h5_no_order = "20230122_055302_1p0a_SO_A_I_"

# LNO 0p3a
h5_no_order = "20240120_173338_0p3a_LNO_1_E_"

# orders = [186]
orders = [195]
# orders = [194]
# orders = np.arange(186, 196)
h5s = ["%s%i" % (h5_no_order, order) for order in orders]
bin_ixs = [0, 1, 2, 3]

# apply baseline als or manual baseline correction?
apply_als = True


# low_alt_ix = 10

plot_type = []
plot_type = ["trans_raw_nu"]  # raw transmittances with wavenumbers
# plot_type = ["trans_raw"] #raw transmittances with pixel numbers
# plot_type = ["pdf"] #save baseline-corrected and smoothed transmittances to file
# plot_type = ["trans_norm"] #check baseline fitting and normalisation
# plot_type = ["trans_raw", "trans_norm"]

# plot_type = ["trans_raw_nu", "abs_fitting", "trans_norm", "abs_depth"]
# plot_type = ["trans_raw_nu", "abs_fitting", "trans_norm", "pdf", "abs_depth"]

# save_plots = True
save_plots = False

# orders = [186]#np.arange(183, 187)
# orders = np.arange(183, 187)
# bin_ixs = [0]
# low_alt_ix = 10
# h5_no_order = "20220301_114833_1p0a_SO_A_I_"
# h5s = ["%s%i" %(h5_no_order, order) for order in orders]


# solar calibration
# h5s = ["20211105_155547_1p0a_SO_1_CM"]
# bin_ixs = [6,7,8,9,10,11,12,13,14,15,16,17,18]


# baseline points are calculated for one reference observation,
# assuming first 50 pixels removed
abs_line_points = {
    183: [95.443, 137.048, 177.873, 217.829],
    184: [76.91540426844962, 114.88315004791231, 189.86206428016794],
    185: [3.3050990035902257, 77.30923093995291, 112.16091686858721, 145.67586364327533, 173.18314437648354, 178.61095030677595],
    186: [17.835, 144.926, 190.194],
    194: [24.76336173, 51.87450897, 57.10463264],
    195: [13.9924554, 51.44191521, 234.28596886],  # very faint lines
}

# assuming first 50 pixels removed, corresponding to absorption indices above
baseline_points = {
    183: [4, 8, 26, 47, 65, 81, 88, 109, 123, 129, 149, 160, 170, 194, 203, 224, 229, 239, 246, 262],
    184: [0, 5, 8, 13, 27, 46, 83, 92, 108, 118, 123, 130, 142, 159, 165, 181, 194, 205, 215, 230, 237, 246, 262],
    185: [8, 13, 14, 22, 27, 33, 46, 47, 48, 83, 91, 92, 108, 117, 128, 129, 139, 157, 182, 204, 205, 215, 230, 237, 246, 261],
    186: [6, 10, 22, 27, 48, 66, 72, 82, 92, 108, 120, 130, 158, 165, 182, 194, 206, 217, 247, 261],
    194: [1, 17, 41, 55, 83, 108, 124, 161, 173, 195, 211, 226, 241, 254, 267],
    195: [1, 6, 9, 17, 22, 33, 39, 47, 62, 69, 72, 83, 94, 110, 115, 123, 133, 140, 148, 153, 162, 177, 183, 192, 210, 225, 228, 239, 245, 252, 255, 261, 268],
}


if "trans_raw" in plot_type or "trans_raw_nu" in plot_type:
    fig1, ax1 = plt.subplots()
if "abs_depth" in plot_type:
    fig2, ax2 = plt.subplots()


h5s_d = {}
for h5 in h5s:

    h5o = h5_obj(h5)
    h5o.set_h5_path(r"C:\Users\iant\Documents\DATA\hdf5")
    # h5o.set_h5_path(r"E:\DATA\hdf5")

    h5o.h5_to_dict(bin_ixs)
    h5o.cut_pixels(bin_ixs, good_px_ixs)

    if h5o.calibration:
        h5o.solar_cal(bins_to_average=bin_ixs)
    else:
        h5o.trans_recal(bin_ixs=bin_ixs, top_of_atmosphere=110.0)

    if "trans_raw" in plot_type:
        h5o.plot("trans_raw", "y_mean", bin_ixs, ax=ax1)
        plt.title("%s%s split fullscan" % (h5_no_order, "_".join(["%s" % i for i in orders])))
        plt.grid()
        plt.tight_layout()
    if "trans_raw_nu" in plot_type:
        h5o.plot("trans_raw_nu", "y_mean", bin_ixs, ax=ax1)
        plt.title("%s%s split fullscan" % (h5_no_order, "_".join(["%s" % i for i in orders])))
        plt.grid()
        plt.tight_layout()
        if save_plots:
            plt.savefig()

    # loop through bins i.e. 0-3
    for bin_ix in bin_ixs:
        h5_d = h5o.h5_d[bin_ix]

        y = h5_d["y_mean"]
        alts_all = h5_d["alt"]
        order = h5o.h5_d["orders"][0]
        x = h5o.h5_d["x"]

        """spectral calibration"""
        # fit to abs lines
        # find local minima

        # find indices where 5% < Trans <95%
        atmos_ixs = np.where((np.max(y[:, 160:240], axis=1) > 0.2) & (np.max(y[:, 160:240], axis=1) < 0.95))[0]

        # quick baseline fit
        y_simp_norms = np.zeros((len(atmos_ixs[2:]), y.shape[1]))
        for i, atmos_ix in enumerate(atmos_ixs[2:]):
            y_simp_baseline = np.polyval(np.polyfit(np.arange(y.shape[1]), y[atmos_ix, :], 9), np.arange(y.shape[1]))
            y_simp_norm = y[atmos_ix, :] / y_simp_baseline
            y_simp_norms[i, :] = y_simp_norm

        y_av_norm = np.mean(y_simp_norms, axis=0)

        y_av_std = np.std(y_av_norm)
        y_av_mean = np.mean(y_av_norm)

        low_alt_ix = atmos_ixs[0]-1
        high_alt_ix = np.where(alts_all < 100.0)[0][-1]

        # find absorption line indices - iterate until at least 3 lines found
        abs_ixs = []
        n_stds = 5

        while len(abs_ixs) < 3:

            n_stds *= 0.95

            below_std_ixs = np.where(y_av_norm < (y_av_mean - y_av_std*n_stds))[0]
            minima_ixs = get_local_minima_or_equals(y_av_norm)

            abs_ixs = sorted(list(set(below_std_ixs).intersection(minima_ixs)))

        # now find exact pixel number of centre of lines
        # plt.figure()
        min_ixs = []
        min_nus = []
        for abs_ix in abs_ixs:
            abs_ix_around = [abs_ix-1, abs_ix, abs_ix+1]

            # quadratic to find minima
            abs_polyfit = np.polyfit([-1., 0., 1.], y_av_norm[abs_ix_around], 2)
            abs_polyval = np.polyval(abs_polyfit, [-1., 0., 1.])
            abs_min = -abs_polyfit[1]/(2.0 * abs_polyfit[0])

            # plt.scatter(abs_ix_around, abs_polyval)
            # plt.scatter(np.arange(abs_ix-1, abs_ix+1, 0.01), np.polyval(abs_polyfit, np.arange(-1, 1, 0.01)))
            # plt.axvline(abs_min + abs_ix)

            min_ixs.append(abs_min + abs_ix)
            min_nus.append(x[abs_ix] + abs_min * (x[abs_ix]-x[abs_ix-1]))

        min_ixs = np.asarray(min_ixs)

        print(h5, "pixel indices of minima:", min_ixs)

        if order in baseline_points.keys():

            # correct for temperature shift
            # get corresponding absorption indices from a reference observation
            abs_line_ixs = np.asarray(abs_line_points[order])

            # find difference between current observation and the reference obs
            # check if number of elements is the same
            if min_ixs.shape[0] == abs_line_ixs.shape[0]:
                if np.max(np.abs(abs_line_ixs - min_ixs)) < 20.0:
                    # if fitting is good
                    ix_shifts = min_ixs - abs_line_ixs
                else:
                    # if n elements is correct but mismatching indices
                    good_ixs1 = []
                    good_ixs2 = []
                    for abs_line_ix in abs_line_ixs:
                        for min_ix in min_ixs:
                            if np.abs(abs_line_ix - min_ix) < 10:
                                good_ixs1.append(min_ix)
                                good_ixs2.append(abs_line_ix)
                    min_ixs = np.asarray(good_ixs1)
                    abs_line_ixs = np.asarray(good_ixs2)

            elif min_ixs.shape[0] > abs_line_ixs.shape[0]:
                # need to remove element(s) from min_ixs
                # find nearest indices
                good_ixs = []
                for abs_line_ix in abs_line_ixs:
                    for min_ix in min_ixs:
                        if np.abs(abs_line_ix - min_ix) < 10:
                            good_ixs.append(min_ix)
                min_ixs = np.asarray(good_ixs)

            else:
                # need to remove element(s) from abs_line_ixs
                # find nearest indices
                good_ixs = []
                for min_ix in min_ixs:
                    for abs_line_ix in abs_line_ixs:
                        if np.abs(abs_line_ix - min_ix) < 10:
                            good_ixs.append(abs_line_ix)
                abs_line_ixs = np.asarray(good_ixs)

            if min_ixs.shape[0] == abs_line_ixs.shape[0]:
                ix_shifts = min_ixs - abs_line_ixs
            else:
                print("Error: mismatching indices found")

            # check not too far away (indicates error)
            if np.std(ix_shifts) < 0.22:
                ix_shift = np.mean(ix_shifts)
                print("Temperature shift from reference spectrum: %0.3f pixels, std %0.3f" % (ix_shift, np.std(ix_shifts)))
            else:
                print("Error in temperature shift correction:", np.std(ix_shifts), min_ixs, abs_line_ixs)

            if not apply_als:
                # now shift the continuum indices by this amount
                # get continuum indices
                baseline_ixs = np.asarray(baseline_points[order])
                baseline_ixs += int(np.round(ix_shift))

                # remove negatives (points cut off the left hand side due to the shift)
                baseline_ixs = baseline_ixs[baseline_ixs >= 0.0]

            if "abs_fitting" in plot_type:

                plt.figure()
                plt.title("Fitting absorption lines")
                for atmos_ix in atmos_ixs:
                    plt.plot(y[atmos_ix, :])
                plt.grid()
                plt.xlabel("Pixel number")
                plt.ylabel("Transmittance")
                for min_ix in min_ixs:
                    plt.axvline(min_ix)

        else:
            print("Points not found")
            plt.figure()
            plt.plot(y.T, label=["%0.1fkm" % f for f in alts_all])
            plt.legend()
            sys.exit()

        # ratio_2d = ratio[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :]
        # y *= ratio_2d.T

        """normalise baseline"""
        if apply_als:
            y_baseline = np.zeros_like(y)
            for row_ix in np.arange(y.shape[0]):
                y_baseline[row_ix, :] = baseline_als(y[row_ix, :], lam=125.0, p=0.99)
        else:

            polyfits = np.polyfit(baseline_ixs, y[:, baseline_ixs].T, 5)

            # interpolate to spectral continuum
            y_baseline = np.zeros_like(y)
            for row_ix in np.arange(y.shape[0]):
                y_baseline[row_ix, :] = np.polyval(polyfits[:, row_ix], np.arange(y.shape[1]))

        # plt.figure()
        # plt.title("Y and Y baseline fits")
        # plt.plot(y.T)
        # plt.plot(y_baseline.T)

        # normalise baseline to 1, remove lowest altitudes (noise only)
        y_norm = y[low_alt_ix:, :]/y_baseline[low_alt_ix:, :]
        alts = alts_all[low_alt_ix:]

        # smooth each pixel as the transmittance drops using a high order polynomial
        y_smooth = np.zeros_like(y_norm)
        row_ixs = np.arange(y_norm.shape[0])
        for px_ix in np.arange(y_norm.shape[1]):
            y_smooth[:, px_ix] = np.polyval(np.polyfit(alts, y_norm[:, px_ix], int(np.floor(row_ixs.shape[0]/8.))), alts)

        if "trans_raw" in plot_type:
            # plot baselines on transmittance plots
            ax1.plot(y_baseline.T)

        if "trans_norm" in plot_type:

            plt.figure()
            # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y_norm.shape[0])[np.newaxis, :], y_smooth.T, alpha=0.1)
            plt.plot(y_smooth.T, alpha=0.1)

            plt.xlabel("Pixel number")
            plt.ylabel("Transmittance")
            plt.title("%s: baseline corrected and smoothed" % h5)
            plt.grid()

        if "pdf" in plot_type:
            with PdfPages("%s.pdf" % h5) as pdf:
                ylim = [np.nanmin(y_norm), np.nanmax(y_norm)]
                for i, (alt, y_norm_spec, y_smooth_spec) in enumerate(zip(alts, y_norm, y_smooth)):
                    plt.figure()
                    plt.plot(x, y_norm_spec, alpha=0.5, label="Y Normalised")
                    plt.plot(x, y_smooth_spec, alpha=0.5, label="Y Smoothed")
                    plt.ylim(ylim)
                    plt.text(x[0]+1.0, np.mean(ylim), "i=%i, %0.1fkm (T=%0.2f)" % (i, alt, np.mean(y_baseline[low_alt_ix+i, :])))
                    pdf.savefig()
                    plt.close()

            # anim_d[alt] = y_norm_spec

        # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :], y[:, px_ixs].T)
        # plt.plot(px_ixs[:, np.newaxis] + np.zeros(y.shape[0])[np.newaxis, :], y_baseline.T)

        # plt.figure()
        # plt.plot(y_norm)

        if "abs_depth" in plot_type:
            for i, px_ix in enumerate(np.round(min_ixs)):
                ax2.plot(alts, y_smooth[:, int(px_ix)], label="Selected pixel %i" % px_ix, c="C%i" % i)
                ax2.scatter(alts, y_norm[:, int(px_ix)], c="C%i" % i)
            ax2.legend()


# compare absorption depths to forward model
# from analysis.so_lno_2023.geometry import get_geometry

channel = "so"
# chosen_alt = 40.0 #km
# chosen_alt = 60.0 #km
# chosen_alt = 140.0 #km

molecules = {
    "CO": {"isos": [1, 2, 3, 4]},
    # "CO":{"isos":[1]},
}


# alt_delta = 5.0 #each layer
# alt_max = chosen_alt + 50.0

if orders[0] == [184, 185, 186, 187]:
    orders = np.arange(183, 193)
if orders[0] == [188, 189, 190, 191, 192]:
    orders = np.arange(185, 195)
if orders[0] in [193, 194, 195]:
    orders = np.arange(190, 199)


centre_order = h5o.diffraction_order
aotf_freq = h5o.h5_d["aotfs"][0]
grat_t = h5o.h5_d["T"]
aotf_t = h5o.h5_d["T"]


# get calibration info
# aotf = {"type":"file", "filename":"4500um_closest_aotf.txt"}
# aotf = aotf={"type":"sinc_gauss"}
aotf = aotf = {"type": "custom"}

"""initial parameters"""
aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)

cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)


geom_d = {}

for bin_ix in bin_ixs:
    # h5_d = h5o.h5_d[bin_ix]

    for z_ix in range(1):

        # geom_d["alt"] = h5o.h5_d[bin_ix]["alt"][atmos_ixs]
        geom_d["alt"] = alts_all[low_alt_ix+z_ix:high_alt_ix]
        geom_d["myear"] = 36
        geom_d["ls"] = h5o.h5_d["ls"]
        geom_d["lst"] = h5o.h5_d["lst"]
        geom_d["lat"] = h5o.h5_d[bin_ix]["lat"][0]
        geom_d["lon"] = h5o.h5_d[bin_ix]["lon"][0]

        geom_d["alt_grid"] = geom_d["alt"]
        path_lengths = make_path_lengths(geom_d["alt_grid"])
        geom_d["path_lengths_km"] = path_lengths

        molecule_d = get_molecules(molecules, geom_d)

        # from lmfit import minimize
        from lmfit import Parameters

        fw = forward(raw=False)
        fw.calibrate(cal_d)
        fw.geometry(geom_d)
        fw.molecules(molecule_d)

        params = Parameters()
        params.add('mol_scaler', value=mol_scaler)

        # trans1 = fw.forward_so(params, plot=["hr", "cont"])
        trans1 = fw.forward_so(params)

        plt.figure()
        plt.plot(y_smooth[atmos_ixs[0], :])
        plt.plot(trans1)
        plt.title("Altitude range %0.3f-%0.3f" % (alts_all[low_alt_ix+z_ix], alts_all[high_alt_ix]))
