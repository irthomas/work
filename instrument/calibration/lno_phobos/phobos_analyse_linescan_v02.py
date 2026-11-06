# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:50:38 2022

@author: iant

PHOBOS FOV LINESCAN ANALYSIS

UPDATED TO INCLUDE ITERATIVE BAD PIXEL CORRECTION AND SOLAR SHAPE FITTING
"""

# import os
# import sys
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import spiceypy as sp

from datetime import datetime

# from instrument.nomad_lno_instrument_v01 import m_aotf
from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
# from tools.general.normalise_values_to_range import normalise_values_to_range
from tools.general.get_minima_maxima import get_local_maxima
from tools.spectra.running_mean import running_mean_1d
from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.make_offset_vector import make_offset_vector
from tools.spice.make_perimeter_bounds import make_perimeter_bounds
from tools.spice.rotation_matrix_from_vectors import rotation_matrix_from_vectors
from tools.spice.make_offset_vector import make_offset_vector

from instrument.calibration.lno_phobos.lno_offset_correction_phobos import fit_spectra
from instrument.calibration.lno_phobos.lno_bad_pixel_correction_phobos import bad_pixel_correction

DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5"

SAVE_FIGS = True
# SAVE_FIGS = False

px_range = range(150, 250)  # use mean of these pixels
px_range = [-1]  # take max point of full spectra

# continue this, then loop on vectors to get linear fits to match, check if sum is correct
# PLOT_TYPES = ["raw_spectra", "bin_signals", "angle_vs_signal", "frame_signal", "plot offset"]
# PLOT_TYPES = ["bin_signals", "angle_vs_signal", "frame_signal", "plot offset"]
PLOT_TYPES = ["frame_signal", "plot offset"]
# PLOT_TYPES = ["angle_vs_signal"]

offset_correction = "solar"  # use solar spectrum to fit spectral shape

bad_pixel_type = "iteration"  # iterate removing point with largest standard deviation


obs_types = {
    # "LNO prime Phobos linescan 1": {"h5": "20221113_202115_0p3a_LNO_1_P_169", "bins": [1, 2, 3, 4]},
    "LNO prime Phobos linescan 2": {"h5": "20221222_050103_0p3a_LNO_1_P_169", "bins": [1, 2, 3, 4]},  # low signal phase angle 52.0 degrees
    # "LNO prime Phobos linescan 3": {"h5": "20230109_230206_0p3a_LNO_1_P_169", "bins": [1, 2, 3, 4]},  # 51 degrees
    # "LNO prime Phobos linescan 4": {"h5": "20230113_034115_0p3a_LNO_1_P_169", "bins": [2, 3, 4]},
    # "LNO prime Phobos linescan 5": {"h5": "20230228_184820_0p3a_LNO_1_P_170", "bins": [2, 3, 4]},

    # "UVIS prime Phobos linescan 1":{"h5":"20221116_170814_0p1a_LNO_1", "bins":[3,4,5,6]},
}

# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_PHOBOS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 401
# et2lst: form of longitude supplied by the variable lon
SPICE_LONGITUDE_FORM = "PLANETOCENTRIC"
# spkpos: reference frame relative to which the output position vector
# should be expressed
SPICE_REFERENCE_FRAME = "J2000"
# et2utc: string format flag describing the output time string. 'C' Calendar format, UTC
SPICE_STRING_FORMAT = "C"
# et2utc: number of decimal places of precision to which fractional seconds
# (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed
SPICE_TIME_PRECISION = 3

SPICE_TARGET = "PHOBOS"
SPICE_OBSERVER = "-143"

SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"

# SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
# SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"

DREF = "TGO_NOMAD_LNO_OPS_NAD"

sp.kclear()
load_spice_kernels()

SP_DPR = sp.dpr()
NA_VALUE = np.nan
LNO_DETECTOR_CENTRE_LINE = 152
ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0

# sp.pdpool("INS-143310_FOV_REF_ANGLE", [4.000000] )
# sp.pdpool("INS-143311_FOV_REF_ANGLE", [4.000000] )
# sp.pdpool("INS-143312_FOV_REF_ANGLE", [4.000000] )
# "INS-143311_FOV_REF_ANGLE = (  4.000000 )",
# "INS-143312_FOV_REF_ANGLE = (  4.000000 )",
# ])


PHOBOS_H_OFFSET_ARCMINS = 0.0
# PHOBOS_H_OFFSET_ARCMINS = -3.25

# PHOBOS_V_OFFSET_ARCMINS = 0.0
PHOBOS_V_OFFSET_ARCMINS = -3.0

# add new vector offset in degrees and apply it to kernels
new_h_vector = 0.3158361166823326 + PHOBOS_H_OFFSET_ARCMINS / 60.0
new_v_vector = 0.0600003762020601 + PHOBOS_V_OFFSET_ARCMINS / 60.0
sp.pdpool("TKFRAME_-143311_ANGLES", [new_h_vector, new_v_vector, 0.00000000000000000])


def make_obs_point_dict(ets, new_vectors):

    dp = {}
    # dp_tmp = {}

    for point, fov in enumerate(new_vectors):
        # for point, fov in enumerate(boresightVectorbounds):

        dp[point] = {}
        # dp_tmp[point] = {}

        # dp_tmp[point]["sincpt"] = []
        # dp_tmp[point]["surf"] = []
        # dp_tmp[point]["ilumin"] = []
        dp[point]["ph_angle"] = []
        dp[point]["inc_angle"] = []
        dp[point]["em_angle"] = []

        # dp[point]["lon"] = []
        # dp[point]["lat"] = []

        for et in ets:

            try:
                sincpt = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME,
                                   SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, DREF, fov)[0]
            except sp.stypes.NotFoundError:
                sincpt = (np.zeros(3) + NA_VALUE, NA_VALUE, np.zeros(3) + NA_VALUE)[0]

            if not np.isnan(sincpt[1]):  # != NA_VALUE:
                ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME,
                                   SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, sincpt)
                dp[point]["ph_angle"].append(ilumin[2] * SP_DPR)
                dp[point]["inc_angle"].append(ilumin[3] * SP_DPR)
                dp[point]["em_angle"].append(ilumin[4] * SP_DPR)

                # reclat = sp.reclat(sincpt)
                # dp[point]["lon"].append(reclat[1] * SP_DPR)
                # dp[point]["lat"].append(reclat[2] * SP_DPR)

            else:
                dp[point]["ph_angle"].append(NA_VALUE)
                dp[point]["inc_angle"].append(NA_VALUE)
                dp[point]["em_angle"].append(NA_VALUE)

                # dp[point]["lon"].append(NA_VALUE)
                # dp[point]["lat"].append(NA_VALUE)

    return dp


cal_d = {}
for obs_type in obs_types.keys():

    h5 = obs_types[obs_type]["h5"]

    h5f = open_hdf5_file(h5, path=DATA_PATH)  # no detector offset!)

    # get detector row binning
    top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
    unique_bins = sorted(list(set(top_bins)))
    n_bins = len(unique_bins)
    binning = unique_bins[1] - unique_bins[0]

    # bin_height = h5f["Channel/Binning"][0] + 1  # number of arcminutes per bin

    # get y, reshape to 3d
    y_all = h5f["Science/Y"][...]
    y_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

    obs_dt_str_2d = h5f["Geometry/ObservationDateTime"][...]
    obs_dt_str_3d = np.reshape(obs_dt_str_2d, [-1, n_bins, obs_dt_str_2d.shape[1]])

    obs_dt_str = obs_dt_str_3d[:, 0, :]

    dt_s = [datetime.strptime(s[0].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]
    dt_e = [datetime.strptime(s[1].decode(), "%Y %b %d %H:%M:%S.%f") for s in obs_dt_str]

    dt_means = [((e - s) / 2) + s for s, e in zip(dt_s, dt_e)]

    et_s = np.asfarray([sp.utc2et(s[0]) for s in obs_dt_str])
    et_e = np.asfarray([sp.utc2et(s[1]) for s in obs_dt_str])

    ets = np.mean([et_s, et_s], axis=0)

    phase_angle_2d = np.zeros([y_3d.shape[0], y_3d.shape[1]])
    inc_angle_2d = np.zeros([y_3d.shape[0], y_3d.shape[1]])
    emis_angle_2d = np.zeros([y_3d.shape[0], y_3d.shape[1]])

    for bin_ix, bin_top in enumerate(unique_bins):

        # calculate geometry manually for each bin, so vectors can be modified
        detector_offset = np.float32([bin_top, bin_top + binning]) - np.float32(LNO_DETECTOR_CENTRE_LINE)
        # offset in pixels from detector centre
        detector_offset[1] += 1  # actually is 1 pixel more in -ve direction (down detector)

        binFovSize = detector_offset / ARCMINS_TO_RADIANS  # assume 1 arcmin per pixel

        fovPoints = [[np.mean(binFovSize), 0.0]]

        channelId = sp.bods2c(DREF)  # find channel id number
        [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)
        # boresightVector #should be array([0., 0., 1.]),

        offset_vector = [make_offset_vector(fovPoint) for fovPoint in fovPoints]

        dp = make_obs_point_dict(ets, offset_vector)

        phase_angle_2d[:, bin_ix] = dp[0]["ph_angle"]
        inc_angle_2d[:, bin_ix] = dp[0]["inc_angle"]
        emis_angle_2d[:, bin_ix] = dp[0]["em_angle"]

        # if bin_ix == 4:
        #     stop()

    # # phase angle
    # phase_angle = np.mean(h5f["Geometry/Point0/PhaseAngle"][:, :], axis=1)
    # phase_angle[phase_angle == -999] = np.nan
    # phase_angle_2d = np.reshape(phase_angle, [-1, n_bins])

    # inc_angle = np.mean(h5f["Geometry/Point0/IncidenceAngle"][:, :], axis=1)
    # inc_angle[inc_angle == -999] = np.nan
    # inc_angle_2d = np.reshape(inc_angle, [-1, n_bins])

    # emis_angle = np.mean(h5f["Geometry/Point0/EmissionAngle"][:, :], axis=1)
    # emis_angle[emis_angle == -999] = np.nan
    # emis_angle_2d = np.reshape(emis_angle, [-1, n_bins])

    emis_inc_angle_2d = inc_angle_2d + emis_angle_2d

    mean_phase = np.nanmean(phase_angle_2d)
    print("Mean phase angle:", mean_phase)

    """fit to solar spectrum to correct offset"""
    if offset_correction == "solar":
        # get diffraction orders from filename
        order = int(h5.split("_")[-1])
        # get solar calibration data
        cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
        if order not in cal_d.keys():
            cal_d = rad_cal_order(cal_h5, [order], centre_indices=px_range, path=DATA_PATH, silent=True)
            cal_d[order]["solar_spectrum"] = cal_d[order]["y_spectrum"] / np.max(cal_d[order]["y_spectrum"])
        ideal_spectrum = cal_d[order]["solar_spectrum"]

    if bad_pixel_type == "iteration":
        if "bad_pixel_fits" in PLOT_TYPES:
            y_3d = bad_pixel_correction(y_3d, plot=[[0, 2], [1, 4], [3, 2]])
        else:
            y_3d = bad_pixel_correction(y_3d)

    if offset_correction in ["solar"]:
        if "offset_fits" in PLOT_TYPES:
            title = "NOMAD-LNO observation %s:" % h5
            fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, ideal_spectrum, plot=[[1, 4], [1, 5]], title=title)
        else:
            fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_3d, ideal_spectrum)

    # make offset corrected spectral mean or max, one point per unbinned spectrum
    if px_range[0] == -1:
        y_spectral_mean = np.max(corr_spectra[:, :, :], axis=2)  # / total_integration_time / binning
    else:
        y_spectral_mean = np.mean(corr_spectra[:, :, px_range], axis=2)  # / total_integration_time / binning

    phobos_bins = obs_types[obs_type]["bins"]

    if "raw_spectra" in PLOT_TYPES:
        # plot all good spectra for a specific bin
        for bin_ix in range(y_spectral_mean.shape[1]):
            plt.figure(figsize=(8, 4))
            plt.title("Raw spectra for bin %i before offset correction. Blue = all spectra; black = high signal" % bin_ix)
            plt.xlabel("Pixel number")
            plt.ylabel("Signal (counts)")
            for i, y_spectrum in enumerate(y_3d[:, bin_ix, :]):
                colour = "C0"
                plt.plot(y_spectrum, alpha=0.3, color=colour)  # plot all in blue
                # if np.any(y_spectrum > 100):  # if any with good signal, plot in black
                #     plt.plot(y_spectrum, color="k")
                #     print(unique_bins[bin_ix])

    if "bin_signals" in PLOT_TYPES:

        emis_inc_angle_mean = np.nanmean(emis_inc_angle_2d[:, phobos_bins], axis=1)

        plt.figure(figsize=(14, 8))
        for bin_ix in np.arange(y_spectral_mean.shape[1]):
            colour = "C%0i" % bin_ix
            plt.plot(y_spectral_mean[:, bin_ix], color=colour, label="Bin %i" % bin_ix, alpha=0.3)
            phobos_running_mean = running_mean_1d(y_spectral_mean[:, bin_ix], 19)
            plt.plot(phobos_running_mean, color=colour, label="LNO signal (smoothed)")

        phobos_running_mean = running_mean_1d(np.mean(y_spectral_mean[:, phobos_bins], axis=1), 19)
        plt.plot(phobos_running_mean + 10, "k--", label="LNO mean signal (smoothed)")

        plt.plot(15 - emis_inc_angle_mean / 20, "k.", label="Sum mean emission inc angles")

        plt.title("%s: spectral mean for each bin with running mean" % h5)
        plt.legend()
        plt.grid()
        plt.xlabel("Frame index")
        plt.ylabel("Signal, with offset")

        # stop()

    if "angle_vs_signal" in PLOT_TYPES:

        emis_inc_angle_mean = np.nanmean(emis_inc_angle_2d[:, phobos_bins], axis=1)

        plt.figure(figsize=(14, 8))
        for bin_ix in np.arange(y_spectral_mean.shape[1]):
            colour = "C%0i" % bin_ix

            x = emis_inc_angle_2d[:, bin_ix]
            y = y_spectral_mean[:, bin_ix]

            y = running_mean_1d(y_spectral_mean[:, bin_ix], 19)

            good_ixs = np.where(x < 140)[0]
            if len(good_ixs) == 0:
                continue

            plt.scatter(x[good_ixs], y[good_ixs], color=colour, label="Bin %i" % bin_ix, alpha=0.3)

            # straight line for plotting
            poly = Polynomial.fit(x[good_ixs], y[good_ixs], 1)
            yfit = poly(x[good_ixs])

            plt.plot(x[good_ixs], yfit, color=colour, label="Linear fit bin %i" % bin_ix, alpha=1.0)

        plt.title(h5)
        plt.legend()
        plt.grid()
        plt.xlabel("Emission + Incidence Angle")
        plt.ylabel("Signal, with offset")

    if "frame_signal" in PLOT_TYPES:
        # plot raw y data spectrally binned
        plt.figure(figsize=(12, 3))
        plt.title("%s: Offset-corrected spectral mean" % h5)
        plt.xlabel("Frame index)")
        plt.ylabel("Bin number")
        im = plt.imshow(y_spectral_mean[:, :].T, aspect="auto")
        cbar = plt.colorbar(im)
        cbar.set_label("Signal (counts)", rotation=270, labelpad=10)
        plt.subplots_adjust(bottom=0.15)

        if SAVE_FIGS:
            plt.savefig("LNO_phobos_linescan_frame_%s_%0.2fH_%0.2fV_arcminute_offset.png" % (h5[0:15], PHOBOS_H_OFFSET_ARCMINS, PHOBOS_V_OFFSET_ARCMINS))

    if "plot offset" in PLOT_TYPES:
        phobos_bin_means = y_spectral_mean[:, phobos_bins]

        fig, ax = plt.subplots(figsize=(15, 8))
        phobos_mean = np.nanmean(phobos_bin_means, axis=1)
        for bin_ix, phobos_bin in enumerate(phobos_bins):
            plt.plot(dt_means, phobos_bin_means[:, bin_ix], alpha=0.33)
        # plt.plot(phobos_mean, "k--")
        phobos_running_mean = running_mean_1d(phobos_mean, 19)
        ax.plot(dt_means, phobos_running_mean, "k-", label="LNO signal (smoothed)")

        phobos_local_max_ix = get_local_maxima(phobos_running_mean)

        phobos_obs_ix = np.where(phobos_running_mean > 2.5)[0]

        phobos_max_ix = [i for i in phobos_local_max_ix if i in phobos_obs_ix]
        for ix, frame_ix in enumerate(phobos_max_ix):

            if ix == 0:
                label = "Max LNO signal"
            else:
                label = ""

            # remove local minima
            if phobos_running_mean[frame_ix] > phobos_running_mean[frame_ix - 2] and phobos_running_mean[frame_ix] > phobos_running_mean[frame_ix + 2]:

                ax.axvline(x=dt_means[frame_ix], color="k", linestyle="--", label=label)
                ax.text(dt_means[frame_ix], 4.5 - ix, dt_means[frame_ix])
                print(dt_means[frame_ix])

        ets = [sp.utc2et(datetime.strftime(s, "%Y %b %d %H:%M:%S.%f")) for s in dt_means]
        phobos_pos_lno = np.array([sp.spkpos(SPICE_TARGET, et, DREF, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0] for et in ets])

        phobos_vec_lno = np.array([s / sp.vnorm(s) for s in phobos_pos_lno])

        # plt.figure()
        # plt.plot(phobos_vec_lno[:, 0])
        ax2 = ax.twinx()
        ax2.plot([dt_means[0], dt_means[-1]], [0, 0], "b--", label="Zero offset from Phobos centre")
        ax2.plot(dt_means, phobos_vec_lno[:, 1], "b", label="LNO offset from Phobos centre")
        # plt.plot(phobos_vec_lno[:, 2])

        ax.set_xlabel("Observation datetime")
        ax.set_ylabel("LNO signal in illuminated bins")
        ax.set_title("LNO Phobos linescan with %0.2f arcminute boresight offset" % PHOBOS_H_OFFSET_ARCMINS)
        ax2.set_ylabel("LNO vector offset from Phobos centre with %0.2f arcminute offset" % PHOBOS_H_OFFSET_ARCMINS, color="b")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        # ax.legend()
        # ax2.legend()
        ax2.grid()

        if SAVE_FIGS:
            plt.savefig("LNO_phobos_linescan_%s_%0.2f_arcminute_offset.png" % (h5[0:15], PHOBOS_H_OFFSET_ARCMINS))
