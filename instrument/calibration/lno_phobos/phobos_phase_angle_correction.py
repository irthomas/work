# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:37:18 2025

@author: iant

FULL TREATMENT:
    BAD PIXEL REMOVAL
    SELECT LOW NOISE FRAMES
    CHECK ILLUMINATION OF TOP/BOTTOM ROWS TO REMOVE CORRECT ONE - TBD IF DONE
    SUBTRACT OFFSET
    DEFINE/CORRECT FOR PHASE ANGLE VARIATION


"""
import re
import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2

from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order
from instrument.calibration.lno_phobos.lno_offset_correction_phobos import fit_spectra
from instrument.calibration.lno_phobos.lno_bad_pixel_correction_phobos import bad_pixel_correction


# data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3a"

px_range = range(120, 280)


# choose dataset for finding most/least signal rows (phase angle is not representative)
dset_name = "EmissionAngle"
# dset_name = "IncidenceAngle"
# dset_name = "PhaseAngle"
# dset_name = "PhaseAngle"

PLOT_TYPES = [""]

# PLOT_TYPES = ["raw_frames"]
# PLOT_TYPES = ["angles"]
PLOT_TYPES = ["all_orders", "each_order"]
# PLOT_TYPES = ["each_order"]
# PLOT_TYPES = ["bad_pixel_fits"]
# PLOT_TYPES = ["solar_fits"]


if __name__ == "__main__":

    if "all_orders" in PLOT_TYPES:
        fig1, ax1 = plt.subplots()
        ax1.set_title("LNO Phobos phase angle all orders")

    # get list of all orders measured of Phobos
    # regex = re.compile("(20240[6-9]|20241.|2025..).._.*_0p3a_LNO_1_P_.*")
    regex = re.compile("(20240[6-9]|20241.).._.*_0p3a_LNO_1_P_.*")
    _, h5s, _ = make_filelist2(regex, file_level, path=data_path, open_files=False)

    phobos_orders = [int(s.split("_")[-1]) for s in h5s]
    phobos_unique_orders = sorted(list(set(phobos_orders)))

    # get solar counts from a calibration observation
    """get solar calibration info from an LNO solar cal fullscan. This gives the sensitivity of the instrument in each order and solar radiance"""
    # if "solar_scalars" not in globals() or "solar_spectra" not in globals():
    cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
    # cal_d = {order:rad_cal_order(cal_h5, order, centre_indices=None) for order in unique_orders}
    cal_d = {order: rad_cal_order(cal_h5, order, centre_indices=px_range, path=data_path) for order in phobos_unique_orders}
    solar_scalars = {order: cal_d[order]["y_centre_mean"] / 2.0e6 for order in cal_d.keys()}
    solar_spectra = {order: cal_d[order]["y_spectrum"] / np.max(cal_d[order]["y_spectrum"]) for order in cal_d.keys()}

    order_d = {}

    phobos_unique_orders = [i for i in sorted(list(set(phobos_orders))) if i in range(160, 171, 2)]

    for order_ix, phobos_unique_order in enumerate(phobos_unique_orders):

        # h5s = [
        #     "20250628_083013_0p3a_LNO_1_P_166",  # tracking
        #     "20250622_070511_0p3a_LNO_1_P_166",
        #     "20250619_101826_0p3a_LNO_1_P_166",
        #     "20250619_022647_0p3a_LNO_1_P_166",
        #     "20250411_203101_0p3a_LNO_1_P_166",
        #     "20250408_080101_0p3a_LNO_1_P_166",
        #     "20250405_190547_0p3a_LNO_1_P_166",
        #     "20250402_221907_0p3a_LNO_1_P_166",
        #     "20250402_142717_0p3a_LNO_1_P_166",
        #     "20241229_103444_0p3a_LNO_1_P_166",  # inertial
        #     "20241213_052910_0p3a_LNO_1_P_166",
        #     "20241210_084211_0p3a_LNO_1_P_166",
        #     "20241207_040411_0p3a_LNO_1_P_166",
        #     # "20241011_012228_0p3a_LNO_1_P_174",
        #     # "20241004_235732_0p3a_LNO_1_P_174",
        #     # "20241001_191914_0p3a_LNO_1_P_174",
        #     # "20240922_210734_0p3a_LNO_1_P_174",
        #     # "20240911_224710_0p3a_LNO_1_P_174",
        #     "20240831_034846_0p3a_LNO_1_P_165",  # 6 rows
        #     "20240825_022347_0p3a_LNO_1_P_165",
        #     "20240819_005847_0p3a_LNO_1_P_165",
        #     "20240805_163944_0p3a_LNO_1_P_165",
        #     "20240702_042211_0p3a_LNO_1_P_165",
        #     "20240627_023139_0p3a_LNO_1_P_165",
        #     "20240620_092319_0p3a_LNO_1_P_165",
        # ]

        regex = re.compile("(20240[6-9]|20241.|2025..).._.*_0p3a_LNO_1_P_%i" % phobos_unique_order)
        # regex = re.compile("(20240[6-9]|20241.).._.*_0p3a_LNO_1_P_%i" % phobos_unique_order)
        # regex = re.compile("2025...._.*_0p3a_LNO_1_P_%i" % phobos_unique_order)

        h5fs, h5s, _ = make_filelist2(regex, file_level, path=data_path)

        # if len(h5s) < 10:
        #     continue

        if "each_order" in PLOT_TYPES:
            fig2, ax2 = plt.subplots()
            ax2.set_title("LNO Phobos phase angle order %i " % phobos_unique_order)

        all_mean_angles = []
        all_mean_ys = []

        for h5_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):

            # print(h5)

            obs_dt_strs_all = h5f["Geometry/ObservationDateTime"][...]

            top_bins = h5f["Science/Bins"][:, 0]  # detector row of top of each bin
            unique_bins = sorted(list(set(top_bins)))
            n_bins = len(unique_bins)
            binning = unique_bins[1] - unique_bins[0]

            """correct for different total integration times"""
            integration_time = h5f["Channel/IntegrationTime"][0]  # detector row of top of each bin
            n_accs = h5f["Channel/NumberOfAccumulations"][0]  # detector row of top of each bin
            total_integration_time = integration_time * n_accs / 1000  # seconds

            order = int(h5f["Channel/DiffractionOrder"][0])

            # get y, reshape to 3d
            y_all = h5f["Science/Y"][...]
            y_all_3d = np.reshape(y_all, [-1, n_bins, y_all.shape[1]])

            # get chosen angle data, reshape to 2d
            angle = np.mean(h5f["Geometry/Point0/%s" % dset_name][:, :], axis=1)
            angle[angle == -999] = np.nan  # 90.0
            angle_2d = np.reshape(angle, [-1, n_bins])

            # also get phase angle
            phase_angle = np.mean(h5f["Geometry/Point0/PhaseAngle"][:, :], axis=1)
            phase_angle[phase_angle == -999] = np.nan
            phase_angle_2d = np.reshape(phase_angle, [-1, n_bins])

            """correct bad pixels"""
            if "bad_pixel_fits" in PLOT_TYPES:
                y_all_3d = bad_pixel_correction(y_all_3d, plot=[[0, 2], [1, 4], [3, 2]])
            else:
                y_all_3d = bad_pixel_correction(y_all_3d)

            """fit to solar spectrum to correct offset"""
            if "solar_fits" in PLOT_TYPES:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_all_3d, solar_spectra[phobos_unique_order], plot=[[1, 0], [1, 1], [1, 2]])
            else:
                fitted_spectra, corr_spectra, fitted_params = fit_spectra(y_all_3d, solar_spectra[phobos_unique_order])

            # TODO : check why better fit when binning is multiplied
            y_spectral_mean = np.max(corr_spectra, axis=2) / total_integration_time * binning
            # print(total_integration_time, binning)

            good_angle = angle_2d[:, :]

            # correct nans in phase angle (needs more investigation why geometry seems wrong)
            nan_frames, nan_rows = np.where(np.isnan(phase_angle_2d))
            for frame, row in zip(nan_frames, nan_rows):
                good_rows = np.where(~np.isnan(phase_angle_2d[frame, :]))[0]
                phase_angle_2d[frame, row] = np.mean(phase_angle_2d[frame, good_rows])

            # scale by instrument sensitivity and solar radiance scaler
            y_spectral_mean /= solar_scalars[order]

            if "raw_frames" in PLOT_TYPES:
                fig3, ax3 = plt.subplots()
                im = ax3.imshow(y_spectral_mean.T)
                cb3 = fig3.colorbar(im)
                cb3.set_label("Raw counts", rotation=270, labelpad=10)

            if "angles" in PLOT_TYPES:
                fig6, ax6 = plt.subplots()
                im = ax6.imshow(good_angle.T)
                cb6 = fig6.colorbar(im)
                cb6.set_label(dset_name, rotation=270, labelpad=10)

            # plt.figure()
            # plt.plot(row_stds)
            # plt.scatter(good_ixs, row_stds[good_ixs])

            # find if top or bottom bin is the worst illuminated
            # replace nan by 90 (otherwise counted as 0.0)
            good_angle_90 = good_angle[:, :]
            good_angle_90[np.where(np.isnan(good_angle))] = 90.0
            mean_angle_rows = np.nanmean(good_angle_90[:, :], axis=0)
            mean_y_rows = np.nanmean(y_spectral_mean[:, :], axis=0)

            worst_row_ix = np.argmax(mean_angle_rows)
            best_row_ix = np.argmin(mean_angle_rows)
            worst_row_ix_y = np.argmin(mean_y_rows)
            best_row_ix_y = np.argmax(mean_y_rows)

            print(h5, mean_angle_rows, worst_row_ix, best_row_ix, worst_row_ix_y, best_row_ix_y)

            # plt.figure()
            # # plt.scatter(good_angle[:, best_row_ix], y_good_mean[:, best_row_ix])
            # for i in [2, 3, 1, 4]:
            #     plt.scatter(good_angle[:, i], y_good_mean[:, i], alpha=0.2)
            #     plt.scatter(np.nanmean(good_angle[:, i]), np.mean(y_good_mean[:, i]), c="k")

            # remove worst illuminated row
            # y_good_mean -= y_good_mean[:, worst_row_ix_y][:, np.newaxis]

            # remove mean of top and bottom
            # y_good_mean -= np.mean(y_good_mean[:, [0, -1]], axis=1)[:, np.newaxis]

            # plt.figure()
            # plt.imshow(y_good_mean.T)

            # print(y_good_mean[:, 1])

            if "all_orders" in PLOT_TYPES:
                ax1.scatter(phase_angle_2d[:, best_row_ix_y], y_spectral_mean[:, best_row_ix_y], alpha=0.4, color="C%i" % order_ix)
            if "each_order" in PLOT_TYPES:
                ax2.scatter(phase_angle_2d[:, best_row_ix_y], y_spectral_mean[:, best_row_ix_y], alpha=0.4, label=h5)

            nnan_indices = np.array([i for i, f in enumerate(phase_angle_2d[:, best_row_ix_y]) if not np.isnan(f)], dtype=int)

            all_mean_angles.extend(list(phase_angle_2d[:, best_row_ix_y][nnan_indices]))
            all_mean_ys.extend(list(y_spectral_mean[:, best_row_ix_y][nnan_indices]))

        poly = Polynomial.fit(np.array(all_mean_angles), np.array(all_mean_ys), 1)
        yfit = poly(np.array(all_mean_angles))

        coeffs = poly.convert().coef
        if "all_orders" in PLOT_TYPES:
            ax1.plot(np.array(all_mean_angles), yfit, label=phobos_unique_order, color="C%i" % order_ix)
            ax1.text(10.0, max(yfit)+1, ",".join(["%0.5e" % f for f in poly.convert().coef]))
        if "each_order" in PLOT_TYPES:
            ax2.plot(np.array(all_mean_angles), yfit)
            ax2.text(10.0, max(yfit)+1, ",".join(["%0.5e" % f for f in poly.convert().coef]))
            ax2.set_xlabel("Phase angle (degrees)")
            ax2.set_ylabel("Signal (counts)")
            ax2.legend()
            ax2.grid()
            plt.savefig("lno_phase_angle_correction_order_%i.png" % phobos_unique_order)

        order_d[phobos_unique_order] = {"n_obs": len(h5s), "coeffs": coeffs}

    if "all_orders" in PLOT_TYPES:
        ax1.legend()
        ax1.set_xlabel("Phase angle (degrees)")
        ax1.set_ylabel("Signal scaled by solar spectrum (counts)")
        ax1.grid()
        plt.savefig("lno_phase_angle_correction_all_orders.png")

        # for i in range(1, n_bins-1):
    #     plt.scatter(phase_angle_2d[:, i], y_good_mean[:, i], alpha=0.2)
    #     plt.scatter(np.nanmean(phase_angle_2d[:, i]), np.mean(y_good_mean[:, i]), c="k")

all_coeffs = np.asarray([v["coeffs"] for k, v in order_d.items()])
print(np.mean(all_coeffs[:, 0]), "The gradient is:", np.mean(all_coeffs[:, 1]))
