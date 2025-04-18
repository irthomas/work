# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:48:08 2023

@author: iant

FORWARD MODEL TO FIT TO RAW

"""


# import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime


# from analysis.so_lno_2023.functions.deconvolve_hapi_trans import reduce_resolution
from analysis.so_lno_2023.functions.aotf_blaze_ils import make_ils
# from tools.spectra.hapi_functions import get_abs_coeff, hapi_transmittance
from tools.datasets.get_solar import get_nomad_solar

from tools.spectra.hapi_lut import get_abs_coeff, abs_coeff_pt, hapi_transmittance


logging.basicConfig(filename='fw.log', encoding='utf-8', format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', force=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_px = 159

FIG_SIZE = (15, 8)

OPTIMIZE = False


class forward:

    def __init__(self, raw=False):

        self.raw = raw  # raw signal or transmittance?

    def calibrate(self, cal_d):
        self.cal_d = cal_d

        self.centre_order = cal_d["centre_order"]
        self.orders = list(cal_d["orders"].keys())

        # """need to select an order where absorption lines are present to start with"""
        # if 191 in self.orders:
        #     self.first_order = 191
        # elif 186 in self.orders:
        #     self.first_order = 186
        # else:
        #     print("Error: select a first order")
        self.first_order = self.orders[0]

        # dictionary to keep the correct indices with the orders as the order is changed
        self.order_ixs_d = {order: i for i, order in enumerate(self.orders)}
        # re-order the orders with the one first with absorption liens
        self.ordered_orders = [self.first_order] + [i for i in self.orders if i != self.first_order]

        self.n_px = len(cal_d["orders"][self.centre_order]["px_nus"])
        self.pxs = np.arange(self.n_px)

        self.nu_range = cal_d["aotf"]["aotf_nu_range"]

    def geometry(self, geom_d):
        self.geom_d = geom_d

    def molecules(self, molecule_d):
        self.molecule_d = molecule_d

        self.molecules = ", ".join(list(molecule_d.keys()))

    def forward_so(self, params, plot=[], axes=[]):

        logger.debug("### Starting SO forward model %s###", datetime.now())

        mol_scaler = params["mol_scaler"].value
        logger.debug("mol_scaler = %f", mol_scaler)

        path_lengths_km = self.geom_d["path_lengths_km"]
        alt_grid = self.geom_d["alt_grid"]

        for molecule in self.molecule_d.keys():
            logger.info("molecule = %s", molecule)
            print(molecule, mol_scaler)

            if "hr" in plot:
                if len(axes) == 0:
                    fig1, ax1 = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
                else:
                    ax1 = axes[0]

            isos = self.molecule_d[molecule]["isos"]

            """make HR transmittance spectra for all isotopes and molecules at the given altitudes"""
            hapi_transs = []

            for i in range(len(alt_grid)):
                print("Altitude %0.2fkm, path length %0.2fkm" % (alt_grid[i], path_lengths_km[i]))

                t_raw = self.molecule_d[molecule]["ts"][i]
                # round temperature to nearest 5K
                t = np.round(t_raw / 5.0) * 5.0

                p = self.molecule_d[molecule]["pressures"][i] * self.molecule_d[molecule]["mol_ppmvs"][i] / 1.0e6
                for iso in isos:

                    # print("t:",  molecule_d[molecule]["ts"][i], \
                    # "pressure:",  molecule_d[molecule]["pressures"][i], \
                    # "mol_ppmv:",  molecule_d[molecule]["mol_ppmvs"][i], \
                    # "mol_ppmv_scaled:",  mol_ppmv_scaled, \
                    # "co2_ppmv:",  molecule_d[molecule]["co2_ppmvs"][i])

                    hapi_nus, hapi_abs_coeffs = get_abs_coeff(self.centre_order, molecule, iso, t)

                    if i == 0:
                        logger.debug("%s %i before cat: hapi_nus = %f-%fcm-1", molecule, iso, hapi_nus[0], hapi_nus[-1])

                    # test - expand hapi grid to cover all nus
                    # hapi_nus = np.concatenate((hapi_nus, np.arange(4360.1036, 4600.0, 0.001)))
                    # hapi_abs_coeffs = np.concatenate((hapi_abs_coeffs, np.zeros_like(np.arange(4360.1036, 4600.0, 0.001))))

                    if i == 0:
                        logger.debug("%s %i after cat: hapi_nus = %f-%fcm-1", molecule, iso, hapi_nus[0], hapi_nus[-1])

                    hapi_abs_coeffs_pt = abs_coeff_pt(hapi_abs_coeffs, p, t) * mol_scaler
                    _, hapi_trans = hapi_transmittance(hapi_nus, hapi_abs_coeffs_pt, path_lengths_km[i], t, spec_res=None)

                    if iso == isos[0]:
                        logger.debug("%s %i alt = %f, path_km = %f: hapi_trans min = %f", molecule, iso, alt_grid[i], path_lengths_km[i], np.min(hapi_trans))

                    # reduce spectral resolution
                    # hapi_nus_red, hapi_trans_red = reduce_resolution(hapi_nus, hapi_trans, 0.01)
                    # hapi_nus = hapi_nus_red
                    # hapi_trans = hapi_trans_red

                    # if "hr" in plot:
                    #     plt.plot(hapi_nus, hapi_trans, label="%0.1f km" %alt_grid[i])
                    hapi_transs.append(hapi_trans)

        # multiply transmittances for different molecules and isotopes together to get total atmos trans
        hapi_transs = np.asarray(hapi_transs)
        hapi_trans_total = np.prod(hapi_transs, axis=0)
        logger.debug("hapi_transs[:, log_px] = %s", hapi_transs[:, log_px])
        logger.debug("hapi_trans_total[log_px] = %f", hapi_trans_total[log_px])

        self.hapi_nus = hapi_nus
        self.hapi_trans_total = hapi_trans_total

        if "hr" in plot:
            ax1.plot(hapi_nus, hapi_trans_total, "k")
            for order in self.orders:
                px_nus = self.cal_d["orders"][order]["px_nus"]
                ax1.fill_betweenx([0, 1], px_nus[0], px_nus[-1], alpha=0.3, label="Order %i" % order)
                px_ixs = self.cal_d["orders"][order]["px_ixs"]
                for i in range(0, 320, 50):
                    if i in px_ixs:
                        ix = np.where(i == px_ixs)[0]
                        ax1.axvline(px_nus[ix], color="k", linestyle=":", alpha=0.5)

            ax1.legend(loc="lower left")

        if self.raw:
            # raw solar spectrum
            solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hapi_nus)

        """convolve AOTF function to wavenumber of each pixel in each order"""
        # precompute the blaze_aotf
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        for px in self.pxs:

            # loop through orders, starting with the one preselected to have absorption lines
            for order in self.ordered_orders:

                order_ix = self.order_ixs_d[order]

                blaze = self.cal_d["orders"][order]["F_blaze"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]
                blaze_aotf[order_ix, px] = blaze * aotf

        # ILS convolution
        # sum of ILS for each pixel for T=1
        ils_sum = np.zeros(len(self.pxs))
        ils_sums = np.zeros((len(self.orders), len(self.pxs)))
        # sum of ILS for each pixel including absorptions
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))

        # just for plotting
        if "cont" in plot:
            rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        # loop through pixel
        for px in self.pxs:

            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]

            # loop through orders, starting with the one preselected to have absorption lines
            for order in self.ordered_orders:

                order_ix = self.order_ixs_d[order]

                # pixel centre cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]

                # get bounding indices of hapi grid
                ix_start = np.searchsorted(hapi_nus, px_nu - 0.7)
                ix_end = np.searchsorted(hapi_nus, px_nu + 0.7)

                # TODO: add check to ensure first order is covered by hapi range

                # if raw, no shortcuts available
                if self.raw:
                    hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                    hapi_trans_grid = hapi_trans_total[ix_start:ix_end]

                    ils = make_ils(hapi_grid, width, displacement, amplitude)
                    # summed ils without absorption lines - different for different pixels but v. similar for orders
                    ils_sum[px] = np.sum(ils)
                    ils_sums[order_ix, px] = ils_sum[px]
                    # summed ils with absorption lines - changes with pixel and order
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_grid * solar_hr_rad[ix_start:ix_end])
                    continue

                # first run through, do full calculation
                if order == self.first_order or not OPTIMIZE:

                    logger.debug("Running first order %i", order)

                    hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                    hapi_trans_grid = hapi_trans_total[ix_start:ix_end]

                    ils = make_ils(hapi_grid, width, displacement, amplitude)

                    # summed ils without absorption lines - different for different pixels but v. similar for orders
                    ils_sum[px] = np.sum(ils)
                    ils_sums[order_ix, px] = ils_sum[px]

                    # summed ils with absorption lines - changes with pixel and order
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_grid)

                # other orders, may or may not be covered by hapi range!
                else:

                    # if at the end of the hapi range, if no absorption lines no need to do ils spectrum calculation
                    # if lines present, extend the range and do calculation properly
                    if ix_end == len(hapi_nus):

                        # if completely outside the hapi range, just use the summed ils without lines
                        if ix_start == ix_end:
                            ils_sums[order_ix, px] = ils_sum[px]
                            ils_sums_spectrum[order_ix, px] = ils_sum[px]

                        else:
                            # check if hapi absorptions are present
                            hapi_trans_grid = hapi_trans_total[ix_start:ix_end]

                            logger.info("order %i, px %i (%0.3f): insufficient points for ILS", order, px, px_nu)

                            hapi_trans_grid = hapi_trans_total[ix_start:ix_end]
                            logger.debug(np.min(hapi_trans_grid))

                            # if the lines are negligible or order not covered, just use the summed ils without lines
                            if np.min(hapi_trans_grid) > 0.999:
                                logger.info("order %i, px %i (%0.3f): insufficient points for ILS", order, px, px_nu)
                                ils_sums[order_ix, px] = ils_sum[px]
                                ils_sums_spectrum[order_ix, px] = ils_sum[px]

                            else:
                                # if real absorptions present, extend the ranges
                                extra_nus = np.arange(hapi_nus[ix_end-1]+0.001, hapi_nus[ix_start]+1.4002, 0.001)
                                logger.info("Hapi grid extended from %0.3f to %0.3f", extra_nus[0], extra_nus[-1])
                                hapi_grid = np.concatenate((hapi_nus[ix_start:ix_end], extra_nus)) - px_nu
                                hapi_trans_grid = np.concatenate((hapi_trans_total[ix_start:ix_end], np.ones_like(extra_nus)))

                                ils = make_ils(hapi_grid, width, displacement, amplitude)

                                # summed ils without absorption lines - do calculation if partial coverage of order
                                ils_sums[order_ix, px] = np.sum(ils)
                                # summed ils with absorption lines - changes with pixel and order
                                ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_grid)

                    # if not the first order but within hapi range, do full ils spectrum
                    else:

                        # make ILS function on hapi grid
                        hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                        hapi_trans_grid = hapi_trans_total[ix_start:ix_end]

                        # summed ils with absorption lines - changes with pixel and order
                        ils = make_ils(hapi_grid, width, displacement, amplitude)
                        ils_sums[order_ix, px] = ils_sum[px]
                        ils_sums_spectrum[order_ix, px] = np.sum(ils * hapi_trans_grid)

                if px == log_px:
                    if ix_start == ix_end:
                        hapi_min = 1.0
                    else:
                        hapi_min = np.min(hapi_trans_grid)
                    logger.debug("order %i, px %i: ix_start = %i, ix_end = %i, hapi_grid has %i points, min trans = %0.8f, blaze_aotf = %f, \
                                 ils_sum = %f, ils_sums = %f, ils_sums_spectrum = %f",
                                 order, px, ix_start, ix_end, len(hapi_grid), hapi_min, blaze_aotf[order_ix, px],
                                 ils_sum[px], ils_sums[order_ix, px], ils_sums_spectrum[order_ix, px])

        # expand ils_sums to [n_orders x n_px]
        # ils_sums = np.tile(ils_sum, (len(self.orders), 1))

        # multiply the ils sum for each order and pixel by blaze and aotf
        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf

        logger.debug("ils_sums_spectrum_blaze_aotf[:, log_px] = %s", ils_sums_spectrum_blaze_aotf[:, log_px])

        if self.raw:
            # order addition
            spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)

            # normalise to 1
            spectrum = spectrum_sum / np.max(spectrum_sum)
        else:
            # ils summed without absorption lines
            ils_sums_blaze_aotf = ils_sums * blaze_aotf

            # order addition - sum of all orders with abs lines normalised to sum of ils without abs lines
            spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0) / np.sum(ils_sums_blaze_aotf, axis=0)

            logger.debug("ils_sums_blaze_aotf[:, log_px] = %s", ils_sums_blaze_aotf[:, log_px])
            logger.debug("sum(ils_sums_spectrum_blaze_aotf[log_px]) = %f", np.sum(ils_sums_spectrum_blaze_aotf, axis=0)[log_px])
            logger.debug("sum(ils_sums_blaze_aotf[log_px]) = %f", np.sum(ils_sums_blaze_aotf, axis=0)[log_px])

        logger.debug("spectrum[log_px] = %f", spectrum[log_px])

        if "cont" in plot:
            # plot bar chart for each

            # loop through orders
            for order_ix, order in enumerate(self.orders):
                # 1.0e-9 added to numerator and denominator to avoid divide by zero
                rel_cont[order_ix, :] = 1.0 - (1.0 - ((ils_sums_spectrum[order_ix, :]+1.0e-9) / (ils_sums[order_ix, :]+1.0e-9))
                                               ) * (blaze_aotf[order_ix] / (np.sum(blaze_aotf[:, :], axis=0)))

            rel_1_cont = 1.0 - rel_cont

            if len(axes) < 2:
                fig2, ax2 = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
            else:
                ax2 = axes[1]

            ax2.set_xlabel("Pixel number")
            ax2.set_ylabel("Contribution from each order")
            rel_1_cont_cumul = np.ones(len(self.pxs))  # set to 0 for first bars
            for order_ix, order in enumerate(self.orders):
                rel_1_cont_cumul -= rel_1_cont[order_ix, :]
                # x axis, height of the bar, bottom of the bar
                ax2.bar(self.pxs, rel_1_cont[order_ix, :], bottom=rel_1_cont_cumul, label=order)

            ax2.plot(self.pxs, spectrum, "k-", label="Spectrum")
            ax2.legend()
            ax2.grid()

            self.rel_cont = rel_cont

        return spectrum

    def fit(self, params, y_raw, plot=[]):

        mol_scaler = params["mol_scaler"].value

        # normalise raw SO spectrum
        y_raw /= np.max(y_raw)

        self.spectrum_norm = self.forward_so(params, plot=plot)
        sum_sq = np.sum(np.square(y_raw - self.spectrum_norm))

        if "fit" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber cm-1")
            plt.ylabel("SO transmittance")

            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], y_raw, label="SO raw spectrum")
            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], self.spectrum_norm, label="Simulation")
            plt.grid()
            plt.title("%s: %0.4f %0.4f" % (self.molecules, sum_sq, mol_scaler))
            plt.legend()
            # plt.savefig(("%0.8f" %ssd).replace(".","p")+".png")

        print("sum_sq=", sum_sq)
        return np.square(y_raw - self.spectrum_norm)

    def forward_toa(self, plot=[]):

        hr_nu_grid = self.cal_d["aotf"]["aotf_nus"]

        solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hr_nu_grid)

        # convolve AOTF function to wavenumber of each pixel in each order

        # ILS convolution
        # loop through pixel
        ils_sums = np.zeros((len(self.orders), len(self.pxs)))
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        # if "cont" in plot:
        #     rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        for px in self.pxs:

            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]

            # loop through order
            for order_ix, order in enumerate(self.orders):

                blaze = self.cal_d["orders"][order]["F_blaze"][px]

                # px central cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]

                # get bounding indices of hapi grid
                ix_start = np.searchsorted(hr_nu_grid, px_nu - 0.7)
                ix_end = np.searchsorted(hr_nu_grid, px_nu + 0.7)

                # make ILS function on hapi grid
                hapi_grid = hr_nu_grid[ix_start:ix_end] - px_nu

                ils = make_ils(hapi_grid, width, displacement, amplitude)
                ils_sums[order_ix, px] = np.sum(ils)  # * blaze * aotf

                if self.raw:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_hr_rad[ix_start:ix_end])
                else:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils)

                blaze_aotf[order_ix, px] = blaze * aotf

        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf
        if self.raw:
            spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)
            spectrum = spectrum_sum / np.max(spectrum_sum)

        else:
            ils_sums_blaze_aotf = ils_sums * blaze_aotf
            spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0) / np.sum(ils_sums_blaze_aotf, axis=0)

        if "aotf" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber")
            plt.ylabel("AOTF / solar radiance")
            plt.plot(hr_nu_grid, solar_hr_rad/np.max(solar_hr_rad), label="Solar spectrum")
            plt.plot(hr_nu_grid, self.cal_d["aotf"]["F_aotf"], label="AOTF function")
            plt.legend()
            plt.grid()

        return spectrum


class forward_solar:

    def calibrate(self, cal_d):
        self.cal_d = cal_d

        self.centre_order = cal_d["centre_order"]
        self.orders = list(cal_d["orders"].keys())

        self.first_order = self.orders[0]

        # dictionary to keep the correct indices with the orders as the order is changed
        self.order_ixs_d = {order: i for i, order in enumerate(self.orders)}
        # re-order the orders with the one first with absorption liens
        self.ordered_orders = [self.first_order] + [i for i in self.orders if i != self.first_order]

        self.n_px = len(cal_d["orders"][self.centre_order]["px_nus"])
        self.pxs = np.arange(self.n_px)

        self.nu_range = cal_d["aotf"]["aotf_nu_range"]

    def forward(self, params, plot=[], axes=[]):

        logger.debug("### Starting SO forward model %s###", datetime.now())

        # raw solar spectrum
        hapi_nus = self.cal_d["aotf"]["aotf_nus"]
        solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hapi_nus)

        # plt.figure()
        # plt.plot(solar_hr_nu, solar_hr_rad)

        """convolve AOTF function to wavenumber of each pixel in each order"""
        # precompute the blaze_aotf
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        for px in self.pxs:

            # loop through orders, starting with the one preselected to have absorption lines
            for order in self.ordered_orders:

                order_ix = self.order_ixs_d[order]

                blaze = self.cal_d["orders"][order]["F_blaze"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]
                blaze_aotf[order_ix, px] = blaze * aotf

                if px == log_px:
                    logger.debug("order %i, px %i: blaze = %f, aotf = %f", order, px, blaze, aotf)

        # ILS convolution
        # sum of ILS for each pixel including absorptions
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))

        # just for plotting
        # if "contribution" in plot:
        #     rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        # loop through pixel
        for px in self.pxs:

            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]

            if px == log_px:
                logger.debug("ils width = %f, displacement = %f, amplitude = %f", width, displacement, amplitude)

            # loop through orders, starting with the one preselected to have absorption lines
            for order in self.ordered_orders:

                order_ix = self.order_ixs_d[order]

                # pixel centre cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]

                # get bounding indices of hapi grid
                ix_start = np.searchsorted(hapi_nus, px_nu - 0.7)
                ix_end = np.searchsorted(hapi_nus, px_nu + 0.7)

                # TODO: add check to ensure first order is covered by hapi range

                hapi_grid = hapi_nus[ix_start:ix_end] - px_nu
                # hapi_trans_grid = hapi_trans_total[ix_start:ix_end]

                ils = make_ils(hapi_grid, width, displacement, amplitude)
                # summed ils without absorption lines - different for different pixels but v. similar for orders
                # ils_sum[px] = np.sum(ils)
                # ils_sums[order_ix, px] = ils_sum[px]
                # summed ils with absorption lines - changes with pixel and order
                ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_hr_rad[ix_start:ix_end])

                if px == log_px:
                    # if ix_start == ix_end:
                    #     hapi_min = 1.0
                    # else:
                    hapi_min = np.min(ils_sums_spectrum[order_ix, px])
                    logger.debug("order %i, px %i: px_nu = %f, ix_start = %i, ix_end = %i, grid_start = %f, grid_end = %f, hapi_grid has %i points, min trans = %0.8f, blaze_aotf = %f, \
                                 ils_sums_spectrum = %f",
                                 order, px, px_nu, ix_start, ix_end, hapi_nus[0], hapi_nus[-1], len(hapi_grid), hapi_min, blaze_aotf[order_ix, px],
                                 ils_sums_spectrum[order_ix, px])

        # expand ils_sums to [n_orders x n_px]
        # ils_sums = np.tile(ils_sum, (len(self.orders), 1))

        # multiply the ils sum for each order and pixel by blaze and aotf
        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf

        logger.debug("ils_sums_spectrum_blaze_aotf[:, log_px] = %s", ils_sums_spectrum_blaze_aotf[:, log_px])

        # order addition
        spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)

        # normalise to 1
        spectrum = spectrum_sum / np.max(spectrum_sum)

        logger.debug("sum(ils_sums_spectrum_blaze_aotf[log_px]) = %f", np.sum(ils_sums_spectrum_blaze_aotf, axis=0)[log_px])

        logger.debug("spectrum[log_px] = %f", spectrum[log_px])

        # if "contribution" in plot:
        #     # plot bar chart for each

        #     # loop through orders
        #     for order_ix, order in enumerate(self.orders):
        #         # 1.0e-9 added to numerator and denominator to avoid divide by zero
        #         rel_cont[order_ix, :] = 1.0 - (1.0 - ((ils_sums_spectrum[order_ix, :]+1.0e-9) / (ils_sums[order_ix, :]+1.0e-9))
        #                                        ) * (blaze_aotf[order_ix] / (np.sum(blaze_aotf[:, :], axis=0)))

        #     rel_1_cont = 1.0 - rel_cont

        #     if len(axes) < 2:
        #         fig2, ax2 = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
        #     else:
        #         ax2 = axes[1]

        #     ax2.set_xlabel("Pixel number")
        #     ax2.set_ylabel("Contribution from each order")
        #     rel_1_cont_cumul = np.ones(len(self.pxs))  # set to 0 for first bars
        #     for order_ix, order in enumerate(self.orders):
        #         rel_1_cont_cumul -= rel_1_cont[order_ix, :]
        #         # x axis, height of the bar, bottom of the bar
        #         ax2.bar(self.pxs, rel_1_cont[order_ix, :], bottom=rel_1_cont_cumul, label=order)

        #     ax2.plot(self.pxs, spectrum, "k-", label="Spectrum")
        #     ax2.legend()
        #     ax2.grid()

        #     self.rel_cont = rel_cont

        return spectrum

    def fit(self, params, y_raw, plot=[]):

        mol_scaler = params["mol_scaler"].value

        # normalise raw SO spectrum
        y_raw /= np.max(y_raw)

        self.spectrum_norm = self.forward_so(params, plot=plot)
        sum_sq = np.sum(np.square(y_raw - self.spectrum_norm))

        if "fit" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber cm-1")
            plt.ylabel("SO transmittance")

            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], y_raw, label="SO raw spectrum")
            plt.plot(self.cal_d["orders"][self.centre_order]["px_nus"], self.spectrum_norm, label="Simulation")
            plt.grid()
            plt.title("%s: %0.4f %0.4f" % (self.molecules, sum_sq, mol_scaler))
            plt.legend()
            # plt.savefig(("%0.8f" %ssd).replace(".","p")+".png")

        print("sum_sq=", sum_sq)
        return np.square(y_raw - self.spectrum_norm)

    def forward_toa(self, plot=[]):

        hr_nu_grid = self.cal_d["aotf"]["aotf_nus"]

        solar_hr_nu, solar_hr_rad = get_nomad_solar(self.nu_range, interp_grid=hr_nu_grid)

        # convolve AOTF function to wavenumber of each pixel in each order

        # ILS convolution
        # loop through pixel
        ils_sums = np.zeros((len(self.orders), len(self.pxs)))
        ils_sums_spectrum = np.zeros((len(self.orders), len(self.pxs)))
        blaze_aotf = np.zeros((len(self.orders), len(self.pxs)))

        # if "cont" in plot:
        #     rel_cont = np.zeros((len(self.orders), len(self.pxs)))

        for px in self.pxs:

            width = self.cal_d["ils"]["ils_width"][px]
            displacement = self.cal_d["ils"]["ils_displacement"][px]
            amplitude = self.cal_d["ils"]["ils_amplitude"][px]

            # loop through order
            for order_ix, order in enumerate(self.orders):

                blaze = self.cal_d["orders"][order]["F_blaze"][px]

                # px central cm-1
                px_nu = self.cal_d["orders"][order]["px_nus"][px]
                aotf = self.cal_d["orders"][order]["F_aotf"][px]

                # get bounding indices of hapi grid
                ix_start = np.searchsorted(hr_nu_grid, px_nu - 0.7)
                ix_end = np.searchsorted(hr_nu_grid, px_nu + 0.7)

                # make ILS function on hapi grid
                hapi_grid = hr_nu_grid[ix_start:ix_end] - px_nu

                ils = make_ils(hapi_grid, width, displacement, amplitude)
                ils_sums[order_ix, px] = np.sum(ils)  # * blaze * aotf

                if self.raw:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils * solar_hr_rad[ix_start:ix_end])
                else:
                    ils_sums_spectrum[order_ix, px] = np.sum(ils)

                blaze_aotf[order_ix, px] = blaze * aotf

        ils_sums_spectrum_blaze_aotf = ils_sums_spectrum * blaze_aotf
        if self.raw:
            spectrum_sum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0)
            spectrum = spectrum_sum / np.max(spectrum_sum)

        else:
            ils_sums_blaze_aotf = ils_sums * blaze_aotf
            spectrum = np.sum(ils_sums_spectrum_blaze_aotf, axis=0) / np.sum(ils_sums_blaze_aotf, axis=0)

        if "aotf" in plot:
            plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            plt.xlabel("Wavenumber")
            plt.ylabel("AOTF / solar radiance")
            plt.plot(hr_nu_grid, solar_hr_rad/np.max(solar_hr_rad), label="Solar spectrum")
            plt.plot(hr_nu_grid, self.cal_d["aotf"]["F_aotf"], label="AOTF function")
            plt.legend()
            plt.grid()

        return spectrum
