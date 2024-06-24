# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:20:03 2023

@author: iant

CONVOLUTION FUNCTIONS

"""

import numpy as np
import matplotlib.pyplot as plt


from analysis.so_lno_2023.functions.aotf_blaze_ils import get_aotf_file
from analysis.so_lno_2023.functions.aotf_blaze_ils import get_aotf_sinc_gaussian
from analysis.so_lno_2023.functions.aotf_blaze_ils import get_aotf_custom

from analysis.so_lno_2023.functions.aotf_blaze_ils import get_blaze_file
from analysis.so_lno_2023.functions.aotf_blaze_ils import get_blaze_sinc


from analysis.so_lno_2023.functions.aotf_blaze_ils import get_ils_coeffs
from analysis.so_lno_2023.functions.spectral_cal import get_orders_nu, aotf_peak_nu, nu0_aotf


def get_aotf(channel, aotf_freq, aotf_t, aotf={"type": "sinc_gauss"}):

    if channel == "so":
        aotf_nu_centre = aotf_peak_nu(aotf_freq, aotf_t)
    elif channel == "lno":
        aotf_nu_centre = nu0_aotf(aotf_freq)

    if aotf["type"] == "sinc_gauss":
        if "aotf_d" not in aotf.keys():
            aotf_d = get_aotf_sinc_gaussian(channel, aotf_nu_centre=aotf_nu_centre, aotf_d={})
        else:
            aotf_d = get_aotf_sinc_gaussian(channel, aotf_nu_centre=aotf_nu_centre, aotf_d=aotf["aotf_d"])

    elif aotf["type"] == "custom":
        if "aotf_d" not in aotf.keys():
            aotf_d = get_aotf_custom(channel, aotf_nu_centre=aotf_nu_centre, aotf_d={})
        else:
            aotf_d = get_aotf_custom(channel, aotf_nu_centre=aotf_nu_centre, aotf_d=aotf["aotf_d"])

    elif aotf["type"] == "file":
        if "filename" not in aotf.keys():
            aotf_d = get_aotf_file(channel, aotf_nu_centre=aotf_nu_centre)
        else:
            aotf_d = get_aotf_file(channel, aotf_nu_centre=aotf_nu_centre, filename=aotf["filename"])

    aotf_d["nu_centre"] = aotf_nu_centre

    return aotf_d


def get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=np.arange(320), blaze={"type": "sinc"}):

    orders_d = get_orders_nu(channel, orders, grat_t, px_ixs=px_ixs)

    if blaze["type"] == "sinc":
        for order in orders:
            px_nus = orders_d[order]["px_nus"]

            blazec = blaze.get("blazec", -1)
            blazew = blaze.get("blazew", -1)
            orders_d[order]["F_blaze"] = get_blaze_sinc(channel, px_nus, aotf_nu_centre, order, grat_t, blazec=blazec, blazew=blazew)

    elif blaze["type"] == "file":
        blaze = get_blaze_file(channel)["F_blaze"][px_ixs]
        for order in orders:
            orders_d[order]["F_blaze"] = blaze

    return orders_d


def get_calibration(channel, centre_order, aotf_d, orders_d, plot=[]):

    cal_d = {}
    cal_d["centre_order"] = centre_order

    orders = list(orders_d.keys())

    aotf_nu_centre = aotf_d["nu_centre"]

    cal_d["aotf"] = aotf_d

    cal_d["ils"] = get_ils_coeffs(channel, aotf_nu_centre)
    cal_d["orders"] = orders_d

    # convolve AOTF function to wavenumber of each pixel in each order
    for order in orders:
        px_nus = cal_d["orders"][order]["px_nus"]
        cal_d["orders"][order]["F_aotf"] = np.interp(px_nus, cal_d["aotf"]["aotf_nus"], cal_d["aotf"]["F_aotf"])

    if "aotf" in plot:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_xlabel("Wavenumber cm-1")
        ax.set_ylabel("AOTF function")
        ax.plot(cal_d["aotf"]["aotf_nus"], cal_d["aotf"]["F_aotf"], color="k")
        text = ["%s=%0.04f\n" % (k, cal_d["aotf"][k]) for k in ["aotfa""aotfg", "aotfgw", "aotfo", "aotfs", "aotfw"] if k in cal_d["aotf"].keys()]
        ax.text(0.1, 0.6, "".join(text), transform=ax.transAxes)
        ax.text(0.1, 0.02, "Central wavenumber=%0.2fcm-1" % aotf_nu_centre, transform=ax.transAxes)
        ax.grid()

    if "aotf_blaze" in plot:
        plt.figure(constrained_layout=True)
        plt.xlabel("Wavenumber cm-1")
        plt.ylabel("Line transmittance / normalised response")
        plt.plot(cal_d["aotf"]["aotf_nus"], cal_d["aotf"]["F_aotf"], color="k")

        for order in cal_d["orders"].keys():
            plt.plot(cal_d["orders"][order]["px_nus"], cal_d["orders"][order]["F_blaze"], label=order)
        plt.legend()
        plt.grid()

    return cal_d
