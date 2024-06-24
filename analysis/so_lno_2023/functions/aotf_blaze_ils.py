# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:41:57 2023

@author: iant

READ IN AOTF, BLAZE AND ILS FUNCTIONS
"""

import os
import numpy as np

from analysis.so_lno_2023.config import INPUT_DIR


def get_aotf_file(channel, aotf_nu_centre=None, filename=""):
    """read in AOTF wavenumbers and shape from file"""

    if filename == "":
        aotf_filename = {
            "so": "aotf_so.tsv",
            "lno": "aotf_lno.tsv",
        }[channel]
        aotf_filepath = os.path.join(INPUT_DIR, aotf_filename)
    else:
        aotf_filepath = os.path.join(INPUT_DIR, filename)

    aotf_nus, F_aotf = np.loadtxt(aotf_filepath, skiprows=1, unpack=True, delimiter="\t")

    # normalise to 1
    F_aotf /= np.max(F_aotf)

    # remove zeros
    F_aotf[F_aotf < 0.0] = 0.0

    # flip around
    # F_aotf = F_aotf[::-1]

    max_ix = int(np.mean(np.where(F_aotf == np.max(F_aotf))[0]))
    aotf_nu_peak = aotf_nus[max_ix]

    if aotf_nu_centre:
        aotf_nus += aotf_nu_centre - aotf_nu_peak
    else:
        aotf_nu_centre = aotf_nu_peak

    aotf_nu_range = [aotf_nus[0], aotf_nus[-1]]

    return {"aotf_nus": aotf_nus, "F_aotf": F_aotf, "aotf_nu_range": aotf_nu_range, "aotf_nu_centre": aotf_nu_centre}


def sinc_gd(dx, width, lobe, asym, offset):
    """new spectral calibration functions Aug/Sep 2021"""

    # goddard version
    sinc = (width * np.sin(np.pi * dx / width)/(np.pi * dx))**2.0
    ind = (abs(dx) > width).nonzero()[0]
    if len(ind) > 0:
        sinc[ind] = sinc[ind] * lobe
    ind = (dx <= -width).nonzero()[0]
    if len(ind) > 0:
        sinc[ind] = sinc[ind] * asym
    sinc += offset
    return sinc


def F_aotf_sinc_gaussian(dx, d):
    offset = d["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotfgw"]**2.0))
    sinc = sinc_gd(dx, d["aotfw"], d["aotfs"], d["aotfa"], offset)
    return sinc


def get_aotf_sinc_gaussian(channel, aotf_nu_centre=None, aotf_d={}):
    """make AOTF from coefficients Sep 2021"""

    dx = np.arange(-100., 150.0, 0.01)
    aotf_nus = dx + aotf_nu_centre
    aotf_nu_range = [aotf_nus[0], aotf_nus[-1]]

    aotf_d = {"aotf_nus": aotf_nus, "aotf_nu_range": aotf_nu_range, "aotf_nu_centre": aotf_nu_centre}

    aotfwc = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01]  # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc = [8.10749274e-07, -3.30238496e-03,  4.08845247e+00]  # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00]  # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc = [0.0,             0.0,             0.0]  # Offset [coefficients for AOTF frequency cm-1]
    aotfgc = [1.49266526e-07, -9.63798656e-04,  1.60097815e+00]  # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

    if "aotfw" not in aotf_d.keys():
        aotf_d["aotfw"] = np.polyval(aotfwc, aotf_nu_centre)

    if "aotfs" not in aotf_d.keys():
        aotf_d["aotfs"] = np.polyval(aotfsc, aotf_nu_centre)

    if "aotfa" not in aotf_d.keys():
        aotf_d["aotfa"] = np.polyval(aotfac, aotf_nu_centre)

    if "aotfo" not in aotf_d.keys():
        aotf_d["aotfo"] = np.polyval(aotfoc, aotf_nu_centre)

    if "aotfg" not in aotf_d.keys():
        aotf_d["aotfg"] = np.polyval(aotfgc, aotf_nu_centre)

    if "aotfgw" not in aotf_d.keys():
        aotf_d["aotfgw"] = 50.  # offset width cm-1

    F_aotf = F_aotf_sinc_gaussian(dx, aotf_d)

    # normalise to 1
    F_aotf /= np.max(F_aotf)

    aotf_d["F_aotf"] = F_aotf

    return aotf_d


def F_aotf_custom(centres, heights, widths, super_gaussian=[], ax=None):
    # new aotf
    # 13 coefficients + optional super gaussian for the central peak

    dx = np.arange(-150., 150.0, 0.01)

    aotf = np.zeros_like(dx)
    for centre, height, width in zip(centres, heights, widths):

        dx_sinc = np.arange(-width, width, 0.01) + 0.00001

        if centre == 0.0 and len(super_gaussian) == 3:
            # overwrite central sinc2 with the super gaussian
            dx_sinc = np.arange(-width*1.75, width*1.75, 0.01) + 0.00001
            sup = super_gaussian[0] * np.exp(- 2.0 * (np.abs(dx_sinc - 0.0) / super_gaussian[1])**super_gaussian[2])
            if ax:
                ax.plot(dx_sinc+centre, sup, alpha=0.5, color="grey", ls="--")
        else:
            sinc2 = (width*np.sin(np.pi * dx_sinc / width)/(np.pi * dx_sinc))**2.0 * height
            if ax:
                ax.plot(dx_sinc + centre, sinc2, alpha=0.5, color="grey", ls="--")

        ixs = np.searchsorted(dx, dx_sinc + centre)

        if centre == 0.0 and len(super_gaussian) == 3:
            aotf[ixs] += sup
        else:
            aotf[ixs] += sinc2

    aotf /= np.max(aotf)  # normalise

    if ax:
        ax.plot(dx, aotf, "k--")

    return dx, aotf


def get_aotf_custom(channel, aotf_nu_centre=None, aotf_d={}):

    # centres = np.asarray([-51., -28.5, 0., 29., 52., 75.])
    # heights = np.asarray([0.18, 0.4, 1.0, 0.45, 0.15, 0.08])
    # widths = np.asarray([17.0, 17.0, 25.0, 17.0, 17.0, 17.0])

    centres = np.asarray([-100., -75., -51., -28.5, 0., 29., 52., 75., 100.])
    heights = np.asarray([0.1, 0.18, 0.4, 1.0, 0.45, 0.15, 0.08, 0.05])
    widths = np.asarray([17.0, 17.0, 17.0, 25.0, 17.0, 17.0, 17.0, 17.0])

    super_gaussian = [1.0, 15.0, 2.8]

    dx, F_aotf = F_aotf_custom(centres, heights, widths, super_gaussian=super_gaussian)

    aotf_nus = dx + aotf_nu_centre
    aotf_nu_range = [aotf_nus[0], aotf_nus[-1]]

    return {"aotf_nus": aotf_nus, "F_aotf": F_aotf, "aotf_nu_range": aotf_nu_range, "aotf_nu_centre": aotf_nu_centre}


"""for testing"""
# import matplotlib.pyplot as plt

# centres = np.asarray([-51., -28.5, 0., 29., 52., 75.])
# heights = np.asarray([0.18, 0.4, 1.0, 0.45, 0.15, 0.08])
# widths = np.asarray([17.0, 17.0, 25.0, 17.0, 17.0, 17.0])

# super_gaussian = [1.0, 15.0, 2.8]

# fig1, ax1 = plt.subplots()

# aotf = F_aotf_custom(centres, heights, widths, super_gaussian=super_gaussian, ax=ax1)


def get_blaze_file(channel):

    blaze_filename = {
        "so": "blaze_so.tsv",
        "lno": "blaze_lno.tsv",
    }[channel]

    blaze_filepath = os.path.join(INPUT_DIR, blaze_filename)
    blaze_px, F_blaze = np.loadtxt(blaze_filepath, skiprows=1, unpack=True, delimiter="\t")

    return {"F_blaze": F_blaze}


def fsr_peak_nu(aotf_nu_centre, nomad_t):
    """SO fsr peak Villanueva 23"""

    dv = aotf_nu_centre - 3700.
    fsr_coeffs = [-1.00162255E-11, -7.20616355E-09, 9.79270239E-06, 2.25863468E+01]

    # FSRpeak = P0 + P1dv + P2dv 2 + P3dv 3, where dv is vAOTF-3700 cm-1
    fsr_peak = np.polyval(fsr_coeffs, dv)

    # where K(T) is a scaling correction factor for temperature T (C)
    # K(T) = K0 + K1T + K2T 2
    K_coeffs = [-2.44383699E-07, -2.30708836E-05, -1.90001923E-04]
    K = np.polyval(K_coeffs, nomad_t)
    fsr_t_corr = fsr_peak * (1. + K)

    return fsr_t_corr


def fsr_peak_nu_lno(aotf_nu_centre, nomad_t):
    """SO fsr peak Villanueva 23"""

    fsr_t_corr = 22.47
    return fsr_t_corr


def F_blaze_sinc(px_nus, blaze_nu, blazew):
    dx = px_nus - blaze_nu

    F = np.sinc((dx) / blazew)**2
    return F


def get_blaze_sinc(channel, px_nus, aotf_nu_centre, order, nomad_t, blazec=-1, blazew=-1):
    """get blaze sinc2 function"""

    if blazew == -1:
        if channel == "so":
            blazew = fsr_peak_nu(aotf_nu_centre, nomad_t)
        elif channel == "lno":
            blazew = fsr_peak_nu_lno(aotf_nu_centre, nomad_t)

    if blazec == -1:
        blazec = order * blazew  # center of the blaze

    # make blaze function, one value per pixel
    blaze = F_blaze_sinc(px_nus, blazec, blazew)

    return blaze


def get_ils(aotf_nu_centre):
    """SO Villanueva 23"""

    pixels = np.arange(320.)

    # from ils.py on 6/7/21
    amp = 0.2724371566666666  # intensity of 2nd gaussian
    rp = 16939.80090831571  # resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03]  # displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number

    A_w_nu0 = aotf_nu_centre / rp
    sconv = A_w_nu0/2.355

    disp_3700_nu = np.polyval(disp_3700, pixels)  # displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * aotf_nu_centre  # displacement adjusted for wavenumber

    return {"ils_width": np.tile(sconv, len(pixels)), "ils_displacement": disp_order, "ils_amplitude": np.tile(amp, len(pixels))}


def get_ils_lno(aotf_nu_centre):
    """LNO - to be recalculated"""

    pixels = np.arange(320.)
    rp = 8500.  # resolving power cm-1/dcm-1

    A_w_nu0 = aotf_nu_centre / rp
    sconv = A_w_nu0/2.355

    return {"ils_width": np.tile(sconv, len(pixels))}


def get_ils_coeffs(channel, aotf_nu_centre):

    if channel == "so":
        return get_ils(aotf_nu_centre)
    elif channel == "lno":
        return get_ils_lno(aotf_nu_centre)


def make_ils(hr_grid, width, displacement, amplitude):

    # make ils shape
    a1 = 0.0
    a2 = width
    a3 = 1.0
    a4 = displacement
    a5 = width
    a6 = amplitude

    ils0 = a3 * np.exp(-0.5 * ((hr_grid + a1) / a2) ** 2)
    ils1 = a6 * np.exp(-0.5 * ((hr_grid + a4) / a5) ** 2)
    ils = ils0 + ils1

    return ils
