# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:46:01 2023

@author: iant

STEP 4: PLOT AND ANALYSE MINISCAN AOTF AND BLAZES FROM GENERATED TEXT FILE

COMPARE TO EXISTING AOTF FUNCTIONS

"""


import os
import numpy as np
import matplotlib.pyplot as plt


from tools.plotting.colours import get_colours


# channel = "so"
channel = "lno"

AOTF_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_aotfs.txt" % channel)
BLAZE_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_blazes.txt" % channel)


FIGSIZE = (15, 8)
# FIGSIZE = (9, 5) #for pptx

# plot = ["all", "all_rel_nu", "split_rel_4300", "split_rel_4400", "split_rel_4500"]
# plot = ["split_rel_4500"]
plot = ["all"]


def aotf_func_raw(order_ix, aotf_range=200.0, step_nu=0.1):
    """get sinc gauss aotf function for minimum and maximum expected coefficients in the CO range
    order_ix = 0: minimum
    order_ix = 1: maximum function"""

    def sinc_gd(dx, width, lobe, asym, offset):
        # goddard version
        sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
        ind = (abs(dx) > width).nonzero()[0]
        if len(ind) > 0:
            sinc[ind] = sinc[ind]*lobe
        ind = (dx <= -width).nonzero()[0]
        if len(ind) > 0:
            sinc[ind] = sinc[ind]*asym
        sinc += offset
        return sinc

    def F_aotf3(dx, width, lobe, asym, offset, offset_width):

        offset = offset * np.exp(-dx**2.0 / (2.0 * offset_width**2.0))
        sinc = sinc_gd(dx, width, lobe, asym, offset)
        return sinc

    # #from github
    # aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    # aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
    # aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    # # aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    # aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

    width = [20.2, 19.8][order_ix]
    lobe = [4.8, 5.8][order_ix]
    asym = [1.3, 1.4][order_ix]
    offset = [0.2, 0.35][order_ix]
    offset_width = 50.  # offset width cm-1

    nu = np.arange(-1.0 * aotf_range, aotf_range + step_nu, step_nu)
    F_aotf = F_aotf3(nu, width, lobe, asym, offset, offset_width)

    F_aotf /= np.max(F_aotf)

    return nu, F_aotf


"""read in AOTFs"""
aotfs = {"nu": [], "aotf": []}
with open(AOTF_OUTPUT_PATH, "r") as f:
    lines = f.readlines()

for line in lines:
    line_split = line.split("\t")
    n_nus = int(len(line_split)/2)
    aotfs["nu"].append(np.asfarray(line_split[0:n_nus]))
    aotfs["aotf"].append(np.asfarray(line_split[n_nus:]))

aotfs["peak_ix"] = [np.argmax(aotf) for aotf in aotfs["aotf"]]
aotfs["peak_nu"] = [nu[ix] for ix, nu in zip(aotfs["peak_ix"], aotfs["nu"])]

aotf_min = np.min(aotfs["peak_nu"])
aotf_max = np.max(aotfs["peak_nu"])
aotf_range = np.linspace(np.min(aotfs["peak_nu"]), np.max(aotfs["peak_nu"]), num=20)
aotf_colours = get_colours(len(aotf_range), cmap="brg")


"""plot all aotfs on the same graph"""
if "all" in plot:
    plt.figure()
    plt.title("AOTF functions")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("AOTF transmittance")
    plt.grid()
    loop = 0
    for i, (nu, aotf) in enumerate(zip(aotfs["nu"], aotfs["aotf"])):

        if np.min(aotf) > -0.1:
            peak_nu = aotfs["peak_nu"][i]
            colour = aotf_colours[np.searchsorted(aotf_range, peak_nu)]
            plt.plot(nu, aotf, color=colour, alpha=0.5)
            loop += 1
    plt.axhline(y=0, color="k")
    print(loop)


if "all_rel_nu" in plot:
    plt.figure()
    plt.title("AOTF functions")
    plt.xlabel("Wavenumber from centre (cm-1)")
    plt.ylabel("AOTF transmittance")
    plt.grid()
    for i, (nu, aotf) in enumerate(zip(aotfs["nu"], aotfs["aotf"])):

        if np.min(aotf) > -0.1:

            peak_nu = aotfs["peak_nu"][i]
            colour = aotf_colours[np.searchsorted(aotf_range, peak_nu)]
            aotf_peak_ixs = np.argmax(aotf)
            plt.plot(-(nu - nu[aotf_peak_ixs]), aotf, color=colour, alpha=0.2, label=peak_nu)
    plt.legend()
    plt.axhline(y=0, color="k")

    aotf_23_low_nu, aotf_23_low = aotf_func_raw(0)
    aotf_23_high_nu, aotf_23_high = aotf_func_raw(1)

    plt.plot(-(aotf_23_low_nu - np.mean(aotf_23_low_nu)), aotf_23_low, "k--")
    plt.plot(-(aotf_23_high_nu - np.mean(aotf_23_high_nu)), aotf_23_high, "k--")


"""plot aotfs for 3 different spectral ranges on different figures"""
if "split_rel_4300" in plot:
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    plt.grid()
    plt.title("AOTF functions")
    plt.xlabel("Wavenumber from centre (cm-1)")
    plt.ylabel("AOTF transmittance")

if "split_rel_4400" in plot:
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    plt.grid()
    plt.title("AOTF functions")
    plt.xlabel("Wavenumber from centre (cm-1)")
    plt.ylabel("AOTF transmittance")

if "split_rel_4500" in plot:
    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    plt.grid()
    plt.title("AOTF functions")
    plt.xlabel("Wavenumber from centre (cm-1)")
    plt.ylabel("AOTF transmittance")


aotf_4500_nu = []
aotf_4500 = []
for i, (nu, aotf) in enumerate(zip(aotfs["nu"], aotfs["aotf"])):

    if np.min(aotf) > -0.1:

        peak_nu = aotfs["peak_nu"][i]
        colour = aotf_colours[np.searchsorted(aotf_range, peak_nu)]
        aotf_peak_ixs = np.argmax(aotf)

        if peak_nu < 4350.:
            if "split_rel_4300" in plot:
                ax1.plot(-(nu - nu[aotf_peak_ixs]), aotf, color=colour, alpha=0.2)  # , label=peak_nu)

        elif peak_nu < 4500.:
            if "split_rel_4400" in plot:
                ax2.plot(-(nu - nu[aotf_peak_ixs]), aotf, color=colour, alpha=0.2)  # , label=peak_nu)
        else:
            if "split_rel_4500" in plot:
                ax3.plot(-(nu - nu[aotf_peak_ixs]), aotf, color=colour, alpha=0.2)  # , label=peak_nu)
            aotf_4500_nu.append(-(nu - nu[aotf_peak_ixs]))
            aotf_4500.append(aotf)
# plt.legend()
# plt.axhline(y=0, color="k")


"""plot aotf mean or min/max on the plot"""
# aotf_23_low_nu, aotf_23_low = aotf_func_raw(0)
# aotf_23_high_nu, aotf_23_high = aotf_func_raw(1)

# if "split_rel_4300" in plot:
#     ax1.plot(-(aotf_23_low_nu - np.mean(aotf_23_low_nu)), aotf_23_low, "k--")
# if "split_rel_4400" in plot:
#     ax2.plot(-(aotf_23_low_nu - np.mean(aotf_23_low_nu)), aotf_23_low, "k--")
# if "split_rel_4500" in plot:
#     ax3.plot(-(aotf_23_low_nu - np.mean(aotf_23_low_nu)), aotf_23_low, color="gray", ls="--", label="AOTF minimum")
#     ax3.plot(-(aotf_23_high_nu - np.mean(aotf_23_high_nu)), aotf_23_high, "k--", label="AOTF maximum")
#     ax3.legend()


"""for last figure, get mean AOTF from data - not good, too noisy """
# interpolate
# nu_int_grid = np.arange(-100, 100, 0.01) #wavenumbers
# interps = []
# for nu, aotf in zip(aotf_4500_nu, aotf_4500):

#     interp = np.interp(nu_int_grid, nu[::-1], aotf[::-1])
#     interps.append(interp)

# interps = np.asarray(interps)

# std = np.std(interps, axis=0)
# median = np.median(interps, axis=0)


# if "split_rel_4500" in plot:
#     ax3.plot(nu_int_grid, median)


"""get closest aotf to median - not good, has abrupt ending"""
# sum_sq = []
# for interp in interps:
#     sum_sq.append(np.sum(np.square(interp - median)))

# min_ix = np.argmin(sum_sq)
# closest_aotf = interps[min_ix, :]
# plt.plot(nu_int_grid, closest_aotf)

# #save best fit to file
# np.savetxt("4500um_closest_aotf.txt", np.asarray([nu_int_grid, closest_aotf]).T, fmt="%0.4f", delimiter="\t")


"""instead fit custom AOTF to data"""


def new_aotf(nus, centres, heights, widths, super_gaussian=[], ax=None):
    # new aotf
    # 13 coefficients + optional super gaussian for the central peak

    aotf = np.zeros_like(nus)
    for centre, height, width in zip(centres, heights, widths):

        dx = np.arange(-width, width, 0.01) + 0.00001

        if centre == 0.0 and len(super_gaussian) == 3:
            # overwrite central sinc2 with the super gaussian
            dx = np.arange(-width*1.75, width*1.75, 0.01) + 0.00001
            sup = super_gaussian[0] * np.exp(- 2.0 * (np.abs(dx - 0.0) / super_gaussian[1])**super_gaussian[2])
            if ax:
                ax.plot(dx+centre, sup, alpha=0.5, color="grey", ls="--")
        else:
            sinc2 = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0 * height
            if ax:
                ax.plot(dx+centre, sinc2, alpha=0.5, color="grey", ls="--")

        ixs = np.searchsorted(nus, dx+centre)

        if centre == 0.0 and len(super_gaussian) == 3:
            aotf[ixs] += sup
        else:
            aotf[ixs] += sinc2

    aotf /= np.max(aotf)  # normalise

    if ax:
        ax.plot(nus, aotf, "k--")

    return aotf


if "split_rel_4500" in plot:

    nus = np.arange(-150, 150, 0.01)

    centres = np.asarray([-51., -28.5, 0., 29., 52., 75.])
    heights = np.asarray([0.18, 0.4, 1.0, 0.45, 0.15, 0.08])
    widths = np.asarray([17.0, 17.0, 25.0, 17.0, 17.0, 17.0])

    super_gaussian = [1.0, 15.0, 2.8]

    aotf = new_aotf(nus, centres, heights, widths, super_gaussian=super_gaussian, ax=ax3)


# fig1.savefig("aotf4300.png")
# fig2.savefig("aotf4400.png")
# fig3.savefig("aotf4500.png")


# #read in blazes
# blazes = {"aotf":[], "blaze":[]}
# with open(BLAZE_OUTPUT_PATH, "r") as f:
#     lines = f.readlines()

# for line in lines:
#     line_split = line.split("\t")
#     blazes["aotf"].append(float(line_split[0]))
#     blazes["blaze"].append(np.asfarray(line_split[1:]))

# blaze_min = np.min(blazes["aotf"])
# blaze_max = np.max(blazes["aotf"])
# blaze_range = np.linspace(blaze_min, blaze_max, num=20)
# blaze_colours = get_colours(len(blaze_range), cmap="rainbow")

# mean_blaze = np.mean(np.asfarray(blazes["blaze"]), axis=0)

# plt.figure()
# plt.title("Blaze functions")
# plt.xlabel("Pixel number")
# plt.ylabel("Blaze function")
# plt.grid()
# for aotf_freq, blaze in zip(blazes["aotf"], blazes["blaze"]):
#     colour = blaze_colours[np.searchsorted(blaze_range, aotf_freq)]
#     plt.plot(np.linspace(0., 320., num=len(blaze)), blaze, color=colour, alpha=0.1, label=aotf_freq)
#     plt.plot(np.linspace(0., 320., num=len(blaze)), mean_blaze, color="k", alpha=0.7)

# plt.legend()

# plt.figure()
# colours = get_colours(20)
# for i, px_ix in enumerate(np.linspace(0, len(blaze)-1, num=20)):
#     points = np.asfarray(blazes["blaze"])[:, int(px_ix)]
#     plt.scatter(blazes["aotf"], points / max(points), color=colours[i], label=px_ix)
# plt.legend()
