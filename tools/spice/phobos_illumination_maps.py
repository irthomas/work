# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:26:31 2026

@author: iant
"""
import itertools
import spiceypy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tools.spice.load_spice_kernels import load_spice_kernels


# get these from phobos_analyse_linescan_v02
# dts = ["2022-11-13 20:41:12.495380", "2022-11-13 20:52:37.048230"]
dts = ["2022-12-22 05:21:09.970870", "2022-12-22 05:38:50.180480"]


# SAVE_FIGURES = False
SAVE_FIGURES = True

PLOT_TYPES = ["groundtrack_illum_start", "groundtrack_illum_end", "groundtrack_illum_combined"]
# PLOT_TYPES = ["groundtrack_illum_combined"]

SP_DPR = sp.dpr()

NA_VALUE = np.nan  # -999 #value to be used for NaN

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


load_spice_kernels()

# calculate illumination on each lat/lon pair and shade map appropriately
lons = np.arange(-180, 180, 1.)
lats = np.arange(-90, 90, 1.)

for dt in dts:
    et = sp.utc2et(dt)

    lonlats = np.array(list(itertools.product(lons / SP_DPR, lats / SP_DPR)))
    surf_xyzs = sp.latsrf(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, lonlats)  # doesn't change with time

    # sum up the start and end to give a combined shaded map
    phobos_illumination = np.zeros((len(lats), len(lons)))
    for lon_ix, lon in enumerate(lons):
        for lat_ix, lat in enumerate(lats):
            index = lon_ix * len(lats) + lat_ix
            ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME,
                               SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, surf_xyzs[index])
            # add instead of assign
            phobos_illumination[len(lats) - 1 - lat_ix, lon_ix] += ilumin[3] * SP_DPR

    sun_subpnt = sp.subpnt(SPICE_INTERCEPT_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, "SUN")
    sun_subpnt_xyz = sun_subpnt[0]
    sun_reclat = sp.reclat(sun_subpnt_xyz)
    sun_lon = sun_reclat[1] * SP_DPR
    sun_lat = sun_reclat[2] * SP_DPR

    cmap = LinearSegmentedColormap.from_list("", colors=["white", "black", "black"], N=256)

    fig2, ax2 = plt.subplots(figsize=(12, 7), constrained_layout=True)
    fig2.suptitle("Phobos solar zenith angles for %s" % dt)
    ax2.grid()
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim((-180, 180))
    ax2.set_ylim((-90, 90))

    im2 = ax2.imshow(phobos_illumination, extent=(-180, 180, -90, 90), alpha=1.0, aspect=1.1, cmap=cmap)
    cbar2 = fig2.colorbar(im2, orientation='vertical')
    cbar2.set_label("Mean start and end solar zenith angle (degrees)", rotation=270, labelpad=15)

    ax2.scatter(sun_lon, sun_lat, marker="x")
    ax2.text(sun_lon + 2, sun_lat, "Sub-solar point\n(%0.1f, %0.1f)" % (sun_lon, sun_lat), color="C0")
    print(dt, sun_lon, sun_lat)

    if SAVE_FIGURES:
        plt.savefig("phobos_groundtrack_solar_zenith_%s.png" % (dt.replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")))

    illum_bins = (90.0 - phobos_illumination.reshape(-1, 5, phobos_illumination.shape[1]).mean(axis=1)) / 90.0
