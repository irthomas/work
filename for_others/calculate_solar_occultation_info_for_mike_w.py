# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:26:28 2025

@author: iant
"""


import numpy as np
import spiceypy as sp
import os


# specify base directory and where to find spice kernels
BASE_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Documents", "PROGRAMS", "nomad_obs")

KERNEL_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Documents", "DATA", "local_spice_kernels", "kernels", "mk")
METAKERNEL_NAME = "em16_plan.tm"

# specify start and end times
utc_string_start = "2025-01-01T00:00:00"
utc_string_end = "2025-01-02T00:00:00"


# load spiceypy kernels
print("KERNEL_DIRECTORY=%s, METAKERNEL_NAME=%s" % (KERNEL_DIRECTORY, METAKERNEL_NAME))
os.chdir(KERNEL_DIRECTORY)
sp.furnsh(METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(BASE_DIRECTORY)


# spice constants
SPICE_ABCORR = "None"
SPICE_TARGET = "MARS"
SPICE_METHOD = "Intercept: ellipsoid"
SPICE_FORMATSTR = "C"
SPICE_PRECISION = 0
SPICE_SHAPE = "Ellipsoid"
SPICE_OBSERVER = "-143"
SPICE_REF = "IAU_MARS"
SPICE_MARS_AXES = sp.bodvrd("MARS", "RADII", 3)[1]  # get mars axis values
SPICE_DATETIME_FORMAT = "%Y %b %d %H:%M:%S"


MAXIMUM_SO_ALTITUDE = 250


frontBody = "MARS"
frontShape = "ELLIPSOID"
frontFrame = "IAU_MARS"

backBody = "SUN"
#    backShape="ELLIPSOID"
backShape = "POINT"
backFrame = "IAU_SUN"
stepSize = 1

#    occultationType="ANNULAR"
occultationType = "ANY"


def et2utc(et):
    """function to convert et to utc if float is not -"""
    if et == "-":
        return "-"
    else:
        return sp.et2utc(et, SPICE_FORMATSTR, SPICE_PRECISION)


def getLonLatIncidenceLst(et):
    """get nadir data for a given time"""
    coords = sp.subpnt(SPICE_METHOD, SPICE_TARGET, et, SPICE_REF, SPICE_ABCORR, SPICE_OBSERVER)[0]
    lon, lat = sp.reclat(coords)[1:3] * np.asfarray([sp.dpr(), sp.dpr()])
    lst = sp.et2lst(et, 499, (lon / sp.dpr()), "PLANETOCENTRIC")[3]
    incidence = sp.ilumin(SPICE_SHAPE, SPICE_TARGET, et, SPICE_REF, SPICE_ABCORR, SPICE_OBSERVER, coords)[3] * sp.dpr()
    lst_hours = float(lst[0:2]) + float(lst[3:5])/60.0 + float(lst[6:8])/3600.0
    return lon, lat, incidence, lst_hours


def getTangentAltitude(et):  # returns zero if viewing planet
    """get occultation tangent altitude for a given time"""
    mars2tgoPos = sp.spkpos("-143", et, SPICE_REF, SPICE_ABCORR, "MARS")[0]  # get tgo pos in mars frame
    tgo2sunPos = sp.spkpos("SUN", et, SPICE_REF, SPICE_ABCORR, "-143")[0]  # get sun pos in mars frame

    # calculate tangent point altitude
    tangentAltitude = sp.npedln(SPICE_MARS_AXES[0], SPICE_MARS_AXES[1], SPICE_MARS_AXES[2], mars2tgoPos, tgo2sunPos)[1]
    return tangentAltitude


def findTangentAltitudeTime(desired_altitude, start_time, step_size):
    """find time where tangent altitude matches a given value"""
    calculated_altitude = 0.0
    time = start_time

    while calculated_altitude < desired_altitude:
        time = time + step_size
        calculated_altitude = getTangentAltitude(time)
    return time


def getLonLatLst(et):
    """get occultation data for a given time"""
    mars2tgoPos = sp.spkpos("-143", et, SPICE_REF, SPICE_ABCORR, "MARS")[0]  # get tgo pos in mars frame
    tgo2sunPos = sp.spkpos("SUN", et, SPICE_REF, SPICE_ABCORR, "-143")[0]  # get sun pos in mars frame

    coords = sp.npedln(SPICE_MARS_AXES[0], SPICE_MARS_AXES[1], SPICE_MARS_AXES[2], mars2tgoPos, tgo2sunPos)[0]
    lon, lat = sp.reclat(coords)[1:3] * np.asfarray([sp.dpr(), sp.dpr()])
    lst = sp.et2lst(et, 499, (lon / sp.dpr()), "PLANETOCENTRIC")[3]
    lst_hours = float(lst[0:2]) + float(lst[3:5])/60.0 + float(lst[6:8])/3600.0
    return lon, lat, lst_hours


confinementWindow = sp.stypes.SPICEDOUBLE_CELL(2)
sp.wninsd(sp.utc2et(utc_string_start), sp.utc2et(utc_string_end), confinementWindow)
resultWindow = sp.stypes.SPICEDOUBLE_CELL(1000)
sp.gfoclt(occultationType, frontBody, frontShape, frontFrame, backBody, backShape,
          backFrame, SPICE_ABCORR, SPICE_OBSERVER, stepSize, confinementWindow, resultWindow)

count = sp.wncard(resultWindow)


ingresses = []
egresses = []

for index in range(count):

    # start when the ingress ends
    # end is when the egress starts
    ingress_end, egress_start = sp.wnfetd(resultWindow, index)

    ingress_start_altitude = MAXIMUM_SO_ALTITUDE
    ingress_end_altitude = 0
    egress_end_altitude = MAXIMUM_SO_ALTITUDE
    egress_start_altitude = 0

    ingress_start = findTangentAltitudeTime(ingress_start_altitude, ingress_end, -1.0)

    ingress_start_str = et2utc(ingress_start)
    ingress_end_str = et2utc(ingress_end)
    ingress_duration = ingress_end - ingress_start
    ingress_start_lon, ingress_start_lat, ingress_start_lst = getLonLatLst(ingress_start)
    ingress_end_lon, ingress_end_lat, ingress_end_lst = getLonLatLst(ingress_end)

    egress_end = findTangentAltitudeTime(egress_end_altitude, egress_start, 1.0)
    egress_start_str = et2utc(egress_start)
    egress_end_str = et2utc(egress_end)
    egress_duration = egress_end - egress_start
    egress_start_lon, egress_start_lat, egress_start_lst = getLonLatLst(egress_start)
    egress_end_lon, egress_end_lat, egress_end_lst = getLonLatLst(egress_end)

    ingresses.append({
        "utcStart": ingress_start_str, "utcEnd": ingress_end_str,
        "etStart": ingress_start, "etEnd": ingress_end,
        "lonStart": ingress_start_lon, "lonEnd": ingress_end_lon,
        "latStart": ingress_start_lat, "latEnd": ingress_end_lat,
        "altitudeStart": ingress_start_altitude, "altitudeEnd": ingress_end_altitude,
        "lstStart": ingress_start_lst, "lstEnd": ingress_end_lst,
        "duration": ingress_duration,
    })
    egresses.append({
        "utcStart": egress_start_str, "utcEnd": egress_end_str,
        "etStart": egress_start, "etEnd": egress_end,
        "lonStart": egress_start_lon, "lonEnd": egress_end_lon,
        "latStart": egress_start_lat, "latEnd": egress_end_lat,
        "altitudeStart": egress_start_altitude, "altitudeEnd": egress_end_altitude,
        "lstStart": egress_start_lst, "lstEnd": egress_end_lst,
        "duration": egress_duration
    })

with open("occultations.txt", "w") as f:

    f.write("################ Ingresses ####################\n")
    for ingress in ingresses:
        f.write("######################\n")
        for key, value in ingress.items():
            f.write("%s, %s\n" % (key, value))

    f.write("################ Egresses ####################\n")
    for egress in egresses:
        f.write("######################\n")
        for key, value in egress.items():
            f.write("%s, %s\n" % (key, value))
