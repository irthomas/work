# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:33:09 2022

@author: iant

PHOBOS SPICE ILLUMINATION ANGLES

SPLIT FOV INTO MANY LINES AND CALCULATE SOLAR ILLUMINATION ANGLES FOR EACH THAT HITS PHOBOS
"""

import re
import numpy as np
import spiceypy as sp
from matplotlib import pyplot as plt
import itertools

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.file.hdf5_functions import make_filelist
from tools.plotting.anim import make_line_anim
from tools.plotting.colours import get_colours

load_spice_kernels()


ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

NA_VALUE = -999 #value to be used for NaN

# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_PHOBOS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 401
# et2lst: form of longitude supplied by the variable lon
SPICE_LONGITUDE_FORM = "PLANETOCENTRIC"
# spkpos: reference frame relative to which the output position vector
# should be expressed
SPICE_REFERENCE_FRAME = "J2000"
#et2utc: string format flag describing the output time string. 'C' Calendar format, UTC
SPICE_STRING_FORMAT = "C"
# et2utc: number of decimal places of precision to which fractional seconds
# (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed
SPICE_TIME_PRECISION = 3

SPICE_TARGET = "PHOBOS"
SPICE_OBSERVER = "-143"

# SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
# SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"

SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"



#define detector row at centre of FOV
SO_DETECTOR_CENTRE_LINE = 128
LNO_DETECTOR_CENTRE_LINE = 152



def getFovPointParameters(detectorBin):
    """now get detector properties to calculate true FOV, assuming 1 pixel per arcminute IFOV
    so and lno defined by 4 points, UVIS defined by 8 points making octagon
    SO/LNO detector row 1 views further upwards when NOMAD is placed upright"""
    detectorCentreLine = LNO_DETECTOR_CENTRE_LINE
    fovHalfwidth = 2.0 / ARCMINS_TO_RADIANS
    
    detectorOffset = np.float32(detectorBin) - np.float32(detectorCentreLine) #offset in pixels from detector centre
    detectorOffset[1] += 1 #2nd value is 1 pixel more in -ve direction (down detector)
    binFovSize = detectorOffset / ARCMINS_TO_RADIANS #assume 1 arcmin per pixel
    
    #1st value is in vertical direction; 2nd is horizontal
    #centre is mean of top row and bottom row offset from detector centre
    
    vertical_ranges = np.linspace(binFovSize[0], binFovSize[1], num=5)
    horiz_ranges = np.linspace(-1.0*fovHalfwidth, fovHalfwidth, num=9)
    
    fovPoints = [[r[0], r[1]] for r in itertools.product(vertical_ranges, horiz_ranges)]
    
    # fovPoints = [[np.mean(binFovSize), 0.0], \
    #                  [binFovSize[0], fovHalfwidth], \
    #                  [binFovSize[0], -1.0*fovHalfwidth], \
    #                  [binFovSize[1], -1.0*fovHalfwidth], \
    #                  [binFovSize[1], fovHalfwidth]]
    fovCorners = [[point[0], point[1], (np.sqrt(1.0 - point[0]**2.0 - point[1]**2.0))] for point in fovPoints]



    return fovCorners

#Times:
#2022-Mar-04 14:50:33.001 #Phase 34-54; no offset; mid-poor signal
#2022-Mar-26 21:20:35.001 #Phase 44-36; Z offset -4; best signal
#2022-Mar-29 10:15:33.001 #Phase 62-52; Z offset -2; mid signal
#2022-Mar-29 18:07:33.001 #Phase 50-41; Z offset +2; poor signal
#2022-Apr-01 14:54:33.001 #Phase 54-46; Z offset +4; no signal


# regex = re.compile("2022...._.*_LNO_._P")
regex = re.compile("20220304_.*_LNO_._P")
# regex = re.compile("20220326_.*_LNO_._P")
# regex = re.compile("20220329_10.*_LNO_._P")
file_level = "hdf5_level_0p3a"

h5_files, h5_filenames, _ = make_filelist(regex, file_level)

h5 = h5_filenames[0]
h5_f = h5_files[0]

observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
bins = h5_f["Science/Bins"][...]

x = h5_f["Science/X"][...]
y = h5_f["Science/Y"][...]
ydimensions = y.shape
nSpectra = ydimensions[0]

#make non-point dictionary
d = {}
d_tmp = {}

d["et_s"] = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
d["et_e"] = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
d["et"] = np.vstack((d["et_s"], d["et_e"])).T

dref = "TGO_NOMAD_LNO_OPS_NAD"
channelId = sp.bods2c(dref) #find channel id number
[channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)



fovCorners = getFovPointParameters(bins[0, :])
nPoints = len(fovCorners)








d_tmp["obs_subpnt_s"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_s"]]
d_tmp["obs_subpnt_e"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_e"]]

d_tmp["obs_subpnt_xyz_s"] = [obs_subpnt[0] for obs_subpnt in d_tmp["obs_subpnt_s"]]
d_tmp["obs_subpnt_xyz_e"] = [obs_subpnt[0] for obs_subpnt in d_tmp["obs_subpnt_e"]]

d_tmp["obs_reclat_s"] = [sp.reclat(obs_subpnt_xyz) for obs_subpnt_xyz in d_tmp["obs_subpnt_xyz_s"]]
d_tmp["obs_reclat_e"] = [sp.reclat(obs_subpnt_xyz) for obs_subpnt_xyz in d_tmp["obs_subpnt_xyz_e"]]

d_tmp["obs_lon_s"] = [obs_reclat[1] for obs_reclat in d_tmp["obs_reclat_s"]]
d_tmp["obs_lon_e"] = [obs_reclat[1] for obs_reclat in d_tmp["obs_reclat_e"]]

d_tmp["obs_lat_s"] = [obs_reclat[2] for obs_reclat in d_tmp["obs_reclat_s"]]
d_tmp["obs_lat_e"] = [obs_reclat[2] for obs_reclat in d_tmp["obs_reclat_e"]]

d["obs_lon_s"] = np.asfarray(d_tmp["obs_lon_s"]) * SP_DPR
d["obs_lon_e"] = np.asfarray(d_tmp["obs_lon_e"]) * SP_DPR

d["obs_lat_s"] = np.asfarray(d_tmp["obs_lat_s"]) * SP_DPR
d["obs_lat_e"] = np.asfarray(d_tmp["obs_lat_e"]) * SP_DPR

d_tmp["sun_subpnt_s"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"SUN") for et in d["et_s"]]
d_tmp["sun_subpnt_e"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"SUN") for et in d["et_e"]]

d_tmp["sun_subpnt_xyz_s"] = [sun_subpnt[0] for sun_subpnt in d_tmp["sun_subpnt_s"]]
d_tmp["sun_subpnt_xyz_e"] = [sun_subpnt[0] for sun_subpnt in d_tmp["sun_subpnt_e"]]

d_tmp["sun_reclat_s"] = [sp.reclat(sun_subpnt_xyz) for sun_subpnt_xyz in d_tmp["sun_subpnt_xyz_s"]]
d_tmp["sun_reclat_e"] = [sp.reclat(sun_subpnt_xyz) for sun_subpnt_xyz in d_tmp["sun_subpnt_xyz_e"]]

d_tmp["sun_lon_s"] = [sun_reclat[1] for sun_reclat in d_tmp["sun_reclat_s"]]
d_tmp["sun_lon_e"] = [sun_reclat[1] for sun_reclat in d_tmp["sun_reclat_e"]]

d_tmp["sun_lat_s"] = [sun_reclat[2] for sun_reclat in d_tmp["sun_reclat_s"]]
d_tmp["sun_lat_e"] = [sun_reclat[2] for sun_reclat in d_tmp["sun_reclat_e"]]

d["sun_lon_s"] = np.asfarray(d_tmp["sun_lon_s"]) * SP_DPR
d["sun_lon_e"] = np.asfarray(d_tmp["sun_lon_e"]) * SP_DPR

d["sun_lat_s"] = np.asfarray(d_tmp["sun_lat_s"]) * SP_DPR
d["sun_lat_e"] = np.asfarray(d_tmp["sun_lat_e"]) * SP_DPR

# d_tmp["fov_point_pa1r!ams"] = 
d_tmp["fov_corners"] = [getFovPointParameters(np.asfarray(d_bin)) for d_bin in bins]



#make point dictionaries
dp = {}
dp_tmp = {}
for point in range(nPoints):
    dp[point] = {}
    dp_tmp[point] = {}

    dp_tmp[point]["sincpt_s"] = []
    dp_tmp[point]["surf_s"] = []
    dp_tmp[point]["ilumin_s"] = []
    dp[point]["ph_angle_s"] = []
    dp[point]["inc_angle_s"] = []
    dp[point]["em_angle_s"] = []

    #check each fov corner individually to see which hit phobos
    for et, fov_corner in zip(d["et_s"], d_tmp["fov_corners"]):
        try:
            sincpt = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, dref, fov_corner[point])
        except sp.stypes.NotFoundError:
            sincpt = (np.zeros(3) + NA_VALUE, NA_VALUE, np.zeros(3) + NA_VALUE)
        
        if sincpt[1] != NA_VALUE:
            ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, sincpt[0])
            dp[point]["ph_angle_s"].append(ilumin[2] * SP_DPR)
            dp[point]["inc_angle_s"].append(ilumin[3] * SP_DPR)
            dp[point]["em_angle_s"].append(ilumin[4] * SP_DPR)
        else:
            dp[point]["ph_angle_s"].append(NA_VALUE)
            dp[point]["inc_angle_s"].append(NA_VALUE)
            dp[point]["em_angle_s"].append(NA_VALUE)
            

    dp_tmp[point]["sincpt_e"] = []
    dp_tmp[point]["surf_e"] = []
    dp_tmp[point]["ilumin_e"] = []
    dp[point]["ph_angle_e"] = []
    dp[point]["inc_angle_e"] = []
    dp[point]["em_angle_e"] = []

    #check each fov corner individually to see which hit phobos
    for et, fov_corner in zip(d["et_e"], d_tmp["fov_corners"]):
        try:
            sincpt = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, dref, fov_corner[point])
        except sp.stypes.NotFoundError:
            sincpt = (np.zeros(3) + NA_VALUE, NA_VALUE, np.zeros(3) + NA_VALUE)
        
        if sincpt[1] != NA_VALUE:
            ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, sincpt[0])
            dp[point]["ph_angle_e"].append(ilumin[2] * SP_DPR)
            dp[point]["inc_angle_e"].append(ilumin[3] * SP_DPR)
            dp[point]["em_angle_e"].append(ilumin[4] * SP_DPR)
        else:
            dp[point]["ph_angle_e"].append(NA_VALUE)
            dp[point]["inc_angle_e"].append(NA_VALUE)
            dp[point]["em_angle_e"].append(NA_VALUE)

unique_bins = list(set(bins[:, 0]))

#plot incidence angle for all points in 1 bin
plt.figure(figsize=(10, 5), constrained_layout=True)
plt.title("Incidence angle for points within a detector bin FOV")
plt.xlabel("Frame number")
plt.ylabel("Incidence angle of solar illumination (degrees)")

colours = get_colours(nPoints, "brg")

indices = np.where(bins[:, 0] == unique_bins[4])[0]
for point in range(nPoints):

    indices2 = [i for i in indices if dp[point]["inc_angle_s"][i] > -998.]
    x1 = [dp[point]["inc_angle_s"][i] for i in indices2]
    
    plt.scatter(indices2, x1, c=colours[point], alpha=0.25)
plt.grid()
plt.savefig("%s_phobos_incidence_angle_1_bin.png" %h5)




plt.figure(figsize=(5, 3), constrained_layout=True)
plt.title("Distribution of points within the FOV\nfor one bin")
for i, fov_corner in enumerate(fovCorners):
    plt.scatter(fov_corner[1], fov_corner[0], c=colours[i])
plt.savefig("%s_phobos_points_in_1_bin.png" %h5)




plt.figure(figsize=(10, 5), constrained_layout=True)
plt.title("Incidence angle at centre of each detector bin")
plt.xlabel("Frame number")
plt.ylabel("Incidence angle of solar illumination")
for unique_bin in unique_bins:
    indices = np.where(bins[:, 0] == unique_bin)[0]
    indices = [i for i in indices if dp[0]["inc_angle_s"][i] > -998.]
    x1 = [dp[0]["inc_angle_s"][i] for i in indices]
    y1 = np.mean(y[indices, 160:240], axis=1)
    
    plt.scatter(indices, x1, label=unique_bin)
    # plt.scatter(x1, y1, label=unique_bin)
plt.legend()
plt.grid()
plt.savefig("%s_phobos_incidence_angle.png" %h5)

plt.figure(figsize=(10, 5), constrained_layout=True)
plt.title("Emission angle at centre of each detector bin")
plt.xlabel("Frame number")
plt.ylabel("Emission angle of reflected illumination")
for unique_bin in unique_bins:
    indices = np.where(bins[:, 0] == unique_bin)[0]
    indices = [i for i in indices if dp[0]["em_angle_s"][i] > -998.]
    x1 = [dp[0]["em_angle_s"][i] for i in indices]
    y1 = np.mean(y[indices, 160:240], axis=1)
    
    plt.scatter(indices, x1, label=unique_bin)
    # plt.scatter(x1, y1, label=unique_bin)
plt.legend()
plt.grid()
plt.savefig("%s_phobos_emission_angle.png" %h5)

plt.figure()
plt.title("Phase angle at centre of each detector bin")
for unique_bin in unique_bins:
    indices = np.where(bins[:, 0] == unique_bin)[0]
    indices = [i for i in indices if dp[0]["ph_angle_s"][i] > -998.]
    x1 = [dp[0]["ph_angle_s"][i] for i in indices]
    y1 = np.mean(y[indices, 160:240], axis=1)
    
    plt.scatter(indices, x1, label=unique_bin)
    # plt.scatter(x1, y1, label=unique_bin)
plt.legend()



# d = {
#      "x":{"bin4":x[indices]},
#      "y":{"bin4":np.mean(y[indices], axis=1)},
#      "text":["%i" %i for i in indices],
#      "text_position":[min(x[0,:]), 0],
#      "xlabel":"Wavenumbers",
#      "ylabel":"Counts",
#      "xlim":[min(x[0,:]), max(x[0,:])],
#      "ylim":[-3, 15],
#      "filename":"test",
#      "legend":{},

     
#      }

# #'x':{name:data}, 'y':{name:data}, 'text':[], 'text_position', 'xlabel', 'ylabel', 'xlim', 'ylim', 'filename', 'legend':{}, 'keys':[], 'title'
# make_line_anim(d)