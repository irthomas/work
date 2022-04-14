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

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.file.hdf5_functions import make_filelist


load_spice_kernels()


ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_MARS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 499
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

SPICE_TARGET = "MARS"
SPICE_OBSERVER = "-143"



#define detector row at centre of FOV
SO_DETECTOR_CENTRE_LINE = 128
LNO_DETECTOR_CENTRE_LINE = 152



def getFovPointParameters(detectorBin):
    """now get detector properties to calculate true FOV, assuming 1 pixel per arcminute IFOV
    so and lno defined by 4 points, UVIS defined by 8 points making octagon
    SO/LNO detector row 1 views further upwards when NOMAD is placed upright"""
    detectorCentreLine = LNO_DETECTOR_CENTRE_LINE
    fovHalfwidth = 2.0 / ARCMINS_TO_RADIANS

    detectorOffset = detectorBin - detectorCentreLine
    detectorOffset[1] += 1 #actually is 1 pixel more in -ve direction (down detector)
    binFovSize = detectorOffset / ARCMINS_TO_RADIANS #assume 1 arcmin per pixel
    points = [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]] #points are per bin. 0=centre, 1=corner. Not used in calculation!
    fovWeights = [1.0, 0.5, 0.5, 0.5, 0.5] #weights are per bin
    fovPoints = [[np.mean(binFovSize), 0.0], \
                 [binFovSize[0], fovHalfwidth], \
                 [binFovSize[0], -1.0*fovHalfwidth], \
                 [binFovSize[1], -1.0*fovHalfwidth], \
                 [binFovSize[1], fovHalfwidth]]
    fovCorners = [[point[0], point[1], (np.sqrt(1.0 - point[0]**2.0 - point[1]**2.0))] for point in fovPoints]



    return points,fovWeights,fovCorners


regex = re.compile("2022...._.*_LNO_._P")
file_level = "hdf5_level_1p0a"

h5_files, h5_filenames, _ = make_filelist(regex, file_level)

h5 = h5_filenames[0]
h5_f = h5_files[0]

observationDatetimes = h5_f["Geometry/ObservationDateTime"][...]
bins = h5_f["Science/Bins"][...]


d = {}

d["et_s"] = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
d["et_e"] = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
d["et"] = np.vstack((d["et_s"], d["et_e"])).T

dref = "TGO_NOMAD_LNO_OPS_NAD"
channelId = sp.bods2c(dref) #find channel id number
[channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)



points,fovWeights,fovCorners = getFovPointParameters(bins[0])
nPoints = len(points)


#make point dictionaries
dp = {}
dp_tmp = {}
for point in range(nPoints):
    dp[point] = {}
    dp_tmp[point] = {}
