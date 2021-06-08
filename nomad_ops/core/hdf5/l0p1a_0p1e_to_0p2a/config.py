# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:40:53 2020

@author: iant

CONFIG
"""

import os
import spiceypy as sp
import platform


if platform.system() != "Windows":
    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
else:
    ##for testing
    from tools.file.paths import paths
    PFM_AUXILIARY_FILES = paths["PFM_AUXILIARY_FILES"]
    NOMAD_TMP_DIR = os.path.join(paths["BASE_DIRECTORY"], "tmp")




NA_VALUE = -999 #value to be used for NaN
NA_STRING = "N/A" #string to be used for NaN


ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

# define spacecraft x-direction
OBSERVER_X_AXIS = [1.0,0.0,0.0]



USE_DSK = True
# USE_DSK = False

USE_REDUCED_ELLIPSE = True   # if True, use reduced ellipse for npedln

USE_AREOID = True      # compute areoid and add areoid/topo info to file


if USE_DSK:
    # sincpt: a triaxial ellipsoid to model the surface of the SPICE_TARGET body.
    #SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
    SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
    # ubpnt: sub-SPICE_OBSERVER point is defined as the SPICE_TARGET surface
    # intercept of the line containing the SPICE_OBSERVER and the SPICE_TARGET's center
    SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"
else:
    # sincpt: a triaxial ellipsoid to model the surface of the SPICE_TARGET body.
    SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
    # subpnt: sub-SPICE_OBSERVER point is defined as the SPICE_TARGET surface
    # intercept of the line containing the SPICE_OBSERVER and the SPICE_TARGET's center
    #SPICE_INTERCEPT_METHOD = "Intercept: ellipsoid"
    SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"


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



BORESIGHT_VECTOR_TABLE=os.path.join(PFM_AUXILIARY_FILES,"geometry","boresight_vectors.txt")

AREOID_4PPD = os.path.join(PFM_AUXILIARY_FILES, 'areoid_model', 'mars_areoid_04.h5')


