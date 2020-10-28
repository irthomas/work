# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:11:43 2020

@author: iant
"""

import os
import platform
import numpy as np

if platform.system() == "Windows":
    SYSTEM = "Windows"
else:
    SYSTEM = "Linux"






LNO_FLAGS_DICT = {
    "Y_UNIT_FLAG":4, # 0 = NONE; 1 = RADIANCE FACTOR; 2 = RADIANCE; 3 = BRIGHTNESS TEMPERATURE; 4 = RADIANCE AND RADIANCE FACTOR
    "Y_TYPE_FLAG":5, #0 = NONE; 1 = RADIANCE; 2 = ; 3 = RADIANCE FACTOR; 4 = BRIGHTNESS TEMPERATURE; 5 = RADIANCE AND RADIANCE FACTOR
    "Y_ERROR_FLAG":2, #0 = NONE; 1 = ONE VALUE; 2 = PER PIXEL
#    "Y_UNIT_FLAG":2, # 0 = NONE; 1 = RADIANCE FACTOR; 2 = RADIANCE; 3 = BRIGHTNESS TEMPERATURE; 4 = RADIANCE AND RADIANCE FACTOR
#    "Y_TYPE_FLAG":1, #0 = NONE; 1 = RADIANCE; 2 = ; 3 = RADIANCE FACTOR; 4 = BRIGHTNESS TEMPERATURE; 5 = RADIANCE AND RADIANCE FACTOR
#    "Y_ERROR_FLAG":2, #0 = NONE; 1 = ONE VALUE; 2 = PER PIXEL
    }


SAVE_FILE = True
#SAVE_FILE = False

#RUNNING_MEAN = True
RUNNING_MEAN = False

#REMOVE_NEGATIVES = True
REMOVE_NEGATIVES = False





"""set paths to calibration files"""
if SYSTEM == "Windows":
    from tools.file.paths import paths
    PFM_AUXILIARY_FILES = paths["PFM_AUXILIARY_FILES"]
    NOMAD_TMP_DIR = os.path.join(paths["BASE_DIRECTORY"], "output")
#    ROOT_STORAGE_PATH = os.path.join(paths["BASE_DIRECTORY"], "output")
    
    THUMBNAILS_DESTINATION = os.path.join(paths["BASE_DIRECTORY"], "output")
    trans = [] #don't run on windows
    import other.pipeline.generic_functions as generics

else:
    import matplotlib
    matplotlib.use('Agg')

    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES, THUMBNAILS_DESTINATION
    from nomad_ops.core.hdf5.l0p3a_to_1p0a import l0p3a_to_1p0a_v23_Transmittance as trans

    import nomad_ops.core.hdf5.generic_functions as generics






"""set paths to calibration files"""

#input directories
RADIOMETRIC_CALIBRATION_AUXILIARY_FILES = os.path.join(PFM_AUXILIARY_FILES, "radiometric_calibration")
RADIOMETRIC_CALIBRATION_ORDERS = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, "lno_radiance_factor_order_data")

#coefficient table to make synthetic solar spectrum
LNO_REFLECTANCE_FACTOR_CALIBRATION_TABLE_NAME = "LNO_Radiance_Factor_Calibration_Table_v04"

#radiance look up table
LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME = "LNO_Radiometric_Calibration_Table_v03"




"""set constants"""
NA_VALUE = -999

FIG_X = 17
FIG_Y = 9

#multiply chi sq fit by this value to make it human readable
CHI_SQ_MULTIPLICATION_FACTOR = 1.0

#pixels to consider when fitting nadir and determining std
GOOD_PIXELS = np.arange(50,320)

HDF5_TIME_FORMAT = "%Y %b %d %H:%M:%S.%f"

LNO_HEATER_DB_INDEX = 2

