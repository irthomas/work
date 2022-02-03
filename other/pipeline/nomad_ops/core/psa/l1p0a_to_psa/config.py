# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:58:41 2021

@author: iant

PSA CONFIG

"""
import os.path
import os


from nomad_ops.config import PFM_AUXILIARY_FILES
from nomad_ops.config import PATH_PSA_PAR_UVIS

import platform

windows = platform.system() == "Windows"


### USER MODIFIABLE OPTIONS###

# VALIDATE_OUTPUT = True #use java validator
VALIDATE_OUTPUT = False

VALIDATION_RATIO = 1.0 #1 = validate all files, 0.5 = validate 50%, 0.1 = validation 1 in 10 etc.
#VALIDATION_RATIO = 0.1 #1 = validate all files, 0.5 = validate 50%, 0.1 = validation 1 in 10 etc.

#VALIDATE_WITH_ONLINE_DICTIONARY = True # use ESA online PDS schema
VALIDATE_WITH_ONLINE_DICTIONARY = False # use if NASA or ESA connection problems





"""set versioning variables"""
PSA_MODIFICATION_DATE="2021-11-20"
PSA_VERSION = "3.0" #must start at 1.0 then increment without skipping. Define manually
PSA_VERSION_DESCRIPTION = "Third version following migration to PDS model 1.15"
#PSA_VERSION_DESCRIPTION = "Second version following peer review"
INFORMATION_MODEL_VERSION = "1.15.0.0"

MISSION_PHASE = "Science Phase" #from April 21st 2018 12:00 onwards
MISSION_PHASE_SHORT = "psp" #from April 21st 2018 12:00 onwards











# Should be moved elsewhere. Not used in this level
PIPELINE_VERSION="0.23"
OUTPUT_VERSION = "PSA_CAL" #Not used here



NA_VALUE = -999 #value to be used for NaN
NA_STRING = "N/A" #string to be used for NaN

ASCII_DATE_TIME_YMD_UTC = "%Y-%m-%dT%H:%M:%S.%f"
HDF5_TIME_FORMAT = "%Y %b %d %H:%M:%S.%f"
HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"
PSA_FILENAME_FORMAT = "%Y%m%dT%H%M%S"
PSA_PAR_VERSION_NUMBER = "2.0" #for incoming par xml files




"""set paths to calibration files"""
SO_OCCULTATION_TEMPLATE = os.path.join(PFM_AUXILIARY_FILES,"psa","psa_template_so_occultation_v10.xml")
LNO_NADIR_TEMPLATE = os.path.join(PFM_AUXILIARY_FILES,"psa","psa_template_lno_nadir_v08.xml")
UVIS_OCCULTATION_TEMPLATE = os.path.join(PFM_AUXILIARY_FILES,"psa","psa_template_uvis_occultation_v04.xml")
UVIS_NADIR_TEMPLATE = os.path.join(PFM_AUXILIARY_FILES,"psa","psa_template_uvis_nadir_v03.xml")
#more to be added. Must also add every channel / obs mode combination in code
PSA_DB_PATH = os.path.join(PATH_PSA_PAR_UVIS, "cache.db")


TITLE = "NOMAD Experiment"
