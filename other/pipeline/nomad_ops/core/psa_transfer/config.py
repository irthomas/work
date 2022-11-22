# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:02:28 2021

@author: iant
"""

import os
import sys
import platform
from datetime import datetime


windows = platform.system() == "Windows"




PSA_CAL_VERSION = "3.0"
OLD_CAL_VERSIONS = ["2.0", "1.0"]


#logs to ignore. When a version is old, add the log filenames to this list
#not yet implemented
LOGS_TO_IGNORE = [
    'nmd-pi-delivery.log.20200610',
    'nmd-pi-delivery.log.20200612',
    'nmd-pi-delivery.log.20200616',
    'nmd-pi-delivery.log.20200617',
    'nmd-pi-delivery.log.20200618',
    'nmd-pi-delivery.log.20200620',
    'nmd-pi-delivery.log.20200621',
    'nmd-pi-delivery.log.20200622',
    'nmd-pi-delivery.log.20200623',
    'nmd-pi-delivery.log.20200624',
    'nmd-pi-delivery.log.20200625',
    'nmd-pi-delivery.log.20200626',
    'nmd-pi-delivery.log.20200627',
    'nmd-pi-delivery.log.20200628',
    'nmd-pi-delivery.log.20200827',
    'nmd-pi-delivery.log.20200828',
    'nmd-pi-delivery.log.20200829',
    'nmd-pi-delivery.log.20200830',
    'nmd-pi-delivery.log.20200831',
    'nmd-pi-delivery.log.20200901',
    'nmd-pi-delivery.log.20200902',
    'nmd-pi-delivery.log.20201008',
    'nmd-pi-delivery.log.20210106',
    'nmd-pi-delivery.log.20210107',
    'nmd-pi-delivery.log.20210108',
    'nmd-pi-delivery.log.20210109',
    'nmd-pi-delivery.log.20210115',
    'nmd-pi-delivery.log.20210116',
    'nmd-pi-delivery.log.20210119',
    'nmd-pi-delivery.log.20210120'
 ]


UPLOAD_DATES = [
    [datetime(2020, 6, 1), datetime(2020, 11, 1), "1.0"],
    [datetime(2020, 11, 1), datetime(2021, 1, 20), "2.0"],
    [datetime(2021, 1, 20), datetime(2021, 1, 21), "1.0"],
    [datetime(2021, 1, 21), datetime(2021, 4, 1), "2.0"],
    [datetime(2022, 9, 1), datetime(2030, 1, 1), "3.0"],
    ]


if windows:
    # from tools.file.paths import paths
    # ROOT_DATASTORE_PATH = paths["DATASTORE_ROOT_DIRECTORY"]

    PATH_DB_PSA_CAL_LOG = os.path.normcase(r"C:\Users\iant\Documents\DATA\psa\cal_logs")
    MAKE_PSA_LOG_DIR = os.path.normcase(r"C:\Users\iant\Documents\DATA\psa\logs\psa_cal")
    PSA_FILE_DIR = os.path.normcase(r"C:\Users\iant\Documents\DATA\psa\3.0\data_calibrated")
    LOCAL_UNZIP_TMP_DIR = os.path.normcase(r"C:\Users\iant\Documents\DATA\psa\tmp")
    PATH_PSA_LOG_DB = os.path.normcase(r"C:\Users\iant\Documents\DATA\psa\cal_logs\logs.db")

else:
    os.environ["NMD_OPS_PROFILE"] = "default"
    os.environ["FS_MODEL"] = "False"
    if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
        sys.path.append(".")

    from nomad_ops.config import ROOT_DATASTORE_PATH

    PATH_DB_PSA_CAL_LOG = os.path.join(ROOT_DATASTORE_PATH, "db", "psa", "cal_logs") #where the PSA ingestion logs are moved to
    MAKE_PSA_LOG_DIR = os.path.join(ROOT_DATASTORE_PATH, "logs", "psa_cal")
    PSA_FILE_DIR = os.path.join(ROOT_DATASTORE_PATH, "archive", "psa", PSA_CAL_VERSION, "data_calibrated")
    LOCAL_UNZIP_TMP_DIR = os.path.join(ROOT_DATASTORE_PATH, "tmp")
    PATH_PSA_LOG_DB = os.path.join(PATH_DB_PSA_CAL_LOG, "logs.db")



BIRA_URL = "file:/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/"
ESA_URL = "ssh://exonmd@exoops01.esac.esa.int/home/exonmd/"

BIRA_PSA_CAL_URL = BIRA_URL + "archive/%s/data_calibrated/" %PSA_CAL_VERSION
ESA_PSA_CAL_URL = ESA_URL + "nmd/tmp0/"



LOG_FORMAT_STR = "%Y-%m-%d %H:%M:%S"


N_FILES_PER_ZIP = 200