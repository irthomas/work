# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:03:57 2021

@author: iant

GENERIC FUNCTIONS
"""

import logging
import os.path
# import re
#import sys
#import h5py
# import numpy as np
from datetime import datetime

import os
#import re


from nomad_ops.config import MAKE_PSA_LOG_DIR


from nomad_ops.core.psa.l1p0a_to_psa.config import HDF5_TIME_FORMAT, ASCII_DATE_TIME_YMD_UTC
from nomad_ops.core.psa.l1p0a_to_psa.xml import PSA_PAR_LOGICAL_IDENTIFIER_PREFIX

logger = logging.getLogger( __name__ )



def psaErrorLog(lineToWrite):
    dt = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S")
    logPath = os.path.join(MAKE_PSA_LOG_DIR, "psa_cal_error.log")
    with open(logPath, 'a') as logFile:
        logFile.write(dt + "\t" + lineToWrite + '\n')
        

def checkIfFullscan(hdf5_basename):

    hdf5_basename_split = hdf5_basename.split("_")

    if hdf5_basename_split[-1] in ["F", "S"]:
        logger.info("%s is a fullscan observation" %hdf5_basename)
        return True
    else:
        return False
    


def convert_par_filename_to_lid(psa_par_filename):
    """remove .xml and version number from psa file, add partially processed LID"""

    par_filename_no_xml = psa_par_filename.replace(".xml", "")
    par_name = par_filename_no_xml.rsplit("_", 1)[0]
    par_lid = PSA_PAR_LOGICAL_IDENTIFIER_PREFIX + par_name

    return par_lid.lower() #all must be lowercase


def findAttribute(hdf5FileIn, attribute_name):
    """Search input file for a particular attribute name"""
    for key, value in list(hdf5FileIn.attrs.items()):
        if key == attribute_name:
            return value
    logger.error("Error: Attribute %s not found", attribute_name)



def convert_hdf5_to_psa_time(hdf5_datetime_string):
    """convert utc string in hdf5 file to PSA format"""
    
    utc_datetime = datetime.strptime(hdf5_datetime_string, HDF5_TIME_FORMAT)
    utc_datetime_string = datetime.strftime(utc_datetime, ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z"
    
    return utc_datetime_string


