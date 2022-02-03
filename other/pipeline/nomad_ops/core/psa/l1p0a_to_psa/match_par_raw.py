# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:13:06 2021

@author: iant

GET PAR EXM MATCHING FILENAMES
"""

import logging
# import re
#import sys
#import h5py
# import numpy as np
from datetime import datetime, timedelta

import os

from nomad_ops.core.psa.l1p0a_to_psa.config import HDF5_TIME_FORMAT, PSA_PAR_VERSION_NUMBER, windows
from nomad_ops.core.psa.l1p0a_to_psa.functions import psaErrorLog


if not windows:
    from nomad_ops.core.storage.edds import EDDS_Trees, VERSION
    from nomad_ops.core.storage.incoming_psa import PSA_Trees


logger = logging.getLogger( __name__ )



def get_exm_filename(hdf5_basename, hdf5FileIn):
    """get correct EXM telemetry product filename from database"""
    
    if windows:
        return "EXM_placeholder.exm"
    
    pdhu_fname = hdf5FileIn.attrs["PDHU_filename"]
    edds_spw_dbcon = EDDS_Trees().spacewire._cache_db._conn
    query = "select * from files where path like ? and version = ?"
    res = edds_spw_dbcon.execute(query, ("%%%s%%" % pdhu_fname, VERSION)).fetchall()
    if len(res) > 1:
        exm_fname = os.path.basename(list(sorted(res))[0][0])
        logger.warning("Multiple matching spacewire files found. Using file %s.", exm_fname)
        
    elif len(res) == 0:
        logger.error("No spacewire file found for %s", hdf5_basename)
        psaErrorLog("%s failed: no exm file found" %hdf5_basename)
        
        exm_fname = "None"
    else:
        exm_fname = os.path.basename(res[0][0])
        logger.info("Exm found: %s --> %s", hdf5_basename, exm_fname)

    return exm_fname



def get_psa_par_filename(channel, hdf5_basename, hdf5FileIn):
    """get correct PSA par filename from database"""

    if windows:
        return "PAR_placeholder.xml"
    
    #approximately correct (SO bins may be slightly different)
    hdf5ObsStartTime = hdf5FileIn["Geometry/ObservationDateTime"][0,0].decode()
    hdf5ObsEndTime = hdf5FileIn["Geometry/ObservationDateTime"][-1,0].decode()

    #convert to datetime, add/subtract small offset to avoid rounding/initialisation errors
    obsFirstDatetime = datetime.strptime(hdf5ObsStartTime, HDF5_TIME_FORMAT) #remove small delta errors
    obsLastDatetime = datetime.strptime(hdf5ObsEndTime, HDF5_TIME_FORMAT)

    #check order (ingress are reversed)
    if obsLastDatetime > obsFirstDatetime:
        obsStartDatetime = obsFirstDatetime + timedelta(seconds=20) #remove small delta errors
        obsEndDatetime = obsLastDatetime  - timedelta(seconds=20)
    else:
        obsStartDatetime = obsLastDatetime  + timedelta(seconds=20) #remove small delta errors
        obsEndDatetime = obsFirstDatetime - timedelta(seconds=20)

    
    if channel == "so":
        psa_par_dbcon = PSA_Trees().so._cache_db._conn
    elif channel == "lno":
        psa_par_dbcon = PSA_Trees().lno._cache_db._conn
    elif channel == "uvis":
        psa_par_dbcon = PSA_Trees().uvis._cache_db._conn
    else:
        logger.error("Channel %s is unknown for file %s", channel, hdf5_basename)


    query = """SELECT * FROM files WHERE beg_dtime <= :dtStart AND end_dtime >= :dtEnd"""
        
    queryResult = psa_par_dbcon.execute(query, {"dtStart":obsStartDatetime, "dtEnd":obsEndDatetime}).fetchall()#.fetchone()

    #check for correct version of par filenames
    queryResultVersion = [result[0].split("/")[-1] for result in queryResult if "_%s.xml" %PSA_PAR_VERSION_NUMBER in result[0]]
    if channel == "uvis": #remove TM29s
        queryResultVersion = [parFilename for parFilename in queryResultVersion if "-28-" in parFilename]

#    print(queryResultVersion)

    if len(queryResultVersion) == 0:
        logger.error("Matching PSA filename not found for file %s start time %s", hdf5_basename, hdf5ObsStartTime)
        psaErrorLog("%s failed: no par file found" %hdf5_basename)
        return "None"
    
    elif len(queryResultVersion) == 1:
        parFilename = queryResultVersion[0]
        return parFilename
    
    else: #multiple results found - must select correct one
        logger.error("Multiple matching PSA filenames found for file %s start time %s", hdf5_basename, hdf5ObsStartTime)
        psaErrorLog("%s failed: multiple par files found" %hdf5_basename)
        print(queryResultVersion)
        return "None"


