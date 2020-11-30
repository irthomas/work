# -*- coding: utf-8 -*-


TESTING=True
# TESTING=False


import logging
import os.path

import h5py
import numpy as np
import spiceypy as sp
#import time
import platform

if platform.system() != "Windows":
    import nomad_ops.core.hdf5.generic_functions as generics
    import nomad_ops.core.hdf5.obs_type as obs_type
    from nomad_ops.core.hdf5.l0p1a_to_0p1d.l0p1a_to_0p1d_v23 import write_attrs_from_itl
# else:
    #for testing
    # from tools.file.paths import paths

    # observationType = "I"
    # channel = "so"
    # # hdf5file_path = os.path.join(paths["BASE_DIRECTORY"], "20180421_202111_0p1e_SO_1_E_134.h5")
    # hdf5file_path = os.path.join(paths["BASE_DIRECTORY"], "20180422_092156_0p1e_SO_1_I_191.h5")


    # observationType = "D"
    # channel = "lno"
    # hdf5file_path = os.path.join(paths["BASE_DIRECTORY"], "20180422_003456_0p1e_LNO_1_D_167.h5")


from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import NOMAD_TMP_DIR, NA_VALUE, BORESIGHT_VECTOR_TABLE
from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.areoid import geoid

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import SPICE_INTERCEPT_METHOD, SPICE_OBSERVER, SPICE_TARGET, SPICE_SHAPE_MODEL_METHOD, \
    SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_PLANET_ID, SPICE_LONGITUDE_FORM, SP_DPR, SPICE_PLANET_REFERENCE_FRAME, \
    KILOMETRES_TO_AU, OBSERVER_X_AXIS

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.boresight_functions import getFovPointParameters, getFovName, getObservationMode, \
    readBoresightFile, findBoresightUsed, findSunWobble, writeBoresightQualityFlag
    
# from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.vectorised_functions import getLST, getPosition, getTransMatrix, getFovVector, \
#     getTangentPointsAlt, getSurfaceCoordsLatLons, getSurfaceNormal, getTangentPoint, getTangentPointsAltReduced, getTangentPointsAltSurface, \
#     getTangentPointsAltAreoid, getLosTiltAngles
    
    
from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.units import getUnitMappings, addUnits

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import USE_AREOID, USE_REDUCED_ELLIPSE



__project__   = "NOMAD"
__author__    = "Ian Thomas, Roland Clairquin & Justin Erwin"
__contact__   = "ian.thomas@oma.be"

logger = logging.getLogger( __name__ )
VERSION = 80
OUTPUT_VERSION = "0.2A"

#============================================================================================
# 4. CONVERT HDF5 LEVEL 0.1E TO LEVEL 0.2A
#
# DONE:
#
#
# STILL TO DO:
#
# ADD COMMENT RECORDING SPICE KERNELS USED IN ANALYSIS
# 
#
#============================================================================================


# OBS_TYPES_TO_CONVERT = ["D", "N", "F", "I", "E", "G", "S", "L", "O"]
OBS_TYPES_TO_CONVERT = ["I", "E", "G", "S"]

#t1 = time.clock()


#make non-point dictionary
d = {}
d_tmp = {}


####START CODE#####
def convert(hdf5file_path):
# if True:
    logger.info("convert: %s", hdf5file_path)

    hdf5_basename = os.path.basename(hdf5file_path).split(".")[0]
    #file operations
    hdf5FileIn = h5py.File(hdf5file_path, "r")
    if platform.system() != "Windows":
        channel, channelType = generics.getChannelType(hdf5FileIn)
    # else:
    #     channel = "so"

    #get observation info, timings, dataset size, bins etc. from input file
    channelName = hdf5FileIn.attrs["ChannelName"]
    observationMode = getObservationMode(channelName)
    observationDatetimes = hdf5FileIn["Geometry/ObservationDateTime"][...]
    ydimensions = hdf5FileIn["Science/Y"].shape
    nSpectra = ydimensions[0]

    if channel in ["so","lno"]:
        bins = hdf5FileIn["Science/Bins"][...]
        
        if platform.system() != "Windows":
            observationType = generics.getObservationType(hdf5FileIn)
        # else:
        #     observationType = "I"
    elif channel == "uvis":
        bins = [0]*nSpectra #UVIS doesn't have bins!!
        if platform.system() != "Windows":
            obs_db_res = obs_type.get_obs_type(hdf5file_path)
            if obs_db_res == None:
                observationType = None
            else:
                observationType = obs_db_res[4]

#    logger.info("observationType=%s for %s", observationType, hdf5FileIn.file)
    if observationType is None:
        logger.error("Observation type is not defined for file %s. Update the ITL db.", hdf5_basename)
        return []
        
    if observationMode == "error":
        logger.error("%s: flip mirror position error. Skipping", hdf5_basename)
        # raise RuntimeError("%s: flip mirror position error. Skipping" %hdf5_basename)
        return []


    #check observation type
    obsType_in_DNF = observationType in ["D","N","F"]
    obsType_in_IEGSLO = observationType in ["I","E","G","S","L","O"] #limb measurements are like occultations
    obsType_in_C = observationType in ["C"] #do nothing
    obsType_in_X = observationType in ["X"] #unknown type

    if observationType not in OBS_TYPES_TO_CONVERT:

        """Error"""
        if obsType_in_X:
            logger.error("Error: Observation type unknown for file %s", hdf5_basename)
            # raise RuntimeError("Error: Observation type unknown for file %s" %hdf5_basename)
            return []
            
            """Calibration file"""
        elif obsType_in_C:
            logger.warning("Observation found of type C. No geometric calibration added to file %s except ephemeris time", hdf5_basename)

            if channel=="uvis":
                hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR,"%s_%s.h5" % (hdf5_basename, observationType))
            else:
                hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
        
            #convert datetimes to et
            et_s = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
            et_e = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
            et = np.vstack((et_s, et_e)).T

            with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:
        
                # Copy datasets and attributes
                if platform.system() != "Windows": #for testing locally
                    generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION)
                    for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                        dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                        hdf5FileIn.copy(dset_path, dest)
        
                hdf5FileIn.close()
        
                #write only ephemeris time to calibration files. Rest remains unchanged.
                hdf5FileOut.create_dataset("Geometry/ObservationEphemerisTime", dtype=np.float,
                                        data=et, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                if channel == "uvis":
                    write_attrs_from_itl(hdf5FilepathOut, obs_db_res)

            return [hdf5FilepathOut]

            """if observation type not in list of those to convert, exit"""
        else:
            logger.warning("Observation type %s not in list. Skipping file %s", observationType, hdf5_basename)
            return []

    
    
    #check UVIS. If mode > 2 or acquistion mode = 1, this file cannot be calibrated and should not be created at 0.2A level (except if type C)
    if channel == "uvis":
        mode = hdf5FileIn["Channel/Mode"][0] #1=SO, 2=Nadir. Higher values=Calibration
        acquistionMode = hdf5FileIn["Channel/AcquisitionMode"][0] #0=unbinned, 1=vertical binning, 2=horizontal /square binning
        if not obsType_in_C:
            if mode > 2:
                logger.warning("File %s has mode %i. This file will not be created at 0.2A level", hdf5_basename, mode)
                return []
            if acquistionMode == 1:
                logger.warning("File %s has acquisition mode %i. This file will not be created at 0.2A level", hdf5_basename, acquistionMode)
                return []




        

    #convert datetimes to et
    d["et_s"] = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
    d["et_e"] = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
    d["et"] = np.vstack((d["et_s"], d["et_e"])).T

    # get spice information about name and FOV shape.
    # dvec = FOV centre vector, dvecCorners = FOV corner vectors
    dref = getFovName(channel, observationMode)
    channelId = sp.bods2c(dref) #find channel id number
    [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)



    if obsType_in_DNF or obsType_in_IEGSLO:

        #first get nPoints by running function once
        # logger.info("Science measurement detected. Getting FOV parameters")
        points,fovWeights,fovCorners = getFovPointParameters(channel,bins[0])
        nPoints = len(points)

        
        #make point dictionaries
        dp = {}
        dp_tmp = {}
        for point in range(nPoints):
            dp[point] = {}
            dp_tmp[point] = {}



        #get spacecraft and sun subpoints, convert to lat/lon
        logger.info("Calculating generic geometry")
        
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

        # find obs position/velocity rel to mars in J2000
        d_tmp["obs2mars_spkezr_s"] = [sp.spkezr(SPICE_TARGET, et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER) for et in d["et_s"]]
        d_tmp["obs2mars_spkezr_e"] = [sp.spkezr(SPICE_TARGET, et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER) for et in d["et_e"]]


        # find mars position/velocity rel to sun in J2000
        d_tmp["mars2sun_spkezr_s"] = [sp.spkezr("SUN", et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, "MARS") for et in d["et_s"]]
        d_tmp["mars2sun_spkezr_e"] = [sp.spkezr("SUN", et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, "MARS") for et in d["et_e"]]


        #height of observer above Mars centre
        d["obs_alt_s"] = [sp.vnorm(spkezr[0][0:3]) for spkezr in d_tmp["obs2mars_spkezr_s"]]
        d["obs_alt_e"] = [sp.vnorm(spkezr[0][0:3]) for spkezr in d_tmp["obs2mars_spkezr_e"]]

        # calculate tgo to sun vector in J2000
        d_tmp["obs2sun_spkpos_s"] = [sp.spkpos("SUN", et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER) for et in d["et_s"]]
        d_tmp["obs2sun_spkpos_e"] = [sp.spkpos("SUN", et, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER) for et in d["et_e"]]

        #distance of observer from Sun centre
        d["sun_dist_s"] = [sp.vnorm(spkpos[0]) / KILOMETRES_TO_AU for spkpos in d_tmp["obs2sun_spkpos_s"]]
        d["sun_dist_e"] = [sp.vnorm(spkpos[0]) / KILOMETRES_TO_AU for spkpos in d_tmp["obs2sun_spkpos_e"]]

        #L sub S in degrees
        d["ls_s"] = [sp.lspcn("MARS", et, SPICE_ABERRATION_CORRECTION) * SP_DPR for et in d["et_s"]]
        d["ls_e"] = [sp.lspcn("MARS", et, SPICE_ABERRATION_CORRECTION) * SP_DPR for et in d["et_e"]]

        # find tgo velocity in J2000
        d_tmp["obs_v_ref_s"] = [spkezr[0][3:6] for spkezr in d_tmp["obs2mars_spkezr_s"]]
        d_tmp["obs_v_ref_e"] = [spkezr[0][3:6] for spkezr in d_tmp["obs2mars_spkezr_e"]]
        

        if obsType_in_DNF:
            # Calculate transformation matrix between spacecraft x-direction
            # (i.e. long edge of LNO slit)
            # and J2000 solar system coordinate frame at given time
            d_tmp["obs2ref_trans_s"] = [sp.pxform("TGO_SPACECRAFT", "J2000", et) for et in d["et_s"]]
            d_tmp["obs2ref_trans_e"] = [sp.pxform("TGO_SPACECRAFT", "J2000", et) for et in d["et_e"]]
    
            # transform TGO X axis into j2000
            d_tmp["obs_x_ref_s"] = [np.dot(OBSERVER_X_AXIS, obs2ref_trans) for obs2ref_trans in d_tmp["obs2ref_trans_s"]]
            d_tmp["obs_x_ref_e"] = [np.dot(OBSERVER_X_AXIS, obs2ref_trans) for obs2ref_trans in d_tmp["obs2ref_trans_e"]]
            
            #nadir tilt angle
            #tilt angle for other observations done later
            d["tilt_angle_s"] = [sp.vsep(obs_x_ref, obs_v_ref) * SP_DPR for obs_x_ref, obs_v_ref in zip(d_tmp["obs_x_ref_s"], d_tmp["obs_v_ref_s"])]
            d["tilt_angle_e"] = [sp.vsep(obs_x_ref, obs_v_ref) * SP_DPR for obs_x_ref, obs_v_ref in zip(d_tmp["obs_x_ref_e"], d_tmp["obs_v_ref_e"])]


        # divide by magnitude to get tgo to sun unit vector
        d_tmp["obs2sun_unit_s"] = [spkpos[0] / sp.vnorm(spkpos[0]) for spkpos in d_tmp["obs2sun_spkpos_s"]]
        d_tmp["obs2sun_unit_e"] = [spkpos[0] / sp.vnorm(spkpos[0]) for spkpos in d_tmp["obs2sun_spkpos_e"]]

        # divide by magnitude to get mars to sun unit vector
        d_tmp["mars2sun_unit_s"] = [spkezr[0][0:3] / sp.vnorm(spkezr[0][0:3]) for spkezr in d_tmp["mars2sun_spkezr_s"]]
        d_tmp["mars2sun_unit_e"] = [spkezr[0][0:3] / sp.vnorm(spkezr[0][0:3]) for spkezr in d_tmp["mars2sun_spkezr_e"]]

        
        d_tmp["mars_v_ref_s"] = [spkezr[0][3:6] for spkezr in d_tmp["mars2sun_spkezr_s"]]
        d_tmp["mars_v_ref_e"] = [spkezr[0][3:6] for spkezr in d_tmp["mars2sun_spkezr_e"]]

        #take dot product to find tgo speed towards sun
        d["obs_speed_sun_s"] = [np.dot(obs_v_ref, obs2sun_unit) for obs_v_ref, obs2sun_unit in zip(d_tmp["obs_v_ref_s"], d_tmp["obs2sun_unit_s"])]
        d["obs_speed_sun_e"] = [np.dot(obs_v_ref, obs2sun_unit) for obs_v_ref, obs2sun_unit in zip(d_tmp["obs_v_ref_e"], d_tmp["obs2sun_unit_e"])]

        #take dot product to find mars speed towards sun
        d["mars_speed_sun_s"] = [np.dot(mars_v_ref, mars2sun_unit) for mars_v_ref, mars2sun_unit in zip(d_tmp["mars_v_ref_s"], d_tmp["mars2sun_unit_s"])]
        d["mars_speed_sun_e"] = [np.dot(mars_v_ref, mars2sun_unit) for mars_v_ref, mars2sun_unit in zip(d_tmp["mars_v_ref_e"], d_tmp["mars2sun_unit_e"])]

        d_tmp["fov_point_params"] = [getFovPointParameters(channel, np.asfarray(d_bin)) for d_bin in bins]
        d_tmp["fov_corners"] = [fov_point_params[2] for fov_point_params in d_tmp["fov_point_params"]]

        
        # for key in d.keys():
        #     print(key, d[key][0])
        # for key in d_tmp.keys():
        #     print(key, d_tmp[key][0])


        if obsType_in_DNF:
            #loop through times, storing surface-point independent values
            #initialise empty arrays
            """size is nSpectra x [start,end] x nValues"""
            subObsPoints = np.zeros((nSpectra,2,3)) + NA_VALUE
            subObsCoords = np.zeros((nSpectra,2,2)) + NA_VALUE
            subSolPoints = np.zeros_like(subObsPoints)
            subSolCoords = np.zeros_like(subObsCoords)
    
            ephemerisTimes = np.zeros((nSpectra,2)) + NA_VALUE
            obsAlts = np.zeros_like(ephemerisTimes) + NA_VALUE
            tiltAngles = np.zeros_like(obsAlts) + NA_VALUE
            subObsLons = np.zeros_like(obsAlts) + NA_VALUE
            subObsLats = np.zeros_like(obsAlts) + NA_VALUE
            lSubSs = np.zeros_like(obsAlts) + NA_VALUE
            subSolLons = np.zeros_like(obsAlts) + NA_VALUE
            subSolLats = np.zeros_like(obsAlts) + NA_VALUE
            distToSuns = np.zeros_like(obsAlts) + NA_VALUE
            spdObsSun = np.zeros_like(obsAlts) + NA_VALUE
            spdTargetSun = np.zeros_like(obsAlts) + NA_VALUE
    
    
            for rowIndex in range(nSpectra):
                obsTimeStart = d["et_s"][rowIndex]
                obsTimeEnd = d["et_e"][rowIndex]
                ephemerisTimes[rowIndex,:] = (obsTimeStart,obsTimeEnd)
    
                #get spacecraft and sun subpoints, convert to lat/lon
                logger.debug("Calculating generic geometry for times %i and %i", obsTimeStart,obsTimeEnd)
    
                subObsPoints[rowIndex,:,:] = (sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,obsTimeStart,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0],
                            sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,obsTimeEnd,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0])
    
                subObsCoords[rowIndex,:,:] = (sp.reclat(subObsPoints[rowIndex,0,:])[1:3],sp.reclat(subObsPoints[rowIndex,1,:])[1:3])
                subSolPoints[rowIndex,:,:] = (sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,obsTimeStart,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"SUN")[0],
                            sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,obsTimeEnd,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"SUN")[0])
                subSolCoords[rowIndex,:,:] = (sp.reclat(subSolPoints[rowIndex,0,:])[1:3],sp.reclat(subSolPoints[rowIndex,1,:])[1:3])
    
                subObsLons[rowIndex,:] = (subObsCoords[rowIndex,0,0] * SP_DPR,subObsCoords[rowIndex,1,0] * SP_DPR)
                subObsLats[rowIndex,:] = (subObsCoords[rowIndex,0,1] * SP_DPR,subObsCoords[rowIndex,1,1] * SP_DPR)
                subSolLons[rowIndex,:] = (subSolCoords[rowIndex,0,0] * SP_DPR,subSolCoords[rowIndex,1,0] * SP_DPR)
                subSolLats[rowIndex,:] = (subSolCoords[rowIndex,0,1] * SP_DPR,subSolCoords[rowIndex,1,1] * SP_DPR)
    
                #height of observer above Mars centre
                obsAlts[rowIndex,:] = (sp.vnorm(sp.spkpos(SPICE_TARGET,obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0]),
                        sp.vnorm(sp.spkpos(SPICE_TARGET,obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0]))
    
                #height of observer above Sun centre
                distToSuns[rowIndex,:] = (sp.vnorm(sp.spkpos("SUN",obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0]) / KILOMETRES_TO_AU,
                          sp.vnorm(sp.spkpos("SUN",obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0]) / KILOMETRES_TO_AU)
    
                #L sub S in degrees
                lSubSs[rowIndex,:] = (sp.lspcn("MARS",obsTimeStart,SPICE_ABERRATION_CORRECTION) * SP_DPR,
                      sp.lspcn("MARS",obsTimeEnd,SPICE_ABERRATION_CORRECTION) * SP_DPR)
    
                # Calculate transformation matrix between spacecraft x-direction
                # (i.e. long edge of LNO slit)
                # and J2000 solar system coordinate frame at given time
                obs2SolSysMatrix = (sp.pxform("TGO_SPACECRAFT","J2000",obsTimeStart),sp.pxform("TGO_SPACECRAFT","J2000",obsTimeEnd))
    
                # transform TGO X axis into j2000
                obsInSolSysFrame = (np.dot(OBSERVER_X_AXIS,obs2SolSysMatrix[0]),
                                            np.dot(OBSERVER_X_AXIS,obs2SolSysMatrix[1]))
    
                # find tgo velocity in J2000
                obsVelocityInSolSysFrame = (sp.spkezr(SPICE_TARGET,obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0][3:6],
                                            sp.spkezr(SPICE_TARGET,obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0][3:6])
    
                # calculate relative speeds of tgo and mars w.r.t. sun
                # calculate tgo/mars to sun vector in J2000
                obs2SunVector = (sp.spkpos("SUN",obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0],
                                sp.spkpos("SUN",obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0])
    
                mars2SunVector = (sp.spkpos("SUN",obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"MARS")[0],
                                  sp.spkpos("SUN",obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"MARS")[0])
    
                # divide by magnitude to get sun unit vector
                obs2SunUnitVector = ((obs2SunVector[0] / sp.vnorm(obs2SunVector[0])),
                                      (obs2SunVector[1] / sp.vnorm(obs2SunVector[1])))
    
                mars2SunUnitVector = ((mars2SunVector[0] / sp.vnorm(mars2SunVector[0])),
                                        (mars2SunVector[1] / sp.vnorm(mars2SunVector[1])))
    
                # calculate tgo/mars velocity in J2000 (done above for tgo)
                marsVelocityInSolSysFrame = (sp.spkezr("SUN",obsTimeStart,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"MARS")[0][3:6],
                                              sp.spkezr("SUN",obsTimeEnd,SPICE_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"MARS")[0][3:6])
    
                #take dot product to find tgo/mars velocity towards sun
                spdObsSun[rowIndex,:] = (np.dot(obsVelocityInSolSysFrame[0],obs2SunUnitVector[0]),
                          np.dot(obsVelocityInSolSysFrame[1],obs2SunUnitVector[1]))
    
                spdTargetSun[rowIndex,:] = (np.dot(marsVelocityInSolSysFrame[0],mars2SunUnitVector[0]),
                            np.dot(marsVelocityInSolSysFrame[1],mars2SunUnitVector[1]))
    
    
                tiltAngles[rowIndex,:] = (sp.vsep(obsInSolSysFrame[0],obsVelocityInSolSysFrame[0]) * SP_DPR,
                          sp.vsep(obsInSolSysFrame[1],obsVelocityInSolSysFrame[1]) * SP_DPR)
                #tilt angle for other observations done later





 



        # Loop through array of times for a first time, storing surface intercept points
        # Valid=times when nadir pointed to Mars otherwise nan
        """make empty array nspectra x npoints x [start,end] x nsurface points"""
        surf_points_s = np.zeros((nSpectra,nPoints,3)) + np.nan #use nan here, not -999
        surf_points_e = np.zeros((nSpectra,nPoints,3)) + np.nan #use nan here, not -999
        
        #surface intercepts
        #run in loop - need to catch off-Mars pointing errors
        for j in range(nSpectra):
            et_s = d["et_s"][j]
            et_e = d["et_e"][j]
            for i in range(nPoints):
                fov_corner = d_tmp["fov_corners"][j][i]
                try:
                    sincpt_s = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et_s, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, dref, fov_corner)
                    sincpt_e = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et_e, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, dref, fov_corner)
                except sp.stypes.SpiceyError: #error thrown if lat and lon point not on planet
                    continue
                
                surf_points_s[j, i, :] = sincpt_s[0]
                surf_points_e[j, i, :] = sincpt_e[0]
                

        nans_found = False
        all_nans = True
        
        for i in range(nPoints):
            dp[i]["surf_xyz_s"] = surf_points_s[:, i, :]
            dp[i]["surf_xyz_e"] = surf_points_e[:, i, :]

            dp[i]["point_xy"] = points[i] #save FOV points i.e. [0,0], [1,1], etc.

            #find and store point dataset
            pointXYs = np.zeros((1,nPoints,2)) + NA_VALUE
            for pointIndex,fovCorner in enumerate(fovCorners):
                pointXYs[[0],pointIndex,:] = points[pointIndex]


            
            #check if any nans found
            if np.any(np.isnan(dp[i]["surf_xyz_s"])):
                nans_found = True
            if np.any(np.isnan(dp[i]["surf_xyz_e"])):
                nans_found = True
            
            #check if all nans
            if len(dp[i]["surf_xyz_s"]) != np.sum(np.isnan(dp[i]["surf_xyz_e"])):
                all_nans = False
                
                
                

        #add point xys to dict
        for i in range(nPoints):
            dp[i]["point_xy"] = np.asfarray([list(pointXYs[0][i])]) #must be of size 1 x 2

        """add code to check FOV pointing vs. observation type letter.
        Note that IESMUL can point to planet (ingress, egress) or not (e.g. grazing, limb) making detection difficult"""

        #check if FOV always on planet
        if not nans_found: #if always on planet
            if obsType_in_DNF:
                logger.info("Nadir observation always points to planet")
            if obsType_in_IEGSLO:
                logger.warning("Warning: observation type %s (file %s) always points towards planet", observationType, hdf5_basename)

        if all_nans: #if always off planet
            if obsType_in_DNF:
                logger.warning("Warning: observation type %s (file %s) never points towards planet", observationType, hdf5_basename)
            if obsType_in_IEGSLO:
                logger.info("Observation never points to planet")

        if nans_found and not all_nans: #if mixed (e.g. normal occs)
            if obsType_in_DNF:
                logger.warning("Warning: observation type %s (file %s) only sometimes points towards planet", observationType, hdf5_basename)
            if obsType_in_IEGSLO:
                logger.info("Observation sometimes points to planet")
            






        if obsType_in_DNF:
            logger.info("Adding nadir point geometry for observation of type %s in file %s", observationType, hdf5_basename)
            
            
            fovCornersAll=[]
            for rowIndex in range(nSpectra):
                obsTimeStart = d["et_s"][rowIndex]
                obsTimeEnd = d["et_e"][rowIndex]
    
                detectorBin = np.asfarray(bins[rowIndex])
                points,fovWeights,fovCorners=getFovPointParameters(channel,detectorBin)
                #store fovCorners for all bins in file
                fovCornersAll.append(fovCorners)
    
    
            surfPoints = np.zeros((nSpectra,nPoints,2,3)) + np.nan #use nan here, not -999
            for rowIndex in range(nSpectra):
                for pointIndex,fovCorner in enumerate(fovCorners):
                    try:
                        sincpt = [sp.sincpt(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,d["et_s"][rowIndex],SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,dref,fovCorner),
                                      sp.sincpt(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,d["et_e"][rowIndex],SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,dref,fovCorner)]
                        surfPoints[rowIndex,pointIndex,:,:] = (sincpt[0][0],sincpt[1][0])
                    except sp.stypes.SpiceyError: #error thrown if lat and lon point not on planet
                        continue


            #initialise empty arrays
            surfCoords = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
            surfRadius = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
            areoidRadius = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
            surfTopo = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
            surfLons = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
            surfLats = np.zeros_like(surfLons) + NA_VALUE
            surfLSTs = np.zeros_like(surfLons) + NA_VALUE
            surfLSTHMSs = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
            surfIllumins = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
            surfLOSAngles = np.zeros_like(surfLons) + NA_VALUE
            surfSunSZAs = np.zeros_like(surfLons) + NA_VALUE
        #    surfSunAzis = np.zeros_like(surfLons) + NA_VALUE
            surfIncidenceAngles = np.zeros_like(surfLons) + NA_VALUE
            surfEmissionAngles = np.zeros_like(surfLons) + NA_VALUE
            surfPhaseAngles = np.zeros_like(surfLons) + NA_VALUE
            surfTangentAlts = np.zeros_like(surfLons) + NA_VALUE

            #loop through spectra and points, adding to arrays
            for rowIndex in range(nSpectra):
                obsTimeStart = d["et_s"][rowIndex]
                obsTimeEnd = d["et_e"][rowIndex]

                # Points Datasets
                for pointIndex,fovCorner in enumerate(fovCornersAll[rowIndex]):
                    if not np.isnan(np.min(surfPoints[rowIndex,pointIndex,:,:])):

                        surfCoords[rowIndex,pointIndex,:,:] = (sp.reclat(surfPoints[rowIndex,pointIndex,0,:])[:],
                                  sp.reclat(surfPoints[rowIndex,pointIndex,1,:])[:])
                        surfLons[rowIndex,pointIndex,:] = (surfCoords[rowIndex,pointIndex,0,1] * SP_DPR,
                                surfCoords[rowIndex,pointIndex,1,1] * SP_DPR)
                        surfLats[rowIndex,pointIndex,:] = (surfCoords[rowIndex,pointIndex,0,2] * SP_DPR,
                                surfCoords[rowIndex,pointIndex,1,2] * SP_DPR)
                        #####

                        surfRadius[rowIndex,pointIndex,:] = (surfCoords[rowIndex,pointIndex,0,0],
                                surfCoords[rowIndex,pointIndex,1,0])
                        if USE_AREOID:
                            areoidRadius[rowIndex,pointIndex,:] = (geoid(surfLons[rowIndex,pointIndex,0], surfLats[rowIndex,pointIndex,0]),
                                    geoid(surfLons[rowIndex,pointIndex,1], surfLats[rowIndex,pointIndex,1]))
                            surfTopo[rowIndex,pointIndex,:] = surfRadius[rowIndex,pointIndex,:] - areoidRadius[rowIndex,pointIndex,:]

                        #####
                        surfLSTHMSs[rowIndex,pointIndex,:,:] = (sp.et2lst(obsTimeStart,SPICE_PLANET_ID,surfCoords[rowIndex,pointIndex,0,1],SPICE_LONGITUDE_FORM)[0:3],
                                   sp.et2lst(obsTimeEnd,SPICE_PLANET_ID,surfCoords[rowIndex,pointIndex,1,1],SPICE_LONGITUDE_FORM)[0:3])
                        surfLSTs[rowIndex,pointIndex,:] = ((surfLSTHMSs[rowIndex,pointIndex,0,0] + surfLSTHMSs[rowIndex,pointIndex,0,1]/60.0 + surfLSTHMSs[rowIndex,pointIndex,0,2]/3600.0),
                                (surfLSTHMSs[rowIndex,pointIndex,1,0] + surfLSTHMSs[rowIndex,pointIndex,1,1]/60.0 + surfLSTHMSs[rowIndex,pointIndex,1,2]/3600.0))
                        surfIllumins[rowIndex,pointIndex,:,:] = (sp.ilumin(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeStart,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,surfPoints[rowIndex,pointIndex,0,:])[2:5],
                                    sp.ilumin(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeEnd,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,surfPoints[rowIndex,pointIndex,1,:])[2:5])

                        #line of sight in nadir is just 180 - emission angle
                        surfLOSAngles[rowIndex,pointIndex,:] = ((180.0-surfIllumins[rowIndex,pointIndex,0,2]*SP_DPR),
                                     (180.0-surfIllumins[rowIndex,pointIndex,1,2]*SP_DPR))
                        surfSunSZAs[rowIndex,pointIndex,:] = ((surfIllumins[rowIndex,pointIndex,0,1]*SP_DPR),
                                   (surfIllumins[rowIndex,pointIndex,1,1]*SP_DPR))
            #            surfSunAzis[rowIndex,pointIndex,:,[0]] = ((surfIllumins[rowIndex,pointIndex,0,1]*SP_DPR),
            #                        (surfIllumins[rowIndex,pointIndex,1,3]*SP_DPR))
                        surfIncidenceAngles[rowIndex,pointIndex,:] = ((surfIllumins[rowIndex,pointIndex,0,1]*SP_DPR),
                                           (surfIllumins[rowIndex,pointIndex,1,1]*SP_DPR))
                        surfEmissionAngles[rowIndex,pointIndex,:] = ((surfIllumins[rowIndex,pointIndex,0,2]*SP_DPR),
                                          (surfIllumins[rowIndex,pointIndex,1,2]*SP_DPR))
                        surfPhaseAngles[rowIndex,pointIndex,:] = ((surfIllumins[rowIndex,pointIndex,0,0]*SP_DPR),
                                       (surfIllumins[rowIndex,pointIndex,1,0]*SP_DPR))
                        surfTangentAlts[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                    else:
                        surfLons[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfLats[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfLSTs[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfLOSAngles[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfSunSZAs[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
            #            surfSunAzis[rowIndex,pointIndex,:,[0]] = (NA_VALUE, NA_VALUE)
                        surfIncidenceAngles[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfEmissionAngles[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfPhaseAngles[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)
                        surfTangentAlts[rowIndex,pointIndex,:] = (NA_VALUE, NA_VALUE)





        """test values:
        2017 MAR 01 23:56, limb moves onto planet
        541685199.1416284, 2017 MAR 02 00:05:29.956, 0km edge, 7N, 25E
        moving northwards away from planet in elliptical orbit

        """
        marsAxes = sp.bodvrd("MARS", "RADII", 3)[1]
        if USE_REDUCED_ELLIPSE:
            bodyAxes = marsAxes*(marsAxes[2]-8.)/marsAxes[2] #reduce ellipsoid height by 8km. Anything lower, set to -999.0
        else:
            bodyAxes = marsAxes

        if obsType_in_IEGSLO:


            logger.info("Adding occultation/limb point geometry for observation of type %s in file %s", observationType, hdf5_basename)
    
            #conversion matrix occ channel to mars frame
            d_tmp["so2mars_trans_s"] = [sp.pxform(dref, "IAU_MARS", et) for et in d["et_s"]]
            d_tmp["so2mars_trans_e"] = [sp.pxform(dref, "IAU_MARS", et) for et in d["et_e"]]
            
            #conversion matrix occ channel to ref frame
            d_tmp["so2ref_trans_s"] = [sp.pxform(dref, SPICE_REFERENCE_FRAME, et) for et in d["et_s"]]
            d_tmp["so2ref_trans_e"] = [sp.pxform(dref, SPICE_REFERENCE_FRAME, et) for et in d["et_e"]]
    
            #get TGO in mars reference frame
            d_tmp["obs2mars_mars_spkpos_s"] = [sp.spkpos("-143",et,"IAU_MARS",SPICE_ABERRATION_CORRECTION,"MARS")[0] for et in d["et_s"]]
            d_tmp["obs2mars_mars_spkpos_e"] = [sp.spkpos("-143",et,"IAU_MARS",SPICE_ABERRATION_CORRECTION,"MARS")[0] for et in d["et_e"]]
            
            # for i in [0]:
            for i in range(nPoints):
                #indices of occultation points
                indices_s = [np.any(np.isnan(n)) for n in dp[i]["surf_xyz_s"]]
                indices_e = [np.any(np.isnan(n)) for n in dp[i]["surf_xyz_e"]]
                
                
                
                #fov vectors in SO frame for each spectrum (changes with bin)
                dp_tmp[i]["so_fov_vec"] = [np.asfarray(fov[i]) for fov in d_tmp["fov_corners"]]
                
                #convert fov vector to reference frame
                dp_tmp[i]["fov_vec_ref_s"] = [so2mars_trans.dot(so_fov_vec) for so2mars_trans, so_fov_vec in zip(d_tmp["so2mars_trans_s"], dp_tmp[i]["so_fov_vec"])]
                dp_tmp[i]["fov_vec_ref_e"] = [so2mars_trans.dot(so_fov_vec) for so2mars_trans, so_fov_vec in zip(d_tmp["so2mars_trans_e"], dp_tmp[i]["so_fov_vec"])]
                                   
                #get nearest point on reduced ellipsoid + distance to tangent
                dp_tmp[i]["tan_pt_npedln_s"] = [sp.npedln(bodyAxes[0], bodyAxes[1], bodyAxes[2], spkpos, fov_vec) for spkpos, fov_vec in zip(d_tmp["obs2mars_mars_spkpos_s"], dp_tmp[i]["fov_vec_ref_s"])]
                dp_tmp[i]["tan_pt_npedln_e"] = [sp.npedln(bodyAxes[0], bodyAxes[1], bodyAxes[2], spkpos, fov_vec) for spkpos, fov_vec in zip(d_tmp["obs2mars_mars_spkpos_e"], dp_tmp[i]["fov_vec_ref_e"])]
    
                #nearest point on reduced ellipsoid xyz to tangent
                dp_tmp[i]["tan_pt_rell_xyz_s"] = [tan_pt[0] for tan_pt in dp_tmp[i]["tan_pt_npedln_s"]]
                dp_tmp[i]["tan_pt_rell_xyz_e"] = [tan_pt[0] for tan_pt in dp_tmp[i]["tan_pt_npedln_e"]]
    
                #tangent alt above reduced ellipsoid
                dp_tmp[i]["tan_alt_rell_s"] = [tan_pt[1] for tan_pt in dp_tmp[i]["tan_pt_npedln_s"]]
                dp_tmp[i]["tan_alt_rell_e"] = [tan_pt[1] for tan_pt in dp_tmp[i]["tan_pt_npedln_e"]]
                
                #get dist from reduced ellipsoid to mars centre, lon, lat
                dp_tmp[i]["tan_reclat_s"] = [np.asfarray(sp.reclat(xyz)) for xyz in dp_tmp[i]["tan_pt_rell_xyz_s"]]
                dp_tmp[i]["tan_reclat_e"] = [np.asfarray(sp.reclat(xyz)) for xyz in dp_tmp[i]["tan_pt_rell_xyz_e"]]
                
                #get tangent point lon and lat
                dp[i]["tan_lon_s"] = [reclat[1] * SP_DPR if flag else -999. for flag, reclat in zip(indices_s, dp_tmp[i]["tan_reclat_s"])]
                dp[i]["tan_lon_e"] = [reclat[1] * SP_DPR if flag else -999. for flag, reclat in zip(indices_e, dp_tmp[i]["tan_reclat_e"])]
    
                dp[i]["tan_lat_s"] = [reclat[2] * SP_DPR if flag else -999. for flag, reclat in zip(indices_s, dp_tmp[i]["tan_reclat_s"])]
                dp[i]["tan_lat_e"] = [reclat[2] * SP_DPR if flag else -999. for flag, reclat in zip(indices_e, dp_tmp[i]["tan_reclat_e"])]
                
                #get LST - H, m, s
                dp_tmp[i]["et2lst_s"] = [sp.et2lst(et, SPICE_PLANET_ID, longitude / SP_DPR, SPICE_LONGITUDE_FORM)[0:3] for et, longitude in zip(d["et_s"], dp[i]["tan_lon_s"])]
                dp_tmp[i]["et2lst_e"] = [sp.et2lst(et, SPICE_PLANET_ID, longitude / SP_DPR, SPICE_LONGITUDE_FORM)[0:3] for et, longitude in zip(d["et_e"], dp[i]["tan_lon_e"])]
                
                #get LST in hours
                dp[i]["lst_s"] = [lst[0] + lst[1]/60.0 + lst[2]/3600.0 if flag else -999. for flag, lst in zip(indices_s, dp_tmp[i]["et2lst_s"])]
                dp[i]["lst_e"] = [lst[0] + lst[1]/60.0 + lst[2]/3600.0 if flag else -999. for flag, lst in zip(indices_e, dp_tmp[i]["et2lst_e"])]
                
                
                #get normal vector to surface
                dp_tmp[i]["tan_normal_s"] = [sp.surfnm(bodyAxes[0], bodyAxes[1], bodyAxes[2], tan_pt_xyz) for tan_pt_xyz in dp_tmp[i]["tan_pt_rell_xyz_s"]]
                dp_tmp[i]["tan_normal_e"] = [sp.surfnm(bodyAxes[0], bodyAxes[1], bodyAxes[2], tan_pt_xyz) for tan_pt_xyz in dp_tmp[i]["tan_pt_rell_xyz_e"]]
                
                #get real tangent point in the air xyz from surface point and surface normal
                dp_tmp[i]["tan_pt_xyz_s"] = [tan_pt_rell + tan_normal * tan_alt_rell for tan_pt_rell, tan_normal, tan_alt_rell in zip(dp_tmp[i]["tan_pt_rell_xyz_s"], dp_tmp[i]["tan_normal_s"], dp_tmp[i]["tan_alt_rell_s"])]
                dp_tmp[i]["tan_pt_xyz_e"] = [tan_pt_rell + tan_normal * tan_alt_rell for tan_pt_rell, tan_normal, tan_alt_rell in zip(dp_tmp[i]["tan_pt_rell_xyz_e"], dp_tmp[i]["tan_normal_e"], dp_tmp[i]["tan_alt_rell_e"])]
    
                #get slant path distance
                dp_tmp[i]["slant_path_xyz_s"] = [tan_pt_xyz - obs_mars_xyz for tan_pt_xyz, obs_mars_xyz in zip(dp_tmp[i]["tan_pt_xyz_s"], d_tmp["obs2mars_mars_spkpos_s"])]
                dp_tmp[i]["slant_path_xyz_e"] = [tan_pt_xyz - obs_mars_xyz for tan_pt_xyz, obs_mars_xyz in zip(dp_tmp[i]["tan_pt_xyz_e"], d_tmp["obs2mars_mars_spkpos_e"])]
    
                dp[i]["slant_path_s"] = [sp.vnorm(slant_path_xyz) if flag else -999. for flag, slant_path_xyz in zip(indices_s, dp_tmp[i]["slant_path_xyz_s"])]
                dp[i]["slant_path_e"] = [sp.vnorm(slant_path_xyz) if flag else -999. for flag, slant_path_xyz in zip(indices_e, dp_tmp[i]["slant_path_xyz_e"])]
    
                
                #latsrf maps lat/lon pairs for a single et - need to give 2d array of single lat/lons 
                dp_tmp[i]["tan_pt_ell_xyz_s"] = [sp.latsrf("Ellipsoid", "MARS", et, "IAU_MARS", np.array([reclat[1:]])) if flag else -999. for flag, et, reclat in zip(indices_s, d["et_s"], dp_tmp[i]["tan_reclat_s"])]
                dp_tmp[i]["tan_pt_ell_xyz_e"] = [sp.latsrf("Ellipsoid", "MARS", et, "IAU_MARS", np.array([reclat[1:]])) if flag else -999. for flag, et, reclat in zip(indices_e, d["et_e"], dp_tmp[i]["tan_reclat_e"])]
    
                # #height of ellipsoid above mars centre
                # dp_tmp[i]["tan_pt_ell_dist_s"] = [sp.vnorm(tan_pt_xyz) if flag else -999. for flag, tan_pt_xyz in zip(indices_s, dp_tmp[i]["tan_pt_ell_xyz_s"])]
                # dp_tmp[i]["tan_pt_ell_dist_e"] = [sp.vnorm(tan_pt_xyz) if flag else -999. for flag, tan_pt_xyz in zip(indices_e, dp_tmp[i]["tan_pt_ell_xyz_e"])]
                
                #height of tangent point above real ellipsoid
                dp[i]["tan_alt_ell_s"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(tan_pt_ell_xyz) if flag else -999. for flag, tan_pt_xyz, tan_pt_ell_xyz in zip(indices_s, dp_tmp[i]["tan_pt_xyz_s"], dp_tmp[i]["tan_pt_ell_xyz_s"])]
                dp[i]["tan_alt_ell_e"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(tan_pt_ell_xyz) if flag else -999. for flag, tan_pt_xyz, tan_pt_ell_xyz in zip(indices_e, dp_tmp[i]["tan_pt_xyz_e"], dp_tmp[i]["tan_pt_ell_xyz_e"])]
                
                #get correct xyz of tangent point on surface of DSK
                try:
                    dp_tmp[i]["tan_pt_dsk_xyz_s"] = [sp.latsrf(SPICE_SHAPE_MODEL_METHOD, "MARS", et, "IAU_MARS", np.array([reclat[1:]])) if flag else -999. for flag, et, reclat in zip(indices_s, d["et_s"], dp_tmp[i]["tan_reclat_s"])]
                    dp_tmp[i]["tan_pt_dsk_xyz_e"] = [sp.latsrf(SPICE_SHAPE_MODEL_METHOD, "MARS", et, "IAU_MARS", np.array([reclat[1:]])) if flag else -999. for flag, et, reclat in zip(indices_e, d["et_e"], dp_tmp[i]["tan_reclat_e"])]
                except sp.stypes.SpiceyError:
                    logger.warning("Error in DSK latsrf in file %s", hdf5_basename)
                    
                #height of DSK surface above mars centre
                dp[i]["tan_pt_dsk_dist_s"] = [sp.vnorm(tan_pt_xyz) if flag else -999. for flag, tan_pt_xyz in zip(indices_s, dp_tmp[i]["tan_pt_dsk_xyz_s"])]
                dp[i]["tan_pt_dsk_dist_e"] = [sp.vnorm(tan_pt_xyz) if flag else -999. for flag, tan_pt_xyz in zip(indices_e, dp_tmp[i]["tan_pt_dsk_xyz_e"])]
                
                #height of tangent point above DSK surface
                dp[i]["tan_alt_dsk_s"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(tan_pt_dsk_xyz) if flag else -999. for flag, tan_pt_xyz, tan_pt_dsk_xyz in zip(indices_s, dp_tmp[i]["tan_pt_xyz_s"], dp_tmp[i]["tan_pt_dsk_xyz_s"])]
                dp[i]["tan_alt_dsk_e"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(tan_pt_dsk_xyz) if flag else -999. for flag, tan_pt_xyz, tan_pt_dsk_xyz in zip(indices_e, dp_tmp[i]["tan_pt_xyz_e"], dp_tmp[i]["tan_pt_dsk_xyz_e"])]
                    
                #get areoid dist above mars centre from lon/lat taken from ellipsoid
                dp_tmp[i]["tan_pt_are_dist_s"] = [geoid(lon, lat) if flag else -999. for flag, lon, lat in zip(indices_s, dp[i]["tan_lon_s"], dp[i]["tan_lat_s"])]
                dp_tmp[i]["tan_pt_are_dist_e"] = [geoid(lon, lat) if flag else -999. for flag, lon, lat in zip(indices_e, dp[i]["tan_lon_e"], dp[i]["tan_lat_e"])]
                
                #surface alt areoid
                dp[i]["surf_alt_are_s"] = [dsk_dist - areoid_dist if flag else -999. for flag, dsk_dist, areoid_dist in zip(indices_s, dp[i]["tan_pt_dsk_dist_s"], dp_tmp[i]["tan_pt_are_dist_s"])]
                dp[i]["surf_alt_are_e"] = [dsk_dist - areoid_dist if flag else -999. for flag, dsk_dist, areoid_dist in zip(indices_e, dp[i]["tan_pt_dsk_dist_e"], dp_tmp[i]["tan_pt_are_dist_e"])]
    
                #find xyz of an imaginary ellipsoid at the height of the areoid and lon/lat taken from ellipsoid
                dp_tmp[i]["tan_pt_are_xyz_s"] = [sp.latrec(are_dist, tan_lon, tan_lat) if flag else -999. for flag, are_dist, tan_lon, tan_lat in zip(indices_s, dp_tmp[i]["tan_pt_are_dist_s"], dp[i]["tan_lon_s"], dp[i]["tan_lat_s"])]
                dp_tmp[i]["tan_pt_are_xyz_e"] = [sp.latrec(are_dist, tan_lon, tan_lat) if flag else -999. for flag, are_dist, tan_lon, tan_lat in zip(indices_e, dp_tmp[i]["tan_pt_are_dist_e"], dp[i]["tan_lon_e"], dp[i]["tan_lat_e"])]
    
                #tangent alt of tangent point above areoid
                dp[i]["tan_alt_are_s"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(are_xyz) if flag else -999. for flag, tan_pt_xyz, are_xyz in zip(indices_s, dp_tmp[i]["tan_pt_xyz_s"], dp_tmp[i]["tan_pt_are_xyz_s"])]
                dp[i]["tan_alt_are_e"] = [sp.vnorm(tan_pt_xyz) - sp.vnorm(are_xyz) if flag else -999. for flag, tan_pt_xyz, are_xyz in zip(indices_e, dp_tmp[i]["tan_pt_xyz_e"], dp_tmp[i]["tan_pt_are_xyz_e"])]
    
    
    
                #calculate LOSAngle i.e. the angle between the FOV point and the centre of Mars.
                #next convert FOV from TGO coords to J2000
                dp_tmp[i]["fovSolSysVector_s"] = [so2ref_trans.dot(np.asfarray(so_fov_vec)) if flag else -999. for flag, so2ref_trans, so_fov_vec in zip(indices_s, d_tmp["so2ref_trans_s"], dp_tmp[i]["so_fov_vec"])]
                dp_tmp[i]["fovSolSysVector_e"] = [so2ref_trans.dot(np.asfarray(so_fov_vec)) if flag else -999. for flag, so2ref_trans, so_fov_vec in zip(indices_e, d_tmp["so2ref_trans_e"], dp_tmp[i]["so_fov_vec"])]
    
                #then finally calculate the vector separation in degrees
                dp[i]["tangentSurfLOSAngles_s"] = [sp.vsep(fovSolSysVector, spkezr[0][0:3]) * SP_DPR if flag else -999. for flag, fovSolSysVector, spkezr in zip(indices_s, dp_tmp[i]["fovSolSysVector_s"], d_tmp["obs2mars_spkezr_s"])]
                dp[i]["tangentSurfLOSAngles_e"] = [sp.vsep(fovSolSysVector, spkezr[0][0:3]) * SP_DPR if flag else -999. for flag, fovSolSysVector, spkezr in zip(indices_e, dp_tmp[i]["fovSolSysVector_e"], d_tmp["obs2mars_spkezr_e"])]
    
    
            # #calculate tilt angle of slit
            # #calculate unit vector from fov centre to mars centre
            d_tmp["obs2mars_mars_spkpos_s"] = [sp.spkpos("-143",et,"IAU_MARS",SPICE_ABERRATION_CORRECTION,"MARS")[0] for et in d["et_s"]]

            d_tmp["obs2MarsUnitVector_s"] = [spkpos / sp.vnorm(spkpos) if flag else -999. for flag, spkpos in zip(indices_s, d_tmp["obs2mars_mars_spkpos_s"])]
            d_tmp["obs2MarsUnitVector_e"] = [spkpos / sp.vnorm(spkpos) if flag else -999. for flag, spkpos in zip(indices_e, d_tmp["obs2mars_mars_spkpos_e"])]

            d_tmp["marsCentre2fovCentreUnitVector_s"] = [obs2mars - centre_fov_vec_ref if flag else -999. for flag, obs2mars, centre_fov_vec_ref in zip(indices_s, d_tmp["obs2MarsUnitVector_s"], dp_tmp[0]["fov_vec_ref_s"])]
            d_tmp["marsCentre2fovCentreUnitVector_e"] = [obs2mars - centre_fov_vec_ref if flag else -999. for flag, obs2mars, centre_fov_vec_ref in zip(indices_e, d_tmp["obs2MarsUnitVector_e"], dp_tmp[0]["fov_vec_ref_e"])]

            #calculate unit vector from fov top left to fov bottom left
            d_tmp["fovTopBottomUnitVectors_s"] = [ul_fov_vec_ref - ll_fov_vec_ref if flag else -999. for flag, ul_fov_vec_ref, ll_fov_vec_ref in zip(indices_s, dp_tmp[2]["fov_vec_ref_s"], dp_tmp[3]["fov_vec_ref_s"])]
            d_tmp["fovTopBottomUnitVectors_e"] = [ul_fov_vec_ref - ll_fov_vec_ref if flag else -999. for flag, ul_fov_vec_ref, ll_fov_vec_ref in zip(indices_e, dp_tmp[2]["fov_vec_ref_e"], dp_tmp[3]["fov_vec_ref_e"])]
            
            d["tilt_angle_s"] = [sp.vsep(ul2ll_fov_vec_ref, mars2centre_fov_vec_ref) * SP_DPR if flag else -999. for flag, ul2ll_fov_vec_ref, mars2centre_fov_vec_ref in zip(indices_s, d_tmp["fovTopBottomUnitVectors_s"], d_tmp["marsCentre2fovCentreUnitVector_s"])]
            d["tilt_angle_e"] = [sp.vsep(ul2ll_fov_vec_ref, mars2centre_fov_vec_ref) * SP_DPR if flag else -999. for flag, ul2ll_fov_vec_ref, mars2centre_fov_vec_ref in zip(indices_s, d_tmp["fovTopBottomUnitVectors_e"], d_tmp["marsCentre2fovCentreUnitVector_e"])]
            
   
                    


    if obsType_in_IEGSLO:
        #combine start and end datasets into one 2d array
        for var_name in ["obs_alt", "obs_lon", "obs_lat", "ls", "sun_lon", "sun_lat", "sun_dist", "obs_speed_sun", "mars_speed_sun", "tilt_angle"]:
            d[var_name] = np.vstack((d[var_name+"_s"], d[var_name+"_e"])).T
            
        for var_name in ["tan_lon", "tan_lat", "lst", "slant_path", "tan_alt_ell", "tan_pt_dsk_dist", "tan_alt_dsk", "surf_alt_are", "tan_alt_are", "tangentSurfLOSAngles"]:
            for i in range(nPoints):
                dp[i][var_name] = np.vstack((dp[i][var_name+"_s"], dp[i][var_name+"_e"])).T
    
        dataset_mappings = {
            "Geometry/ObsAlt":{"units":"KILOMETRES", "desc":"Distance between spacecraft and Mars centre", "dtype":np.float, "var":d["obs_alt"]},
            "Geometry/SubObsLon":{"units":"DEGREES", "desc":"Surface longitude below spacecraft", "dtype":np.float, "var":d["obs_lon"]},
            "Geometry/SubObsLat":{"units":"DEGREES", "desc":"Surface latitude below spacecraft", "dtype":np.float, "var":d["obs_lat"]},
            "Geometry/LSubS":{"units":"DEGREES", "desc":"Mars areographic longitude (season)", "dtype":np.float, "var":d["ls"]},
            "Geometry/SubSolLon":{"units":"DEGREES", "desc":"Surface longitude below Sun", "dtype":np.float, "var":d["sun_lon"]},
            "Geometry/SubSolLat":{"units":"DEGREES", "desc":"Surface latitude below Sun", "dtype":np.float, "var":d["sun_lat"]},
            "Geometry/DistToSun":{"units":"ASTRONOMICAL UNITS", "desc":"Distance from spacecraft to Sun", "dtype":np.float, "var":d["sun_dist"]},
            "Geometry/SpdObsSun":{"units":"KILOMETRES PER SECOND", "desc":"Speed of the spacecraft relative to the Sun", "dtype":np.float, "var":d["obs_speed_sun"]},
            "Geometry/SpdTargetSun":{"units":"KILOMETRES PER SECOND", "desc":"Speed of Mars relative to the Sun", "dtype":np.float, "var":d["mars_speed_sun"]},
            # "Geometry/ObservationDateTime":{"units":"NO UNITS", "desc":"", "dtype":np.float, "var":d[""]},
            "Geometry/ObservationEphemerisTime":{"units":"SECONDS", "desc":"", "dtype":np.float, "var":d["et"]},
            }
        
        if obsType_in_DNF:
            dataset_mappings["Geometry/TiltAngle"] = {"units":"DEGREES", "desc":"Angle between spacecraft velocity and long edge of slit", "dtype":np.float, "var":d["tilt_angle"]}
        if obsType_in_IEGSLO:
            dataset_mappings["Geometry/TiltAngle"] = {"units":"DEGREES", "desc":"Angle between line from Mars centre to field of view and long edge of slit", "dtype":np.float, "var":d["tilt_angle"]}
            
                             
    
    
        for i in range(nPoints):
            dataset_mappings["Geometry/Point%s/PointXY" %i] = {"units":"NO UNITS", "desc":"Relative position of point %i within the field of view", "dtype":np.float, "var":dp[i]["point_xy"]}
        #     # dataset_mappings["Geometry/Point%s/FOVWeight" %i] = {"units":"NO UNITS", "desc":"", "dtype":np.float, "var":dp[i][""]}
            dataset_mappings["Geometry/Point%s/Lat" %i] = {"units":"DEGREES", "desc":"Observation surface latitude", "dtype":np.float, "var":dp[i]["tan_lat"]}
            dataset_mappings["Geometry/Point%s/Lon" %i] = {"units":"DEGREES", "desc":"Observation surface longitude", "dtype":np.float, "var":dp[i]["tan_lon"]}
            dataset_mappings["Geometry/Point%s/LST" %i] = {"units":"HOURS", "desc":"Local solar time", "dtype":np.float, "var":dp[i]["lst"]}
            dataset_mappings["Geometry/Point%s/SlantPathDist" %i] = {"units":"HOURS", "desc":"Distance between spacecraft and tangent point", "dtype":np.float, "var":dp[i]["slant_path"]}
            dataset_mappings["Geometry/Point%s/LOSAngle" %i] = {"units":"DEGREES", "desc":"For ASIMUT only", "dtype":np.float, "var":dp[i]["tangentSurfLOSAngles"]}
        #     dataset_mappings["Geometry/Point%s/SunSZA" %i] = {"units":"DEGREES", "desc":"Solar incidence angle on surface", "dtype":np.float, "var":dp[i][""]}
        #     dataset_mappings["Geometry/Point%s/IncidenceAngle" %i] = {"units":"DEGREES", "desc":"Solar incidence angle on surface", "dtype":np.float, "var":dp[i][""]}
        #     dataset_mappings["Geometry/Point%s/EmissionAngle" %i] = {"units":"DEGREES", "desc":"Surface emission angle", "dtype":np.float, "var":dp[i][""]}
        #     dataset_mappings["Geometry/Point%s/PhaseAngle" %i] = {"units":"DEGREES", "desc":"Surface solar phase angle", "dtype":np.float, "var":dp[i][""]}
            dataset_mappings["Geometry/Point%s/TangentAlt" %i] = {"units":"KILOMETRES", "desc":"Tangent altitude above ellipsoid", "dtype":np.float, "var":dp[i]["tan_alt_ell"]}
            dataset_mappings["Geometry/Point%s/TangentAltAreoid" %i] = {"units":"KILOMETRES", "desc":"Tangent altitude above areoid", "dtype":np.float, "var":dp[i]["tan_alt_are"]}
            dataset_mappings["Geometry/Point%s/TangentAltSurface" %i] = {"units":"KILOMETRES", "desc":"Tangent altitude above DSK surface", "dtype":np.float, "var":dp[i]["tan_alt_dsk"]}
            dataset_mappings["Geometry/Point%s/SurfaceRadius" %i] = {"units":"KILOMETRES", "desc":"Height of DSK surface above Mars centre", "dtype":np.float, "var":dp[i]["tan_pt_dsk_dist"]}
            dataset_mappings["Geometry/Point%s/SurfaceAltAreoid" %i] = {"units":"KILOMETRES", "desc":"Height of DSK surface above areoid", "dtype":np.float, "var":dp[i]["surf_alt_are"]}



    if channel=="uvis":
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR,"%s_%s.h5" % (hdf5_basename, observationType))
    else:
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))

#    logger.info("Writing to file: %s", hdf5FilepathOut)
    with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:

        # Copy datasets and attributes
        if platform.system() != "Windows": #for testing locally
            generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION)
            for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                hdf5FileIn.copy(dset_path, dest)

        hdf5FileIn.close()

        if obsType_in_C:
            #write only ephemeris time to calibration files. Rest remains unchanged.
            hdf5FileOut.create_dataset("Geometry/ObservationEphemerisTime", dtype=np.float,
                                    data=d["et"], fillvalue=NA_VALUE, compression="gzip", shuffle=True)


        #write new datasets for science files
        if obsType_in_IEGSLO:
            for dset_path, mapping in dataset_mappings.items():
                dset = hdf5FileOut.create_dataset(dset_path, dtype=mapping["dtype"],
                                    data=mapping["var"], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                dset.attrs["Units"] = mapping["units"]
                dset.attrs["Description"] = mapping["desc"]



        if obsType_in_DNF:
            # logger.info("Writing generic geometry for observation of type %s in file %s", observationType, hdf5FilepathOut)
            
            
            hdf5FileOut.create_dataset("Geometry/ObservationEphemerisTime", dtype=np.float,
                                    data=ephemerisTimes, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            #Geometry/ObservationDateTime  already written
            a = hdf5FileOut.create_dataset("Geometry/ObsAlt", dtype=np.float, data=obsAlts, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            a.attrs["Description"] = "Distance between spacecraft and Mars centre"

            b = hdf5FileOut.create_dataset("Geometry/TiltAngle", dtype=np.float, data=tiltAngles, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            if obsType_in_DNF:
                b.attrs["Description"] = "Angle between spacecraft velocity and long edge of slit" 
            if obsType_in_IEGSLO:
                b.attrs["Description"] = "Angle between line from Mars centre to field of view and long edge of slit" 


            c = hdf5FileOut.create_dataset("Geometry/SubObsLon", dtype=np.float, data=subObsLons, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            c.attrs["Description"] = "Surface longitude below spacecraft"

            dd = hdf5FileOut.create_dataset("Geometry/SubObsLat", dtype=np.float, data=subObsLats, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            dd.attrs["Description"] = "Surface latitude below spacecraft"

            e = hdf5FileOut.create_dataset("Geometry/LSubS", dtype=np.float, data=lSubSs, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            e.attrs["Description"] = "Mars areographic longitude (season)"

            f = hdf5FileOut.create_dataset("Geometry/SubSolLon", dtype=np.float, data=subSolLons, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            f.attrs["Description"] = "Surface longitude below Sun"

            g = hdf5FileOut.create_dataset("Geometry/SubSolLat", dtype=np.float, data=subSolLats, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            g.attrs["Description"] = "Surface latitude below Sun"

            h = hdf5FileOut.create_dataset("Geometry/DistToSun", dtype=np.float, data=distToSuns, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            h.attrs["Description"] = "Distance from spacecraft to Sun"

            i = hdf5FileOut.create_dataset("Geometry/SpdObsSun", dtype=np.float, data=spdObsSun, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            i.attrs["Description"] = "Speed of the spacecraft relative to the Sun"

            j = hdf5FileOut.create_dataset("Geometry/SpdTargetSun", dtype=np.float, data=spdTargetSun, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            j.attrs["Description"] = "Speed of Mars relative to the Sun"

            #set boresight quality flags to zero for nadir observation
            writeBoresightQualityFlag(hdf5FileOut, "NO OCCULTATION")


#                logger.info("Writing nadir geometry for observation of type %s in file %s", observationType, hdf5_basename)
            for pointIndex in range(nPoints):
                k = hdf5FileOut.create_dataset("Geometry/Point%i/PointXY" %pointIndex, dtype=np.float, data=pointXYs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                k.attrs["Description"] = "Relative position of point %i within the field of view" %pointIndex

                l = hdf5FileOut.create_dataset("Geometry/Point%i/Lon" %pointIndex, dtype=np.float, data=surfLons[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                l.attrs["Description"] = "Observation surface longitude"

                m = hdf5FileOut.create_dataset("Geometry/Point%i/Lat" %pointIndex, dtype=np.float, data=surfLats[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                m.attrs["Description"] = "Observation surface latitude"

                n = hdf5FileOut.create_dataset("Geometry/Point%i/LST" %pointIndex, dtype=np.float, data=surfLSTs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                n.attrs["Description"] = "Local solar time"
                
                o = hdf5FileOut.create_dataset("Geometry/Point%i/LOSAngle" %pointIndex, dtype=np.float, data=surfLOSAngles[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                o.attrs["Description"] = "For ASIMUT only"
                
                p = hdf5FileOut.create_dataset("Geometry/Point%i/SunSZA" %pointIndex, dtype=np.float, data=surfSunSZAs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                p.attrs["Description"] = "Solar incidence angle on surface"
                
                q = hdf5FileOut.create_dataset("Geometry/Point%i/IncidenceAngle" %pointIndex, dtype=np.float, data=surfIncidenceAngles[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                q.attrs["Description"] = "Solar incidence angle on surface"
                
                r = hdf5FileOut.create_dataset("Geometry/Point%i/EmissionAngle" %pointIndex, dtype=np.float, data=surfEmissionAngles[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                r.attrs["Description"] = "Surface emission angle"
                
                s = hdf5FileOut.create_dataset("Geometry/Point%i/PhaseAngle" %pointIndex, dtype=np.float, data=surfPhaseAngles[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                s.attrs["Description"] = "Surface solar phase angle"
                
#                    t = hdf5FileOut.create_dataset("Geometry/Point%i/TangentAlt" %pointIndex, dtype=np.float, data=surfTangentAlts[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
#                    t.attrs["Description"] = "Tangent altitude above areoid"
                
                u = hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceRadius" %pointIndex, dtype=np.float, data=surfRadius[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                u.attrs["Description"] = "Height of DSK surface above Mars centre"

                if USE_AREOID:
                    v = hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceAltAreoid" %pointIndex, dtype=np.float,data=surfTopo[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                    v.attrs["Description"] = "Height of DSK surface above areoid"

                unitMappings = getUnitMappings(nPoints)
                addUnits(hdf5FileOut, unitMappings)


        #write new datasets for occultation/limb only files
        if obsType_in_IEGSLO:
            # logger.info("Writing occultation/limb geometry for observation of type %s in file %s", observationType, hdf5FilepathOut)

            if observationType != "L":
                #find boresight and write quality flag
                all_boresight_vectors, all_boresight_names = readBoresightFile(BORESIGHT_VECTOR_TABLE)
                #get mid point et of occultation
                #TODO: replace by median sun pointing vector
                obsTimeMid = np.mean([d["et_s"][0],d["et_s"][-1]])
                boresight_name_found, boresight_vector_found, v_sep_min = findBoresightUsed(obsTimeMid, all_boresight_vectors, all_boresight_names)
                logger.info("Boresight determination: closest match to %s, vsep = %0.3f arcmins", boresight_name_found, v_sep_min)
                
                #get angular separation in arcmins between sun centre and FOV centre
                fovSunCentreAngle = np.asfarray([findSunWobble(d["et_s"], boresight_vector_found), findSunWobble(d["et_e"], boresight_vector_found)]).T
                dset = hdf5FileOut.create_dataset("Geometry/FOVSunCentreAngle", dtype=np.float,
                                    data=fovSunCentreAngle, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                dset.attrs["Units"] = "ARCMINUTES"
                dset.attrs["Description"] = "Angular separation between centre of the field of view and centre of the Sun"

                
                writeBoresightQualityFlag(hdf5FileOut, boresight_name_found)
            else: #if limb measurement, no sun boresight
                #set boresight quality flags to zero for nadir observation
                writeBoresightQualityFlag(hdf5FileOut, "NO OCCULTATION")




        if obsType_in_DNF or obsType_in_IEGSLO:   
            hdf5FileOut.attrs["GeometryPoints"] = nPoints
            
            
    if channel == "uvis":
        write_attrs_from_itl(hdf5FilepathOut, obs_db_res)
    hdf5FileOut.close()
    return [hdf5FilepathOut]


#t2 = time.clock()
#
#print("Processing time = %0.2f seconds" %(t2 - t1))

# hdf5file_path = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p1e/2018/04/22/20180422_120404_0p1e_SO_1_E_134.h5")
# convert(hdf5file_path)
