# -*- coding: utf-8 -*-

TESTING=True
#TESTING=False



import logging
import os
import h5py
import numpy as np
#import spiceypy as sp
#from scipy import interpolate
#import re
#import matplotlib.pyplot as plt

logger = logging.getLogger( __name__ )



from instrument.calibration.lno_radiance_factor.config import \
    NOMAD_TMP_DIR, LNO_FLAGS_DICT, RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, \
    LNO_RADIOMETRIC_CALIBRATION_TABLE_NAME, LNO_RADIANCE_FACTOR_CALIBRATION_TABLE_NAME, \
    PFM_AUXILIARY_FILES, THUMBNAIL_DIRECTORY, SAVE_FILES, NA_VALUE, trans, generics
from instrument.calibration.lno_radiance_factor.functions.check_flags import check_flags
from instrument.calibration.lno_radiance_factor.convert_lno_radiance import convert_lno_radiance
from instrument.calibration.lno_radiance_factor.convert_lno_rad_fac import convert_lno_rad_fac


__project__   = "NOMAD"
__author__    = "Ian Thomas, Roland Clairquin, Loïc Trompet"
__contact__   = "ian.thomas@oma.be"




#============================================================================================
# 5. CONVERT HDF5 LEVEL 0.3A TO LEVEL 1.0A
# RADIOMETRIC CALIBRATION OF LNO CHANNEL:
# NADIR AND LIMB
# NADIR FULLSCAN
# SO FULLSCAN
#============================================================================================
# DONE:
# SWAP Y AND YRADIANCE
# ADD FIT COEFFICIENTS TO FILE
# TO DO:
# ADD RADIOMETRIC CALIBRATION ERROR
# ADD FILE ATTRIBUTE CALIBRATION REFERENCES
# ADD TEMPERATURE DB DATA
# USE ONLY MATCHING ABSORPTIONS TO CALCULATE N FITS AND ERROR IN FILE AND ON PLOT
#============================================================================================


VERSION = 80
OUTPUT_VERSION = "1.0A"




def convert(hdf5file_path):
    
    hdf5_basename = os.path.basename(hdf5file_path)
    logger.info("convert: %s", hdf5_basename)


    with h5py.File(hdf5file_path, "r") as hdf5FileIn:
        if TESTING:
            observationType = "D"
            channel = "lno"
        else:
            observationType = generics.getObservationType(hdf5FileIn)
            channel, channelType = generics.getChannelType(hdf5FileIn)

        if channel == "uvis": #should not happen, but just in case
            logger.error("%s is a UVIS observation", hdf5_basename)
            return []
    
        if observationType is None:
            logger.error("% has no observation type", hdf5_basename)
            return []
    
    
    
        #if science measurement, do radiometric calibration
        #E and I do not pass here
        if channel in ["so", "lno"] and observationType in ["S"]:
            """SO fullscan transmittance calibration"""
            hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
            return trans.doTransmittanceCalibrationForS(hdf5file_path, hdf5FilepathOut)
    
    
    
        elif channel in ["lno"] and observationType in ["F"]:
            logger.info("Radiometrically calibrating LNO nadir fullscan observation. Not yet implemented")
            logger.info("Y_UNIT_FLAG=%s, Y_TYPE_FLAG=%s, Y_ERROR_FLAG=%s", LNO_FLAGS_DICT["Y_UNIT_FLAG"],LNO_FLAGS_DICT["Y_TYPE_FLAG"],LNO_FLAGS_DICT["Y_ERROR_FLAG"])
            """RADIANCE CALIBRATION"""
            #hdf5FilepathOut = doFullscanRadiometricCalibration(hdf5file_path)
            return []
    
    
    
        elif channel in ["lno"] and observationType in ["D"]:
            """LNO nadir calibration"""
            logger.info("Radiometrically calibrating LNO dayside nadir observation")
            logger.info("Y_UNIT_FLAG=%s, Y_TYPE_FLAG=%s, Y_ERROR_FLAG=%s", LNO_FLAGS_DICT["Y_UNIT_FLAG"],LNO_FLAGS_DICT["Y_TYPE_FLAG"],LNO_FLAGS_DICT["Y_ERROR_FLAG"])
    
    
            #check combination of flags is valid
            calibrationType, errorType = check_flags(LNO_FLAGS_DICT)
        
            if calibrationType == "":
                logger.error("Error: Calibration type unknown for Y_UNIT_FLAG = %i and Y_TYPE_FLAG = %i", LNO_FLAGS_DICT["Y_UNIT_FLAG"], LNO_FLAGS_DICT["Y_TYPE_FLAG"])
            elif errorType == "":
                logger.error("Error: Error type unknown for Y_ERROR_FLAG = %i", LNO_FLAGS_DICT["Y_ERROR_FLAG"])
    
            """radiance calibration"""
            if calibrationType in ["Radiance"]:
                radiance_cal_dict, radiance_refs = convert_lno_radiance(hdf5_basename, hdf5FileIn, errorType)
                rad_fac_cal_dict = {}
                rad_fac_refs = {"error":True}
    
            elif calibrationType in ["Radiance & Radiance Factor"]:
                radiance_cal_dict, radiance_refs = convert_lno_radiance(hdf5_basename, hdf5FileIn, errorType)
                rad_fac_cal_dict, rad_fac_refs = convert_lno_rad_fac(hdf5_basename, hdf5FileIn, errorType)
                
            else:
                logger.error("Calibration type %s not yet implemented", calibrationType)

            y_calib_ref = radiance_refs["calib_ref"] + ". " + rad_fac_refs["calib_ref"]
            y_error_ref = radiance_refs["error_ref"] + ". " + rad_fac_refs["error_ref"]
            logger.info("%s: %s", hdf5_basename, y_calib_ref)

            
            if calibrationType in ["Radiance & Radiance Factor"]:
                #make output filename based on radiance factor pass/fail
                if rad_fac_refs["error"]:
                    hdf5_basename_split = hdf5_basename.split("_")
                    hdf5_basename_split[5] = "DF"
                else:
                    hdf5_basename_split = hdf5_basename.split("_")
                    hdf5_basename_split[5] = "DP"
                hdf5_basename_new = "_".join(hdf5_basename_split)

            else:
                hdf5_basename_new = hdf5_basename

            hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, hdf5_basename_new)





            #specify datasets/attributes are to be modified/removed
            DATASETS_TO_BE_REMOVED = [
            #    "Science/X",
                "Science/Y",
                "Science/YTypeFlag",
                "Science/YUnitFlag",
                "Science/YError",
                "Science/YErrorFlag",
                "Science/YUnmodified",
                ]
            
            #if x values have been recalibrated from nadir solar/molecular lines
            if "Science/X" in rad_fac_cal_dict.keys():
                DATASETS_TO_BE_REMOVED.append("Science/X")
            
            ATTRIBUTES_TO_BE_REMOVED = ["YCalibRef", "YErrorRef"]



            
            #make file from dictionaries
            YTypeFlagOld = hdf5FileIn["Science/YTypeFlag"]
            YTypeFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_TYPE_FLAG"]
            YUnitFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_UNIT_FLAG"]
            YErrorFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_ERROR_FLAG"]
                    
            logger.info("Writing new datasets to output file")
            with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:
            
                generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
                for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                    if dset_path in DATASETS_TO_BE_REMOVED:
                        continue
                    dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                    hdf5FileIn.copy(dset_path, dest)
            
                hdf5FileOut.attrs["YCalibRef"] = y_calib_ref
                hdf5FileOut.attrs["YErrorRef"] = y_error_ref
    
    #            hdf5FileOut.create_dataset("Science/X", dtype=np.float32, data=XOut, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.int16, data=YTypeFlag, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.int16, data=YUnitFlag, compression="gzip", shuffle=True)
                hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.int16, data=YErrorFlag, compression="gzip", shuffle=True)
    
#                hdf5FileOut.create_dataset("Science/YNormCounts", dtype=np.float32, data=Y, compression="gzip", shuffle=True)
#                hdf5FileOut.create_dataset("Science/YUnmodified", dtype=np.float32, data=yIn, compression="gzip", shuffle=True)
                
                for hdf5_dataset_path, dataset_dict in radiance_cal_dict.items():
                    if dataset_dict["compression"]:
                        hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"], compression="gzip", shuffle=True)
                    else:
                        hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"])
                for hdf5_dataset_path, dataset_dict in rad_fac_cal_dict.items():
                    if dataset_dict["compression"]:
                        hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"], compression="gzip", shuffle=True)
                    else:
                        hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"])
    
    
#                if calibrationType in ["Radiance"]:
#                    hdf5FileOut.create_dataset("Science/YRadiance", dtype=np.float32, data=YOut, compression="gzip", shuffle=True)
#                    hdf5FileOut.create_dataset("Science/YRadianceSimple", dtype=np.float32, data=YOutAtWavenumber, compression="gzip", shuffle=True)
#                    hdf5FileOut.create_dataset("Science/YRadianceError", dtype=np.float32, data=YError, compression="gzip", shuffle=True)
#                    hdf5FileOut.create_dataset("Science/SNRRadiance", dtype=np.float32, data=SNR, compression="gzip", shuffle=True)
#                if calibrationType in ["Radiance", "Radiance & Radiance Factor"]:
#                    hdf5FileOut.create_dataset("Science/Y", dtype=np.float32, data=YRadFac, compression="gzip", shuffle=True)
#                    hdf5FileOut.create_dataset("Criteria/LineFit/NumberOfLinesFit", dtype=np.int16, data=len(Criteria))
#                    hdf5FileOut.create_dataset("Criteria/LineFit/ChiSqError", dtype=np.float32, data=Criteria, compression="gzip", shuffle=True)
    



        else:
            logger.error("%s: %s calibration cannot be performed on %s data", hdf5_basename, calibrationType, channel)
        
    return [hdf5FilepathOut]



#if TESTING:
#
#    
#    """this part gets observation files for testing"""
#
##    diffractionOrder = 118
##    diffractionOrder = 120
##    diffractionOrder = 126
##    diffractionOrder = 130
##    diffractionOrder = 133
##    diffractionOrder = 142
##    diffractionOrder = 151
##    diffractionOrder = 156
#
#
##    diffractionOrder = 162
##    diffractionOrder = 163
##    diffractionOrder = 167
##    diffractionOrder = 168
##    diffractionOrder = 169
#    diffractionOrder = 189
##    diffractionOrder = 194
##    diffractionOrder = 196
#    
#    
#    from database_functions_v01 import obsDB, makeObsDict
#    dbName = "lno_0p3a"
#    db_obj = obsDB(dbName)
#    #CURIOSITY = -4.5895, 137.4417
#    
#    
#    #min_lat, max_lat, min_lon, max_lon, max_incidence_angle, min_temperature, max_temperature, max_orders
#    obsSearchDict = {
#            118:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            120:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            126:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            130:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            133:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            142:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            151:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            156:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#
#
#            162:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
#            163:[-90, 90, -180, 180, 90, -30, 30, 4], #minimal data
#            164:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            166:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            167:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            168:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            169:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#
#            173:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            174:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            178:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            179:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            180:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            182:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            184:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            189:[-15, 5, 127, 147, 90, -30, 30, 4], #good
#            194:[-90, 90, 127, 147, 90, -30, 30, 4], #good
#            195:[-90, 90, -180, 180, 90, -30, 30, 4], #none
#            196:[-90, 90, 127, 147, 90, -30, 30, 4], #could be improved
#            }
#
#
##    bestOrderMolecules = {118:"CO2", 120:"CO2", 126:"CO2", 130:"CO2", 133:"H2O", 142:"CO2", 151:"CO2", 156:"CO2", 160:"CO2", 162:"CO2", \
##                          163:"CO2", 164:"CO2", 166:"CO2", 167:"H2O", 168:"H2O", 169:"H2O", 173:"H2O", 174:"H2O", 178:"H2O", 179:"H2O", \
##                          180:"H2O", 182:"H2O", 184:"CO", 189:"CO", 194:"CO", 195:"CO", 196:"CO"}
#
#    
#    if diffractionOrder in obsSearchDict.keys():
#        min_lat, max_lat, min_lon, max_lon, max_incidence_angle, min_temperature, max_temperature, max_orders = obsSearchDict[diffractionOrder]
#    else:
#        print("Error: diffraction order %i not in dictionary" %diffractionOrder)
#    
#    
#    searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir WHERE latitude < %i AND latitude > %i AND longitude < %i AND longitude > %i AND n_orders < %i AND incidence_angle < %i AND temperature > %i AND temperature < %i AND diffraction_order == %i" \
#                                     %(max_lat, min_lat, max_lon, min_lon, max_orders, max_incidence_angle, min_temperature, max_temperature, diffractionOrder))
#    
#    obsDict = makeObsDict("lno", searchQueryOutput)
#    db_obj.close()
#    #plt.figure()
#    #plt.scatter(obsDict["longitude"], obsDict["latitude"])
#    
#    n_files = len(set(obsDict["filename"]))
#    print("%i LNO files found" %n_files)
#    
#    #load spiceypy kernels
#    BASE_DIRECTORY = ROOT_STORAGE_PATH
#    KERNEL_DIRECTORY = os.path.join("C:", os.sep, "Users", "iant", "Documents", "DATA", "local_spice_kernels", "kernels", "mk")
#    METAKERNEL_NAME = "em16_plan.tm"
#    print("KERNEL_DIRECTORY=%s, METAKERNEL_NAME=%s" %(KERNEL_DIRECTORY, METAKERNEL_NAME))
#    os.chdir(KERNEL_DIRECTORY)
#    sp.furnsh(METAKERNEL_NAME)
#    print(sp.tkvrsn("toolkit"))
#    os.chdir(BASE_DIRECTORY)
#
#
#
#    for hdf5file_path in set(obsDict["filepath"]):
#        convert(hdf5file_path)
