# -*- coding: utf-8 -*-


import logging
import os
import h5py
import numpy as np

    
from nomad_ops.core.hdf5.l0p3a_to_1p0a.convert_lno_radiance import convert_lno_radiance
from nomad_ops.core.hdf5.l0p3a_to_1p0a.convert_lno_ref_fac import convert_lno_ref_fac

from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.output_filename import output_filename
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.check_flags import check_flags

from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import NOMAD_TMP_DIR, LNO_FLAGS_DICT, SAVE_FILE, trans, generics

__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian . thomas @ aeronomie .be"

logger = logging.getLogger( __name__ )


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
    
    hdf5_basename = os.path.splitext(os.path.basename(hdf5file_path))[0]
    logger.info("convert: %s", hdf5_basename)


    with h5py.File(hdf5file_path, "r") as hdf5FileIn:
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
 
    
            """radiance calibration only"""
            if calibrationType in ["Radiance"]:
                radiance_cal_dict, radiance_refs = convert_lno_radiance(hdf5_basename, hdf5FileIn, errorType)
                ref_fac_cal_dict = {}
                ref_fac_refs = {"error":True}

    
                """radiance and radiance factor calibration"""
            elif calibrationType in ["Radiance & Radiance Factor"]:
                radiance_cal_dict, radiance_refs = convert_lno_radiance(hdf5_basename, hdf5FileIn, errorType)
                ref_fac_cal_dict, ref_fac_refs = convert_lno_ref_fac(hdf5_basename, hdf5FileIn, errorType)
                
            else:
                logger.error("Calibration type %s not yet implemented", calibrationType)

            #make attribute labels
            y_calib_ref = radiance_refs["calib_ref"] + ". " + ref_fac_refs["calib_ref"]
            y_error_ref = radiance_refs["error_ref"] + ". " + ref_fac_refs["error_ref"]
            logger.info("%s: %s", hdf5_basename, y_calib_ref)

            
            if calibrationType in ["Radiance & Radiance Factor"]:
                #make output filename based on radiance factor pass/fail
                hdf5_basename_new = output_filename(hdf5_basename, ref_fac_refs["error"])

            else:
                hdf5_basename_new = hdf5_basename

            hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, hdf5_basename_new+".h5")



            if SAVE_FILE:

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
                if "Science/X" in ref_fac_cal_dict.keys():
                    DATASETS_TO_BE_REMOVED.append("Science/X")
                
                ATTRIBUTES_TO_BE_REMOVED = ["YCalibRef", "YErrorRef"]
    
    
    
                
                #make file from dictionaries
                YTypeFlagOld = hdf5FileIn["Science/YTypeFlag"]
                YTypeFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_TYPE_FLAG"]
                YUnitFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_UNIT_FLAG"]
                YErrorFlag = np.zeros_like(YTypeFlagOld) + LNO_FLAGS_DICT["Y_ERROR_FLAG"]
                        
                logger.info("Writing new datasets to output file %s", hdf5_basename_new)
                with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:
                
                    generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
                    for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                        if dset_path in DATASETS_TO_BE_REMOVED:
                            continue
                        dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                        hdf5FileIn.copy(dset_path, dest)
                
                    hdf5FileOut.attrs["YCalibRef"] = y_calib_ref
                    hdf5FileOut.attrs["YErrorRef"] = y_error_ref
        
                    hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.int16, data=YTypeFlag, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.int16, data=YUnitFlag, compression="gzip", shuffle=True)
                    hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.int16, data=YErrorFlag, compression="gzip", shuffle=True)
                    
                    for hdf5_dataset_path, dataset_dict in radiance_cal_dict.items():
                        if dataset_dict["compression"]:
                            hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"], compression="gzip", shuffle=True)
                        else:
                            hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"])
                    for hdf5_dataset_path, dataset_dict in ref_fac_cal_dict.items():
                        if dataset_dict["compression"]:
                            hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"], compression="gzip", shuffle=True)
                        else:
                            hdf5FileOut.create_dataset(hdf5_dataset_path, dtype=dataset_dict["dtype"], data=dataset_dict["data"])
        
            if SAVE_FILE:
                return [hdf5FilepathOut]
            else:
                return []
        
        else:
            logger.error("%s: Calibration cannot be performed on %s data", hdf5_basename, channel)
            return []
    


# convert("20200705_031603_0p3a_LNO_1_D_189.h5")