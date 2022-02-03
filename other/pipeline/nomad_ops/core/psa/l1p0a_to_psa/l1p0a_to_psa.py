# -*- coding: utf-8 -*-
"""


OUTPUT PRODUCTS:

LEVEL 1.0B HDF5 FILE
ZIP FILE CONTAINING 4 X PRODUCTS

XML LABEL
TAB PRODUCT

BROWSE LABEL
PNG FILE


The following changes have been required in order to successfully validate a product provided in the test set:


"""

# TODO: pipeline fails for observations with 5x dark spectra and 1x order per rhythm. Geometry/ObservationDateTime is not correctly resized somewhere in pipeline
# This is not a problem for now, as these are calibration measurements, but should be addressed in future updates
# Quality flags are not yet implemented in l1.0a hdf5, especially for UVIS.
#Missing SO/LNO flags:
#"HSKDisturbed"
#"HSKOutOfBounds"
#"DiscardedPackets"
#"DetectorSaturated"
#"PointingError"


import time
import logging
import os.path
# import re
import numpy as np
from lxml import etree
# import zipfile
from datetime import datetime
# try:
#     import zlib
#     compression = zipfile.ZIP_DEFLATED
# except:
#     compression = zipfile.ZIP_STORED

import os
#import re
import shutil

# from nomad_ops.config import NOMAD_TMP_DIR
import nomad_ops.core.hdf5.generic_functions as generics


from nomad_ops.core.psa.l1p0a_to_psa.psa_plotting_functions import plot_so_lno_occultation, plot_lno_uvis_nadir, plot_uvis_occultation
from nomad_ops.core.psa.l1p0a_to_psa.config import \
    HDF5_TIME_FORMAT, ASCII_DATE_TIME_YMD_UTC, PSA_MODIFICATION_DATE, \
    PSA_VERSION, TITLE, INFORMATION_MODEL_VERSION, PSA_VERSION_DESCRIPTION, MISSION_PHASE, MISSION_PHASE_SHORT, \
    VALIDATE_OUTPUT, VALIDATION_RATIO
from nomad_ops.core.psa.l1p0a_to_psa.mappings import \
    get_dataset_info, get_mappings, get_default_flag_values, get_channel_observation_type#, getYUnits
from nomad_ops.core.psa.l1p0a_to_psa.xml import \
    MODEL_HREF, MODEL_SCHEMATRON, SCHEMA_LOCATION, XMLNS, XMLNS_GEOM, XMLNS_PSA, XMLNS_XSI, XMLNS_EM16_TGO_NMD, \
    makeBrowseXmlLabel
from nomad_ops.core.psa.l1p0a_to_psa.functions import \
    psaErrorLog, checkIfFullscan, convert_par_filename_to_lid, findAttribute

from nomad_ops.core.psa.l1p0a_to_psa.paths import make_path_dict
from nomad_ops.core.psa.l1p0a_to_psa.validate import validate_data
from nomad_ops.core.psa.l1p0a_to_psa.match_par_raw import get_exm_filename, get_psa_par_filename
from nomad_ops.core.psa.l1p0a_to_psa.template import read_psa_template_file



__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian.thomas@aeronomie.be"


logger = logging.getLogger( __name__ )






"""start program"""
if True:
    import h5py
    # hdf5file_path = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\04\22\20180422_001650_1p0a_SO_A_E_164.h5"
    hdf5file_path = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\04\22\20180422_003456_1p0a_LNO_1_DP_169.h5"
    # hdf5file_path = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\04\22\20180422_023247_1p0a_UVIS_D.h5"
    # hdf5file_path = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\04\22\20180422_001650_1p0a_UVIS_E.h5"
    # hdf5file_path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_1p0a/2019/11/10/20191110_003310_1p0a_SO_A_E_168.h5"
    hdf5FileIn = h5py.File(hdf5file_path, "r")
# def convert(hdf5file_path, hdf5FileIn):

    error = False

    
    hdf5_basename = os.path.basename(hdf5file_path).split(".")[0]
    logger.info("convert: %s", hdf5_basename)


    channel, channelType = generics.getChannelType(hdf5FileIn)
    observationType = generics.getObservationType(hdf5FileIn)

    if observationType is None:
        error = True
        psaErrorLog("%s failed: observation type error" %hdf5_basename)
        logger.error("%s: observation type is invalid", hdf5_basename)
    else:
        channel_obs = get_channel_observation_type(channel, observationType)
        if channel_obs == "":
            error = True
            psaErrorLog("%s failed: channel and observation type invalid" %hdf5_basename)
            logger.error("%s: Channel (%s) and observation type (%s) combination invalid", hdf5_basename, channel, observationType)


    #get closest PSA PAR and EXM filenames from database"""
    exm_filename = get_exm_filename(hdf5_basename, hdf5FileIn)
    par_filename = get_psa_par_filename(channel, hdf5_basename, hdf5FileIn)

    if exm_filename == "None":
        error = True
        psaErrorLog("%s failed: EXM reference error" %hdf5_basename)
    if par_filename == "None":
        error = True
        psaErrorLog("%s failed: PSA par reference error" %hdf5_basename)
    
    
    #now get dictionary containing filenames and paths
    paths = make_path_dict(channel_obs, observationType, hdf5_basename, hdf5FileIn)


    #define observation mode
    ChannelName = findAttribute(hdf5FileIn, "ChannelName") #write observation mode
    if "Mirror Position" in ChannelName:
        error = True
        psaErrorLog("%s failed: flip mirror position error" %hdf5_basename)
        logger.error("%s: flip mirror position error", hdf5_basename)
    elif "flip mirror" in ChannelName:
        error = True
        psaErrorLog("%s failed: flip mirror position error" %hdf5_basename)
        logger.error("%s: flip mirror position error", hdf5_basename)
    elif "Selector" in ChannelName:
        error = True
        psaErrorLog("%s failed: selector mechanism test" %hdf5_basename)
        logger.error("%s: selector mechanism test", hdf5_basename)
    elif "Test Card" in ChannelName:
        error = True
        psaErrorLog("%s failed: CCD test card" %hdf5_basename)
        logger.error("%s: CCD test card", hdf5_basename)
        

    #read in filename and data from closest psa file
    #read in psa calibrated template from auxilliary files/psa
    psaTemplateElementList = read_psa_template_file(channel_obs)
    if psaTemplateElementList == "":
        error = True
        psaErrorLog("%s failed: template could not be found" %hdf5_basename)
        logger.error("%s: template not found for channel_obs %s", hdf5_basename)



    """part to check for calibration files where 1x light frame and 5x dark frames.
    These can be found where the Geometry/ObservationDateTime field has not been correctly resized to match the other fields"""
    
    observationDateTimeLength = len(hdf5FileIn["Geometry/ObservationDateTime"][:,0])
    lsLength = len(hdf5FileIn["Geometry/LSubS"][:,0])

    if observationDateTimeLength != lsLength:
        logger.error("ObservationDateTime field has not been correctly resized. File %s is likely a 5x dark frame calibration file", hdf5_basename)
        psaErrorLog("%s failed: 5x dark calibration observation" %hdf5_basename)
        error = True


        
    #only run if no errors in template, exm or par filenames
    if not error:
            
        #reformat observation times
        
        obsStartDateTimes = [datetime.strptime(dt.decode(), HDF5_TIME_FORMAT) for dt in hdf5FileIn["Geometry/ObservationDateTime"][:,0]]
        obsEndDateTimes = [datetime.strptime(dt.decode(), HDF5_TIME_FORMAT) for dt in hdf5FileIn["Geometry/ObservationDateTime"][:,1]]
        
        #check if timings reversed (for ingress observations). Can't just use first/last values
        obsStartDateTime = min(obsStartDateTimes)
        obsEndDateTime = max(obsEndDateTimes)
        
        psaObsStartTime = datetime.strftime(obsStartDateTime, ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z"
        psaObsEndTime = datetime.strftime(obsEndDateTime, ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z"
        
        
        psaMeasurementStartTimes = [datetime.strftime(dt, ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z" for dt in obsStartDateTimes]
        psaMeasurementEndTimes = [datetime.strftime(dt, ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z" for dt in obsEndDateTimes]
    
    
        #make dummy fields for those that don't exist yet in the HDF5 files
        dummyGeometry1Column = np.zeros_like(hdf5FileIn["Channel/IntegrationTime"][...])
        
        
        #determine temperatures
        if channel=="lno":
            detectorTemperatures = hdf5FileIn["Housekeeping/FPA1_FULL_SCALE_TEMP_LNO"][...]
            instrumentTemperature = hdf5FileIn["Channel/MeasurementTemperature"][0][0]
        elif channel=="so":
            detectorTemperatures = hdf5FileIn["Housekeeping/FPA1_FULL_SCALE_TEMP_SO"][...]
            instrumentTemperature = hdf5FileIn["Channel/MeasurementTemperature"][0][0]
        elif channel=="uvis":
            detectorTemperatures = hdf5FileIn["Housekeeping/TEMP_2_CCD"][...]
        detectorTemperature = np.mean(detectorTemperatures[-10:-1])
        
    
        
        nPoints = findAttribute(hdf5FileIn, "GeometryPoints")
        psa_mappings = get_mappings(channel_obs)
        default_flags = get_default_flag_values(hdf5_basename, hdf5FileIn, channel_obs)
        psaFieldNames = list(psa_mappings.keys())

        orbitNumber = findAttribute(hdf5FileIn, "Orbit")
        git_tagged_version = findAttribute(hdf5FileIn, "git-tagged_version")
        internal_pipeline_version = git_tagged_version.rsplit("-", 1)[0] + " (%s)" %PSA_MODIFICATION_DATE

    
    
    
    
        
        """make list of variables to add to xml products"""
        
        metadataVariables = {
            "logical_identifier":paths["data_lid_full"],
            "logical_identifier_browse":paths["brow_lid_full"],
#            "logical_identifier_aotf":paths["aotfLogicalIdentifier"],
            "version_id":PSA_VERSION,
            "title":TITLE,
            "information_model_version":INFORMATION_MODEL_VERSION,
            "psa_version":PSA_VERSION,
            "psa_version_description":PSA_VERSION_DESCRIPTION,
            "psa_modification_date":PSA_MODIFICATION_DATE,
            "obs_start_time":psaObsStartTime,
            "obs_end_time":psaObsEndTime,
            "product_description":"NOMAD "+(channel).upper()+" Calibrated Science Product",
            
            "clock_start_count":paths["data_lid_start_time"],
            "clock_stop_count":paths["data_lid_end_time"],
            "mission_phase":MISSION_PHASE,
            "mission_phase_short":MISSION_PHASE_SHORT,
            "orbit_start_number":orbitNumber,
            "lid_psa_par_input":convert_par_filename_to_lid(par_filename),
            "exm_input_file":exm_filename,
            "internal_pipeline_version":internal_pipeline_version,
    
            "Geometry":channel_obs.split("_")[1],
    
            "tab_filename":paths["data_tab_filename"], 
            "local_identifier":paths["data_tab_filename"], 
            "creation_datetime":datetime.strftime(datetime.now(),ASCII_DATE_TIME_YMD_UTC)[:-3]+"Z",
            "number_of_records":findAttribute(hdf5FileIn, "NSpec"),
            "table_name":"CAL_NOMAD_" + channel.upper(),
                             }
    
        metadataVariables["COPTableVersion"] = findAttribute(hdf5FileIn, "COPTableVersion")
        metadataVariables["XCalibRef"] = findAttribute(hdf5FileIn, "XCalibRef")
        metadataVariables["ChannelName"] = findAttribute(hdf5FileIn, "ChannelName")
        metadataVariables["NSpec"] = findAttribute(hdf5FileIn, "NSpec")
        metadataVariables["YCalibRef"] = findAttribute(hdf5FileIn, "YCalibRef").replace("CalibrationTime=b'","").replace("'","")
        metadataVariables["YErrorRef"] = metadataVariables["YCalibRef"] #calib ref is better

        #no subdomains in UVIS
        if channel_obs in ["so_occultation", "lno_occultation", "lno_nadir"]:
            metadataVariables["NSubdomains"] = findAttribute(hdf5FileIn, "NSubdomains")



        #Description is too long. Cut out extra info.
        if not checkIfFullscan(hdf5_basename):
            metadataVariables["Desc"] = hdf5FileIn.attrs["Desc"].split("#")[1].split("&")[0].strip()
        else:
            metadataVariables["Desc"] = "FULLSCAN"
    
    
    
        """add extra variables to dictionary, directly translating hdf5 rows to psa metadata variables"""
        for psaFieldName in psaFieldNames:
            
            if channel_obs in ["lno_nadir", "uvis_nadir"]:
                dictionary = {
                    "lon":["surface longitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lon"],
                    "lat":["surface latitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lat"],
                    "lst":["surface local solar time in hours at centre of field of view", ["f","l"], "hours", "Geometry/PointN/LST"],
                
                    "sun_sza":["surface solar zenith angle", ["f","l","m","x"], "deg", "Geometry/PointN/SunSZA"],
                    "incidence_angle":["surface solar incidence angle", ["f","l","m","x"], "deg", "Geometry/PointN/IncidenceAngle"],
                    "emission_angle":["surface emission angle", ["f","l","m","x"], "deg", "Geometry/PointN/EmissionAngle"],
                    "phase_angle":["surface phase angle", ["f","l","m","x"], "deg", "Geometry/PointN/PhaseAngle"],
                
                
                    "sub_obs_lon":["sub-satellite longitude", ["f","l","m","x"], "deg", "Geometry/SubObsLon"],
                    "sub_obs_lat":["sub-satellite latitude", ["f","l","m","x"], "deg", "Geometry/SubObsLat"],
                
                    "lsubs":["planetocentric longitude Ls", ["f","l"], "deg", "Geometry/LSubS"],
                
                    "sub_sol_lon":["sub-solar longitude", ["f","l"], "deg", "Geometry/SubSolLon"],
                    "sub_sol_lat":["sub-solar latitude", ["f","l"], "deg", "Geometry/SubSolLat"],
                        }
            elif channel_obs in ["so_occultation", "uvis_occultation"]:
                dictionary = {
                    "lon":["tangent point longitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lon"],
                    "lat":["tangent point latitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lat"],
                    "lst":["tangent point local solar time in hours at centre of field of view", ["f","l"], "hours", "Geometry/PointN/LST"],
                    "alt":["tangent altitude at centre of field of view", ["f","l"], "deg", "Geometry/PointN/TangentAlt"],
                    "slant_path_dist":["distance between satellite and tangent point", ["f","l"], "deg", "Geometry/PointN/SlantPathDist"],
                
                    "sub_obs_lon":["sub-satellite longitude", ["f","l","m","x"], "deg", "Geometry/SubObsLon"],
                    "sub_obs_lat":["sub-satellite latitude", ["f","l","m","x"], "deg", "Geometry/SubObsLat"],
                
                    "lsubs":["planetocentric longitude Ls", ["f","l"], "deg", "Geometry/LSubS"],
                
                    "sub_sol_lon":["sub-solar longitude", ["f","l"], "deg", "Geometry/SubSolLon"],
                    "sub_sol_lat":["sub-solar latitude", ["f","l"], "deg", "Geometry/SubSolLat"],


                    "pointing_deviation":["pointing deviation from centre of solar disk", ["m","x"], "deg", "Geometry/FOVSunCentreAngle"],
                        }
    
    
            geom_times = {"f":["first","First"], "l":["last","Last"], "m":["min","Minimum"], "x":["max","Maximum"]}
    
            for geom_type in dictionary.keys():
                
                hdf5_field_name = dictionary[geom_type][3]
                
                #write xml
                for geom_letter in dictionary[geom_type][1]:
                    geom_time_short = geom_times[geom_letter][0]
        
                #then point data
                if "PointN" in dictionary[geom_type][3]:
                    hdf5_field_point_names = [
#                            dictionary[geom_type][3].replace("PointN","Point%i" %point) for point in range(nPoints)
                            dictionary[geom_type][3].replace("PointN","Point0") #just use FOV centre Point0 for metadata
                            ]
                else:
                    hdf5_field_point_names = [hdf5_field_name]
        
        
                for hdf5_field_point_name in hdf5_field_point_names: #only Point0
                    for geom_letter in dictionary[geom_type][1]:
                        geom_time_short = geom_times[geom_letter][0]


                        if geom_letter == "f":
                            metadataVariables[f"{geom_time_short}_{geom_type}"] = "%0.3f" %hdf5FileIn[hdf5_field_point_name][0,0]
                        if geom_letter == "l":
                            metadataVariables[f"{geom_time_short}_{geom_type}"] = "%0.3f" %hdf5FileIn[hdf5_field_point_name][-1,-1]
                        if geom_letter == "m":
                            # get minimum of data that isn't -999.0
                            geom_data = np.ndarray.flatten(hdf5FileIn[hdf5_field_point_name][...])
                            metadataVariables[f"{geom_time_short}_{geom_type}"] = "%0.3f" %np.min(geom_data[geom_data>-998.0])
                        if geom_letter == "x":
                            metadataVariables[f"{geom_time_short}_{geom_type}"] = "%0.3f" %np.max(hdf5FileIn[hdf5_field_point_name][...])
    
    
    
            if psaFieldName in ["AOTFFrequency",
                "DiffractionOrder",
                "NumberOfAccumulations",
                "IntegrationTime",

                "AcquisitionMode",
                "BiasAverage",
                "DarkAverage",
                "ScienceAverage",
                "VStart",
                "VEnd",
                "HStart",
                "HEnd",
                "StartDelay",
                "AcquisitionDelay",
                "NumberOfAcquisitions",
                "NumberOfFlushes",
                "DarkToObservationSteps",
                "ObservationToDarkSteps",
                "HorizontalAndCombinedBinningSize"]:
                metadataVariables[psaFieldName] = hdf5FileIn[psa_mappings[psaFieldName]][...][0] #just take first value

    
            elif psaFieldName in ["SpectralResolution"]: #array not a vector
                metadataVariables[psaFieldName] = "%0.5f" %hdf5FileIn[psa_mappings[psaFieldName]][...][0] #just take first value
    
    
            elif psaFieldName == "InstrumentTemperature":
                metadataVariables[psaFieldName] = instrumentTemperature
    
    
            elif psaFieldName == "DetectorTemperature":
                metadataVariables[psaFieldName] = detectorTemperature
    
            elif psaFieldName == "ObservationType":
                metadataVariables[psaFieldName] = observationType
            
            #default mappings e.g. uvis bad pixel masking always 1
            elif psaFieldName in default_flags.keys():
                metadataVariables[psaFieldName] = default_flags[psaFieldName]

            elif "QualityFlag" in psa_mappings[psaFieldName]:
                if psa_mappings[psaFieldName] in hdf5FileIn: #check if flag exists in file
                    if int(hdf5FileIn[psa_mappings[psaFieldName]][...]) > 0: #catch non boolean values - set to 1
                        metadataVariables[psaFieldName] = 1
                    else:
                        metadataVariables[psaFieldName] = 0
                        
                else:
                    metadataVariables[psaFieldName] = 0 #set unused flags to zero for the time being
                    logger.info("%s: %s not found --> setting to 0", hdf5_basename, psa_mappings[psaFieldName])
                    
            else:
                metadataVariables[psaFieldName] = hdf5FileIn[psa_mappings[psaFieldName]][...]
    
        
        
        #Name, Type, Length, Unit (optional), Format (optional), Description, Data
        tableVariables = [
        ["ObservationDatetimeStart","ASCII_Date_Time_YMD_UTC",28,"","-27s","","UTC start datetime of measurement",psaMeasurementStartTimes],
        ["ObservationDatetimeEnd","ASCII_Date_Time_YMD_UTC",28,"","-27s","","UTC end datetime of measurement",psaMeasurementEndTimes],
        ]
        
        if channel_obs in ["so_occultation", "lno_occultation", "lno_nadir"]:
            
            tableVariables.extend([
                ["AOTFFrequency","ASCII_Real",11,"kilohertz","10.2f","","AOTF frequency used for this spectrum",hdf5FileIn["Channel/AOTFFrequency"][:]],
                ["BinTop","ASCII_Integer",4,"","3i","","Detector starting row number",hdf5FileIn["Channel/WindowTop"][:]],
                ["BinHeight","ASCII_Integer",4,"","3i","","Detector row height",hdf5FileIn["Channel/WindowHeight"][:]],
                ["BinStart","ASCII_Integer",4,"","3i","","Detector start row number",hdf5FileIn["Science/Bins"][:,0]],
                ["BinEnd","ASCII_Integer",4,"","3i","","Detector end row number",hdf5FileIn["Science/Bins"][:,1]],
                ["DiffractionOrder","ASCII_Integer",4,"","3i","","Measured diffraction order",hdf5FileIn["Channel/DiffractionOrder"][:]],
#                ["Exponent","ASCII_Integer",4,"","3i","","Exponent value for each spectrum",hdf5FileIn["Channel/Exponent"][:]],
                ["InstrumentTemperature","ASCII_Real",13,"celsius","12.5e","","Temperature of instrument",dummyGeometry1Column+instrumentTemperature],#hdf5FileIn["Housekeeping/AOTF_TEMP_LNO"]],
            ])

        tableVariables.extend([
            ["DetectorTemperature","ASCII_Real",13,"kelvin","12.5e","","Temperature of detector during measurement",dummyGeometry1Column+detectorTemperature],#hdf5FileIn["Housekeeping/FPA1_FULL_SCALE_TEMP_LNO"]],
            ["YValidFlag","ASCII_Boolean",2,"","0i","","1 = spectrum ok, 0 = error in spectrum",hdf5FileIn["Science/YValidFlag"][:]],
    
            ["StartObsAlt","ASCII_Real",9,"kilometres","8.2f","","Starting altitude of observer above centre of Mars",hdf5FileIn["Geometry/ObsAlt"][:,0]],
            ["EndObsAlt","ASCII_Real",9,"kilometres","8.2f","","Ending altitude of observer above centre of Mars",hdf5FileIn["Geometry/ObsAlt"][:,1]],
        #    ["TiltAngleStart","ASCII_Real",9,"deg","","Starting angle between surface normal and long edge of slit (occultation) or direction of movement and long edge of slit (nadir)",hdf5FileIn["Geometry/TiltAngle"][:,0]],
        #    ["TiltAngleEnd","ASCII_Real",9,"deg","","Ending angle between surface normal and long edge of slit (occultation) or direction of movement and long edge of slit (nadir)",hdf5FileIn["Geometry/TiltAngle"][:,1]],
            ["StartSubObsLon","ASCII_Real",9,"deg","8.3f","","Starting sub-satellite longitude",hdf5FileIn["Geometry/SubObsLon"][:,0]],
            ["EndSubObsLon","ASCII_Real",9,"deg","8.3f","","Ending sub-satellite longitude",hdf5FileIn["Geometry/SubObsLon"][:,1]],
            ["StartSubObsLat","ASCII_Real",9,"deg","8.3f","","Starting sub-satellite latitude",hdf5FileIn["Geometry/SubObsLat"][:,0]],
            ["EndSubObsLat","ASCII_Real",9,"deg","8.3f","","Ending sub-satellite latitude",hdf5FileIn["Geometry/SubObsLat"][:,1]],
            ["StartLSubS","ASCII_Real",9,"deg","8.4f","","Starting planetocentric longitude Ls",hdf5FileIn["Geometry/LSubS"][:,0]],
            ["EndLSubS","ASCII_Real",9,"deg","8.4f","","Ending planetocentric longitude Ls",hdf5FileIn["Geometry/LSubS"][:,1]],
            ["StartSubSolLon","ASCII_Real",9,"deg","8.3f","","Starting sub-solar longitude",hdf5FileIn["Geometry/SubSolLon"][:,0]],
            ["EndSubSolLon","ASCII_Real",9,"deg","8.3f","","Ending sub-solar longitude",hdf5FileIn["Geometry/SubSolLon"][:,1]],
            ["StartSubSolLat","ASCII_Real",9,"deg","8.3f","","Starting sub-solar latitude",hdf5FileIn["Geometry/SubSolLat"][:,0]],
            ["EndSubSolLat","ASCII_Real",9,"deg","8.3f","","Ending sub-solar latitude",hdf5FileIn["Geometry/SubSolLat"][:,1]],
        #    ["StartDistToSun","ASCII_Real",9,"au","","Starting distance from observer to Sun",hdf5FileIn["Geometry/DistToSun"][:,0]],
        #    ["EndDistToSun","ASCII_Real",9,"au","","Ending distance from observer to Sun",hdf5FileIn["Geometry/DistToSun"][:,1]],
        #    ["StartSpdObsSun","ASCII_Real",9,"kilometres per second","","Starting speed of observer (TGO) w.r.t. Sun",hdf5FileIn["Geometry/SpdObsSun"][:,0]],
        #    ["EndSpdObsSun","ASCII_Real",9,"kilometres per second","","Ending speed of observer (TGO) w.r.t. Sun",hdf5FileIn["Geometry/SpdObsSun"][:,1]],
        #    ["StartSpdTargetSun","ASCII_Real",9,"kilometres per second","","Starting speed of target (Mars) w.r.t. Sun",hdf5FileIn["Geometry/SpdTargetSun"][:,0]],
        #    ["EndSpdTargetSun","ASCII_Real",9,"kilometres per second","","Ending speed of target (Mars) w.r.t. Sun",hdf5FileIn["Geometry/SpdTargetSun"][:,1]],
        #    ["BetaAngleStart","ASCII_Real",9,"deg","","Starting beta angle of satellite orbit",dummyGeometry2Columns[:,0]+NA_VALUE],
        #    ["BetaAngleEnd","ASCII_Real",9,"deg","","Ending beta angle of satellite orbit",dummyGeometry2Columns[:,1]+NA_VALUE],
        ])

        if channel_obs in ["so_occultation", "lno_occultation", "uvis_occultation"]:

            tableVariables.extend([
                ["StartPointingDeviation","ASCII_Real",9,"arcmin","8.3f","","Starting pointing deviation from centre of solar disk",hdf5FileIn["Geometry/FOVSunCentreAngle"][:,0]],
                ["EndPointingDeviation","ASCII_Real",9,"arcmin","8.3f","","Ending pointing deviation from centre of solar disk",hdf5FileIn["Geometry/FOVSunCentreAngle"][:,1]],
            ])
    
    
    
        for point in range(nPoints):
            pointx = hdf5FileIn["Geometry/Point%i/PointXY" %point][...][0][0]
            pointy = hdf5FileIn["Geometry/Point%i/PointXY" %point][...][0][1]
            #name, type, length, units, _, description, values
            if channel_obs in ["lno_nadir", "uvis_nadir"]:
                extraTableColumns = [
                    ["PointX%i" %point,"ASCII_Real",9,"","8.3f","","Relative X position of point w.r.t. bin or slit, where -1 to 1 define edges",dummyGeometry1Column+pointx],
                    ["PointY%i" %point,"ASCII_Real",9,"","8.3f","","Relative Y position of point w.r.t. bin or slit, where -1 to 1 define edges",dummyGeometry1Column+pointy],
                    ["LonStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting surface longitude (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lon" %point][:,0]],
                    ["LonEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending surface longitude (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lon" %point][:,1]],
                    ["LatStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting surface latitude (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lat" %point][:,0]],
                    ["LatEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending surface latitude (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lat" %point][:,1]],
                    ["LSTStart%i" %point,"ASCII_Real",9,"hours","8.3f","-999.0","Starting local solar time of surface point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/LST" %point][:,0]],
                    ["LSTEnd%i" %point,"ASCII_Real",9,"hours","8.3f","-999.0","Ending local solar time of surface point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/LST" %point][:,1]],
    #                    ["LOSAngleStart%i" %point,"ASCII_Real",9,"hours","","Starting LOS angle (180 - emission angle for nadir), angle between planet centre and point (occ) for this point",hdf5FileIn["Geometry/Point%i/LOSAngle" %point][:,0]],
    #                    ["LOSAngleEnd%i" %point,"ASCII_Real",9,"hours","","Ending LOS angle (180 - emission angle for nadir), angle between planet centre and point (occ) for this point",hdf5FileIn["Geometry/Point%i/LOSAngle" %point][:,1]],
    #                ["TangentAltStart%i" %point,"ASCII_Real",9,"hours","","Starting tangent point altitude for this point (0 for nadir)",hdf5FileIn["Geometry/Point%i/TangentAlt" %point][:,0]],
    #                ["TangentAltEnd%i" %point,"ASCII_Real",9,"hours","","Ending tangent point altitude for this point (0 for nadir)",hdf5FileIn["Geometry/Point%i/TangentAlt" %point][:,1]],
#                    ["SolarZenithAngleStart%i" %point,"ASCII_Real",9,"hours","","Starting solar zenith angle for this point",hdf5FileIn["Geometry/Point%i/SunSZA" %point][:,0]],
#                    ["SolarZenithAngleEnd%i" %point,"ASCII_Real",9,"hours","","Ending solar zenith angle for this point",hdf5FileIn["Geometry/Point%i/SunSZA" %point][:,1]],
    #                    ["SunAziStart%i" %point,"ASCII_Real",9,"hours","","Starting sun azimuthal angle for this point",hdf5FileIn["Geometry/Point%i/SunAzi" %point][:,0]],
    #                    ["SunAziEnd%i" %point,"ASCII_Real",9,"hours","","Ending sun azimuthal angle for this point",hdf5FileIn["Geometry/Point%i/SunAzi" %point][:,1]],
                    ["IncidenceAngleStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting incidence angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/IncidenceAngle" %point][:,0]],
                    ["IncidenceAngleEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending incidence angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/IncidenceAngle" %point][:,1]],
                    ["EmissionAngleStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting emission angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/EmissionAngle" %point][:,0]],
                    ["EmissionAngleEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending emission angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/EmissionAngle" %point][:,1]],
                    ["PhaseAngleStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting phase angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/PhaseAngle" %point][:,0]],
                    ["PhaseAngleEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending phase angle for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/PhaseAngle" %point][:,1]],

                    ["SurfaceAltAreoidStart%i" %point,"ASCII_Real",9,"kilometres","8.3f","-999.0","Starting height of the DSK surface above the areoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SurfaceAltAreoid" %point][:,0]],
                    ["SurfaceAltAreoidEnd%i" %point,"ASCII_Real",9,"kilometres","8.3f","-999.0","Ending height of the DSK surface above the areoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SurfaceAltAreoid" %point][:,1]],
                    ["SurfaceRadiusStart%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Starting DSK surface radius for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SurfaceRadius" %point][:,0]],
                    ["SurfaceRadiusEnd%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Ending DSK surface radius for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SurfaceRadius" %point][:,1]],
                                ]
    
            elif channel_obs in ["so_occultation", "lno_occultation", "uvis_occultation"]:
                extraTableColumns = [
                    ["PointX%i" %point,"ASCII_Real",9,"","8.3f","","Relative X position of point w.r.t. bin or slit, where -1 to 1 define edges",dummyGeometry1Column+pointx],
                    ["PointY%i" %point,"ASCII_Real",9,"","8.3f","","Relative Y position of point w.r.t. bin or slit, where -1 to 1 define edges",dummyGeometry1Column+pointy],
                    ["LonStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting tangent point longitude for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lon" %point][:,0]],
                    ["LonEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending tangent point longitude for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lon" %point][:,1]],
                    ["LatStart%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Starting tangent point latitude for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lat" %point][:,0]],
                    ["LatEnd%i" %point,"ASCII_Real",9,"deg","8.3f","-999.0","Ending tangent point latitude for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/Lat" %point][:,1]],
                    ["LSTStart%i" %point,"ASCII_Real",9,"hours","8.3f","-999.0","Starting local solar time of tangent point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/LST" %point][:,0]],
                    ["LSTEnd%i" %point,"ASCII_Real",9,"hours","8.3f","-999.0","Ending local solar time of tangent point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/LST" %point][:,1]],
    #                    ["LOSAngleStart%i" %point,"ASCII_Real",9,"hours","","Starting LOS angle (180 - emission angle for nadir), angle between planet centre and point (occ) for this point",hdf5FileIn["Geometry/Point%i/LOSAngle" %point][:,0]],
    #                    ["LOSAngleEnd%i" %point,"ASCII_Real",9,"hours","","Ending LOS angle (180 - emission angle for nadir), angle between planet centre and point (occ) for this point",hdf5FileIn["Geometry/Point%i/LOSAngle" %point][:,1]],
                    ["TangentAltEllipsoidStart%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Starting tangent point altitude above the Mars ellipsoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAlt" %point][:,0]],
                    ["TangentAltEllipsoidEnd%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Ending tangent point altitude above the Mars ellipsoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAlt" %point][:,1]],
                    ["TangentAltAreoidStart%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Starting tangent point altitude above the Mars areoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAltAreoid" %point][:,0]],
                    ["TangentAltAreoidEnd%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Ending tangent point altitude above the Mars areoid for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAltAreoid" %point][:,1]],
                    ["TangentAltSurfaceStart%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Starting tangent point altitude above the Mars DSK surface for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAltSurface" %point][:,0]],
                    ["TangentAltSurfaceEnd%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Ending tangent point altitude above the Mars DSK surface for this point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/TangentAltSurface" %point][:,1]],
                    ["SlantPathDistanceStart%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Starting distance between satellite and tangent point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SlantPathDist" %point][:,0]],
                    ["SlantPathDistanceEnd%i" %point,"ASCII_Real",9,"kilometres","8.2f","-999.0","Ending distance between satellite and tangent point (-999 = invalid value)",hdf5FileIn["Geometry/Point%i/SlantPathDist" %point][:,1]],
                                ]
    
            for extra_table_column in extraTableColumns:
                tableVariables.append(extra_table_column)
    
        """write X, Y and YError arrays to file"""
        #first get units
        # outputType, outputUnits, spectralUnitsDesc, spectralUnits = getYUnits(channel_obs)
        
        
        
        groups_with_units = {"group_location":"byte", "group_length":"byte", "field_location":"byte", "field_length":"byte"}

        n_pixels = hdf5FileIn["Science/X"][...].shape[-1]
        starting_byte = sum([i[2] for i in tableVariables if i[0] != ""])+1
        
        d = get_dataset_info(channel_obs)

        if "y2" in d.keys(): #if two y datasets
            group_length = [n_pixels*9, n_pixels*13, n_pixels*13, n_pixels*13, n_pixels*2]
            group_location = [starting_byte, \
                              starting_byte + group_length[0], \
                              starting_byte + group_length[0] + group_length[1], \
                              starting_byte + group_length[0] + group_length[1] + group_length[2], \
                              starting_byte + group_length[0] + group_length[1] + group_length[2] + group_length[3]]
        else:
            group_length = [n_pixels*9, n_pixels*13, n_pixels*13, n_pixels*2]
            group_location = [starting_byte, \
                              starting_byte + group_length[0], \
                              starting_byte + group_length[0] + group_length[1], \
                              starting_byte + group_length[0] + group_length[1] + group_length[2]]

            
        
        logger.info("Adding datasets for product %s", paths["data_lid"])
        
        group_variables = []
        
        if d["x"]["valid"]:
            group_variables.append(
                {"name":d["x"]["unit_desc"], "group_number":"1", "repetitions":str(n_pixels), "fields":"1", \
                 "groups":"0", "group_location":str(group_location[0]), "group_length":str(group_length[0]), "Field_Character":\
                     {"name":"Pixel %s" %d["x"]["unit_desc"].lower(), "field_location":"1", "data_type":"ASCII_Real", "field_length":"9", \
                      "unit":d["x"]["units"], "description":"%s at the centre of the pixel" %d["x"]["unit_desc"]}
                }
            )
                
            """add data only for tab file output"""
            #X can be in the form Nspectra x Npixels or just Npixels
            array = hdf5FileIn[d["x"]["path"]][...]
            if array.ndim == 2:
                for columnNumber in range(array.shape[-1]):
                    tableVariables.append(["","",9,"","8.3f","","", array[:, columnNumber]])
            else:
                for columnNumber in range(array.shape[-1]):
                    tableVariables.append(["","",9,"","8.3f","","", dummyGeometry1Column + array[columnNumber]])

                    



        if d["y"]["valid"]:
            group_variables.append(
                {"name":d["y"]["unit_desc"], "group_number":"1", "repetitions":str(n_pixels), "fields":"1", \
                 "groups":"0", "group_location":str(group_location[1]), "group_length":str(group_length[1]), "Field_Character":\
                     {"name":"Pixel %s" %d["y"]["unit_desc"].lower(), "field_location":"1", "data_type":"ASCII_Real", "field_length":"13", \
                      "description":"Pixel %s" %d["y"]["unit_desc"].lower()}
                }
            )

            #add y data
            array = hdf5FileIn[d["y"]["path"]][...]
            array[np.isnan(array)] = -999.0 #replace all nans with negative values
            for columnNumber in range(array.shape[1]):
                tableVariables.append(["","",13,"","12.5e","","", array[:, columnNumber]])


        if "y2" in d.keys():
            if d["y2"]["valid"]:
                group_variables.append(
                    {"name":d["y2"]["unit_desc"], "group_number":"1", "repetitions":str(n_pixels), "fields":"1", \
                     "groups":"0", "group_location":str(group_location[1]), "group_length":str(group_length[1]), "Field_Character":\
                         {"name":"Pixel %s" %d["y2"]["unit_desc"].lower(), "field_location":"1", "data_type":"ASCII_Real", "field_length":"13", \
                          "description":"Pixel %s" %d["y2"]["unit_desc"].lower()}
                    }
                )

                #add second y dataset (optional)
                array = hdf5FileIn[d["y2"]["path"]][...]
                array[np.isnan(array)] = -999.0 #replace all nans with negative values
                for columnNumber in range(array.shape[1]):
                    tableVariables.append(["","",13,"","12.5e","","", array[:, columnNumber]])


        if d["error"]["valid"]:
            group_variables.append(
                {"name":d["error"]["unit_desc"], "group_number":"1", "repetitions":str(n_pixels), "fields":"1", \
                 "groups":"0", "group_location":str(group_location[2]), "group_length":str(group_length[2]), "Field_Character":\
                     {"name":"Pixel %s" %d["error"]["unit_desc"].lower(), "field_location":"1", "data_type":"ASCII_Real", "field_length":"13", \
                      "description":"Pixel %s" %d["error"]["unit_desc"].lower()}
                 }
            )

            #error is optional
            array = hdf5FileIn[d["error"]["path"]][...]
            array[np.isnan(array)] = -999.0 #replace all nans with negative values
            for columnNumber in range(array.shape[1]):
                tableVariables.append(["","",13,"","12.5e","","", array[:, columnNumber]])



        if d["mask"]["valid"]:
            group_variables.append(
                {"name":d["mask"]["unit_desc"], "group_number":"1", "repetitions":str(n_pixels), "fields":"1", \
                 "groups":"0", "group_location":str(group_location[3]), "group_length":str(group_length[3]), "Field_Character":\
                     {"name":"Pixel %s" %d["mask"]["unit_desc"].lower(), "field_location":"1", "data_type":"ASCII_Boolean", "field_length":"2", \
                      "description":"Mask on pixel (0 = OK, 1 = pixel to be rejected)"}
                }
            )

            #mask is optional (UVIS only)
            array = hdf5FileIn[d["mask"]["path"]][...]
            for columnNumber in range(array.shape[1]):
                tableVariables.append(["","",2,"","1i","","", array[:, columnNumber]])

    
    
    
            
        formats = []
        recordLength = 2 #account for CRLF on each line
        for tableVariable in tableVariables:
            recordLength += tableVariable[2]
            fmt = "%#" + "%s" %(tableVariable[4])
                
            formats.append(fmt)
            
        metadataVariables["number_of_fields"] = len([i for i in tableVariables if i[0] != ""]),
        metadataVariables["number_of_groups"] = len(group_variables),
        metadataVariables["record_length"] = recordLength
        metadataVariables["file_size"] = recordLength * metadataVariables["number_of_records"] #total file size
    
    #        for metadataVariable in metadataVariables.keys():
    #            print(metadataVariable)
    #        stop()
              
    
        """write xml data from predefined dictionary"""
        subElements = []
        index = -1
        
        for level,name,etext,attribute in psaTemplateElementList:
            if level==0:
                index += 1
                level0 = index
                # if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
    
                subElements.append(etree.Element("{" + XMLNS + "}Product_Observational", attrib={"{" + XMLNS_XSI + "}schemaLocation" : SCHEMA_LOCATION}, \
                             nsmap={None:XMLNS, "geom":XMLNS_GEOM, "psa":XMLNS_PSA, "xsi":XMLNS_XSI, "em16_tgo_nmd":XMLNS_EM16_TGO_NMD}))
                psaCalDoc = etree.ElementTree(subElements[level0])
                
                for MODEL in MODEL_HREF:
                    psaCalDoc.getroot().addprevious(etree.ProcessingInstruction("xml-model", "href=%s schematypens=%s" %(MODEL,MODEL_SCHEMATRON)))

            elif level==1:
                index += 1
                level1 = index
#                if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level0], name))
            elif level==2:
                index += 1
                level2 = index
#                if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level1], name))
            elif level==3:
                index += 1
                level3 = index
    #            if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level2], name))
            elif level==4:
                index += 1
                level4 = index
    #            if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level3], name))
            elif level==5:
                index += 1
                level5 = index
    #            if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level4], name))
            elif level==6:
                index += 1
    #            level6 = index
    #            if PRINT_FLAG: print("Making %s on level %i, index=%i" %(name,level,index))
                subElements.append(etree.SubElement(subElements[level5], name))
                
            if attribute != {}:
                """assume there can only be one key!"""
                attributeName = attribute.keys()[0] 
                attributeValue = attribute[attributeName]
                
                subElements[index].set(attributeName, attributeValue)
    
            if etext != "":
    #            if PRINT_FLAG: print("Writing %s to element no %i" %(etext,index))
                lineToWrite = etext
                if lineToWrite[0]=="%": #find variables in template file
                    variableName = lineToWrite.replace("%","") #get variable name
                    
                    if variableName in metadataVariables.keys():
                        valueToWrite = metadataVariables[variableName]
                        subElements[index].text = "%s" %valueToWrite
                    else:
                        print("Error: Variable %s not found in list" %variableName)
                    
                else:
                    subElements[index].text = lineToWrite
        
    
        """write table data from predefined list"""
        fieldNumber = 1
        fieldLocation = 1 #bytes start at 1 in xml
        
        allFieldValues = []
    
        for fieldName,fieldDtype,fieldLength,fieldUnit,fieldFormat,fieldNan,fieldDescription,fieldValues in tableVariables: #loop through columns in tab file
            allFieldValues.append(fieldValues)

            if fieldName != "":
                subElements.append(etree.SubElement(subElements[level3], "Field_Character"))
                index += 1
                level4 = index
                
                subElements.append(etree.SubElement(subElements[level4], "name"))
                index += 1
                subElements[index].text = fieldName
                subElements.append(etree.SubElement(subElements[level4], "field_number"))
                index += 1
                subElements[index].text = "%s" %fieldNumber
                fieldNumber += 1
                subElements.append(etree.SubElement(subElements[level4], "field_location"))
                index += 1
                subElements[index].set("unit", "byte")
                subElements[index].text = "%s" %fieldLocation
                subElements.append(etree.SubElement(subElements[level4], "data_type"))
                index += 1
                subElements[index].text = fieldDtype
                subElements.append(etree.SubElement(subElements[level4], "field_length"))
                index += 1
                subElements[index].text = "%s" %fieldLength
                subElements[index].set("unit", "byte")
                fieldLocation += fieldLength
                if fieldUnit != "":
                    subElements.append(etree.SubElement(subElements[level4], "unit"))
                    index += 1
                    subElements[index].text = fieldUnit
                subElements.append(etree.SubElement(subElements[level4], "description"))
                index += 1
                subElements[index].text = fieldDescription
                if fieldNan != "":
                    subElements.append(etree.SubElement(subElements[level4], "Special_Constants"))
                    index += 1
                    subElements.append(etree.SubElement(subElements[index], "error_constant"))
                    index += 1
                    subElements[index].text = fieldNan
            
            


        """new version to make group fields"""
        for group_field in group_variables:
            subElements.append(etree.SubElement(subElements[level3], "Group_Field_Character"))
            index += 1
            level4 = index
            
            for group_field_name in ["name", "group_number", "repetitions", "fields", "groups", "group_location", "group_length"]:
                if group_field_name in group_field.keys():
                    subElements.append(etree.SubElement(subElements[level4], group_field_name))
                    index += 1
                    subElements[index].text = group_field[group_field_name]
                    if group_field_name in groups_with_units.keys():
                        subElements[index].set("unit", groups_with_units[group_field_name])

            subElements.append(etree.SubElement(subElements[level4], "Field_Character"))
            index += 1
            level5 = index

            for field_name in ["name", "field_location", "data_type", "field_length", "unit", "description"]:
                if field_name in group_field["Field_Character"].keys():
                    subElements.append(etree.SubElement(subElements[level5], field_name))
                    index += 1
                    subElements[index].text = group_field["Field_Character"][field_name]
                    if field_name in groups_with_units.keys():
                        subElements[index].set("unit", groups_with_units[field_name])
                    

            
            
    
    
        psaXmlFileOut = etree.tostring(psaCalDoc, xml_declaration=True, pretty_print=True, encoding="utf-8").decode()

        psaTabFileOut = ""
        for rowIndex in range(len(allFieldValues[0])): #loop through rows in tab file
            lineToWrite=""
            for columnIndex,fieldColumn in enumerate(allFieldValues): #loop through columns in tab file appending values to line
                lineToWrite = lineToWrite + formats[columnIndex] %fieldColumn[rowIndex] + " "
            psaTabFileOut += lineToWrite+"\r\n" #write CRLF at end of each line

            
        #make browse label text
        browseXmlFileOut = makeBrowseXmlLabel(paths)
            
    
        #delete existing tmp/lid/ dir and make new one
        if os.path.isdir(paths["tmp_dir_path"]):
            shutil.rmtree(paths["tmp_dir_path"])
        os.makedirs(paths["tmp_dir_path"])
        
        #write to files, save to tmp/lid/ folder
        with open(paths["data_xml_tmp_path"], "w") as f:
            f.write(psaXmlFileOut)
        with open(paths["data_tab_tmp_path"], "w") as f:
            f.write(psaTabFileOut)
        with open(paths["brow_xml_tmp_path"], "w") as f:
            f.write(browseXmlFileOut)


        #save browse png, save to tmp/lid/ folder"""
        title = paths["data_lid"]
        brow_path = paths["brow_png_tmp_path"]
        
        x = hdf5FileIn[d["x"]["path"]][...]
        #if x is 2D, take 1 line only
        if x.ndim == 2:
            x = x[0, :]

        #only use y for browse product (ignore y2)
        y = hdf5FileIn[d["y"]["path"]][...]
        y[np.isnan(y)] = -999.0 #replace all nans with negative values

        if np.max(y) == -999.0:
            logger.warning("%s: No useable Y data found in file", hdf5_basename)
        
        else:
        
            if channel_obs in ["so_occultation", "lno_occultation"]:
                logger_out = plot_so_lno_occultation(channel_obs, hdf5FileIn, title, brow_path, x, y)
            if channel_obs in ["uvis_occultation"]:
                logger_out = plot_uvis_occultation(channel_obs, hdf5FileIn, title, brow_path, x, y)
            elif channel_obs in ["lno_nadir"]:
                logger_out = plot_lno_uvis_nadir(channel_obs, hdf5FileIn, title, brow_path, x, y)
            elif channel_obs in ["uvis_nadir"]:
                logger_out = plot_lno_uvis_nadir(channel_obs, hdf5FileIn, title, brow_path, x, y)
            
            if logger_out != "":
                logger.warning("%s: %s", hdf5_basename, logger_out)


        
        error = False
        
        #generate random number to decide if to validate or not
        validate = False
        random_number = np.random.rand()
        if VALIDATE_OUTPUT:
            validate = random_number < VALIDATION_RATIO

        
        #validate files
        if validate:
            
            data_error_count, output = validate_data(paths["data_xml_tmp_path"])
            brow_error_count, output = validate_data(paths["brow_xml_tmp_path"])
            
            if data_error_count == 0:
                logger.info("Data product %s successfully passed validation", paths["data_lid"])
            else:
                error = True
                logger.error("Product %s did not pass validation", paths["data_lid"])
                logger.error(output)
                
            
            if brow_error_count == 0:
                logger.info("Browse product %s successfully passed validation", paths["brow_lid"])
            else:
                error = True
                logger.error("Browse product %s did not pass validation", paths["brow_lid"])
                logger.error(output)
        else:
            logger.info("Not validating PSA or browse product for %s", paths["data_lid"])


        #make final directory in datastore
        if not os.path.isdir(paths["zip_final_dir_path"]):
            os.makedirs(paths["zip_final_dir_path"], exist_ok=True)

        #rename files -> append version
        if not error:
            time.sleep(1) #wait 1 second for file processes to finish
            os.rename(paths["data_xml_tmp_path"], paths["data_xml_tmp_ver_path"])
            os.rename(paths["data_tab_tmp_path"], paths["data_tab_tmp_ver_path"])
            os.rename(paths["brow_xml_tmp_path"], paths["brow_xml_tmp_ver_path"])
            os.rename(paths["brow_png_tmp_path"], paths["brow_png_tmp_ver_path"])
    
            shutil.make_archive(paths["zip_final_path"], 'zip', paths["tmp_dir_path"])
            shutil.rmtree(paths["tmp_dir_path"])


    hdf5FileIn.close()
    # return [] #return empty (PSA products don't get stored in db)

    
    
