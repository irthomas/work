# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:07:54 2021

@author: iant

PATHS AND FILENAMES
"""

# import logging
import os
# import re
#import sys
#import h5py
# import numpy as np
#import spiceypy as sp
from datetime import datetime, timedelta



from nomad_ops.config import PSA_DATA_CAL_FILE_DESTINATION, NOMAD_TMP_DIR


from nomad_ops.core.psa.l1p0a_to_psa.functions import checkIfFullscan
from nomad_ops.core.psa.l1p0a_to_psa.config import \
    PSA_FILENAME_FORMAT, PSA_VERSION

from nomad_ops.core.psa.l1p0a_to_psa.xml import PSA_LOGICAL_IDENTIFIER_PREFIX, BROWSE_LOGICAL_IDENTIFIER_PREFIX



def prepare_psa_tree(hdf5_basename):
    
    year = hdf5_basename[0:4] #get the date from the filename to find the file
    month = hdf5_basename[4:6]
    day = hdf5_basename[6:8]

    data_path = os.path.join(PSA_DATA_CAL_FILE_DESTINATION, year, month, day)  
    
    return data_path







def make_path_dict(channel_obs, observation_type_letter, hdf5_basename, hdf5FileIn):
    """Prepare filenames and directories"""
    """filename must be of the form nmd_cal_sc_so_20180421T000000-20211101T000000-so-a-e-121_1.0.xml"""


    observation_type_letter = observation_type_letter.lower() #lowercase if not already

    hdf5_basename_split = hdf5_basename.split("_")
    channel = channel_obs.split("_")[0]
    
    isFullscan = checkIfFullscan(hdf5_basename)
    
    if channel_obs in ["so_occultation", "lno_occultation", "lno_nadir"]:
        if isFullscan:
            altitudeLetter = "a"
        else:
            altitudeLetter = hdf5_basename_split[4].lower()
            diffractionOrderText = hdf5_basename_split[6]
    tcDuration = int(hdf5FileIn["Telecommand20/%sStartTime" %channel.upper()][...]) + int(hdf5FileIn["Telecommand20/%sDurationTime" %channel.upper()][...])

    psaFilenameStartTime = hdf5_basename[0:8]+"T"+hdf5_basename[9:15] #get start time from filename
    zipFilenameTime = hdf5_basename[0:8]+"-"+hdf5_basename[9:15]+"00" #get zip file time from filename

    #convert to datetime, add tcDuration, convert back to text
    psaFilenameEndTime = datetime.strftime(datetime.strptime(psaFilenameStartTime, PSA_FILENAME_FORMAT) + timedelta(seconds=tcDuration), PSA_FILENAME_FORMAT)
    
    if channel_obs in ["so_occultation"]:
        if isFullscan:
            psaFilename = "nmd_cal_sc_so_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+altitudeLetter+"-"+observation_type_letter
            zipFilename = "nmd_cal_sc_so_"+zipFilenameTime+"-"+altitudeLetter+"-"+observation_type_letter
            browseFilename = "nmd_cal_sc_browse_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+altitudeLetter+"-"+observation_type_letter+"-so"
        else:
            psaFilename = "nmd_cal_sc_so_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+altitudeLetter+"-"+observation_type_letter+"-"+diffractionOrderText
            zipFilename = "nmd_cal_sc_so_"+zipFilenameTime+"-"+altitudeLetter+"-"+observation_type_letter+"-"+diffractionOrderText
            browseFilename = "nmd_cal_sc_browse_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+altitudeLetter+"-"+observation_type_letter+"-"+diffractionOrderText+"-so"

    elif channel_obs in ["lno_nadir", "lno_occultation"]:
        if isFullscan:
            psaFilename = "nmd_cal_sc_lno_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter
            zipFilename = "nmd_cal_sc_lno_"+zipFilenameTime+"-"+observation_type_letter
            browseFilename = "nmd_cal_sc_browse_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter+"-lno"
        else:
            psaFilename = "nmd_cal_sc_lno_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter+"-"+diffractionOrderText
            zipFilename = "nmd_cal_sc_lno_"+zipFilenameTime+"-"+observation_type_letter+"-"+diffractionOrderText
            browseFilename = "nmd_cal_sc_browse_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter+"-"+diffractionOrderText+"-lno"

    elif channel_obs in ["uvis_nadir", "uvis_occultation"]:
        psaFilename = "nmd_cal_sc_uvis_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter
        zipFilename = "nmd_cal_sc_uvis_"+zipFilenameTime+"-"+observation_type_letter
        browseFilename = "nmd_cal_sc_browse_"+psaFilenameStartTime+"-"+psaFilenameEndTime+"-"+observation_type_letter+"-uvis"

    
    zipDirpath = prepare_psa_tree(hdf5_basename)
    # psaFilepath = os.path.join(zipDirpath, psaFilename)
    zipFilepath = os.path.join(zipDirpath, zipFilename)

    tmp_dir_path = os.path.join(NOMAD_TMP_DIR, psaFilename)
    
    data_xml_tmp_path = os.path.join(tmp_dir_path, psaFilename + ".xml")
    data_tab_tmp_path = os.path.join(tmp_dir_path, psaFilename + ".tab")
    brow_xml_tmp_path = os.path.join(tmp_dir_path, browseFilename + ".xml")
    brow_png_tmp_path = os.path.join(tmp_dir_path, browseFilename + ".png")

    data_xml_tmp_ver_path = os.path.join(tmp_dir_path, psaFilename + "_"+PSA_VERSION + ".xml")
    data_tab_tmp_ver_path = os.path.join(tmp_dir_path, psaFilename + "_"+PSA_VERSION + ".tab")
    brow_xml_tmp_ver_path = os.path.join(tmp_dir_path, browseFilename + "_"+PSA_VERSION + ".xml")
    brow_png_tmp_ver_path = os.path.join(tmp_dir_path, browseFilename + "_"+PSA_VERSION + ".png")

    
    # tabFilename = psaFilename + ".tab"

    #use lowercase file base name for logical identifier
    #e.g. 'urn:esa:psa:em16_tgo_nmd:data_calibrated:20180421_183121_1p0a_lno_1_d_196'
    psaLogicalIdentifier = (PSA_LOGICAL_IDENTIFIER_PREFIX + psaFilename).lower()
    browseLogicalIdentifier = (BROWSE_LOGICAL_IDENTIFIER_PREFIX + browseFilename).lower()
    # psaLocalIdentifier = psaFilename + ".xml" #must start with a letter, not a number (apparently)
    # tabLocalIdentifier = psaFilename + ".tab" #must start with a letter, not a number (apparently)
    # browseLocalIdentifier = browseFilename + ".xml" #must start with a letter, not a number (apparently)
    # pngLocalIdentifier = browseFilename + ".png" #must start with a letter, not a number (apparently)


#    if isFullscan:
#        aotfLogicalIdentifier = "aotf_function_all_orders"
#    else:
#        diffractionOrder = hdf5FileIn["Channel/DiffractionOrder"][0]
#        aotfLogicalIdentifier = AOTF_FUNCTION_LOGICAL_IDENTIFIER_PREFIX + "nmd_so_aotf_function_order_%03i" %diffractionOrder
                
            
    return {
            "data_lid":psaFilename,
            "data_lid_ver":psaFilename+"_"+PSA_VERSION,
            "data_lid_full":psaLogicalIdentifier,
            "data_lid_start_time":psaFilenameStartTime,
            "data_lid_end_time":psaFilenameEndTime,

            "data_xml_filename":psaFilename + ".xml", #local identifier must start with a letter, not a number
            "data_tab_filename":psaFilename + ".tab", #local identifier must start with a letter, not a number

            "brow_lid":browseFilename,
            "brow_lid_ver":browseFilename+"_"+PSA_VERSION,
            "brow_lid_full":browseLogicalIdentifier,

            "brow_xml_filename":browseFilename + ".xml", #local identifier must start with a letter, not a number
            "brow_png_filename":browseFilename + ".png", #local identifier must start with a letter, not a number

            "tmp_dir_path":tmp_dir_path,
            "data_xml_tmp_path":data_xml_tmp_path,
            "data_tab_tmp_path":data_tab_tmp_path,
            "brow_xml_tmp_path":brow_xml_tmp_path,
            "brow_png_tmp_path":brow_png_tmp_path,

            "data_xml_tmp_ver_path":data_xml_tmp_ver_path,
            "data_tab_tmp_ver_path":data_tab_tmp_ver_path,
            "brow_xml_tmp_ver_path":brow_xml_tmp_ver_path,
            "brow_png_tmp_ver_path":brow_png_tmp_ver_path,
            
            
            "zip_final_dir_path":zipDirpath,
            "zip_final_path":zipFilepath+"_"+PSA_VERSION,

            # "data_lid_path":psaFilepath,
            # "tabFilename":tabFilename,
            # "zip_name":zipFilename+"_"+PSA_VERSION+".zip",
            # "zipDirpath":zipDirpath, #for uvis only


#            "aotfLogicalIdentifier":aotfLogicalIdentifier,
            # "psaLocalIdentifier":psaLocalIdentifier,
            # "tabLocalIdentifier":tabLocalIdentifier,
            # "browseLocalIdentifier":browseLocalIdentifier,
            # "pngLocalIdentifier":pngLocalIdentifier,


            }
