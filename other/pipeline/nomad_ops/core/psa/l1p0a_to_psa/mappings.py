# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:00:42 2021

@author: iant

HDF5 TO PSA DATASET NAME MAPPINGS
"""
import re
import numpy as np


import logging
logger = logging.getLogger( __name__ )

__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian.thomas@aeronomie.be"


def get_dataset_info(channel_obs):
    """list the primary and primary errors HDF5 datasets to put in file"""

    if channel_obs in ["so_occultation", "lno_occultation"]:
        info = {
            "x":    {"valid":True, "unit_desc":"Wavenumber", "units":"wavenumbers", "path":"Science/X"}, 
            "y":    {"valid":True, "unit_desc":"Transmittance", "units":"", "path":"Science/Y"},
            "error":{"valid":True, "unit_desc":"Transmittance error", "units":"", "path":"Science/YErrorNorm"},
            "mask": {"valid":False, "unit_desc":"", "units":"", "path":""}
        }

    elif channel_obs in ["uvis_occultation"]:
        info = {
            "x":    {"valid":True, "unit_desc":"Wavelength", "units":"nanometres", "path":"Science/X"}, 
            "y":    {"valid":True, "unit_desc":"Transmittance", "units":"", "path":"Science/YMean"},
            "error":{"valid":True, "unit_desc":"Transmittance error", "units":"", "path":"Science/YErrorMeanRandom"},
            "mask": {"valid":True, "unit_desc":"Mask", "units":"", "path":"Science/YMask"}
        }
        
    elif channel_obs in ["lno_nadir"]:
        info = {
            "x":    {"valid":True, "unit_desc":"Wavenumber", "units":"wavenumbers", "path":"Science/X"}, 
            "y":    {"valid":True, "unit_desc":"Reflectance factor", "units":"", "path":"Science/YReflectanceFactorFlat"},
            "y2":   {"valid":True, "unit_desc":"Reflectance factor (baseline removed)", "units":"", "path":"Science/YReflectanceFactorBaselineRemoved"},
            "error":{"valid":False, "unit_desc":"", "units":"", "path":""},
            "mask": {"valid":False, "unit_desc":"", "units":"", "path":""}
        }

    elif channel_obs in ["uvis_nadir"]:
        info = {
            "x":    {"valid":True, "unit_desc":"Wavelength", "units":"nanometres", "path":"Science/X"}, 
            "y":    {"valid":True, "unit_desc":"Radiance", "units":"W/m**2/sr/nm", "path":"Science/Y"},
            "error":{"valid":True, "unit_desc":"Radiance error", "units":"", "path":"Science/YError"},
            "mask": {"valid":True, "unit_desc":"Mask", "units":"", "path":"Science/YMask"}
        }

    else:
        logger.error("Error: channel_obs (%s) not recognised", channel_obs)
        
    return info




def get_mappings(channel_obs):

    #all channels all observations
    psa_mappings = {
    "LNOCopGeneral":"Telecommand20/LNOCopGeneral", \
    "LNOCopPrecooling":"Telecommand20/LNOCopPrecooling", \
    "LNOCopScience1":"Telecommand20/LNOCopScience1", \
    "LNOCopScience2":"Telecommand20/LNOCopScience2", \
    "LNODurationReference1":"Telecommand20/LNODurationReference1", \
    "LNODurationReference2":"Telecommand20/LNODurationReference2", \
    "LNODurationTime":"Telecommand20/LNODurationTime", \
    "LNOStartScience1":"Telecommand20/LNOStartScience1", \
    "LNOStartScience2":"Telecommand20/LNOStartScience2", \
    "LNOStartTime":"Telecommand20/LNOStartTime", \
    "SOCopGeneral":"Telecommand20/SOCopGeneral", \
    "SOCopPrecooling":"Telecommand20/SOCopPrecooling", \
    "SOCopScience1":"Telecommand20/SOCopScience1", \
    "SOCopScience2":"Telecommand20/SOCopScience2", \
    "SODurationReference1":"Telecommand20/SODurationReference1", \
    "SODurationReference2":"Telecommand20/SODurationReference2", \
    "SODurationTime":"Telecommand20/SODurationTime", \
    "SOStartScience1":"Telecommand20/SOStartScience1", \
    "SOStartScience2":"Telecommand20/SOStartScience2", \
    "SOStartTime":"Telecommand20/SOStartTime", \
    "UVISCopRow":"Telecommand20/UVISCopRow", \
    "UVISDurationTime":"Telecommand20/UVISDurationTime", \
    "UVISStartTime":"Telecommand20/UVISStartTime", \

    "HSKDisturbed":"QualityFlag/HSKDisturbed", \
    "HSKOutOfBounds":"QualityFlag/HSKOutOfBounds", \
    "BadPixelsHInterpolated":"QualityFlag/BadPixelsHInterpolated", \
    "BadPixelsVInterpolated":"QualityFlag/BadPixelsVInterpolated", \
    "BadPixelsMasked":"QualityFlag/BadPixelsMasked", \
    "DetectorSaturated":"QualityFlag/DetectorSaturated", \
    "DiscardedPackets":"QualityFlag/DiscardedPackets", \
    "LNOStraylight":"QualityFlag/LNOStraylight", \
    "HighInstrumentTemperature":"QualityFlag/HighInstrumentTemperature", \
    "PointingError":"QualityFlag/PointingError", \
    "UVISOccBoresight":"QualityFlag/UVISOccBoresight", \
    "LNOOccBoresight":"QualityFlag/LNOOccBoresight", \
    "MIROccBoresight":"QualityFlag/MIROccBoresight", \
    "TIRVIMOccBoresight":"QualityFlag/TIRVIMOccBoresight", \
    "CalibrationFailed":"QualityFlag/CalibrationFailed", \

#    "BadPixelsPresent":"QualityFlag/BadPixelsPresent", \
#    "DetectorFirstPacketDisturbed":"QualityFlag/DetectorFirstPacketDisturbed", \
#    "HSKFirstPacketDisturbed":"QualityFlag/HSKFirstPacketDisturbed", \
#    "OtherPacketIssue":"QualityFlag/OtherPacketIssue", \

    "IntegrationTime":"Channel/IntegrationTime", \
    "DetectorTemperature":"", \
    "ObservationType":"", \

    }
    
    if channel_obs in ["so_occultation", "lno_occultation", "lno_nadir"]:
        psa_mappings["AOTFFrequency"] = "Channel/AOTFFrequency"
        psa_mappings["DiffractionOrder"] = "Channel/DiffractionOrder"
        psa_mappings["NumberOfAccumulations"] = "Channel/NumberOfAccumulations"
        psa_mappings["SpectralResolution"] = "Channel/SpectralResolution"
        psa_mappings["InstrumentTemperature"] = ""

    elif channel_obs in ["uvis_occultation", "uvis_nadir"]:
        psa_mappings["AcquisitionMode"] = "Channel/AcquisitionMode"
        psa_mappings["BiasAverage"] = "Channel/BiasAverage"
        psa_mappings["DarkAverage"] = "Channel/DarkAverage"
        psa_mappings["ScienceAverage"] = "Channel/ScienceAverage"
        psa_mappings["VStart"] = "Channel/VStart"
        psa_mappings["VEnd"] = "Channel/VEnd"
        psa_mappings["HStart"] = "Channel/HStart"
        psa_mappings["HEnd"] = "Channel/HEnd"
        psa_mappings["StartDelay"] = "Channel/StartDelay"
        psa_mappings["AcquisitionDelay"] = "Channel/AcquisitionDelay"
        psa_mappings["NumberOfAcquisitions"] = "Channel/NumberOfAccumulations"
        psa_mappings["NumberOfFlushes"] = "Channel/NumberOfFlushes"
        psa_mappings["DarkToObservationSteps"] = "Channel/DarkToObservationSteps"
        psa_mappings["ObservationToDarkSteps"] = "Channel/ObservationToDarkSteps"
        psa_mappings["HorizontalAndCombinedBinningSize"] = "Channel/HorizontalAndCombinedBinningSize"
    
    return psa_mappings



def get_default_flag_values(hdf5_basename, hdf5_file, channel_obs):
    """set defaults for some flags that are not correct in HDF5"""
    
    default_flags = {}
    if channel_obs in ["uvis_occultation", "uvis_nadir"]:
        default_flags["BadPixelsMasked"] = 1 #bad pixels always masked in uvis
        
    if channel_obs in ["lno_nadir"]:
        default_flags["CalibrationFailed"] = 0 #0 by default

        #check filename
        passfail = re.findall("\d*_\d*_...._LNO_._D(\w).*", hdf5_basename)
        if len(passfail) == 1:
            if passfail == "F":
                default_flags["CalibrationFailed"] = 1


    if channel_obs in ["so_occultation", "lno_occultation", "uvis_occultation"]:
        default_flags["CalibrationFailed"] = 0 #0 by default

        #check file
        bins_accepted = hdf5_file["Criteria/Transmittance/BinAccepted"][...]
        if not np.all(bins_accepted == 1):
            default_flags["CalibrationFailed"] = 1
    
    return default_flags




def get_channel_observation_type(channel, observation_type):
    
    
    if channel == "so" and observation_type in ["I", "E"]:
        channel_obs = "so_occultation"
    elif channel == "lno" and observation_type in ["I", "E"]:
        channel_obs = "lno_occultation"
    elif channel == "lno" and observation_type in ["D"]:
        channel_obs = "lno_nadir"
    elif channel == "uvis" and observation_type in ["I", "E"]:
        channel_obs = "uvis_occultation"
    elif channel == "uvis" and observation_type in ["D"]:
        channel_obs = "uvis_nadir"
    else:
        channel_obs = ""

    return channel_obs


