# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:40:17 2020

@author: iant

BORESIGHT FUNCTIONS
"""
import numpy as np
import spiceypy as sp
import logging

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import SO_DETECTOR_CENTRE_LINE, LNO_DETECTOR_CENTRE_LINE, ARCMINS_TO_RADIANS

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import SPICE_OBSERVER, SPICE_ABERRATION_CORRECTION


logger = logging.getLogger( __name__ )



def getFovName(channel, observationMode):
    """SPICE kernel boresight names and properties"""

    if channel=="so" and observationMode=="solar occultation":
        dref = "TGO_NOMAD_SO"

    elif channel=="lno" and observationMode=="solar occultation":
        dref = "TGO_NOMAD_LNO_OPS_OCC"
    elif channel=="lno" and observationMode=="nadir":
        dref = "TGO_NOMAD_LNO_OPS_NAD"

    elif channel=="uvis" and observationMode=="solar occultation":
        dref = "TGO_NOMAD_UVIS_OCC"
    elif channel=="uvis" and observationMode=="nadir":
        dref = "TGO_NOMAD_UVIS_NAD"
    elif channel=="uvis" and observationMode=="calibration":
        dref = "TGO_NOMAD_UVIS_NAD"
    else:
        mess_tmp = "Channel %s and Observation Mode %s are not defined in pipeline_mappings"
        raise RuntimeError(mess_tmp % (channel, observationMode))

#    logger.info("Boresight name=%s",dref)
    return dref



def getObservationMode(ChannelName):
    #find observation mode
    if "Occultation" in ChannelName:
        observationMode = "solar occultation"
    elif "Nadir" in ChannelName:
        observationMode = "nadir"
    elif "Mirror Position" in ChannelName or "flip mirror" in ChannelName:
        observationMode = "error"
    elif "Selector" in ChannelName or "Test Card" in ChannelName:
        observationMode = "calibration"

#    if observationMode=="error":
#        raise RuntimeError("Error in Flip Mirror Position")

#    logger.info("Observation mode=%s",observationMode)
    return observationMode


def getFovPointParameters(channel,detectorBin):
    """now get detector properties to calculate true FOV, assuming 1 pixel per arcminute IFOV
    so and lno defined by 4 points, UVIS defined by 8 points making octagon
    SO/LNO detector row 1 views further upwards when NOMAD is placed upright"""
    if channel in ["so","lno"]:
        #X = long edge of slit
        if channel == "lno":
            detectorCentreLine = LNO_DETECTOR_CENTRE_LINE
            fovHalfwidth = 2.0 / ARCMINS_TO_RADIANS
#            spiceKernelFovSize = LNO_FOV_IN_KERNEL
        elif channel =="so":
            detectorCentreLine = SO_DETECTOR_CENTRE_LINE
            fovHalfwidth = 1.0 / ARCMINS_TO_RADIANS
#            spiceKernelFovSize = SO_FOV_IN_KERNEL

        detectorOffset = detectorBin - detectorCentreLine
        detectorOffset[1] += 1 #actually is 1 pixel more in -ve direction (down detector)
        binFovSize = detectorOffset / ARCMINS_TO_RADIANS #assume 1 arcmin per pixel
        points = [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]] #points are per bin. 0=centre, 1=corner. Not used in calculation!
        fovWeights = [1.0, 0.5, 0.5, 0.5, 0.5] #weights are per bin
        fovPoints = [[np.mean(binFovSize), 0.0], \
                     [binFovSize[0], fovHalfwidth], \
                     [binFovSize[0], -1.0*fovHalfwidth], \
                     [binFovSize[1], -1.0*fovHalfwidth], \
                     [binFovSize[1], fovHalfwidth]]
        fovCorners = [[point[0], point[1], (np.sqrt(1.0 - point[0]**2.0 - point[1]**2.0))] for point in fovPoints]


    elif channel=="uvis":
        #X = long edge of slit
        fovSize = [21.5 / ARCMINS_TO_RADIANS, 21.5 / ARCMINS_TO_RADIANS]
        points = [[0.0,0.0],[0.0,1.0],[0.707106,0.707106],[1.0,0.0],[0.707106,-0.707106],
                  [0.0,-1.0],[-0.707106,-0.707106],[-1.0,0.0],[-0.707106,0.707106]]
        fovWeights = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        fovCorners = [[(point[0] * fovSize[0]),(point[1] * fovSize[1]),
                        (np.sqrt(1.0 - (point[0] * fovSize[0]) ** 2 + (point[1] * fovSize[1]) ** 2))]
                       for point in points]

    return points,fovWeights,fovCorners



def readBoresightFile(boresight_vector_file_path):
    """read auxilliary text file containing all boresight vectors"""
#    logger.info("Opening boresight vector file %s for reading", boresight_vector_file_path)

    with open(boresight_vector_file_path, "r") as f:
        lines = f.readlines()


    all_boresight_vectors = []
    all_boresight_names = []

    for line in lines:
        content = line.split(",")
        boresight_name_in = content[0].strip()
        boresight_vector_in = [
                np.float32(content[1].strip()),
                np.float32(content[2].strip()),
                np.float32(content[3].strip()),
                ]
        if boresight_name_in == "SO_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("SO_BORESIGHT")
        elif boresight_name_in == "UVIS_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("UVIS_BORESIGHT")
        elif boresight_name_in == "LNO_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("LNO_BORESIGHT")
        elif boresight_name_in == "MIR_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("MIR_BORESIGHT")
        elif boresight_name_in == "TIRVIM_BORESIGHT":
            all_boresight_vectors.append(boresight_vector_in)
            all_boresight_names.append("TIRVIM_BORESIGHT")
        else:
            logger.error("Boresight file cannot be read in correctly (%s boresight unknown)" %boresight_name_in)


    return all_boresight_vectors, all_boresight_names



def findBoresightUsed(et, boresight_vectors, boresight_names):
    """find which boresight in auxiliary file is closest to sun pointing vector at the given time"""

    obs2SunVector = sp.spkpos("SUN", et, "TGO_SPACECRAFT", SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)

    v_norm = obs2SunVector[0]/sp.vnorm(obs2SunVector[0])

    boresight_name_found = "NONE"
    boresight_vector_found = [0.0,0.0,0.0]
    v_sep_min = 999.0

    for boresight_vector, boresight_name in zip(boresight_vectors, boresight_names):
        v_sep = sp.vsep(v_norm, boresight_vector) * sp.dpr() * 60.0 #arcmins difference
        if v_sep < v_sep_min:
            v_sep_min = v_sep
            boresight_name_found = boresight_name
            boresight_vector_found = boresight_vector
    return boresight_name_found, boresight_vector_found, v_sep_min




def findSunWobble(ets, boresight_vector):
    """find angular separation between chosen boresight vector and sun centre"""
    
    v_seps = []
    for et in ets:
        obs2SunVector = sp.spkpos("SUN", et, "TGO_SPACECRAFT", SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)
        v_norm = obs2SunVector[0]/sp.vnorm(obs2SunVector[0])
        v_sep = sp.vsep(v_norm, boresight_vector) * sp.dpr() * 60.0 #arcmins difference
        v_seps.append(v_sep)

    return v_seps
    


def writeBoresightQualityFlag(hdf5_file_out, boresight_name_in):
    """set the corresponding quality flags to the correct values in the hdf5 file"""
    error=False

    if boresight_name_in == "SO_BORESIGHT":
        boresights = {
            "UVISOccBoresight":0,
            "LNOOccBoresight":0,
            "MIROccBoresight":0,
            "TIRVIMOccBoresight":0,
            }
    elif boresight_name_in == "UVIS_BORESIGHT":
        boresights = {
            "UVISOccBoresight":1,
            "LNOOccBoresight":0,
            "MIROccBoresight":0,
            "TIRVIMOccBoresight":0,
            }
    elif boresight_name_in == "LNO_BORESIGHT":
        boresights = {
            "UVISOccBoresight":0,
            "LNOOccBoresight":1,
            "MIROccBoresight":0,
            "TIRVIMOccBoresight":0,
            }
    elif boresight_name_in == "MIR_BORESIGHT":
        boresights = {
            "UVISOccBoresight":0,
            "LNOOccBoresight":0,
            "MIROccBoresight":1,
            "TIRVIMOccBoresight":0,
            }
    elif boresight_name_in == "TIRVIM_BORESIGHT":
        boresights = {
            "UVISOccBoresight":0,
            "LNOOccBoresight":0,
            "MIROccBoresight":0,
            "TIRVIMOccBoresight":1,
            }
    elif boresight_name_in == "NO OCCULTATION":
        boresights = {
            "UVISOccBoresight":0,
            "LNOOccBoresight":0,
            "MIROccBoresight":0,
            "TIRVIMOccBoresight":0,
            }
    else:
        error=True
        logger.error("Boresight detected is not correct (%s boresight unknown)" %boresight_name_in)

    if not error:
        for flag_name, flag_value in boresights.items():
            hdf5_file_out.create_dataset("QualityFlag/%s" %flag_name, dtype=np.int,
                                    data=flag_value)
    return

