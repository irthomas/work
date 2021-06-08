# -*- coding: utf-8 -*-


TESTING=True
TESTING=False


import logging
import os.path

import h5py
import numpy as np
import spiceypy as sp
#import time

from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
import nomad_ops.core.hdf5.generic_functions as generics
import nomad_ops.core.hdf5.obs_type as obs_type
from nomad_ops.core.hdf5.l0p1a_to_0p1d.l0p1a_to_0p1d_v23 import write_attrs_from_itl
from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.areoid import geoid



__project__   = "NOMAD"
__author__    = "Ian Thomas, Roland Clairquin & Justin Erwin"
__contact__   = "roland.clairquin@oma.be"

#============================================================================================
# 4. CONVERT HDF5 LEVEL 0.1E TO LEVEL 0.2A
#
# DONE:
#
# ADDED GEOMETRY FOR NADIR OBSERVATIONS
# ADDED NEW POINT GEOMETRIES
# ADDED END OBSERVATION TIME GEOMETRY
# GEOMETRY FOR EACH BIN
# CHECK THAT OBSERVATION TYPE LETTER IS CORRECT AT START
#
# STILL TO DO:
#
# ADD COMMENT RECORDING SPICE KERNELS USED IN ANALYSIS
# CALCULATE TILT ANGLE I.E. ANGLE BETWEEN FOV LONG EDGE AND MARS LIMB
#
#============================================================================================



#t1 = time.clock()

logger = logging.getLogger( __name__ )

VERSION = 80
OUTPUT_VERSION = "0.2A"
NA_VALUE = -999 #value to be used for NaN
NA_STRING = "N/A" #string to be used for NaN


ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

#USE_VECTORISED_FUNCTIONS = True #speeds up calculation, only for occultations/. Doesn't work yet at present
USE_VECTORISED_FUNCTIONS = False

USE_DSK = True
#USE_DSK = False
if USE_DSK:
    # sincpt: a triaxial ellipsoid to model the surface of the SPICE_TARGET body.
    #SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
    SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
    # ubpnt: sub-SPICE_OBSERVER point is defined as the SPICE_TARGET surface
    # intercept of the line containing the SPICE_OBSERVER and the SPICE_TARGET's center
    SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"
else:
    # sincpt: a triaxial ellipsoid to model the surface of the SPICE_TARGET body.
    SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
    # subpnt: sub-SPICE_OBSERVER point is defined as the SPICE_TARGET surface
    # intercept of the line containing the SPICE_OBSERVER and the SPICE_TARGET's center
    #SPICE_INTERCEPT_METHOD = "Intercept: ellipsoid"
    SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"

USE_REDUCED_ELLIPSE = True   # if True, use reduced ellipse for npedln
USE_AREOID = True      # compute areoid and add areoid/topo info to file

# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_MARS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 499
# et2lst: form of longitude supplied by the variable lon
SPICE_LONGITUDE_FORM = "PLANETOCENTRIC"
# spkpos: reference frame relative to which the output position vector
# should be expressed
SPICE_REFERENCE_FRAME = "J2000"
#et2utc: string format flag describing the output time string. 'C' Calendar format, UTC
SPICE_STRING_FORMAT = "C"
# et2utc: number of decimal places of precision to which fractional seconds
# (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed
SPICE_TIME_PRECISION = 3

SPICE_TARGET = "MARS"
SPICE_OBSERVER = "-143"



#define detector row at centre of FOV
SO_DETECTOR_CENTRE_LINE = 128
LNO_DETECTOR_CENTRE_LINE = 152



BORESIGHT_VECTOR_TABLE=os.path.join(PFM_AUXILIARY_FILES,"geometry","boresight_vectors.txt")




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


def getUnitMappings(number_of_points):
    UNIT_MAPPINGS={
                "Geometry/ObsAlt":"KILOMETRES", \
                "Geometry/TiltAngle":"DEGREES", \
                "Geometry/SubObsLon":"DEGREES", \
                "Geometry/SubObsLat":"DEGREES", \
                "Geometry/LSubS":"DEGREES", \
                "Geometry/SubSolLon":"DEGREES", \
                "Geometry/SubSolLat":"DEGREES", \
                "Geometry/DistToSun":"ASTRONOMICAL UNITS", \
                "Geometry/SpdObsSun":"KILOMETRES PER SECOND", \
                "Geometry/SpdTargetSun":"KILOMETRES PER SECOND", \
                "Geometry/ObservationDateTime":"NO UNITS", \
                "Geometry/ObservationEphemerisTime":"SECONDS", \
                }
    for index in range(number_of_points):
        UNIT_MAPPINGS["Geometry/Point%s/PointXY" %index] = "NO UNITS"
        UNIT_MAPPINGS["Geometry/Point%s/FOVWeight" %index] = "NO UNITS"
        UNIT_MAPPINGS["Geometry/Point%s/Lat" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/Lon" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/LST" %index] = "HOURS"
        UNIT_MAPPINGS["Geometry/Point%s/LOSAngle" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/SunSZA" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/IncidenceAngle" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/EmissionAngle" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/PhaseAngle" %index] = "DEGREES"
        UNIT_MAPPINGS["Geometry/Point%s/TangentAlt" %index] = "KILOMETRES"
        UNIT_MAPPINGS["Geometry/Point%s/TangentAltAreoid" %index] = "KILOMETRES"
        UNIT_MAPPINGS["Geometry/Point%s/TangentAltSurface" %index] = "KILOMETRES"
        UNIT_MAPPINGS["Geometry/Point%s/SurfaceRadius" %index] = "KILOMETRES"
        UNIT_MAPPINGS["Geometry/Point%s/SurfaceAltAreoid" %index] = "KILOMETRES"
    return UNIT_MAPPINGS



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



def addUnits(hdf5FileOut, unitMappings):
#    logger.info("Adding units to hdf5 datasets")
    topKeys=hdf5FileOut.keys()
    for topKey in topKeys:
        if isinstance(hdf5FileOut[topKey], h5py.Group): #find datasets within groups
            for subGroup in hdf5FileOut[topKey].keys():
                if isinstance(hdf5FileOut[topKey+"/"+subGroup], h5py.Group): #for datasets within sub-groups (3rd level down)
                    for subSubGroup in hdf5FileOut[topKey+"/"+subGroup].keys():
                        if topKey+"/"+subGroup+"/"+subSubGroup in unitMappings:
                            hdf5FileOut[topKey+"/"+subGroup+"/"+subSubGroup].attrs["Units"] = unitMappings[topKey+"/"+subGroup+"/"+subSubGroup]
                elif isinstance(hdf5FileOut[topKey+"/"+subGroup], h5py.Dataset): #for datasets on the second level of file
                    if topKey+"/"+subGroup in unitMappings:
                        hdf5FileOut[topKey+"/"+subGroup].attrs["Units"] = unitMappings[topKey+"/"+subGroup]
        elif isinstance(hdf5FileOut[topKey], h5py.Dataset): #for datasets on the top level of file
            if topKey in unitMappings:
                hdf5FileOut[topKey].attrs["Units"] = unitMappings[topKey]




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


####VECTORISED SPICE FUNCTIONS######


def getLST(ets, lons):
    LONGITUDE_TYPE =    "degrees"
#    LONGITUDE_TYPE =    "radians"

    ets_flat = ets.flatten()
    lons_flat = lons.reshape(-1, 5)

    N_CORNERS = lons.shape[2]
    N_SPECTRA = lons_flat.shape[0]

    lst_hours_flat = np.zeros((N_SPECTRA, N_CORNERS))
    lst_spice_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))

    #check if data is valid
    if len(ets_flat) != len(lons_flat):
        return [0]


    for corner_index in range(N_CORNERS):
        if LONGITUDE_TYPE == "degrees":
            lst_spice_flat[:, corner_index, :] = np.asfarray([sp.et2lst(et, 499, lon / SP_DPR, SPICE_LONGITUDE_FORM)[0:3] for et, lon in zip(ets_flat, lons_flat[:, corner_index])])
        elif LONGITUDE_TYPE == "radians":
            lst_spice_flat[:, corner_index, :] = np.asfarray([sp.et2lst(et, 499, lon, SPICE_LONGITUDE_FORM)[0:3] for et, lon in zip(ets_flat, lons_flat[:, corner_index])])

        lst_hours_flat[:, corner_index] = lst_spice_flat[:, corner_index, 0] + lst_spice_flat[:, corner_index, 1] / 60.0 + lst_spice_flat[:, corner_index, 2] / 3600.0

    lst_hours = lst_hours_flat.reshape(-1, 2, 5)

    return lst_hours


def getPosition(ets, observer, target):

    ets_flat = ets.flatten()
    coords_flat = np.asfarray(sp.spkpos(observer, ets_flat, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, target)[0])
    coords = coords_flat.reshape(-1, 2, 3)
    return coords


def getTransMatrix(d_ref, ets):

    ets_flat = ets.flatten()
    matrices_flat = np.asfarray([sp.pxform(d_ref, SPICE_PLANET_REFERENCE_FRAME, et) for et in ets_flat])
    matrices = matrices_flat.reshape(-1, 2, 3, 3)
    return matrices


def getFovVector(matrices, fov_corners):

    matrices_flat = matrices.reshape(-1,3,3)

    N_CORNERS = len(fov_corners)
    N_SPECTRA = matrices_flat.shape[0]

    fov_vectors_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    for corner_index, fov_corner in enumerate(fov_corners):
        fov_vectors_flat[:, corner_index, :] = matrices_flat.dot(np.asfarray(fov_corner))
    fov_vectors = fov_vectors_flat.reshape(-1, 2, N_CORNERS, 3)
    return fov_vectors



def getTangentPointsAlt(observer_to_mars_vectors, fov_vectors, body_axes):

    observer_to_mars_vectors_flat = observer_to_mars_vectors.reshape(-1, 3)

    N_CORNERS = fov_vectors.shape[2]
    N_SPECTRA = observer_to_mars_vectors_flat.shape[0]

    fov_vectors_flat = fov_vectors.reshape(-1, 5, 3)
    surface_coords_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    tangent_alts_flat = np.zeros((N_SPECTRA, N_CORNERS))
    for fov_index in range(N_CORNERS):
        surface_coords_flat[:, fov_index, :] = np.asfarray([sp.npedln(body_axes[0], body_axes[1], body_axes[2], obs_vector, fov_vector)[0]
        for obs_vector, fov_vector in zip(observer_to_mars_vectors_flat, fov_vectors_flat[:,fov_index,:])])
        tangent_alts_flat[:, fov_index] = np.asfarray([sp.npedln(body_axes[0], body_axes[1], body_axes[2], obs_vector, fov_vector)[1]
        for obs_vector, fov_vector in zip(observer_to_mars_vectors_flat, fov_vectors_flat[:,fov_index,:])])
    surface_coords = surface_coords_flat.reshape(-1, 2, N_CORNERS, 3)
    tangent_alts = tangent_alts_flat.reshape(-1, 2, N_CORNERS)

    return surface_coords, tangent_alts

def getSurfaceCoordsLatLons(tangent_surface_points):

    tangent_surface_points_flat = tangent_surface_points.reshape(-1, 3)

    tangent_surface_coords_flat = np.asfarray([sp.reclat(tangent_surface_point) for tangent_surface_point in tangent_surface_points_flat])
    tangent_surface_lons_flat = tangent_surface_coords_flat[:,1] * SP_DPR
    tangent_surface_lats_flat = tangent_surface_coords_flat[:,2] * SP_DPR

    tangent_surface_coords = tangent_surface_coords_flat.reshape(-1, 2, 5, 3)
    tangent_surface_lons = tangent_surface_lons_flat.reshape(-1, 2, 5)
    tangent_surface_lats = tangent_surface_lats_flat.reshape(-1, 2, 5)

    return tangent_surface_coords, tangent_surface_lons, tangent_surface_lats




def getSurfaceNormal(tangent_surface_points, body_axes):

    tangent_surface_points_flat = tangent_surface_points.reshape(-1, 3)
    surface_normals_flat = np.asfarray([sp.surfnm(body_axes[0], body_axes[1], body_axes[2], tangent_surface_point) for tangent_surface_point in tangent_surface_points_flat])
    surface_normals = surface_normals_flat.reshape(-1, 2, 5, 3)

    return surface_normals


def getTangentPoint(tangent_surface_points, surface_normals, tangent_altitudes):
    """get point where LOS closest to planet"""
    tangent_surface_points_flat = tangent_surface_points.reshape(-1, 3)
    surface_normals_flat = surface_normals.reshape(-1, 3)
    tangent_altitudes_flat = np.repeat(tangent_altitudes.flatten()[:,np.newaxis], 3, 1)

    tangent_points_flat = tangent_surface_points_flat + surface_normals_flat * tangent_altitudes_flat
    tangent_points = tangent_points_flat.reshape(-1, 2, 5, 3)

    return tangent_points



def getTangentPointsAltReduced(tangent_surface_coords, tangent_points, ets):

    N_CORNERS = tangent_surface_coords.shape[2]

    tangent_surface_coords_flat = tangent_surface_coords.reshape(-1, 5, 3)
    N_SPECTRA = tangent_surface_coords_flat.shape[0]
    tangent_points_flat = tangent_points.reshape(-1, 5, 3)

    ets_flat = ets.flatten()

    tangent_surface_points_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
#    tangent_surface_coords_flat_new = np.zeros((N_SPECTRA, N_CORNERS))
    tangent_altitudes_flat = np.zeros((N_SPECTRA, N_CORNERS))
    for fov_index in range(N_CORNERS):
        #use lat/lon coords to find surface points on ellipse
        tangent_surface_points_flat[:, fov_index, :] = np.asfarray([sp.latsrf("Ellipsoid", "MARS", et, SPICE_PLANET_REFERENCE_FRAME, tangent_surface_coord)[0] \
        for et, tangent_surface_coord in zip(ets_flat, tangent_surface_coords_flat[:, fov_index, 1:])])

        #update radius coord value with reduced ellipsoid point
        tangent_surface_coords_flat[:, fov_index, 0] = np.asfarray([sp.vnorm(tangent_surface_point) \
        for tangent_surface_point in tangent_surface_points_flat[:, fov_index, :]])

        #calculate tangent altitudes from old tangent point - tangent surface point
        tangent_altitudes_flat[:, fov_index] = np.asfarray([sp.vnorm(tangent_point) - sp.vnorm(tangent_surface_point) \
        for tangent_point, tangent_surface_point in zip(tangent_points_flat[:, fov_index, :], tangent_surface_points_flat[:, fov_index, :])])


    tangent_surface_points = tangent_surface_points_flat.reshape(-1, 2, 5, 3)
    tangent_surface_coords_new = tangent_surface_coords_flat.reshape(-1, 2, 5, 3)
    tangent_altitudes = tangent_altitudes_flat.reshape(-1, 2, 5)


    return tangent_surface_points, tangent_surface_coords_new, tangent_altitudes




def getTangentPointsAltSurface(tangent_surface_coords, tangent_points, ets):

    N_CORNERS = tangent_surface_coords.shape[2]
    tangent_surface_coords_flat = tangent_surface_coords.reshape(-1, 5, 3)
    N_SPECTRA = tangent_surface_coords_flat.shape[0]
    tangent_points_flat = tangent_points.reshape(-1, 5, 3)

    ets_flat = ets.flatten()

    tangent_surface_points_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    tangent_surface_radius_flat = np.zeros((N_SPECTRA, N_CORNERS))
    tangent_altitudes_surface_flat = np.zeros((N_SPECTRA, N_CORNERS))
    for fov_index in range(N_CORNERS):
        #use lat/lon coords to find surface points on DSK
        tangent_surface_points_flat[:, fov_index, :] = np.asfarray([sp.latsrf(SPICE_SHAPE_MODEL_METHOD, "MARS", et, SPICE_PLANET_REFERENCE_FRAME, tangent_surface_coord)[0] \
        for et, tangent_surface_coord in zip(ets_flat, tangent_surface_coords_flat[:, fov_index, 1:])])

        #find radius of DSK surface point
        tangent_surface_radius_flat[:, fov_index] = np.asfarray([sp.vnorm(tangent_surface_point) \
        for tangent_surface_point in tangent_surface_points_flat[:, fov_index, :]])

        #find DSK surface altitude
        tangent_altitudes_surface_flat[:, fov_index] = np.asfarray([sp.vnorm(tangent_point) - tangent_surface_radius \
        for tangent_point, tangent_surface_radius in zip(tangent_points_flat[:, fov_index, :], tangent_surface_radius_flat[:, fov_index])])

    tangent_surface_points = tangent_surface_points_flat.reshape(-1, 2, 5, 3)
    tangent_surface_radius = tangent_surface_radius_flat.reshape(-1, 2, 5)
    tangent_altitudes_surface = tangent_altitudes_surface_flat.reshape(-1, 2, 5)

    return tangent_surface_points, tangent_surface_radius, tangent_altitudes_surface



def getTangentPointsAltAreoid(tangent_surface_coords, tangent_points, tangent_surface_radius, tangent_surface_lons, tangent_surface_lats):

    N_CORNERS = tangent_surface_coords.shape[2]
    tangent_surface_coords_flat = tangent_surface_coords.reshape(-1, 5, 3)
    N_SPECTRA = tangent_surface_coords_flat.shape[0]
    tangent_points_flat = tangent_points.reshape(-1, 5, 3)

    tangent_surface_radius_flat = tangent_surface_radius.reshape(-1, 5)
    tangent_surface_lons_flat = tangent_surface_lons.reshape(-1, 5)
    tangent_surface_lats_flat = tangent_surface_lats.reshape(-1, 5)

    tangent_areoid_radius_flat = np.zeros((N_SPECTRA, N_CORNERS))
    tangent_surface_topography_flat = np.zeros((N_SPECTRA, N_CORNERS))
    tangent_surface_points_areoid_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    tangent_altitudes_areoid_flat = np.zeros((N_SPECTRA, N_CORNERS))
    for fov_index in range(N_CORNERS):
        """areoid calc doesn't accept vectors yet"""
#        tangent_areoid_radius_flat[:, fov_index] = geoid(tangent_surface_lons_flat[:, fov_index], tangent_surface_lats_flat[:, fov_index])

        tangent_areoid_radius_flat[:, fov_index] = np.asfarray([geoid(lon, lat) \
        for lon, lat in zip(tangent_surface_lons_flat[:, fov_index], tangent_surface_lats_flat[:, fov_index])])

        tangent_surface_topography_flat[:, fov_index] = tangent_surface_radius_flat[:, fov_index] - tangent_areoid_radius_flat[:, fov_index]

        tangent_surface_points_areoid_flat[:, fov_index, :] = np.asfarray([sp.latrec(tangent_areoid_radius, tangent_surface_coord[1], tangent_surface_coord[2]) \
        for tangent_areoid_radius, tangent_surface_coord in zip(tangent_areoid_radius_flat[:, fov_index], tangent_surface_coords_flat[:, fov_index, :])])

        #find areoid surface altitude
        tangent_altitudes_areoid_flat[:, fov_index] = np.asfarray([sp.vnorm(tangent_point) - sp.vnorm(tangent_surface_point_areoid) \
        for tangent_point, tangent_surface_point_areoid in zip(tangent_points_flat[:, fov_index, :], tangent_surface_points_areoid_flat[:, fov_index, :])])

    tangent_areoid_radius = tangent_areoid_radius_flat.reshape(-1, 2, 5)
    tangent_surface_topography = tangent_surface_topography_flat.reshape(-1, 2, 5)
    tangent_surface_points_areoid = tangent_surface_points_areoid_flat.reshape(-1, 2, 5, 3)
    tangent_altitudes_areoid = tangent_altitudes_areoid_flat.reshape(-1, 2, 5)

    return tangent_areoid_radius, tangent_surface_topography, tangent_surface_points_areoid, tangent_altitudes_areoid


def getLosTiltAngles(observer_to_mars_vectors, ets, d_ref, fov_corners, fov_vectors):

    #calculate LOSAngle i.e. the angle between the FOV point and the centre of Mars.
    N_CORNERS = len(fov_corners)

    ets_flat = ets.flatten()
    N_SPECTRA = len(ets_flat)

    observer_to_mars_vectors_flat = observer_to_mars_vectors.reshape(-1, 3)
    fov_vectors_flat = fov_vectors.reshape(-1, 5, 3)

    mars_to_observer_vectors_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    channel_to_j2000_transforms_flat = np.zeros((N_SPECTRA, N_CORNERS, 3, 3))
    fov_to_j2000_vectors_flat = np.zeros((N_SPECTRA, N_CORNERS, 3))
    tangent_surface_los_angles_flat = np.zeros((N_SPECTRA, N_CORNERS))

    #calculate vector from mars to TGO in J2000 frame
    mars_to_observer_vector = np.asfarray(sp.spkpos("MARS", ets_flat, SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0])
    mars_to_observer_vectors_flat = np.repeat(mars_to_observer_vector[:, np.newaxis], 5, axis=1)

    #next transformation matrix from occultation channel to J2000
    channel_to_j2000_transform = np.asfarray([sp.pxform(d_ref, SPICE_REFERENCE_FRAME, et) for et in ets_flat])
    channel_to_j2000_transforms_flat = np.repeat(channel_to_j2000_transform[:, np.newaxis], 5, axis=1)

    for corner_index, fov_corner in enumerate(fov_corners):
        #next convert FOV from TGO coords to J2000
        fov_to_j2000_vectors_flat[:, corner_index, :] = channel_to_j2000_transforms_flat[:, corner_index, :, :].dot(np.asfarray(fov_corner))
        #then finally calculate the vector separation in degrees
        tangent_surface_los_angles_flat[:, corner_index] = np.asfarray([sp.vsep(fov_to_j2000_vector, mars_to_observer_vector) * SP_DPR \
        for fov_to_j2000_vector, mars_to_observer_vector in zip(fov_to_j2000_vectors_flat[:, corner_index, :], mars_to_observer_vectors_flat[:, corner_index, :])])


    #calculate tilt angle of slit
    #calculate unit vector from fov centre to mars centre
    observer_to_mars_unit_vector_magnitudes = np.linalg.norm(observer_to_mars_vectors_flat, axis=1)
    observer_to_mars_unit_vectors_flat = observer_to_mars_vectors_flat / np.repeat(observer_to_mars_unit_vector_magnitudes[:, np.newaxis], 3, axis=1)
    mars_centre_to_fov_unit_vectors_flat = observer_to_mars_unit_vectors_flat - fov_vectors_flat[:, 0, :]
    #calculate unit vector from fov top left to fov bottom left
    fov_top_to_bottom_unit_vectors_flat = fov_vectors_flat[:, 2, :] - fov_vectors_flat[:, 3, :]

    tilt_angles_flat = np.asfarray([sp.vsep(fov_top_to_bottom_unit_vector, mars_centre_to_fov_unit_vector) * SP_DPR \
    for fov_top_to_bottom_unit_vector, mars_centre_to_fov_unit_vector in zip(fov_top_to_bottom_unit_vectors_flat, mars_centre_to_fov_unit_vectors_flat)])


    tangent_point_los_angles = tangent_surface_los_angles_flat.reshape(-1, 2, 5)
    tilt_angles = tilt_angles_flat.reshape(-1, 2)

    return tangent_point_los_angles, tilt_angles




####START CODE#####
def convert(hdf5file_path):
#if True:
    logger.info("convert: %s", hdf5file_path)

    hdf5_basename = os.path.basename(hdf5file_path).split(".")[0]
    #file operations
    hdf5FileIn = h5py.File(hdf5file_path, "r")
    channel, channelType = generics.getChannelType(hdf5FileIn)

    #get observation info, timings, dataset size, bins etc. from input file
    channelName = hdf5FileIn.attrs["ChannelName"]
    observationMode = getObservationMode(channelName)
    observationDTimes = hdf5FileIn["Geometry/ObservationDateTime"][...]
    ydimensions = hdf5FileIn["Science/Y"].shape
    nSpectra = ydimensions[0]

    if channel in ["so","lno"]:
        bins = hdf5FileIn["Science/Bins"][...]
        observationType = generics.getObservationType(hdf5FileIn)
    elif channel == "uvis":
        bins = [0]*nSpectra #UVIS doesn't have bins!!
        obs_db_res = obs_type.get_obs_type(hdf5file_path)
        observationType = obs_db_res[4]

#    logger.info("observationType=%s for %s", observationType, hdf5FileIn.file)
    if observationType is None:
        raise RuntimeError("Observation type is not defined. Update the ITL db.")
        
    if observationMode == "error":
        logger.warning("%s: flip mirror position error. Skipping", hdf5_basename)
        return []


        

    #convert datetimes to et
    observationDTimesStart = [str(observationDTime[0]) for observationDTime in observationDTimes]
    observationDTimesEnd = [str(observationDTime[1]) for observationDTime in observationDTimes]
    obsTimesStart = [sp.utc2et(datetime.strip("<b>").strip("'")) for datetime in observationDTimesStart]
    obsTimesEnd = [sp.utc2et(datetime.strip("<b>").strip("'")) for datetime in observationDTimesEnd]


    # get spice information about name and FOV shape.
    # dvec = FOV centre vector, dvecCorners = FOV corner vectors
    dref = getFovName(channel, observationMode)
    channelId = sp.bods2c(dref) #find channel id number
    [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)


    #check observation type
    obsType_in_DNFV = observationType in ["D","N","F"]
    obsType_in_IESMULO = observationType in ["I","E","G","S","L","O"] #limb measurements are like occultations
    obsType_in_C = observationType in ["C"] #do nothing
    obsType_in_X = observationType in ["X"] #unknown type
    if obsType_in_X:
        logger.error("Error: Observation type unknown for file %s", hdf5file_path)
        return []
    if obsType_in_C:
        logger.warning("Observation found of type C. No geometric calibration added to file %s except ephemeris time", hdf5_basename)

        #short part to add ephemeris times to file
        ephemerisTimes = np.zeros((nSpectra,2)) + NA_VALUE
        for rowIndex in range(nSpectra):
            obsTimeStart = obsTimesStart[rowIndex]
            obsTimeEnd = obsTimesEnd[rowIndex]
            ephemerisTimes[rowIndex,:] = (obsTimeStart,obsTimeEnd)


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

    if obsType_in_DNFV or obsType_in_IESMULO:

        #first get nPoints by running function once
#        logger.info("Science measurement detected. Getting FOV parameters")
        points,fovWeights,fovCorners = getFovPointParameters(channel,bins[0])
        nPoints = len(points)

        unitMappings = getUnitMappings(nPoints)



        # loop through times, storing surface-point independent values
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

        # define spacecraft x-direction
        OBSERVER_X_AXIS = [1.0,0.0,0.0]

        for rowIndex in range(nSpectra):
            obsTimeStart = obsTimesStart[rowIndex]
            obsTimeEnd = obsTimesEnd[rowIndex]
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


            if obsType_in_DNFV:
                tiltAngles[rowIndex,:] = (sp.vsep(obsInSolSysFrame[0],obsVelocityInSolSysFrame[0]) * SP_DPR,
                          sp.vsep(obsInSolSysFrame[1],obsVelocityInSolSysFrame[1]) * SP_DPR)
                #tilt angle for other observations done later


        # Loop through array of times for a first time, storing surface intercept points
        # Valid=times when nadir pointed to Mars otherwise nan
        """make empty array nspectra x npoints x [start,end] x nsurface points"""
        surfPoints = np.zeros((nSpectra,nPoints,2,3)) + np.nan #use nan here, not -999

        fovCornersAll=[]
        for rowIndex in range(nSpectra):
            obsTimeStart = obsTimesStart[rowIndex]
            obsTimeEnd = obsTimesEnd[rowIndex]

            detectorBin = np.asfarray(bins[rowIndex])
            points,fovWeights,fovCorners=getFovPointParameters(channel,detectorBin)
            #store fovCorners for all bins in file
            fovCornersAll.append(fovCorners)


            for pointIndex,fovCorner in enumerate(fovCorners):
                try:
                    sincpt = [sp.sincpt(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeStart,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,dref,fovCorner),
                                 sp.sincpt(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeEnd,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,dref,fovCorner)]
                    surfPoints[rowIndex,pointIndex,:,:] = (sincpt[0][0],sincpt[1][0])
                except sp.stypes.SpiceyError: #error thrown if lat and lon point not on planet
                    continue




        """add code to check FOV pointing vs. observation type letter.
        Note that IESMUL can point to planet (ingress, egress) or not (e.g. grazing, limb) making detection difficult"""
        #check if FOV always on planet
        if not np.isnan(np.min(surfPoints)): #if always on planet
#            logger.info("FOV always pointed towards planet")
    #        fovOnPlanet=True
    #        fovChanges = False
            if obsType_in_IESMULO:
                logger.warning("Warning: off planet observation type %s (file %s) always points towards planet", observationType, hdf5file_path)

        elif np.any(np.isnan(surfPoints)): #if always off planet (e.g. limbs, grazing occs)
#            logger.info("FOV never pointed towards planet")
    #        fovOnPlanet=False
    #        fovChanges = False
            if obsType_in_DNFV:
                logger.warning("Warning: on-planet observation type %s (file %s) never points towards planet", observationType, hdf5file_path)

        else: #if mixed (e.g. normal occs)
#            logger.info("FOV sometimes pointed towards planet, sometimes not")
    #        fovOnPlanet=False
    #        fovChanges = True
            if obsType_in_DNFV:
                logger.warning("Warning: on-planet observation type %s (file %s) only sometimes points towards planet", observationType, hdf5file_path)







        #find and store point dataset
        pointXYs = np.zeros((1,nPoints,2)) + NA_VALUE
        for pointIndex,fovCorner in enumerate(fovCorners):
            pointXYs[[0],pointIndex,:] = points[pointIndex]


        if obsType_in_DNFV:
#            logger.info("Adding nadir geometry for observation of type %s in file %s", observationType, hdf5file_path)
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
                obsTimeStart = obsTimesStart[rowIndex]
                obsTimeEnd = obsTimesEnd[rowIndex]

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

        if obsType_in_IESMULO:


            if USE_VECTORISED_FUNCTIONS:
                logger.info("Adding vectorised occultation/limb geometry for observation of type %s in file %s", observationType, hdf5file_path)

                obsTimes = np.asfarray([obsTimesStart,obsTimesEnd]).T

                obs2MarsVector = getPosition(obsTimes, SPICE_OBSERVER, "MARS")

                occ2MarsTransform = getTransMatrix(dref,obsTimes)
                fovVectors = getFovVector(occ2MarsTransform, fovCorners)

                tangentSurfPoints, tangentAltitudes = getTangentPointsAlt(obs2MarsVector, fovVectors, bodyAxes)
                tangentSurfCoords, tangentSurfLons, tangentSurfLats = getSurfaceCoordsLatLons(tangentSurfPoints)
                tangentSurfLSTs = getLST(obsTimes, tangentSurfLons)

                surfNormals = getSurfaceNormal(tangentSurfPoints, bodyAxes)

                tangentPoints = getTangentPoint(tangentSurfPoints, surfNormals, tangentAltitudes)


                tangentSurfPoints2, tangentSurfCoords2, tangentAltitudes2 = getTangentPointsAltReduced(tangentSurfCoords, tangentPoints, obsTimes)

                tangentSurfPoints3, tangentSurfRadius, tangentAltitudesSurface = getTangentPointsAltSurface(tangentSurfCoords2, tangentPoints, obsTimes)

                tangentAreoidRadius, tangentSurfTopo, surfPointAreoid, tangentAltitudesAreoid = \
                    getTangentPointsAltAreoid(tangentSurfCoords2, tangentPoints, tangentSurfRadius, tangentSurfLons, tangentSurfLats)

                tangentSurfLOSAngles, tiltAngles = getLosTiltAngles(obs2MarsVector, obsTimes, dref, fovCorners, fovVectors)

            else:
#                logger.info("Adding non-vectorised occultation/limb geometry for observation of type %s in file %s", observationType, hdf5file_path)
                #initialise empty arrays
                tangentSurfPoints = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
                tangentSurfCoords = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
                fovVectors = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
                tangentAltitudes = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
                tangentSurfLons = np.zeros_like(tangentAltitudes) + NA_VALUE
                tangentSurfLats = np.zeros_like(tangentSurfLons) + NA_VALUE
                tangentSurfLSTs = np.zeros_like(tangentSurfLons) + NA_VALUE
                tangentSurfLSTHMSs = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
        #        tangentSurfIllumins = np.zeros((nSpectra,nPoints,2,3)) + NA_VALUE
                tangentSurfLOSAngles = np.zeros_like(tangentSurfLons) + NA_VALUE
        #        tangentSurfSunSZAs = np.zeros_like(tangentSurfLons) + NA_VALUE
        #        tangentSurfIncidenceAngles = np.zeros_like(tangentSurfLons) + NA_VALUE
        #        tangentSurfEmissionAngles = np.zeros_like(tangentSurfLons) + NA_VALUE
        #        tangentSurfPhaseAngles = np.zeros_like(tangentSurfLons) + NA_VALUE

                tangentSurfRadius = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
                tangentAltitudesSurface = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
                tangentAreoidRadius = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
                tangentSurfTopo = np.zeros((nSpectra,nPoints,2)) + NA_VALUE
                tangentAltitudesAreoid = np.zeros((nSpectra,nPoints,2)) + NA_VALUE

                #loop through spectra and points, adding to arrays
                for rowIndex in range(nSpectra):
                    obsTimeStart = obsTimesStart[rowIndex]
                    obsTimeEnd = obsTimesEnd[rowIndex]

                    if np.isnan(surfPoints[rowIndex,:,:,:]).any():

                        obs2MarsVector = (sp.spkpos(SPICE_OBSERVER,obsTimeStart,"IAU_MARS",SPICE_ABERRATION_CORRECTION,"MARS")[0],
                                            sp.spkpos(SPICE_OBSERVER,obsTimeEnd,"IAU_MARS",SPICE_ABERRATION_CORRECTION,"MARS")[0])

                        occ2MarsTransform = (sp.pxform(dref,"IAU_MARS",obsTimeStart),
                                                sp.pxform(dref,"IAU_MARS",obsTimeEnd))
                        # Points Datasets
                        for pointIndex,fovCorner in enumerate(fovCornersAll[rowIndex]):

                            fovVectors[rowIndex,pointIndex,:,:] = (occ2MarsTransform[0].dot(np.asfarray(fovCorner)),
                                          occ2MarsTransform[1].dot(np.asfarray(fovCorner)))
                            tangentPointAlts = (sp.npedln(bodyAxes[0], bodyAxes[1], bodyAxes[2], obs2MarsVector[0], fovVectors[rowIndex,pointIndex,0,:]),
                                                       sp.npedln(bodyAxes[0], bodyAxes[1], bodyAxes[2], obs2MarsVector[1], fovVectors[rowIndex,pointIndex,1,:]))

                            # if either hit ellipse, skip this point
                            if tangentPointAlts[0][1] <= 0. or tangentPointAlts[1][1] <= 0.:
                                continue

                            # convert the Alt, Lon, Lat
                            tangentSurfPoints[rowIndex,pointIndex,:,:] = (tangentPointAlts[0][0],
                                             tangentPointAlts[1][0])
                            tangentAltitudes[rowIndex,pointIndex,:] = (tangentPointAlts[0][1],
                                            tangentPointAlts[1][1])

                            tangentSurfCoords[rowIndex,pointIndex,:,:] = (sp.reclat(tangentSurfPoints[rowIndex,pointIndex,0,:]),
                                                         sp.reclat(tangentSurfPoints[rowIndex,pointIndex,1,:]))
                            tangentSurfLons[rowIndex,pointIndex,:] = (tangentSurfCoords[rowIndex,pointIndex,0,1]*SP_DPR,
                                           tangentSurfCoords[rowIndex,pointIndex,1,1]*SP_DPR)
                            tangentSurfLats[rowIndex,pointIndex,:] = (tangentSurfCoords[rowIndex,pointIndex,0,2]*SP_DPR,
                                           tangentSurfCoords[rowIndex,pointIndex,1,2]*SP_DPR)

                            # compute tanget point in LOS
                            surfNormal = (sp.surfnm(bodyAxes[0], bodyAxes[1], bodyAxes[2], tangentSurfPoints[rowIndex,pointIndex,0,:]),
                                    sp.surfnm(bodyAxes[0], bodyAxes[1], bodyAxes[2], tangentSurfPoints[rowIndex,pointIndex,1,:]))
                            tangentPoint = (tangentSurfPoints[rowIndex,pointIndex,0,:] + surfNormal[0]*tangentAltitudes[rowIndex,pointIndex,0],
                                    tangentSurfPoints[rowIndex,pointIndex,1,:] + surfNormal[1]*tangentAltitudes[rowIndex,pointIndex,1])

    #                        print("obsTimeStart=%0.1f" %obsTimeStart)
    #                        print("tangentSurfCoords[rowIndex,pointIndex,0,0]=%0.1f" %(tangentSurfCoords[rowIndex,pointIndex,0,0]))
    #                        print("tangentSurfCoords[rowIndex,pointIndex,1,0]=%0.1f" %(tangentSurfCoords[rowIndex,pointIndex,1,0]))
                            tangentSurfLSTHMSs[rowIndex,pointIndex,:,:] = (sp.et2lst(obsTimeStart,SPICE_PLANET_ID,tangentSurfCoords[rowIndex,pointIndex,0,1],SPICE_LONGITUDE_FORM)[0:3],
                                       sp.et2lst(obsTimeEnd,SPICE_PLANET_ID,tangentSurfCoords[rowIndex,pointIndex,1,1],SPICE_LONGITUDE_FORM)[0:3])
    #                        print(tangentSurfLSTHMSs[rowIndex,pointIndex,0,:])
                            tangentSurfLSTs[rowIndex,pointIndex,:] = ((tangentSurfLSTHMSs[rowIndex,pointIndex,0,0] + tangentSurfLSTHMSs[rowIndex,pointIndex,0,1]/60.0 + tangentSurfLSTHMSs[rowIndex,pointIndex,0,2]/3600.0),
                                    (tangentSurfLSTHMSs[rowIndex,pointIndex,1,0] + tangentSurfLSTHMSs[rowIndex,pointIndex,1,1]/60.0 + tangentSurfLSTHMSs[rowIndex,pointIndex,1,2]/3600.0))


                            # if using reduced ellipse, recompute altitudes
                            if USE_REDUCED_ELLIPSE:
                                #correct tangent surf to standard ellipse
                                tangentSurfPoints[rowIndex,pointIndex,:,:] = (sp.latsrf("Ellipsoid", "MARS", obsTimeStart, "IAU_MARS", tangentSurfCoords[rowIndex,pointIndex,0,1:])[0],
                                        sp.latsrf("Ellipsoid", "MARS", obsTimeEnd, "IAU_MARS", tangentSurfCoords[rowIndex,pointIndex,1,1:])[0])
                                tangentSurfCoords[rowIndex,pointIndex,:,0] = (sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,0,:]),
                                        sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,1,:]))
                                tangentAltitudes[rowIndex,pointIndex,:] = (sp.vnorm(tangentPoint[0]) - sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,0,:]),
                                        sp.vnorm(tangentPoint[1])-sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,1,:]))

                            # find surface and tangent altitude relative to on DSK
                            try:
                                tangentSurfPoints[rowIndex,pointIndex,:,:] = (sp.latsrf(SPICE_SHAPE_MODEL_METHOD, "MARS", obsTimeStart, "IAU_MARS", tangentSurfCoords[rowIndex,pointIndex,0,1:])[0],
                                            sp.latsrf(SPICE_SHAPE_MODEL_METHOD, "MARS", obsTimeEnd, "IAU_MARS", tangentSurfCoords[rowIndex,pointIndex,1,1:])[0])
                            except sp.stypes.SpiceyError:
                                logger.warning("Error in latsrf in file %s at times %s & %s. Tangent surf coords = %0.4f, %0.4f and %0.4f, %0.4f", hdf5file_path, obsTimeStart, obsTimeEnd,
                                             tangentSurfCoords[rowIndex,pointIndex,0,1], tangentSurfCoords[rowIndex,pointIndex,0,2], tangentSurfCoords[rowIndex,pointIndex,1,1], tangentSurfCoords[rowIndex,pointIndex,1,2])

                            tangentSurfRadius[rowIndex,pointIndex,:] = (sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,0,:]),
                                    sp.vnorm(tangentSurfPoints[rowIndex,pointIndex,1,:]))
                            tangentAltitudesSurface[rowIndex,pointIndex,:] = (sp.vnorm(tangentPoint[0]) - tangentSurfRadius[rowIndex,pointIndex,0],
                                        sp.vnorm(tangentPoint[1])-tangentSurfRadius[rowIndex,pointIndex,1])


                            # if using areoid, compute Topo and AltAreoid
                            if USE_AREOID:
                                tangentAreoidRadius[rowIndex,pointIndex,:] = (geoid(tangentSurfLons[rowIndex,pointIndex,0], tangentSurfLats[rowIndex,pointIndex,0]),
                                        geoid(tangentSurfLons[rowIndex,pointIndex,1], tangentSurfLats[rowIndex,pointIndex,1]))
                                tangentSurfTopo[rowIndex,pointIndex,:] = tangentSurfRadius[rowIndex,pointIndex,:] - tangentAreoidRadius[rowIndex,pointIndex,:]

                                surfPointAreoid = (sp.latrec(tangentAreoidRadius[rowIndex,pointIndex,0], tangentSurfCoords[rowIndex,pointIndex,0,1], tangentSurfCoords[rowIndex,pointIndex,0,2]),
                                        sp.latrec(tangentAreoidRadius[rowIndex,pointIndex,1], tangentSurfCoords[rowIndex,pointIndex,1,1], tangentSurfCoords[rowIndex,pointIndex,1,2]))

                                tangentAltitudesAreoid[rowIndex,pointIndex,:] = (sp.vnorm(tangentPoint[0]) - sp.vnorm(surfPointAreoid[0]),
                                        sp.vnorm(tangentPoint[1])-sp.vnorm(surfPointAreoid[1]))


                            #tangentSurfIllumins[rowIndex,pointIndex,:,:] = (sp.ilumin(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeStart,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,tangentSurfCoords[rowIndex,pointIndex,0,:])[2:5],
                            #            sp.ilumin(SPICE_SHAPE_MODEL_METHOD,SPICE_TARGET,obsTimeEnd,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER,tangentSurfCoords[rowIndex,pointIndex,1,:])[2:5])

                            #calculate LOSAngle i.e. the angle between the FOV point and the centre of Mars.
                            #calculate vector from mars to TGO in J2000 frame
                            mars2ObsVector =  (sp.spkpos("MARS",obsTimeStart,"J2000",SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0],
                                             sp.spkpos("MARS",obsTimeEnd,"J2000",SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER)[0])
                            #next transformation matrix from occultation channel to J2000
                            lnoOcc2SolSysTransform = (sp.pxform(dref,"J2000",obsTimeStart),
                                                   sp.pxform(dref,"J2000",obsTimeEnd))
                            #next convert FOV from TGO coords to J20000
                            fovSolSysVector = (lnoOcc2SolSysTransform[0].dot(np.asfarray(fovCorner)),
                                          lnoOcc2SolSysTransform[1].dot(np.asfarray(fovCorner)))
                            #then finally calculate the vector separation in degrees
                            tangentSurfLOSAngles[rowIndex,pointIndex,:] = (sp.vsep(fovSolSysVector[0],mars2ObsVector[0])*SP_DPR,
                                                sp.vsep(fovSolSysVector[1],mars2ObsVector[1])*SP_DPR)

                            #"""not for occultation"""
                            #tangentSurfSunSZAs[rowIndex,pointIndex,:] = ((tangentSurfIllumins[rowIndex,pointIndex,0,1]*SP_DPR),
                            #                   (tangentSurfIllumins[rowIndex,pointIndex,1,1]*SP_DPR))
                            #tangentSurfIncidenceAngles[rowIndex,pointIndex,:] = ((tangentSurfIllumins[rowIndex,pointIndex,0,1]*SP_DPR),
                            #                          (tangentSurfIllumins[rowIndex,pointIndex,1,1]*SP_DPR))
                            #tangentSurfEmissionAngles[rowIndex,pointIndex,:] = ((tangentSurfIllumins[rowIndex,pointIndex,0,2]*SP_DPR),
                            #                         (tangentSurfIllumins[rowIndex,pointIndex,1,2]*SP_DPR))
                            #tangentSurfPhaseAngles[rowIndex,pointIndex,:] = ((tangentSurfIllumins[rowIndex,pointIndex,0,0]*SP_DPR),
                            #                      (tangentSurfIllumins[rowIndex,pointIndex,1,0]*SP_DPR))




                        #calculate tilt angle of slit
                        #calculate unit vector from fov centre to mars centre
                        obs2MarsUnitVector = (obs2MarsVector[0]/np.sqrt(obs2MarsVector[0][0]**2 + obs2MarsVector[0][1]**2 + obs2MarsVector[0][2]**2),
                                              obs2MarsVector[1]/np.sqrt(obs2MarsVector[1][0]**2 + obs2MarsVector[1][1]**2 + obs2MarsVector[1][2]**2),)

                        marsCentre2fovCentreUnitVector = (obs2MarsUnitVector[0] - fovVectors[rowIndex,0,0,:],
                                                          obs2MarsUnitVector[1] - fovVectors[rowIndex,0,1,:])

                        #calculate unit vector from fov top left to fov bottom left
                        fovTopBottomUnitVectors = (fovVectors[rowIndex,2,0,:] - fovVectors[rowIndex,3,0,:],
                                                   fovVectors[rowIndex,2,1,:] - fovVectors[rowIndex,3,1,:])
                        tiltAngles[rowIndex,:] = (sp.vsep(fovTopBottomUnitVectors[0],marsCentre2fovCentreUnitVector[0]) * SP_DPR,
                                  sp.vsep(fovTopBottomUnitVectors[1],marsCentre2fovCentreUnitVector[1]) * SP_DPR)








    if channel=="uvis":
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR,"%s_%s.h5" % (hdf5_basename, observationType))
    else:
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))

#    logger.info("Writing to file: %s", hdf5FilepathOut)
    with h5py.File(hdf5FilepathOut, "w") as hdf5FileOut:

        # Copy datasets and attributes
        generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION)
        for dset_path, dset in generics.iter_datasets(hdf5FileIn):
            dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            hdf5FileIn.copy(dset_path, dest)

        hdf5FileIn.close()

        if obsType_in_C:
            #write only ephemeris time to calibration files. Rest remains unchanged.
            hdf5FileOut.create_dataset("Geometry/ObservationEphemerisTime", dtype=np.float,
                                    data=ephemerisTimes, fillvalue=NA_VALUE, compression="gzip", shuffle=True)


        #write new datasets for science files
        if obsType_in_DNFV or obsType_in_IESMULO:
#            logger.info("Writing generic geometry for observation of type %s in file %s", observationType, hdf5FilepathOut)
            #write new datasets for all files
            hdf5FileOut.create_dataset("Geometry/ObservationEphemerisTime", dtype=np.float,
                                    data=ephemerisTimes, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            #Geometry/ObservationDateTime  already written
            a = hdf5FileOut.create_dataset("Geometry/ObsAlt", dtype=np.float, data=obsAlts, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            a.attrs["Description"] = "Distance between spacecraft and Mars centre"

            b = hdf5FileOut.create_dataset("Geometry/TiltAngle", dtype=np.float, data=tiltAngles, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            if obsType_in_DNFV:
                b.attrs["Description"] = "Angle between spacecraft velocity and long edge of slit" 
            if obsType_in_IESMULO:
                b.attrs["Description"] = "Angle between line from Mars centre to field of view and long edge of slit" 


            c = hdf5FileOut.create_dataset("Geometry/SubObsLon", dtype=np.float, data=subObsLons, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            c.attrs["Description"] = "Surface longitude below spacecraft"

            d = hdf5FileOut.create_dataset("Geometry/SubObsLat", dtype=np.float, data=subObsLats, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
            d.attrs["Description"] = "Surface latitude below spacecraft"

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

            #write new datasets for nadir only files
            if obsType_in_DNFV:
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

            #write new datasets for occultation/limb only files
            if obsType_in_IESMULO:

                if observationType != "L":
                    #find boresight and write quality flag
                    all_boresight_vectors, all_boresight_names = readBoresightFile(BORESIGHT_VECTOR_TABLE)
                    #get mid point et of occultation
                    obsTimeMid = np.mean([obsTimesStart[0],obsTimesStart[-1]])
                    boresight_name_found, boresight_vector_found, v_sep_min = findBoresightUsed(obsTimeMid, all_boresight_vectors, all_boresight_names)
                    logger.info("Boresight determination: closest match to %s, vsep = %0.3f arcmins", boresight_name_found, v_sep_min)
                    
                    #get angular separation in arcmins between sun centre and FOV centre
                    fovSunCentreAngle = np.asfarray([findSunWobble(obsTimesStart, boresight_vector_found), findSunWobble(obsTimesEnd, boresight_vector_found)]).T
                    dset = hdf5FileOut.create_dataset("Geometry/FOVSunCentreAngle", dtype=np.float,
                                        data=fovSunCentreAngle, fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                    dset.attrs["Units"] = "ARCMINUTES"
                    dset.attrs["Description"] = "Angular separation between centre of the field of view and centre of the Sun"

                    
                    writeBoresightQualityFlag(hdf5FileOut, boresight_name_found)
                else: #if limb measurement, no sun boresight
                    #set boresight quality flags to zero for nadir observation
                    writeBoresightQualityFlag(hdf5FileOut, "NO OCCULTATION")


#                logger.info("Writing occultation/limb geometry for observation of type %s in file %s", observationType, hdf5FilepathOut)
                for pointIndex in range(nPoints):
                    if USE_VECTORISED_FUNCTIONS:
                        hdf5FileOut.create_dataset("Geometry/Point%i/PointXY" %pointIndex, dtype=np.float,
                                            data=pointXYs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/Lon" %pointIndex, dtype=np.float,
                                            data=tangentSurfLons[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/Lat" %pointIndex, dtype=np.float,
                                            data=tangentSurfLats[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/LST" %pointIndex, dtype=np.float,
                                            data=tangentSurfLSTs[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/LOSAngle" %pointIndex, dtype=np.float,
                                            data=tangentSurfLOSAngles[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/TangentAlt" %pointIndex, dtype=np.float,
                                            data=tangentAltitudes[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)

                        hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceRadius" %pointIndex, dtype=np.float,
                                            data=tangentSurfRadius[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        hdf5FileOut.create_dataset("Geometry/Point%i/TangentAltSurface" %pointIndex, dtype=np.float,
                                            data=tangentAltitudesSurface[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)


                        if USE_AREOID:
                            hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceAltAreoid" %pointIndex, dtype=np.float,
                                                data=tangentSurfTopo[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)

                            hdf5FileOut.create_dataset("Geometry/Point%i/TangentAltAreoid" %pointIndex, dtype=np.float,
                                                data=tangentAltitudesAreoid[:,:,pointIndex], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                    else:
                        k = hdf5FileOut.create_dataset("Geometry/Point%i/PointXY" %pointIndex, dtype=np.float, data=pointXYs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        k.attrs["Description"] = "Relative position of point %i within the field of view" %pointIndex

                        l = hdf5FileOut.create_dataset("Geometry/Point%i/Lon" %pointIndex, dtype=np.float, data=tangentSurfLons[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        l.attrs["Description"] = "Surface longitude above tangent altitude"
                        
                        m = hdf5FileOut.create_dataset("Geometry/Point%i/Lat" %pointIndex, dtype=np.float, data=tangentSurfLats[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        m.attrs["Description"] = "Surface latitude above tangent altitude"
                        
                        n = hdf5FileOut.create_dataset("Geometry/Point%i/LST" %pointIndex, dtype=np.float, data=tangentSurfLSTs[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        n.attrs["Description"] = "Local solar time"
                        
                        o = hdf5FileOut.create_dataset("Geometry/Point%i/LOSAngle" %pointIndex, dtype=np.float, data=tangentSurfLOSAngles[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        o.attrs["Description"] = "For ASIMUT only"
                        
                        p = hdf5FileOut.create_dataset("Geometry/Point%i/TangentAlt" %pointIndex, dtype=np.float, data=tangentAltitudes[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        p.attrs["Description"] = "Tangent altitude above ellipsoid"

                        q = hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceRadius" %pointIndex, dtype=np.float, data=tangentSurfRadius[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        q.attrs["Description"] = "Height of DSK surface above Mars centre"
                        
                        r = hdf5FileOut.create_dataset("Geometry/Point%i/TangentAltSurface" %pointIndex, dtype=np.float, data=tangentAltitudesSurface[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                        r.attrs["Description"] = "Tangent altitude above DSK surface"


                        if USE_AREOID:
                            s = hdf5FileOut.create_dataset("Geometry/Point%i/SurfaceAltAreoid" %pointIndex, dtype=np.float, data=tangentSurfTopo[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                            s.attrs["Description"] = "Height of DSK surface above areoid"

                            t = hdf5FileOut.create_dataset("Geometry/Point%i/TangentAltAreoid" %pointIndex, dtype=np.float, data=tangentAltitudesAreoid[:,pointIndex,:], fillvalue=NA_VALUE, compression="gzip", shuffle=True)
                            t.attrs["Description"] = "Tangent altitude above areoid"



            addUnits(hdf5FileOut, unitMappings)
            hdf5FileOut.attrs["GeometryPoints"] = nPoints
    if channel == "uvis":
        write_attrs_from_itl(hdf5FilepathOut, obs_db_res)
    return [hdf5FilepathOut]


#t2 = time.clock()
#
#print("Processing time = %0.2f seconds" %(t2 - t1))

#hdf5file_path = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p1e/2018/04/22/20180422_120404_0p1e_SO_1_E_134.h5")
#convert(hdf5file_path)
