# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:57:26 2020

@author: iant

VECTORISED SPICE FUNCTIONS

"""

import numpy as np
import spiceypy as sp

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.config import SPICE_INTERCEPT_METHOD, SPICE_OBSERVER, SPICE_TARGET, SPICE_SHAPE_MODEL_METHOD, \
    SPICE_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_PLANET_ID, SPICE_LONGITUDE_FORM, SP_DPR, SPICE_PLANET_REFERENCE_FRAME, \
    KILOMETRES_TO_AU, OBSERVER_X_AXIS

from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.areoid import geoid




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


