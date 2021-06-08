# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:53:06 2020

@author: iant

UNITS

"""

import h5py

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


