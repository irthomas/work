# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:19:23 2019

@author: iant
"""

solar_geometry_dict = {
    "lon":["tangent point longitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lon"],
    "lat":["tangent point latitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lat"],
    "lst":["tangent point local solar time in hours at centre of field of view", ["f","l"], "hours", "Geometry/PointN/LST"],

    "alt":["tangent altitude at centre of field of view", ["f","l"], "deg", "Geometry/PointN/TangentAlt"],

    "sub_obs_lon":["sub-satellite longitude", ["f","l","m","x"], "deg", "Geometry/SubObsLon"],
    "sub_obs_lat":["sub-satellite latitude", ["f","l","m","x"], "deg", "Geometry/SubObsLat"],

    "lsubs":["planetocentric longitude Ls", ["f","l"], "deg", "Geometry/LSubS"],

    "sub_sol_lon":["sub-solar longitude", ["f","l"], "deg", "Geometry/SubSolLon"],
    "sub_sol_lat":["sub-solar latitude", ["f","l"], "deg", "Geometry/SubSolLat"],
        }




nadir_geometry_dict = {
    "lon":["surface longitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lon"],
    "lat":["surface latitude at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/Lat"],
    "lst":["surface local solar time in hours at centre of field of view", ["f","l"], "hours", "Geometry/PointN/LST"],

    "sun_sza":["surface solar zenith angle at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/SunSZA"],
    "incidence_angle":["surface solar incidence angle at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/IncidenceAngle"],
    "emission_angle":["surface emission angle at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/EmissionAngle"],
    "phase_angle":["surface phase angle at centre of field of view", ["f","l","m","x"], "deg", "Geometry/PointN/PhaseAngle"],


    "sub_obs_lon":["sub-satellite longitude", ["f","l","m","x"], "deg", "Geometry/SubObsLon"],
    "sub_obs_lat":["sub-satellite latitude", ["f","l","m","x"], "deg", "Geometry/SubObsLat"],

    "lsubs":["planetocentric longitude Ls", ["f","l"], "deg", "Geometry/LSubS"],

    "sub_sol_lon":["sub-solar longitude", ["f","l"], "deg", "Geometry/SubSolLon"],
    "sub_sol_lat":["sub-solar latitude", ["f","l"], "deg", "Geometry/SubSolLat"],
        }


geom_times = {"f":["first","First"], "l":["last","Last"], "m":["min","Minimum"], "x":["max","Maximum"]}


h = "xml code\n"

for dictionary in [solar_geometry_dict, nadir_geometry_dict]:
    h += "####################################\n"
    metadataVariables = {}
    c = "tableVariables\n"

    for geom_type in dictionary.keys():
        desc = dictionary[geom_type][0]
        
        hdf5_field_name = dictionary[geom_type][3]
        
        #write xml
        for geom_letter in dictionary[geom_type][1]:
            geom_time_short = geom_times[geom_letter][0]
            geom_time_full = geom_times[geom_letter][1]
            h += f"                <nmd:{geom_time_short}_{geom_type}_description>{geom_time_full} {desc} of the observation</nmd:{geom_time_short}_{geom_type}_description>\n"
            h += f"                <nmd:{geom_time_short}_{geom_type}>%{geom_time_short}_{geom_type}</nmd:{geom_time_short}_{geom_type}>\n"

        #then point data
        if "PointN" in dictionary[geom_type][3]:
            hdf5_field_point_names = [
                    dictionary[geom_type][3].replace("PointN","Point0"),
                    dictionary[geom_type][3].replace("PointN","Point1"),
                    dictionary[geom_type][3].replace("PointN","Point2"),
                    dictionary[geom_type][3].replace("PointN","Point3"),
                    dictionary[geom_type][3].replace("PointN","Point4"),
                    ]
        else:
            hdf5_field_point_names = [hdf5_field_name]


        for hdf5_field_point_name in hdf5_field_point_names:
            for geom_letter in dictionary[geom_type][1]:
                geom_time_short = geom_times[geom_letter][0]
                geom_time_full = geom_times[geom_letter][1]
                
                

                c += f"[\"{geom_time_short}_{geom_type}\", \"ASCII_Real\", 9, \"deg\", \"\", \"description\", hdf5FileIn[\"{hdf5_field_point_name}\"][:,0]],\n"
#                print(f"{geom_time_short}_{geom_type} - {hdf5_field_point_name}")
                
#                if geom_letter == "f":
#                    metadataVariables[f"{geom_time_short}_{geom_type}"] = hdf5FileIn[hdf5_field_point_name][0,0]
#                if geom_letter == "l":
#                    metadataVariables[f"{geom_time_short}_{geom_type}"] = hdf5FileIn[hdf5_field_point_name][-1,-1]
#                if geom_letter == "m":
#                    metadataVariables[f"{geom_time_short}_{geom_type}"] = np.min(hdf5FileIn[hdf5_field_point_name][...])
#                if geom_letter == "x":
#                    metadataVariables[f"{geom_time_short}_{geom_type}"] = np.max(hdf5FileIn[hdf5_field_point_name][...])







        h += "\n"
        c += "\n"

#    print("####################")
#    print(h)

#    metadataVariables["First%s" %psaFieldName] = hdf5FileIn[psaMappings[psaFieldName]][0,0]
    #["EndSubObsLat","ASCII_Real",9,"deg","","Ending sub-satellite latitude",hdf5FileIn["Geometry/SubObsLat"][:,1]],



